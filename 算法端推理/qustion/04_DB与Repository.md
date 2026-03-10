# 04 DB 与 Repository（db 模块）

本章解释数据库连接池、Repository 抽象、任务“原子领取”机制、以及任务执行时的状态迁移与乐观锁。

  

## 1. 数据库连接池：`AsyncDatabaseConnectionManager`

连接管理在：

- [db/connection.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/connection.py#L19-L156)

  

### 1.0 读代码时的几个语法点

- `self`：实例方法的第一个参数，代表“当前对象实例”。例如 `__init__` 里 `self._pool` 就是这个连接管理器实例自己的属性：[db/connection.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/connection.py#L19-L27)

- `db_manager`：模块级创建的 `AsyncDatabaseConnectionManager()` 实例，是一个对象引用，可以被 `import` 到其它模块里使用：[db/connection.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/connection.py#L155-L156)

- 多进程 `spawn`：每个进程都会重新 import 模块，所以“每个进程各自有一个 db_manager 实例”（不是跨进程共享同一个对象）。

  

### 1.1 初始化流程

`initialize()` 负责创建 asyncpg 连接池并做一次 `SELECT version()` 验证：

- [db/connection.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/connection.py#L28-L71)

  

关键参数（对性能/稳定性影响大）：

- `min_size/max_size`：连接池大小

- `command_timeout`：单条 SQL 的超时

- `statement_cache_size=0`：禁用预处理缓存（用于兼容某些事务代理/连接池模式）

  

### 1.2 使用方式：连接上下文与事务上下文

两种上下文管理器：

- `get_connection()`：拿到连接，自动 acquire/release

- `transaction()`：在连接上开启事务

  

位置：

- [db/connection.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/connection.py#L83-L121)

  

Repository 层通常用：

- 读操作：`async with db_manager.get_connection() as conn: ...`

- 写操作：`async with db_manager.transaction() as conn: ...`

  

补充：为什么 `async with ... as conn` 能自动释放/提交？

- `get_connection()` 和 `transaction()` 使用了 `@asynccontextmanager`：`yield conn` 之前相当于进入（借连接/开事务），`yield conn` 把 `conn` 交给 `as conn`，代码块结束后自动退出（归还连接；若是事务则 commit/rollback）。对应实现：[db/connection.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/connection.py#L83-L121)

  

## 2. Repository 抽象：BaseRepository

基础抽象在：

- [db/repositories/base_repository.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/repositories/base_repository.py#L23-L278)

  

### 2.1 设计意图

它提供通用 CRUD（find_by_id/find_all/find_by_conditions/count/update...），并要求子类提供：

- `primary_key`：表主键字段

- `_to_model(data)`：把 dict 转成强类型模型对象

  

对应抽象方法位置：

- [base_repository.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/repositories/base_repository.py#L56-L75)

  

### 2.2 update() 与“乐观锁”

`update(id, data, expected_version=...)` 的关键点：

- 若 `expected_version` 不为空，会在 WHERE 子句加上 `version = $X`

- 若版本不匹配，更新将返回 `None`（表示冲突/不存在）

  

对应逻辑：

- [base_repository.py:update](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/repositories/base_repository.py#L208-L278)

  

这就是任务框架中“抢占同一任务时只有一个执行者能成功进入 processing”的基础能力。

  

## 3. 任务仓储基类：BaseTaskRepository 与原子领取

任务仓储基类：

- [db/repositories/base_task_repository.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/repositories/base_task_repository.py#L19-L80)

  

### 3.1 `claim_pending(limit)`：原子领取 pending → queued

核心 SQL 在同一函数中生成：

- 使用 CTE 先 `SELECT task_id FROM ... WHERE status='pending' ... FOR UPDATE SKIP LOCKED`

- 再 `UPDATE ... SET status='queued' FROM cte WHERE t.task_id = cte.task_id RETURNING t.*`

  

位置：

- [base_task_repository.py:claim_pending](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/repositories/base_task_repository.py#L29-L80)

  

这个模式的效果：

- 多实例并发领取时不会互相阻塞（`SKIP LOCKED`）

- 同一条任务不会被重复领取（行锁 + 状态变更）

  

## 4. Image_cell 任务仓储：ImageCellInferenceTaskRepository

实现：

- [db/repositories/image_cell_inference_task_repository.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/repositories/image_cell_inference_task_repository.py#L16-L37)

  

它继承 BaseTaskRepository，因此天然具备：

- `claim_pending()` 原子领取能力

- 通用 CRUD（继承 BaseRepository）

  

额外提供了一些查询/更新方法：

例如 `find_by_inference_result_id`：

- [image_cell_inference_task_repository.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/repositories/image_cell_inference_task_repository.py#L38-L52)

  

## 5. 任务状态机：从“抢到执行权”到“成功/失败落库”

真正的状态迁移不是在业务执行器里手写的，而是在 `BaseTaskContext` 里统一处理：

- [task_core/execution/context.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/execution/context.py#L71-L199)

  

这里的“状态迁移”指数据库任务记录的 `status` 字段在生命周期中的变化，典型包括：

- `pending`（待领取）→ `queued`（已领取/排队）→ `processing`（执行中）→ `completed`（成功）或 `failed`（失败）

  

### 5.1 进入上下文：processing（带乐观锁）

`__aenter__`：

- `repository.update(task_id, {'status':'processing'}, expected_version=task.version)`

- 若返回 `None`：抛 `TaskCancelledException`，表示任务已被其他进程拿走

  

位置：

- [context.py:__aenter__](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/execution/context.py#L71-L108)

  

### 5.2 退出上下文：completed / failed（统一落库）

`__aexit__`：

- 无异常：更新为 `completed`，并在有 progress 字段时置 100

- 异常：更新为 `failed` 并写入 `error_message`（通常也带 expected_version）

- 最终：调用资源清理钩子 + 从 TaskManager 移除，防止内存泄漏

  

位置：

- [context.py:__aexit__](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/execution/context.py#L110-L199)

  

你可以把 `BaseTaskContext` 看作“任务的事务边界”：它把状态更新、异常分类、资源清理、内存移除统一起来。

  

## 6. 本章“代码流转路径”小结

1. 连接池初始化与事务： [db/connection.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/connection.py#L28-L121)

2. Repository update 乐观锁： [base_repository.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/repositories/base_repository.py#L208-L278)

3. 任务原子领取 pending： [base_task_repository.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/db/repositories/base_task_repository.py#L29-L80)

4. 执行期状态机： [task_core/execution/context.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/execution/context.py#L71-L199)