# 06 task_core 任务机制（扫描 / 派发 / 执行 / 状态机）
本章从“通用框架”的角度解释 `task_core/`：它并不关心 Image_cell 的业务细节，只负责把任务从数据库“变成一次可控的执行”。

## 1. 核心对象关系图（概念）
```
BaseOrchestrator
  ├─ scanner: BaseTaskScanCoordinator
  │    ├─ scanners: [BaseScanner...]  (通知驱动/定时)
  │    └─ executor: BaseScanExecutor  (perform_scan 从 DB 领取)
  ├─ dispatch_strategy: DispatchStrategy (Queue/Semaphore)
  └─ executor: BaseExecutor (execute_task 执行业务)
```

关键点：
- **scanner** 产出的是 `task_ids` 批次（列表）
- **dispatch_strategy** 决定如何调度这些 `task_ids`（队列串行/并发）
- **executor** 执行单个任务，并通过 **BaseTaskContext** 统一做状态迁移与清理

## 2. 扫描协调器：BaseTaskScanCoordinator
实现：
- [task_core/scanning/coordinator/base.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/scanning/coordinator/base.py#L22-L178)

### 2.1 scan_stream()：统一“批次扫描流”
入口：
- [coordinator/base.py:scan_stream](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/scanning/coordinator/base.py#L127-L171)

核心循环（按代码顺序）：
1. `await _wait_for_trigger()`：等待触发事件并清除标记  
   - [coordinator/base.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/scanning/coordinator/base.py#L91-L95)
2. `_determine_limit()`：结合 `scan_limit` 与 `capacity_provider()` 计算本轮领取上限  
   - [coordinator/base.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/scanning/coordinator/base.py#L96-L114)
3. `executor.perform_scan(limit)`：执行一次批量扫描（通常是 DB 原子领取）  
   - 约束接口：[scanning/executor/base.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/scanning/executor/base.py)
   - 通用实现：[repository_executor.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/scanning/executor/repository_executor.py)
4. `yield batch`：产出给 orchestrator 主循环消费

### 2.2 capacity_provider：把“背压”传播回扫描
`capacity_provider()` 的作用是：当下游繁忙（队列满、任务创建并发达到上限）时，返回 0 让扫描跳过，避免继续从 DB 领取更多任务。

在 Image_cell 中，该能力会被用于构建完整反压链路（详见第 7 章）。

## 3. 编排器：BaseOrchestrator
实现：
- [task_core/orchestrator.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/orchestrator.py#L19-L473)

### 3.1 start()：启动顺序
- 启动监听器（可选）
- 启动 scanner
- 启动 executor（可选生命周期）
- 创建主循环任务 `_run_loop()`
- 若策略支持空闲事件，则启动 `_monitor_dispatch_idle()`

对应代码：
- [orchestrator.py:start](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/orchestrator.py#L88-L115)

### 3.2 _run_loop()：消费扫描批次并派发
主循环：
- [orchestrator.py:_run_loop](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/orchestrator.py#L413-L473)

逻辑（简化）：
1. `async for task_ids in scanner.scan_stream():`
2. `await dispatch_strategy.dispatch(task_ids)`
3. 错误处理：  
   - dispatch 错误：走 error_handler，决定是否退避/继续  
   - scan_stream 错误：支持重启 scanner（带超时保护）

### 3.3 stop_after_current()：排空停止
排空停止的意义是：停止继续扫描/派发新任务，但尽量让当前任务执行完再退出。
- [orchestrator.py:stop_after_current](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/orchestrator.py#L168-L227)

## 4. 派发策略：QueueDispatchStrategy / SemaphoreDispatchStrategy
实现：
- [task_core/dispatch/strategies.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/dispatch/strategies.py#L106-L365)

### 4.1 QueueDispatchStrategy（长任务更常用）
特征：
- `dispatch()` 将 task_id 放入 asyncio.Queue（可能阻塞，形成反压）
- 消费者 `_consumer_loop()` 持续从队列取 task_id，创建执行任务 `_run_task()`
- `drain_and_stop()` 支持排空队列并等待执行完成

关键代码：
- [strategies.py:QueueDispatchStrategy](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/dispatch/strategies.py#L196-L365)

#### 4.1.1 提前切换（Early Switch）
当启用 early_switch：
- consumer 会同时等待 “当前任务完成” 或 “early_switch_event”
- 若提前切换触发，则允许下一个 task_id 更早开始（减少流水线气泡）

代码：
- [strategies.py:_wait_for_completion_or_switch](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/dispatch/strategies.py#L275-L315)

### 4.2 SemaphoreDispatchStrategy（短任务并发）
特征：
- 用 `asyncio.Semaphore` 控制并发上限
- 对每个 task_id 创建一个 `_run_single()` 任务

代码：
- [strategies.py:SemaphoreDispatchStrategy](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/dispatch/strategies.py#L106-L195)

## 5. 执行器：BaseExecutor
实现：
- [task_core/execution/executor.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/execution/executor.py#L25-L88)

`execute_task(task_id)` 的关键流程：
1. 从 TaskManager 取任务对象（内存态）
2. `async with self._create_context(task) as ctx:` 进入上下文（写 processing）
3. 调用子类 `_execute_logic(ctx.task)` 执行业务逻辑
4. 上下文退出时自动写 completed/failed，并从 TaskManager 移除

## 6. 任务上下文：BaseTaskContext（状态机 + 清理）
实现：
- [task_core/execution/context.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/execution/context.py#L34-L246)

它统一解决了三个系统问题：
1. 并发安全：通过 `expected_version` 乐观锁抢执行权  
   - [context.py:__aenter__](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/execution/context.py#L71-L108)
2. 统一落库：无异常 → completed；有异常 → failed（含 error_message）  
   - [context.py:__aexit__](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/execution/context.py#L110-L199)
3. 资源回收：调用清理钩子 + 从内存移除，避免泄漏  
   - [context.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/execution/context.py#L186-L197)

## 7. 本章“代码流转路径”小结
当你想理解“一个 task_id 从 DB 到执行完”的最短路径，可以按下面顺序阅读：
1. 扫描批次流： [coordinator/base.py:scan_stream](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/scanning/coordinator/base.py#L127-L171)
2. 编排器主循环： [orchestrator.py:_run_loop](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/orchestrator.py#L413-L473)
3. 派发策略： [dispatch/strategies.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/dispatch/strategies.py#L196-L365)
4. 任务执行： [execution/executor.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/execution/executor.py#L54-L88)
5. 状态机/清理： [execution/context.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/execution/context.py#L71-L199)

