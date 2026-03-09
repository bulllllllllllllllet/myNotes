# 07 Image_cell 扫描与任务创建（scanner / preprocessor / task_manager / orchestrator）
本章进入 Image_cell 业务侧：解释系统如何从数据库领取任务、并发创建 Task 对象、通过反压控制吞吐、并把 task_id 派发到执行器。

## 1. Image_cell 编排器：TaskOrchestrator 组合关系
实现：
- [image_cell_processing/orchestration/orchestrator.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/orchestration/orchestrator.py#L31-L127)

它在构造函数里完成“业务组装”：
1. 创建业务 TaskManager（内存任务存储）
2. 创建 TaskScanner（扫描 + 任务创建流水线）
3. 创建 TaskExecutor（真正执行推理）
4. 创建 QueueDispatchStrategy（派发策略）
5. （可选）把 PipelineService 的 early_switch_event 交给派发策略
6. 配置 DB 监听器（超时清理 / 删除通知）
7. 调用 `BaseOrchestrator.__init__` 完成基类装配

对应代码位置：
- TaskManager： [orchestrator.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/orchestration/orchestrator.py#L56-L58)
- TaskScanner： [orchestrator.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/orchestration/orchestrator.py#L72-L82)
- TaskExecutor： [orchestrator.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/orchestration/orchestrator.py#L84-L85)
- QueueDispatchStrategy： [orchestrator.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/orchestration/orchestrator.py#L86-L91)

## 2. TaskScanner：扫描协调器 + 任务创建器
实现：
- [image_cell_processing/scanner/scanner.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/scanner.py#L38-L167)

它继承自 `BaseTaskScanCoordinator[int]`，对外产出的是 `task_id` 列表（批次）。

### 2.1 触发器：通知驱动 + 定时兜底
构造函数里创建 `scanners`：
- NotificationDrivenScanner：监听 `cell_inference_task_change` 频道
- ScheduledScanner：按 interval 触发扫描（兜底）

位置：
- [scanner.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/scanner.py#L85-L96)

### 2.2 扫描执行器：RepositoryScanExecutor（统一领取）
默认使用：
- [task_core/scanning/executor/repository_executor.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/scanning/executor/repository_executor.py)

它内部会调用 repository 的 `claim_pending(limit)`（原子领取）。

### 2.3 关键：capacity_provider 把“队列容量”变成“扫描上限”
TaskScanner 使用了一个 `asyncio.Queue(maxsize=queue_size)` 作为“DB任务入队 → 任务创建器消费”的缓冲区：
- [scanner.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/scanner.py#L77-L83)

capacity_provider 的计算方式（本质是“剩余可用容量”）：
- [scanner.py:_capacity_provider](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/scanner.py#L97-L103)

效果：
- 当队列接近满时，capacity 变小甚至为 0
- `BaseTaskScanCoordinator._determine_limit()` 发现 capacity<=0 → 跳过本轮扫描  
  - [task_core/scanning/coordinator/base.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/scanning/coordinator/base.py#L96-L113)

这就是“反压链路”的第一段：**队列满 → 不再从 DB 领取更多任务**。

## 3. TaskPreprocessor：真正的任务创建与资源准备（并发受控）
实现：
- [scanner/preprocessor.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/preprocessor.py#L30-L273)

它通过 `asyncio.Semaphore(max_concurrent_creations)` 控制并发：
- [preprocessor.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/preprocessor.py#L43-L75)

### 3.1 为什么说它实现了“真正的反压”？
调度器循环 `_task_scheduler()` 的关键策略：
1. 先 `await semaphore.acquire()`（达到上限会阻塞）
2. 再 `db_task = await task_queue.get()`（队列空会阻塞）
3. 再创建 `asyncio.create_task(self._run_task_wrapper(db_task))` 并发执行

位置：
- [preprocessor.py:_task_scheduler](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/preprocessor.py#L114-L169)

这意味着：
- 当并发创建达到上限时，它会停止从队列取新任务
- 队列逐渐被填满后，反压会进一步传递回 scanner 的 capacity_provider

### 3.2 资源准备：ResourcePreparer
创建任务前会做资源准备：
- MinIO URL → 预下载到临时目录
- 非 MinIO → 可配置是否预下载

实现：
- [resource_preparer.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/utils/resource_preparer.py#L11-L131)

在 `_create_task()` 中使用：
- [preprocessor.py:_create_task](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/preprocessor.py#L187-L213)

返回值：
- `resolved_source_url`：最终可读的本地路径（后续推理只读这个）
- `predownloaded_files`：后续清理用的临时文件列表

### 3.3 TaskFactory：把 DB 模型变成“运行时 Task 对象”
实现：
- [scanner/factory.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/factory.py#L15-L106)

它创建的是：
- [image_cell_processing/task.py:Task](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/task.py#L35-L97)

Task 的关键字段：
- `db_task`：数据库模型（组合）
- `resolved_source_url`：本地文件路径（推理入口）
- `predownloaded_files`：临时文件列表（用于清理）

## 4. TaskManager：内存态任务存储 + “创建完成事件”
实现：
- [image_cell_processing/task_manager.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/task_manager.py#L18-L145)

关键设计：
- `add_task()` 成功后会触发 `creation_event.set()`  
  - [task_manager.py:add_task](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/task_manager.py#L46-L62)
- `wait_for_task_creation_completion(task_id)` 会等待该事件  
  - [task_manager.py:wait_for_task_creation_completion](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/task_manager.py#L64-L75)

这让 scanner 能保证：**把 task_id 派发出去前，Task 对象已经在内存里可被 executor 获取**。

## 5. 任务从“扫描领取”到“可执行 task_id 批次”的关键连接点
TaskScanner 重写了 `_process_and_collect_batch(raw_records)`：
- [scanner.py:_process_and_collect_batch](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/scanner.py#L128-L157)

它做的事情（按代码顺序）：
1. 过滤掉 TaskManager 已存在的 task（健壮性）
2. 对每个 task_id：
   - `wait_for_task_creation_completion(task_id)`（等待创建完成事件）
   - `task_queue.put(db_task)`（把 DB 任务放进创建队列）
3. 返回 `ids_to_process`（该批次的 task_id 列表）

你会注意到这里的 `asyncio.gather(...)` 是“同时等待创建完成 + 入队 DB 任务”。这依赖 TaskPreprocessor 的调度器并发运行，从而实现批量吞吐。

## 6. 本章“代码流转路径”小结（从 DB 到 task_id 批次）
1. 编排器装配： [orchestration/orchestrator.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/orchestration/orchestrator.py#L40-L127)
2. 触发扫描：通知/定时 → [scanner/scanner.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/scanner.py#L85-L112)
3. DB 原子领取：repository_executor → base_task_repository.claim_pending  
4. DB 任务入队 + 等待创建完成： [scanner.py:_process_and_collect_batch](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/scanner.py#L128-L157)
5. 任务创建/预下载： [preprocessor.py:_create_task](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/scanner/preprocessor.py#L187-L273)
6. Task 进入内存： [task_manager.py:add_task](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/task_manager.py#L46-L62)
7. 批次 task_id 被 yield 给 orchestrator： [task_core/scanning/coordinator/base.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/task_core/scanning/coordinator/base.py#L127-L171)

