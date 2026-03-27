# 01 - CUDA 核心概念：Warp、Shared Memory 与 Bank Conflict

在编写高性能 CUDA 算子之前，必须深刻理解 GPU 的硬件执行模型和内存层级。

## 1. Warp 调度机制 (Warp Scheduling)

* **什么是 Warp？** Warp 是 GPU 执行的基本单位。在 NVIDIA GPU 中，一个 Warp 包含 32 个线程 (Threads)。这 32 个线程在同一个物理核心 (SM) 上以 **SIMT (单指令多线程)** 的方式执行相同的指令，但处理不同的数据。
* **Warp Divergence (分支发散)：**
  如果一个 Warp 内的 32 个线程遇到 `if-else` 分支，且部分线程走向 `if`，部分走向 `else`，GPU 无法同时执行两个分支。它会先执行 `if` 分支（挂起 `else` 线程），再执行 `else` 分支（挂起 `if` 线程）。这会导致性能减半甚至更低。
  * **优化原则：** 尽量保证同一个 Warp 内的线程执行路径一致，避免条件分支依赖于线程 ID (`threadIdx.x`)。

## 2. Shared Memory (共享内存) 深度剖析

* **层级位置：** 位于 SM (流多处理器) 内部，类似于 CPU 的 L1 Cache，但由开发者手动管理。速度比 Global Memory (HBM) 快上百倍。
* **作用：** 作为 Block 内线程通信的桥梁，也是缓存重复读取数据的“中转站”。
* **生命周期：** 与 Thread Block 绑定。Block 启动时分配，Block 结束时销毁。
* **声明方式：** `__shared__ float s_data[256];`

## 3. Bank Conflict (存储体冲突) 规避

* **什么是 Bank？** Shared Memory 被划分为 32 个大小相等的内存块，称为 Banks。设计目的是允许 32 个线程 (一个 Warp) 同时访问 Shared Memory 时，每个线程访问不同的 Bank，从而实现极高的内存带宽。
* **什么是 Conflict？** 当一个 Warp 内的多个线程尝试访问**同一个 Bank 中的不同地址**时，请求会被序列化（排队处理），这就是 Bank Conflict，会导致访存延迟成倍增加。
* **规避策略 (Padding)：**
  常见的冲突场景是按列读取二维数组。通过在数组末尾增加一个冗余的填充位 (Padding)，可以错开不同行的内存地址，打破冲突。
  * *冲突定义*：`__shared__ float tile[32][32];`
  * *解决思路*：`__shared__ float tile[32][33];` （Padding 技术）