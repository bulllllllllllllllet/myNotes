# 第 1 个月：算子开发与工程接入 (Compute & Integration)

**学习目标：** 不仅要掌握底层算子开发（CUDA/Triton），还要打通工程落地的“最后一公里”，将其封装为高效、易用的 Python 库。

## 学习目录

请按照以下顺序阅读和实践学习文档：

### 阶段一：CUDA 基础与核心概念
* [📄 01_CUDA核心概念](./01_CUDA_Core_Concepts.md)
  * 内容：Warp 调度机制、Shared Memory (共享内存) 深度剖析、Bank Conflict (存储体冲突) 规避指南。

### 阶段二：CUDA 算子手撕与优化
* [📄 02_CUDA_GEMM_Optimization.md](./02_CUDA_GEMM_Optimization.md)
  * 内容：矩阵乘法 (GEMM) 从 Naive (朴素实现) 到 Tiling (分块优化) 的完整推演。
* [📄 03_CUDA_Reduce_WarpShuffle.md](./03_CUDA_Reduce_WarpShuffle.md)
  * 内容：归约算法 (Sum/Max) 的实现，以及利用 Warp Shuffle 指令进行极速线程间通信。

### 阶段三：OpenAI Triton 弯道超车
* [📄 04_Triton_FlashAttention.md](./04_Triton_FlashAttention.md)
  * 内容：Triton 的 Block 级编程思想，以及如何手写一个简易版的前向 FlashAttention。

### 阶段四：工程化接入与性能验证
* [📄 05_PyTorch_Extension.md](./05_PyTorch_Extension.md)
  * 内容：使用 Pybind11 和 PyTorch C++ Extension 封装算子，并进行数值正确性验证 (Allclose) 与耗时对比测试 (Benchmark)。