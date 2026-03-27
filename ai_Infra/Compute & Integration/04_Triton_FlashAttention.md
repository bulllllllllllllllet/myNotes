# 04 - OpenAI Triton：Block 级编程与 FlashAttention

写纯 CUDA C++ 门槛高、容易出错。OpenAI Triton 改变了游戏规则。

## 1. Triton 编程思想 (弯道超车)

* **从 Thread 到 Block：** CUDA 关注的是“一个线程做什么”（SIMT）。Triton 关注的是“一个 Block 处理哪一块张量”。
* **自动优化：** 在 Triton 中，你直接对指针块 (Pointers to Blocks) 进行操作，所有的 Shared Memory 分配、Warp 同步、内存对齐，Triton 编译器都会自动帮你搞定。
* **语法：** 使用纯 Python 编写，通过 `@triton.jit` 装饰器编译为高效的 GPU 机器码。

## 2. 手写 FlashAttention 前向传播 (概念验证)

FlashAttention 的核心突破是 **Memory-Aware (内存感知)**，目标是干掉标准的 Self-Attention 中庞大的中间矩阵 (如 $QK^T$ 的 $N \times N$ 矩阵) 对 HBM (显存) 的读写。

**Triton 实现步骤思路：**
1. **分块 (Tiling)：** 将巨大的 Q, K, V 矩阵按照硬件 SRAM (Shared Memory) 的大小进行分块。
2. **外层循环加载 K, V：** 遍历 K 和 V 的块。
3. **内层循环加载 Q：** 遍历 Q 的块。
4. **SRAM 内计算：** 在 SRAM 中计算 `S = Q_block @ K_block.T`。
5. **Online Softmax：** 使用在线 Softmax 技巧，维护局部的最大值 (`m`) 和指数和 (`l`)，边计算边更新，不把庞大的 S 矩阵写回显存。
6. **计算 Output：** `O_block += P_block @ V_block`。最后将结果写回 HBM。

*通过 Triton 的 `tl.dot` 和 `tl.load/store`，你可以用几十行 Python 代码实现比 PyTorch 原生快几倍的 Attention。*