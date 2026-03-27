# 02 - 手撕算子：GEMM 矩阵乘法从 Naive 到 Tiling

矩阵乘法 $C = A \times B$ 是深度学习最基础的算子。假设矩阵 A 大小为 $M \times K$，B 大小为 $K \times N$。

## 1. Naive GEMM (朴素实现)

* **核心思想：** 每个线程计算矩阵 C 中的一个元素 $C_{i,j}$。
* **过程：** 线程 $(i, j)$ 需要读取 A 的第 $i$ 行 (共 K 个元素) 和 B 的第 $j$ 列 (共 K 个元素)，做内积。
* **性能瓶颈：** 对 Global Memory 的访问极其昂贵。A 的同一行被读取了 N 次，B 的同一列被读取了 M 次。内存带宽成为巨大的瓶颈 (Memory Bound)。

## 2. Tiling GEMM (分块优化)

* **核心思想：** 利用 Shared Memory 减少 Global Memory 的重复读取。把大矩阵划分为小块 (Tiles)，比如 $32 \times 32$。
* **执行步骤：**
  1. **协同加载 (Load)：** Block 内的线程协作，将 A 和 B 当前对应的小块 (Tile) 从 Global Memory 加载到 Shared Memory 中。
  2. **同步 (Sync)：** 调用 `__syncthreads()`，确保 Block 内所有线程都完成了加载。
  3. **计算 (Compute)：** 线程从极快的 Shared Memory 中读取数据，计算当前子块的部分内积。
  4. **同步 (Sync)：** 再次 `__syncthreads()`，确保所有线程计算完毕，准备加载下一个子块。
  5. **循环滑动：** 沿着 K 维度滑动，重复上述步骤，累加结果。

* **性能提升：** 如果分块大小为 $32 \times 32$，Global Memory 的访存量将直接下降约 32 倍！这就是 Tiling 技术的魅力。