# 03 - 手写 Reduce：归约算法与 Warp Shuffle

归约操作 (Reduce) 如 Sum、Max 是 Softmax 和 LayerNorm 等复杂算子的基础。

## 1. 传统 Reduce (基于 Shared Memory 的树状归约)

* **核心思想：** 两两相加/比较。
* **过程：**
  将数组加载到 Shared Memory 中。第一轮：步长为 1，相邻元素相加；第二轮：步长为 2... 直到得出最终结果。
* **缺点：** 需要使用 Shared Memory，且伴随着频繁的 `__syncthreads()` 同步开销，以及可能引发的 Warp Divergence。

## 2. 现代极速优化：Warp Shuffle 指令

* **什么是 Warp Shuffle？** 从 Kepler 架构开始引入的硬件级指令。允许**同一个 Warp 内的 32 个线程直接读取彼此的寄存器数据**，完全不需要经过 Shared Memory，也不需要 `__syncthreads()`。
* **核心指令 `__shfl_down_sync`：**
```python
  // 伪代码：向高 ID 线程借数据
  float val = ...;
  for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
  }
  // 最终，线程 0 的 val 将包含整个 Warp 的和。
```


**实现 Block 级 Reduce：**

1. 每个 Warp 使用 Shuffle 指令求出 Warp 内部的局部和。
    
2. 每个 Warp 的 0 号线程将局部和写入 Shared Memory (最多 32 个值，如果 Block 是 1024 线程)。
    
3. 第一个 Warp 读取这 32 个局部和，再次进行一次 Warp Shuffle，得出整个 Block 的最终结果。