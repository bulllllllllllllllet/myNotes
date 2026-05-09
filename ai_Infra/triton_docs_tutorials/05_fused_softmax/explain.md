# Triton Fused Softmax 详解

> 对 `fused_softmax.py` 中 Triton 融合 Softmax 实现的核心概念进行视觉化解读。

---

## 一、核心概念：为什么要"融合（Fuse）"？

朴素（Naive）Softmax 虽然在 Python 中只有几行代码，但在 GPU 硬件层面会导致频繁且昂贵的 DRAM 数据传输。

![[Generated Image May 09, 2026 - 10_50PM.jpg]]

### 左图：朴素实现（Unfused）

由多个独立的子操作组成——算最大值、做减法、做 Exp、求和、做除法。每一步操作（K1, K2, K3）都必须从慢速的 **GPU DRAM** 中读取输入，并将中间结果写回 DRAM。红色粗箭头代表高延迟的 DRAM 访问，数据不停地在 DRAM 和计算核心之间"往返"（共 `8MN + 4M` 次内存操作），DRAM 成为性能瓶颈。

### 右图：Triton 融合 Kernel（Fused）

所有操作合并进**一个 Kernel**。每个并行实例只从 DRAM 读取**一次**输入行数据，存入快速的 **GPU SRAM**。所有数学计算（Max、Sub、Exp、Sum、Div）都在 SRAM 内部高效完成。绿色箭头代表低延迟的片内访问。最终结果才写回 DRAM，访问次数降至 `2MN`，理论上可带来 **4 倍**提速。

---

## 二、Triton Kernel 内部做了什么？

Triton 的核心理念是让程序员以"块（Block）"为单位编写代码。下图展示了 `_softmax_kernel` 处理一行数据的完整流程：

![[Generated Image May 09, 2026 - 10_52PM.jpg]]

### 阶段 1：数据加载与掩码（LOAD & MASK）

Triton 要求 Block 大小为 2 的幂（如 `BLOCK_SIZE=1024`），但实际列数可能不足（如 `n_cols=781`）。`tl.load` 使用**掩码（mask）**区分有效/无效位置。

- 越界位置填充为 **`-inf`**（负无穷大）而非 `0`
- 为什么用 `-inf`？如果填充 `0`，当一行全为负数时，`max` 会错误取到 `0`，导致计算偏移。`-inf` 确保无效位置在 `max` 和 `sum` 中被完全忽略（`max(-inf, x) = x`，`exp(-inf) = 0`）

### 阶段 2：安全最大值与减法（SAFE MAX & SUB）

在 SRAM 内部 `tl.max` 快速算出该行最大值，执行 `row - max`。所有结果 $\le 0$，保证了数值稳定性（详见第四节）。

### 阶段 3：指数运算与求和（EXP & SUM）

- `tl.exp()` 是 Triton 的优化指令，对应 GPU 硬件层更快的近似指数计算（Fast Math），用微小精度损耗换取更高吞吐量
- 原来填充 `-inf` 的位置变为 `0.0`，完全消失在后续计算中
- `tl.sum` 算出分母

### 阶段 4：除法与回写（DIV & STORE）

- 执行除法得到最终 Softmax 结果行
- `tl.store` 再次使用相同**掩码**，只将有效的 `n_cols` 列写回 DRAM，避免破坏无关内存数据

---

## 三、性能优化：获取硬件参数与 Occupancy

Triton 允许在 Python 中获取 GPU 硬件规格，据此调整 Kernel 的元参数以最大化**占有率（Occupancy）**。

![[Generated Image May 09, 2026 - 10_49PM.jpg]]

### 阶段 1：输入与启发式算法（INPUT & HEURISTICS）

检查 GPU 硬件属性（如 `TOTAL_SRAM_PER_SM`），应用启发式算法静态决定 `num_warps` 和 `num_stages` 初始值（见代码中的 `if n_cols >= 2048...` 分支）。

### 阶段 2：Kernel 预热（KERNEL WARMUP）

调用 `.warmup()` —— Triton 编译 Kernel 并返回**性能报告（metadata）**，包括实际消耗的寄存器数（`n_regs`）和 SRAM 用量（`shared`）。

### 阶段 3：占有率计算（OCCUPANCY CALCULATION）

受两个硬件极限约束：

- **寄存器限制**：`reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)`。每 SM 寄存器总量固定（如 65536），Kernel 每线程用越多寄存器，可并行的 Program 越少
- **SRAM 限制**：`sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program`。每 SM 的 SRAM 固定（如 128KB），BLOCK_SIZE 越大，SRAM 消耗越多

**实际并行数** `programs_per_sm = min(reg_occupancy, sram_occupancy)` —— 受两者中较严格的约束。

### 阶段 4：Grid 启动（GRID LAUNCH）

`num_programs = min(NUM_SM * programs_per_sm, n_rows)` —— 充分利用所有 SM 又不超出矩阵行数。最终用优化后的 `grid` 配置启动 Kernel。

---

## 四、`row - max` 保证数值稳定性的原理

### 核心原理：平移不变性

Softmax 原始定义：
$$Softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}}$$

对任意常数 $C$：
$$
\frac{e^{x_i - C}}{\sum_{j=1}^N e^{x_j - C}}
= \frac{e^{x_i} \cdot e^{-C}}{e^{-C} \cdot \sum_{j=1}^N e^{x_j}}
= \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}}
$$

### 问题的根源：指数函数 $e^x$

- **指数爆炸（Overflow）**：$x_i = 1000$ 时，$e^{1000}$ 远超 float32 最大值（约 $3.4 \times 10^{38}$），结果为 `inf`，`inf / inf = NaN`
- **下溢（Underflow）**：所有元素都很小时，分子分母全为 0

### "减去最大值"的妙处

令 $C = \max(x)$，则该行所有元素：
- $x_i - \max(x) \le 0$，最大值为 $0$
- $e^{x_i - \max(x)}$ 最大值是 $e^0 = 1$
- 所有分子项被限制在 $(0, 1]$ 范围内，**彻底避免 $e^{1000}$ 式指数爆炸**
- 至少有一项为 1，分母 $\ge 1$，**避免分母为 0**

---

## 五、三张图总结

| 图表 | 核心问题 | 回答 |
|------|----------|------|
| **图像 1** | 为什么要融合？ | 减少昂贵的 DRAM 访问，避免数据在 DRAM 和计算核心间反复往返 |
| **图像 2** | 如何编写融合 Kernel？ | 使用 Triton Block 编程模型，配合 Mask 技术处理非 2 的幂次数据 |
| **图像 3** | 如何让 Kernel 跑最快？ | 利用硬件参数和 Warmup 预热，动态计算 Occupancy 以优化资源利用率 |

Triton 在执行前先"侦察"硬件环境和代码需求（Warmup），计算出当前 GPU 的最优负载（Occupancy），然后按最优配置启动 Kernel。这种**动态规划式**启动让 Triton Kernel 能在不同规格 GPU（如 A100 vs H100）上都保持高性能。
