# AI Infra 面试与实战核心资料包

本资料汇总了 CUDA 算子优化、大模型架构演进以及 AI 背景下的系统常考点。

---

## 一、CUDA 算子优化 (算子手撕高频点)

### 1. 通用优化手段
- **Vectorized Memory Access (向量化访存):** 使用 `float4` 等类型一次读取 128 bit 数据，最大化带宽利用率。
- **Shared Memory Tiling (访存分块):** 将数据分块加载到 Shared Memory 提高复用，减少对 Global Memory 的访问。
- **Padding:** 在 Shared Memory 声明中增加一列（如 `[TILE][TILE+1]`）以规避 Bank Conflict。
- **Loop Unrolling:** 使用 `#pragma unroll` 减少循环开销并增加指令并行度。

### 2. 核心算子优化点
| 算子 | 优化关键词 | 关键挑战 |
| :--- | :--- | :--- |
| **GEMM** | Double Buffering, Register Tiling | 隐藏数据传输延迟，提高计算强度 |
| **Reduce** | Warp Shuffle (`__shfl_down_sync`) | 减少同步开销，利用寄存器通信 |
| **Softmax** | Online Softmax (2-pass) | 显存带宽受限，计算数值稳定性 |
| **Transpose** | Coalesced Access, Shared Memory | 解决读取连续而写入不连续的问题 |

---

## 二、大模型 (LLM) 架构演进

### 1. 注意力机制 (Attention)
- **MHA (Multi-Head Attention):** 原始形态，KV Cache 显存占用极大。
- **MQA (Multi-Query Attention):** 所有 Head 共享一对 KV，大幅减少显存，但可能损失精度。
- **GQA (Grouped Query Attention):** 分组共享 KV，如 Llama-3 所选用的折中方案。
- **MLA (Multi-head Latent Attention):** **DeepSeek V2/V3 核心**。通过低秩压缩 KV 矩阵，大幅降低推理时的显存占用，同时不牺牲表达能力。

### 2. 位置编码 (PE)
- **RoPE (Rotary Position Embedding):** 目前大模型标配。优点是具有远程衰减特性且支持长度外推。

### 3. MoE (混合专家模型)
- **DeepSeek 创新：** 引入 **Shared Experts** 处理共性知识，**Fine-grained Experts** 提升细粒度专家感知。
- **负载均衡：** 现代 MoE 趋向于 Aux-loss free 路由，避免辅助损失对模型性能的负面影响。

---

## 三、AI Infra 面试常考面试题 (FAQ)

### 1. 系统基础
- **Q: 如何判断一个算子是算力受限 (Compute-bound) 还是带宽受限 (Memory-bound)？**
  - **A:** 使用 Roofline Model。计算算子的运算强度（Operational Intensity = Ops / Bytes），对比硬件的峰值算力与带宽比值。
- **Q: 为什么 Transformer 的推理会变慢？KV Cache 是什么？**
  - **A:** 推理是 Autoregressive（自回归）过程。KV Cache 存储了之前所有 token 的键值对，避免重复计算。随着序列增长，KV Cache 会迅速占满显存，成为内存墙瓶颈。

### 2. 分布式训练
- **Q: ZeRO-1/2/3 之间的区别是什么？**
  - **A:** ZeRO-1 只切分 Optimizer States；ZeRO-2 增加切分 Gradients；ZeRO-3 进一步切分 Weights。级别越高，显存节省越多，但通信开销也越大。
- **Q: Tensor Parallelism (TP) 和 Pipeline Parallelism (PP) 的区别？**
  - **A:** TP 是在 Layer 内部横向切分（如切分 GEMM）；PP 是在 Layer 之间纵向切分（将不同的层放在不同 GPU 上）。

### 3. 前沿技术
- **Q: DeepSeek-V3 为什么要用 FP8 训练？**
  - **A:** 相比 FP16/BF16，FP8 能节省一半的显存并在支持的硬件（如 H100）上提供双倍计算量。其核心难题在于动态缩放的数值稳定性。

---
> [!IMPORTANT]
> **面试提醒：** 现在的 AI Infra 面试非常看重对 **DeepSeek** 等最新国产自研架构的理解。请务必深入研究其 MLA 和 MoE 相关的技术细节。
