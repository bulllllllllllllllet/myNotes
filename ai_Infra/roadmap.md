# AI Infra 4-5个月大厂实习冲刺学习路径

针对已有计算机基础（C++、OS、体系结构）的小白用户，目标冲击大厂 AI Infra 实习岗位。

## 核心思维转变：从“调包侠”到“算力榨汁机”
AI Infra 的核心价值在于：在硬件资源受限的情况下，通过跨层（框架、编译器、通信、算子）优化，实现高吞吐、低延迟的训练与推理。

---

## 阶段一：并行计算与 CUDA 算子手撕 (第 1 个月)
**目标：** 理解 GPU 架构，能手写并优化核心算子。

- **核心知识点：**
    - **GPU 架构：** SM (Streaming Multiprocessor)、Warp 调度、Memory Hierarchy (Global, Shared, Register, Constant)。
    - **访存优化：** 合并访存 (Coalesced Access)、向量化访存 (`float4`)、Bank Conflict 规避。
    - **计算优化：** 循环展开、Register Tiling、Double Buffering。
- **必做实验：**
    - 手写 **Matrix Multiplication (GEMM)**：从 Naive 到 Tiling，再到使用 Register Tiling 优化到 cuBLAS 80% 以上性能。
    - 手写 **Reduce (Sum/Max)**：理解访存密集型算子的优化技巧（Warp Shuffle）。
    - 手写 **Softmax**：理解 Online Softmax 算法及其数值稳定性。
- **推荐资源：**
    - 《Programming Massively Parallel Processors》
    - 知乎：CUDA 算子手撕系列

---

## 阶段二：分布式训练框架与并行策略 (第 2 个月)
**目标：** 理解单机多卡、多机多卡如何协同训练千亿参数模型。

- **核心知识点：**
    - **三大并行：** Data Parallelism (DP/DDP), Tensor Parallelism (TP), Pipeline Parallelism (PP)。
    - **显存优化：** ZeRO (Zero Redundancy Optimizer) 1/2/3, Recomputation (Check-pointing), Offloading。
    - **主流框架：** Megatron-LM 技术栈, DeepSpeed 优化库。
- **必做内容：**
    - 通读 **Megatron-LM** 源码：重点看 TP 和 PP 的实现逻辑。
    - 实验：在实验室集群或 2-4 卡机上跑通 LLaMA 的分布式微调，观察显存变化。
- **重要概念：** 3D Parallelism, Sequence Parallelism, Mixture of Experts (MoE) 路由。

---

## 阶段三：通信协议与集群工程 (第 3 个月)
**目标：** 解决大规模集群中的“长尾效应”与通信瓶颈。

- **核心知识点：**
    - **通信库：** NCCL (All-Reduce, All-Gather, Reduce-Scatter) 集合通信算法。
    - **底层网络：** RDMA (RoCE/InfiniBand) 协议, NVLink 拓扑。
    - **工程实战：** 故障容错 (Fault Tolerance), 弹性训练, 异构算力适配。
- **重点关注：**
    - 为什么需要 Ring All-Reduce 和 Tree All-Reduce？
    - 大规模训练中节点掉线如何快速恢复？
- **推荐关注：** 华为昇腾等国产芯片的 NPU 通信库适配。

---

## 阶段四：推理加速与系统集成 (第 4 个月)
**目标：** 解决大模型推理的“算力墙”与“内存墙”问题。

- **核心知识点：**
    - **推理引擎：** vLLM (PagedAttention), Text-Generation-Inference (TGI), TensorRT-LLM。
    - **Attention 演进：** MHA -> MQA -> GQA -> MLA (DeepSeek V2/V3)。
    - **量化技术：** FP16, BF16, INT8, INT4, 以及最新的 FP8 (DeepSeek V3 使用)。
- **必读项目：**
    - **vLLM**：深入理解 PagedAttention 如何管理 KV Cache 减少碎片。
    - **FlashAttention 1/2/3**：理解如何通过算子融合减少显存读写。
- **性能评估：** 首 token 延迟 (TTFT), Token 吞吐量 (TPS/Throughput)。

---

## 阶段五：面试冲刺与简历复盘 (第 5 个月)
**目标：** 形成自己的技术壁垒，准备高频系统面试题。

- **复盘重点：**
    - 讲清楚一个你解决过的最复杂的 **性能瓶颈/系统 Bug**。
    - 能够从端到端（Request -> Bridge -> Scheduler -> Kernel）描述推理全链路。
- **模拟题目：**
    - “给一个长文本背景，如何优化 KV Cache 导致的 OOM ？”
    - “如何判断一个算子是 Memory-bound 还是 Compute-bound？”
    - “DeepSeek-V3 的 MLA 相较于 GQA 的优势在哪里？”

---
> [!TIP]
> **学习建议：** 不要只看论文，AI Infra 是工程学科，必须有 **Read Code** 和 **Profiling (Nsight Compute/Systems)** 的过程。
