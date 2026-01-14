# 第一阶段：Attention 机制详解

## 1. 核心概念：Query, Key, Value

Transformer 的核心在于 **Self-Attention (自注意力机制)**。想象你在图书馆找书：
*   **Query (查询)**: 你手里拿着的书单（比如“深度学习”）。
*   **Key (键)**: 书架上每本书的标签（比如“机器学习”、“神经网络”、“烹饪”）。
*   **Value (值)**: 书原本的内容。
*   **Score (分数)**: 你的 Query 和书架上 Key 的匹配程度。

**流程**：
1.  你拿着 Query 去和所有的 Key 进行匹配（点积运算）。
2.  匹配程度高的（Score 大），你就多关注对应的 Value。
3.  匹配程度低的（Score 小），你就少关注甚至忽略。
4.  最后把你“关注”到的所有 Value 加权融合起来，就是输出。

---

## 2. 数学公式 (Scaled Dot-Product Attention)

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

1.  **$QK^T$**: 矩阵乘法。计算 Query 和 Key 的相似度。
2.  **$/ \sqrt{d_k}$**: 缩放。防止点积结果过大导致 Softmax 梯度消失（梯度变小，学习变慢）。
3.  **Softmax**: 归一化。把相似度变成概率（权重），所有权重加起来等于 1。
4.  **$\times V$**: 加权求和。根据权重提取 Value 的信息。

---

## 3. Multi-Head Attention (多头注意力)

为什么要“多头”？
*   **类比**: 就像你看一篇文章，你可能同时关注“语法结构”和“情感色彩”。
*   **机制**: 把 Q, K, V 拆分成多个小份（头），让每个头去关注不同的特征子空间。
*   **结果**: 模型能学到更丰富的信息。

---

## 4. 维度变化图解 (代码中会反复出现)

假设：
*   Batch Size ($B$) = 2
*   Sequence Length ($L$) = 10 (句子的长度)
*   Model Dimension ($D$) = 512 (每个词向量的维度)
*   Heads ($H$) = 8
*   Head Dimension ($d_k$) = $512 / 8 = 64$

**数据流向**:
1.  **Input**: `[B, L, D]` (原始词向量)
2.  **Linear**: `[B, L, D]` (分别生成 Q, K, V)
3.  **Reshape (分头)**: `[B, L, H, d_k]` -> 转置为 `[B, H, L, d_k]`
    *   *解释*: 把 $D$ 拆成 $H \times d_k$，并把 $H$ 放到前面，方便并行计算。
4.  **Attention**:
    *   $Q \times K^T$: `[B, H, L, d_k] @ [B, H, d_k, L]` -> `[B, H, L, L]` (注意力分数矩阵)
    *   $\times V$: `[B, H, L, L] @ [B, H, L, d_k]` -> `[B, H, L, d_k]`
5.  **Reshape (合并)**: `[B, H, L, d_k]` -> 转置为 `[B, L, H, d_k]` -> `[B, L, D]`
6.  **Output**: `[B, L, D]` (与输入形状一致，包含上下文信息)