  

## 1. 整体架构图解
 
Transformer 是一个 **Encoder-Decoder (编码器-解码器)** 结构。

*   **Encoder (左边)**: 负责“理解”输入序列。它将输入的一串词变成一串富含语义的向量。

    *   *输入*: 比如 "I love you"

    *   *输出*: 3个向量，每个向量包含该位置词语及其上下文的信息。

*   **Decoder (右边)**: 负责“生成”输出序列。它根据 Encoder 的输出和已经生成的词，预测下一个词。

    *   *输入*: Encoder 的输出 + 已经翻译出来的词 (比如 "我", "爱")

    *   *输出*: 下一个词的概率分布 (预测 "你")

  

---

  

## 2. 关键组件

  
### 2.1 Positional Encoding (位置编码)

Attention 机制本身是**没有顺序概念**的（"我爱你"和"你爱我"在 Attention 看来只是词的组合）。

为了让模型知道词的顺序，我们必须把位置信息“加”到词向量里。

*   **方法**: 使用正弦和余弦函数生成固定的位置向量，直接加到 Input Embedding 上。

  

### 2.2 Residual Connection (残差连接) & Layer Normalization

*   **公式**: `LayerNorm(x + Sublayer(x))`

*   **残差连接 (`x + ...`)**: 防止网络过深导致梯度消失，让信息能直接流向深层。

*   **Layer Normalization**: 对每一层神经元的输出做归一化，让训练更稳定。

  

### 2.3 Position-wise Feed-Forward Networks (FFN)

在 Attention 层之后，每个位置的向量都会独立地通过一个全连接网络。

*   **结构**: `Linear -> ReLU -> Linear`

*   **作用**: 增加非线性变换能力，整合特征。

  

---

  

## 3. Encoder 与 Decoder 的区别

  

| 组件 | Encoder Layer | Decoder Layer |

| :--- | :--- | :--- |

| **Self-Attention** | **双向**可见 (能看到整个句子) | **Masked** (只能看到自己之前的词，不能偷看后面) |

| **Cross-Attention** | 无 | **有** (Query来自Decoder, Key/Value来自Encoder) |

| **输入** | 源语言句子 | 目标语言句子 (Shifted Right) |

  

### 3.1 Masked Self-Attention (Decoder 专用)

在训练翻译 "I love you" -> "我爱你" 时：

*   预测 "爱" 时，Decoder 输入是 "我"。

*   此时 Decoder **绝对不能**看到 "爱" 和 "你"，否则就等于作弊了。

*   **实现**: 使用一个上三角矩阵 Mask (Look-ahead Mask)，把未来的位置设为 $-\infty$。

  

### 3.2 Cross-Attention (交互注意力)

连接 Encoder 和 Decoder 的桥梁。

*   **Query**: 来自 Decoder (我想知道翻译到这里需要关注原文的哪里？)

*   **Key / Value**: 来自 Encoder (原文的所有信息)

  

---

  

## 4. 总结：数据流向

  

1.  **输入** -> Embedding + Positional Encoding

2.  **Encoder**: N 层堆叠

    *   Self-Attention -> Add & Norm -> FFN -> Add & Norm

3.  **Decoder**: N 层堆叠

    *   Masked Self-Attention -> Add & Norm

    *   Cross-Attention (看 Encoder) -> Add & Norm

    *   FFN -> Add & Norm

4.  **输出**: Linear -> Softmax -> 预测下一个词的概率