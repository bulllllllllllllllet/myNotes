# XCellAligner 集成评估与实施分析方案

## 1. 概述与背景

本方案旨在评估 **XCellAligner** 项目最新核心提交（Commit `117bf2f`）对 **medlabel_image_cell_inference** 推理系统的影响，并制定详细的升级与集成计划。

**核心变更点**：
XCellAligner 对 `TransformerEncoder` 进行了架构重构，引入了多潜在变量（Multi-Latents）、反向交叉注意力（Reverse Cross-Attention）和细胞注意力偏置（Cell Attention Bias）。此外，组织推理（Tissue Inference）流程中明确引入了手工特征增强步骤。

---

## 2. 差异分析与评估 (Gap Analysis)

### 2.1 核心模型架构差异

| 组件 | XCellAligner (`updated_models.py`) | medlabel 推理工程 (`xcell_lib/models.py`) | 评估结论 |
| :--- | :--- | :--- | :--- |
| **类定义** | `TransformerEncoder` (新版) | `TransformerEncoder` (旧版) | **不兼容** |
| **参数** | `num_cls_latents=32` | 无 | 无法加载新权重 |
| **注意力机制** | `reverse_cross_attention` | `nn.MultiheadAttention` (标准) | 逻辑根本性改变 |
| **辅助层** | `cell_attn_bias` | 无 | 缺失关键计算层 |

**风险**: 若不更新 `models.py`，加载新训练的 `he_best_model.pth` 时将直接抛出 `RuntimeError: Error(s) in loading state_dict`，且维度不匹配。

### 2.2 推理流程差异：整图 vs 组织

通过对 `slide_inference/extract_feature.py` (WSI) 和 `he_transformer_inference.py` (Tissue) 的深度代码走查，确认了两种模式在**特征输出**上的严格区分：

| 特征维度 | WSI 推理 (整图) | 组织推理 (单图/ROI) |
| :--- | :--- | :--- |
| **Encoder 输出** | 8 维 (Deep Features) | 8 维 (Deep Features) |
| **特征增强** | **无** | **有** (`build_cell_features`) |
| **增强内容** | N/A | +5 维 (面积, R, G, B, Gray) |
| **最终维度** | **8 维** | **13 维** |
| **medlabel 现状** | 支持 (需更新模型) | **不支持** (缺失增强步骤) |

**结论**:
1.  **整图推理 (WSI)**: 算法流程逻辑无需变更，仅需更新模型定义以适配权重。
2.  **组织推理 (Tissue)**: 必须在推理管道中新增“特征增强”环节，将 8维 特征扩展为 13维。

---

## 3. 实施方案 (Implementation Plan)

### 3.1 阶段一：核心算法库升级 (Core Upgrade)

**目标**: 确保推理工程能够正确加载和运行最新的 XCellAligner 模型。

*   **操作 1.1**: 更新模型定义
    *   **源文件**: `XCellAligner/updated_models.py`
    *   **目标文件**: `medlabel_image_cell_inference/src/image_cell_processing/core/algorithms/xcell_lib/models.py`
    *   **动作**: 完全替换文件内容。

*   **操作 1.2**: 验证工具库
    *   **目标文件**: `medlabel_image_cell_inference/src/image_cell_processing/core/algorithms/xcell_lib/utils.py`
    *   **检查**: 确认 `build_cell_features` 函数是否存在且逻辑与 XCellAligner 一致。
    *   **动作**: 若缺失或逻辑不一致，进行同步更新。

### 3.2 阶段二：整图推理适配 (WSI Adaptation)

**目标**: 修复因模型参数变更导致的初始化错误。

*   **操作 2.1**: 适配 `SlideSegmenter` 初始化
    *   **文件**: `medlabel_image_cell_inference/src/image_cell_processing/core/algorithms/slide_segmenter.py`
    *   **修改**: 在 `load_models` 方法中，实例化 `TransformerEncoder` 时增加 `num_cls_latents=32` 参数（建议设为默认值或从配置读取）。

### 3.3 阶段三：组织推理集成 (Tissue Integration)

**目标**: 新增对 13维 特征输出的支持。

*   **操作 3.1**: 扩展 `SlideSegmenter.process_tile` 接口
    *   **文件**: `medlabel_image_cell_inference/src/image_cell_processing/core/algorithms/slide_segmenter.py`
    *   **修改**:
        1.  增加参数 `enable_handcrafted_features: bool = False`。
        2.  引入 `xcell_lib.utils.build_cell_features`。
        3.  当开关开启时：
            *   利用当前的 `tile_np` (原图) 和 `masks` (分割结果)。
            *   调用 `build_cell_features(tile_np, masks, final_features)`。
            *   替换 `final_features` 为增强后的 13维 向量。

*   **操作 3.2**: 新增/更新推理组件
    *   **方案 A (复用)**: 在 `TileInferencer` 的输入 `FilterResult` 或配置中增加 `inference_mode` 字段，透传给 `SlideSegmenter`。
    *   **方案 B (独立)**: 新增 `TissueInferencer` 组件 (如需处理非瓦片的大图或特殊逻辑)，但鉴于 ROI 通常较小且已被切片，方案 A 更为轻量。
    *   **推荐**: 采用 **方案 A**，通过任务配置驱动特征模式。

---

## 4. 验证策略 (Verification Strategy)

1.  **单元测试**:
    *   测试 `TransformerEncoder` 能否成功加载 `he_best_model.pth`。
    *   测试输入 dummy data 后，输出维度是否符合预期（8维）。

2.  **集成测试 (WSI)**:
    *   运行现有整图推理流程，确保无报错，且输出特征维度保持为 8。

3.  **集成测试 (Tissue)**:
    *   构造一个 `enable_handcrafted_features=True` 的调用。
    *   验证输出特征维度是否为 13。
    *   验证手工特征部分（后5维）是否非零且数值合理（如面积 > 0, 颜色在 0-255 范围内）。

---

## 5. 附录：关键代码片段参考

**SlideSegmenter 改造示例**:

```python
def process_tile(self, tile_np, ..., enable_handcrafted_features=False):
    # ... (原有逻辑：分割 -> CTransPath -> Transformer) ...
    # final_features shape: [N, 8]
    
    if enable_handcrafted_features:
        from .xcell_lib.utils import build_cell_features
        # 注意：build_cell_features 需要 masks 为 int 类型
        # tile_np 应为 [H, W, 3] uint8
        enhanced = build_cell_features(tile_np, masks, final_features)
        final_features = enhanced # shape: [N, 13]
        
    # ... (原有逻辑：打包结果) ...
```
