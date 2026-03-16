# XCellAligner 算法升级与集成评估分析报告

## 1. 核心差异评估

通过对比 `XCellAligner` 最新代码（Commit `117bf2f`）与 `medlabel_image_cell_inference` 现有代码，发现核心模型定义与特征处理流程存在显著差异。

### 1.1 模型文件 (`models.py`)

XCellAligner 的 `TransformerEncoder` 已重构，而 `medlabel` 中的版本较旧。

| 特性 | XCellAligner (最新) | medlabel (现有) | 影响 |
| :--- | :--- | :--- | :--- |
| **Latent Variables** | `num_cls_latents=32` | 无 | **严重**: 权重加载失败 (Shape Mismatch) |
| **Attention Mechanism** | `reverse_cross_attention` | 标准 Attention | **严重**: 前向传播逻辑错误 |
| **Attention Bias** | `cell_attn_bias` | 无 | **严重**: 缺失关键计算路径 |
| **Output Tuple** | `(cls_out, x_out, logits)` | `(cls_linear, x, logits)` | 接口需适配 |

**结论**: 必须将 `medlabel` 中的 `xcell_lib/models.py` 替换为 `XCellAligner/updated_models.py`，否则无法加载新训练的模型权重。

### 1.2 工具库 (`utils.py`)

*   **Handcrafted Features**: `XCellAligner` 新增了 `build_cell_features` 函数，用于生成 5维 手工特征（面积、RGB均值、灰度）。`medlabel` 需确认是否包含此函数以支持组织推理。

---

## 2. 推理流程分析：整图 vs 组织

### 2.1 整图推理 (Whole Slide Inference)
*   **XCellAligner 逻辑**: `slide_inference/extract_feature.py`
    *   仅使用 `TransformerEncoder` 输出的 **8维深度特征**。
    *   **不使用** `build_cell_features` 进行增强。
*   **medlabel 现状**: `SlideSegmenter` 输出 8维 特征。
*   **兼容性结论**: 流程逻辑一致。仅需更新 `TransformerEncoder` 模型定义即可支持。不需要引入后续的聚类步骤（由下游系统处理）。

### 2.2 组织推理 (Tissue Inference)
*   **XCellAligner 逻辑**: `he_transformer_inference.py`
    *   输入: 单张 ROI 图像。
    *   特征提取: `TransformerEncoder` 输出 (8维)。
    *   **关键步骤**: 调用 `build_cell_features` 将 8维 特征扩展为 **13维** (8维深度 + 5维手工)。
    *   聚类: 基于 13维 特征进行 K-Means。
*   **medlabel 现状**: 不支持输出 13维 特征。
*   **兼容性结论**: 需要在 `SlideSegmenter` 或新组件中添加“特征增强”选项，以支持输出 13维 特征向量。

---

## 3. 升级实施方案

### 3.1 步骤一：核心算法库同步
1.  **替换 `models.py`**:
    *   将 `medlabel_image_cell_inference/src/image_cell_processing/core/algorithms/xcell_lib/models.py` 替换为 `XCellAligner/updated_models.py`。
2.  **更新 `utils.py`**:
    *   确保 `medlabel_image_cell_inference/src/image_cell_processing/core/algorithms/xcell_lib/utils.py` 包含 `build_cell_features` 函数。

### 3.2 步骤二：SlideSegmenter 改造
修改 `medlabel_image_cell_inference/src/image_cell_processing/core/algorithms/slide_segmenter.py`：

1.  **初始化适配**:
    ```python
    self.transformer_encoder = TransformerEncoder(
        ...,
        num_cls_latents=32  # 新增参数
    )
    ```
2.  **增加组织推理模式**:
    在 `process_tile` 方法中增加 `inference_mode` 参数或 `enhance_features` 标志。
    ```python
    def process_tile(self, tile_np, ..., inference_mode='wsi'):
        # ... 获取 8维 final_features ...
        
        if inference_mode == 'tissue':
            # 调用 utils.build_cell_features 进行增强
            # 输入: 原图 tile_np, 分割 masks, 8维特征
            # 输出: 13维特征
            final_features = build_cell_features(tile_np, masks, final_features)
            
        return results # 包含 8维 或 13维 特征
    ```

### 3.3 步骤三：推理服务集成
1.  在 `TileInferencer` 中透传任务类型（WSI 或 Tissue）。
2.  根据任务类型调用 `SlideSegmenter` 的相应模式。

## 4. 总结

`medlabel` 工程的整图推理流程无需算法层面的逻辑变更，仅需**更新模型定义**以匹配新权重。对于新增的**组织推理**需求，需要在特征提取阶段引入**手工特征融合**，输出 13维 特征向量。
