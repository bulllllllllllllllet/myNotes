# XCellAligner 集成评估与升级方案

## 1. 现状评估

经过对 `XCellAligner` 最新代码（Commit `117bf2f`）与 `medlabel_image_cell_inference` 工程的对比分析，发现现有推理工程中的算法模块存在版本滞后和功能差异。

### 1.1 模型文件差异 (`models.py`)

*   **XCellAligner (`updated_models.py`)**:
    *   引入了 `num_cls_latents` (默认32)，用于生成多路潜在向量。
    *   `cls_to_kv` 层的输出维度变为 `num_cls_latents * adjusted_output_dim`。
    *   新增了 `cell_attn_bias` (Cell-dependent attention bias)。
    *   使用了 `reverse_cross_attention` 机制。
*   **推理工程 (`xcell_lib/models.py`)**:
    *   代码结构较旧，缺少 `num_cls_latents` 参数。
    *   `cls_to_kv` 输出维度仅为 `adjusted_output_dim`。
    *   缺少 `cell_attn_bias` 层。
    *   **结论**: 模型定义不匹配，无法加载新训练的 `he_best_model.pth` 权重（会导致 shape mismatch 错误）。

### 1.2 推理流程差异 (WSI vs Tissue)

通过分析 `XCellAligner` 的源码，发现**整图推理 (Whole Slide)** 和 **组织推理 (Tissue/Patch)** 的特征输出逻辑存在显著差异：

| 特性 | 整图推理 (WSI Inference) | 组织推理 (Tissue Inference) |
| :--- | :--- | :--- |
| **核心脚本** | `slide_inference/extract_feature.py` | `he_transformer_inference.py` |
| **特征提取** | TransformerEncoder Output (8维) | TransformerEncoder Output (8维) |
| **后处理** | **无** (直接使用 8维 特征聚类) | **特征增强** (`build_cell_features`) |
| **最终特征** | **8 维** (深度特征) | **13 维** (8维深度特征 + 5维手工特征) |
| **手工特征** | N/A | 细胞面积, R均值, G均值, B均值, 灰度均值 |

*   **结论**:
    *   `medlabel` 目前的 `SlideSegmenter` 输出的是 8维 特征，符合 XCellAligner 的 WSI 推理流程。
    *   若要支持 **组织推理**，必须引入 `build_cell_features` 逻辑，将特征扩展为 13维。

---

## 2. 升级与集成方案

为了同步 XCellAligner 的最新算法并支持组织推理，建议执行以下更新：

### 2.1 核心算法库更新 (`medlabel.../core/algorithms/xcell_lib`)

1.  **更新 `models.py`**:
    *   使用 `XCellAligner/updated_models.py` 的代码完全替换现有的 `TransformerEncoder` 类。
    *   确保 `__init__` 参数包含 `num_cls_latents`。

2.  **更新 `utils.py`**:
    *   确保包含 `build_cell_features` 函数（现有代码似乎已有，需核对逻辑一致性）。
    *   核对 `extract_cell_features` 逻辑。

### 2.2 推理器改造 (`SlideSegmenter`)

需要在 `medlabel.../core/algorithms/slide_segmenter.py` 中进行适配：

1.  **模型初始化适配**:
    *   更新 `load_models` 方法中 `TransformerEncoder` 的实例化参数，增加 `num_cls_latents=32` (或从配置读取)。

2.  **增强特征输出模式**:
    *   修改 `process_tile` 方法，增加一个参数 `enhance_features: bool = False`。
    *   当 `enhance_features=True` 时，在获取 Transformer 输出后，调用 `utils.build_cell_features`。
    *   **注意**: `build_cell_features` 需要原始图像数据，`process_tile` 内部拥有 `tile_np`，可以满足需求。

### 2.3 新增组织推理能力 (Tissue Inference Support)

针对“组织推理”需求（即处理单张 ROI 图片而非全切片），建议在 `medlabel` 工程中新增组件：

1.  **新增 `TissueInferencer` 组件**:
    *   路径: `medlabel.../core/components/tissue_inferencer.py` (可选，或复用 `TileInferencer`)。
    *   如果组织推理是离线的、单张图的任务，可以直接在 `SlideSegmenter` 上封装一个 `process_image` 方法。
    *   **流程**:
        1.  读取图像。
        2.  调用 `SlideSegmenter.process_tile(img, enhance_features=True)`。
        3.  返回包含 13维 特征的细胞数据。

---

## 3. 详细实施步骤

### 步骤 1: 同步模型定义
*   **操作**: 复制 `XCellAligner/updated_models.py` 内容覆盖 `medlabel.../xcell_lib/models.py`。
*   **验证**: 检查 `num_cls_latents` 参数是否存在。

### 步骤 2: 改造 SlideSegmenter
*   **修改**: `medlabel.../core/algorithms/slide_segmenter.py`。
*   **初始化**:
    ```python
    self.transformer_encoder = TransformerEncoder(
        ...,
        num_cls_latents=32  # 新增参数
    )
    ```
*   **特征增强**:
    在 `process_tile` 中：
    ```python
    # ... 获取 final_features (8维) ...
    
    if self.enable_handcrafted_features: # 新增控制开关
        # 调用 utils.build_cell_features
        # 注意：build_cell_features 需要 masks，目前 process_tile 中 masks 是局部的
        # 需要确保传入正确的 mask 和原图
        enhanced = build_cell_features(tile_np, masks, final_features)
        final_features = enhanced # 变为 13维
    ```

### 步骤 3: 配置与接口更新
*   在 `TaskContext` 或配置中增加 `inference_mode` ("wsi" vs "tissue")。
*   如果模式为 "tissue"，则开启手工特征融合。

## 4. 总结

`medlabel` 工程的推理流程主要需要更新 **TransformerEncoder 模型结构** 以匹配最新的训练权重。对于 **组织推理** 的支持，核心在于在特征提取后增加 **手工特征融合 (Handcrafted Features)** 步骤，将输出维度从 8 扩展到 13。整图推理流程目前逻辑正确（仅需更新模型），无需大幅修改。
