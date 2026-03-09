# 11 WSI 加载与 KFB 支持（OpenSlide / ThreadLocalKFBSlide）
本章解释：为什么系统要抽象一个 `get_slide()` 工厂函数；KFB 文件是如何被打开；以及为什么 KFB 需要“线程本地句柄”。

## 1. 统一入口：get_slide(wsi_path)
实现：
- [utils/slide_loader.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/utils/slide_loader.py#L49-L103)

### 1.1 为什么需要 get_slide
TileReader 处在流水线最上游，它只需要一个能力：
- 给定 slide 文件路径，读取某个区域的像素（read_region）

不同 WSI 格式由不同库支持：
- 大多数格式：OpenSlide
- KFB：项目内置的 `libkfbreader.so` + Python 包装

为了让 TileReader 不关心格式差异，系统提供了 `get_slide()` 做统一分发。

### 1.2 OpenSlide 路径
当扩展名在 `OPENSLIDE_EXTENSIONS`（svs/tif/...）：
- 使用 `OpenSlide(wsi_path)`
- 并调用 `_validate_openslide` 确认金字塔层级信息

位置：
- [slide_loader.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/utils/slide_loader.py#L79-L96)

### 1.3 KFB 路径
当扩展名是 `.kfb`：
- 返回 `ThreadLocalKFBSlide(wsi_path)`

位置：
- [slide_loader.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/utils/slide_loader.py#L97-L103)

## 2. TileReader 如何使用 get_slide
TileReader 在处理瓦片时延迟打开 slide：
- [core/components/tile_reader.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/core/components/tile_reader.py#L159-L170)

它将 slide handle 缓存在任务级 context 中（一个 task_id 一个 handle），并复用 handle 读取多个瓦片：
- [tile_reader.py:SlideContext](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/core/components/tile_reader.py#L35-L50)

这有两个好处：
- 避免每个瓦片都重复打开文件（开销巨大）
- 控制生命周期：TaskEndSignal 到来时在 on_finalize 关闭 handle
  - [tile_reader.py:on_finalize](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/core/components/tile_reader.py#L212-L233)

## 3. KFBReader：C 库封装与 ThreadLocalKFBSlide
KFB 读取实现文件：
- [kfb_reader/kfbreader.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/kfb_reader/kfbreader.py#L1-L38)

它会加载共享库：
- `libkfbreader.so`
- `libImageOperationLib.so`

位置：
- [kfbreader.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/kfb_reader/kfbreader.py#L28-L41)

### 3.1 ThreadLocalKFBSlide：为什么要线程本地
实现：
- [kfbreader.py:ThreadLocalKFBSlide](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/kfb_reader/kfbreader.py#L494-L574)

设计意图（从代码结构可读出）：
- TileReader 使用 ThreadPoolExecutor 多线程并行读取瓦片
- KFB 原始句柄很可能不是线程安全的（或线程安全成本高）
- 因此 ThreadLocalKFBSlide 为“每个线程”提供独立的 KFBSlide 句柄

关键实现点：
- `self._local = threading.local()` 保存线程本地变量
- `_get_slide()`：若当前线程没有 slide，就创建一个 KFBSlide 并缓存  
  - [kfbreader.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/kfb_reader/kfbreader.py#L508-L517)
- `close()`：关闭所有线程创建过的句柄  
  - [kfbreader.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/kfb_reader/kfbreader.py#L523-L529)

这意味着：
- 线程并发读取时不会争用同一个底层句柄
- 任务结束时能集中释放所有线程句柄，避免泄漏

## 4. 与 tiling 坐标生成的关系
tiling 在生成坐标网格时也会短暂打开 slide 获取尺寸：
- [core/algorithms/tiling.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/core/algorithms/tiling.py#L45-L55)

它在拿到尺寸后立刻 `slide.close()`：
- [tiling.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/core/algorithms/tiling.py#L53-L56)

注意：这与 TileReader 的“任务级缓存 handle”是两条独立路径：
- tiling 只为计算坐标服务，不参与推理读图
- TileReader 才是推理读图的长期句柄持有者

## 5. 本章“代码流转路径”小结
1. Runner/TileReader 使用 slide 的统一入口： [slide_loader.py:get_slide](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/utils/slide_loader.py#L49-L103)
2. KFB 分支：返回 ThreadLocalKFBSlide： [slide_loader.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/utils/slide_loader.py#L97-L103)
3. 线程本地句柄： [kfbreader.py](file:///home/zyh/NewMedLabel/medlabel_image_cell_inference/src/image_cell_processing/kfb_reader/kfbreader.py#L494-L574)

