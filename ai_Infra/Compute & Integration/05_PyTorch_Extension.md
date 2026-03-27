# 05 - 工程化接入：Pybind11 与 PyTorch Benchmark

写出优化的 `.cu` 代码或 Triton 脚本只是第一步，大厂需要的是可以直接 `import` 的 Python 包。

## 1. PyTorch C++ Extension 工作流

将 CUDA 代码包装进 PyTorch 主要包含三个核心文件：

**文件一：`kernel.cu` (CUDA 核心代码)**
这里存放你的实际计算逻辑（比如你手写的 GEMM Kernel）。

**文件二：`binding.cpp` (C++ 桥接层)**
使用 Pybind11 和 PyTorch 的 ATen 库，将 C++ 函数暴露给 Python。代码示例如下：

```cpp
#include <torch/extension.h>

// 声明你的 CUDA 函数
void run_my_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor c);

// Pybind11 绑定宏
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_gemm", &run_my_gemm, "My custom GEMM");
}
```


**文件三：`setup.py` (编译脚本)** 使用 `torch.utils.cpp_extension.BuildExtension` 进行编译。写好后，在终端执行 `python setup.py install` 即可将其编译并安装到你的 Python 环境中。

---

## 2.数值正确性验证

- 永远不要跳过验证环节！算子跑得再快，算错了也没用。
    
- 在 Python 端，生成相同的随机输入张量，分别传入你的 `my_custom_op` 和官方的 `torch.matmul`。
    
- 使用以下代码来判断数值误差是否在接受范围内：

	torch.allclose(out_custom, out_torch, atol=1e-5, rtol=1e-5)

## 性能测试(benchmark)

**错误做法：** 直接用 Python 的 `time.time()` 包裹 CUDA 函数。因为 CUDA 是异步执行的，这样测出来的仅仅是 CPU 下发 Kernel 的极短时间，毫无参考价值。

**正确做法 (PyTorch Event)：** 必须使用 CUDA Event 并强行同步等待，才能测出真实的耗时。
```python
	import torch
	
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	
	start_event.record()
	# 执行你的自定义算子
	my_custom_op.my_gemm(A, B, C)
	end_event.record()
	
	# 这一步非常关键：必须等待 GPU 真正把活干完！
	torch.cuda.synchronize() 
	
	# 打印真实的毫秒级耗时
	print(start_event.elapsed_time(end_event))
```
**进阶做法：** 推荐使用 Triton 提供的 `@triton.testing.perf_report` 和 `do_bench`。它可以自动帮你处理 Warmup（预热）和多次测量取平均，还能直接画出不同矩阵维度下的性能折线图，非常适合在写技术报告或文档时使用。