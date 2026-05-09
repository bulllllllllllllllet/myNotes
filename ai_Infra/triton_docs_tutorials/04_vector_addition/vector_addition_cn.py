"""
本文件实现一个最简单的 Triton GPU Kernel：向量逐元素加法（entry-wise addition）。

你将学到：
- 如何编写单元测试来验证 Triton kernel 的数值正确性
- Triton kernel 的基础知识（语法、指针、launch grid、DRAM vs SRAM 等）
- 如何将 Triton kernel 与 PyTorch 做性能对比

推荐阅读顺序：
Step 1 - 单元测试
Step 2 - wrapper 封装
Step 3 - kernel 实现
Step 4 - 性能测试 benchmark

参考官方教程：
https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py
"""
import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

######### Step 3 #########
# 使用 triton.jit 装饰器，将该函数编译为 GPU kernel
@triton.jit  # 注意：在 Triton kernel 中只能使用 Python 的一个子集
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr): 
    """
    这是一个逐元素加法 kernel，设计上比较简单：
    - 只支持向量输入
    - 不支持 broadcasting

    传入 Triton kernel 的 torch.tensor 会自动转换为指向首元素的指针

    参数说明：
    x_ptr:      输入向量 x 的首地址（长度为 n_elements）
    y_ptr:      输入向量 y 的首地址（长度为 n_elements）
    output_ptr: 输出向量的首地址（长度为 n_elements）
    n_elements: 向量长度
    BLOCK_SIZE: 每个 kernel 实例处理的元素数量（建议为 2 的幂）

    tl.constexpr 表示 BLOCK_SIZE 是编译期常量（不是运行时变量），
    每一个不同的 BLOCK_SIZE 都会生成一个独立的 kernel。
    这种参数有时也称为“元参数（meta-parameters）”
    """

    # Triton 会并行启动多个“program”（可以理解为 kernel 实例）
    # 每个 program 是该 kernel 的一次独立执行
    # program 可以是多维的（由 launch grid 决定），这里是一维
    pid = tl.program_id(axis=0)

    # 计算当前块的起始位置
    block_start = pid * BLOCK_SIZE

    # offsets 表示当前 program 要处理的索引位置
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # mask 用于防止越界访问（非常关键）
    # 假设 offsets = [1024, 1025, 1026, 1027]
    # 假设总元素数量 n_elements = 1026
    # 当你执行 offsets < n_elements 时，GPU 会把这个 < 作用于数组里的每一个元素，最终返回一个布尔数组：
    # mask = [True, True, False, False]
    mask = offsets < n_elements

    # 从全局显存（DRAM / VRAM）加载数据到片上内存（SRAM）
    # DRAM：容量大但访问慢
    # SRAM：容量小但访问快
    # 这不是取一个数，而是取出一个长度为 1024 的向量
    x = tl.load(x_ptr + offsets, mask=mask, other=None)
    y = tl.load(y_ptr + offsets, mask=mask, other=None)

    # mask 确保不会访问越界数据
    # other 表示 mask 为 False 时填充值（默认 None）

    # 这不是两个数相加，是 1024 对数字在硬件层面瞬间加完
    # 在 SRAM 中执行计算（速度快）
    output = x + y

    # 写回结果到 DRAM（同样需要 mask 防止越界）
    tl.store(output_ptr + offsets, output, mask=mask)


######### Step 2 #########
def add(x: torch.Tensor, y: torch.Tensor):
    '''
    wrapper 函数，用于：
    1）分配输出 tensor
    2）配置并启动 Triton kernel（设置 grid / block）

    注意：该函数不接入 PyTorch 的计算图，因此不支持反向传播
    '''
    # 预分配输出
    #写 Triton kernel 时，预分配输出内存最标准的做法就是用 empty_like。
    #它不仅能立刻开辟内存而不浪费时间去填 0（比 zeros_like 快），还能自动继承 x 的 shape、dtype 和 device。
    output = torch.empty_like(x)

    # 确保所有 tensor 在同一个 GPU 上
    # assert语法为：assert [条件判断], [错误信息提示]
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE,\
        f'DEVICE: {DEVICE}, x.device: {x.device}, y.device: {y.device}, output.device: {output.device}'
    
    # 元素总数
    n_elements = output.numel()


    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # 启动 kernel
    # [grid] 指定并行 program 数量。注：Triton 自动管理硬件线程(Thread/Warp)，
    # 因此不需要像 CUDA 那样指定 block 线程数，处理规模由 BLOCK_SIZE 元参数控制。
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


######### Step 1 #########
def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    """
    单元测试：
    使用 PyTorch 作为参考结果，验证 Triton kernel 的正确性
    """
    torch.manual_seed(0)
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    z_tri = add(x, y)
    z_ref = x + y

    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")


######### Step 4 #########
# 使用 Triton 自带工具做性能测试
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # 横轴变量名，会自动传给下方 benchmark 函数的 size 参数
        x_vals=[2**i for i in range(12, 28, 1)],  # 测试规模的取值范围（从 4096 到 1.34亿个元素）
        x_log = True,  # 开启对数坐标，适合观察跨度巨大的数据量变化
        line_arg='provider',  # 图例分类变量名，会自动传给 benchmark 函数的 provider 参数
        line_vals=['triton', 'torch'],  # 对比的两大实现分类
        line_names=['Triton', 'Torch'],  # 图例中显示的名称
        styles=[('blue', '-'), ('green', '-')],  # 设置线条颜色和样式
        ylabel='GB/s',  # 纵轴单位（内存带宽吞吐量）
        plot_name='vector-add-performance',  # 生成图表或文件的名称前缀
        args={},  # 其他静态参数（此处为空）
    )
)
def benchmark(size, provider):
    # 构造输入
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.05, 0.95]  # 设置分位数（中位数、P5、P95），用于衡量执行时间的统计分布和稳定性

    # 运行不同实现
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)  # 使用 Triton 的 do_bench 进行高精度计时，自动处理 Warmup 和多次重复
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)  # 测量 Triton 实现的耗时

    # 吞吐量公式：(3次访存 * 总字节数 / 1e9) / (耗时毫秒 / 1e3)。3代表：x读 + y读 + output写
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # 先跑单元测试
    test_add_kernel(size=98432)

    # 可选：运行 benchmark
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)