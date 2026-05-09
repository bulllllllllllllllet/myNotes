import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask = mask , other=None)
    y = tl.load(y_ptr + offsets, mask = mask , other=None)
    output = x + y 
    
    tl.store(output_ptr + offsets, output, mask = mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE, f'DEVICE:{DEVICE}, x.device:{x.device}, y.device:{y.device}, output.device:{output.device}'

    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    x = torch.rand(size, device = DEVICE)
    y = torch.rand(size, device = DEVICE)

    z_tri = add(x, y)
    z_ref = x + y

    torch.testing.assert_close(z_ref, z_tri, atol=atol, rtol=rtol)
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
    x = torch.rand(size, device = DEVICE , dtype = float32)
    y = torch.rand(size, device = DEVICE , dtype = float32)

    quantiles = [0.5, 0.05, 0.95]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles = quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles = quantiles)

    gbps = lambda ms : 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)




if __name__ = '__main__':
    test_add_kernel(size=83198)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark(save_path = '.', print_data=False)


