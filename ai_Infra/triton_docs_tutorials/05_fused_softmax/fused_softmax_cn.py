"""
本文件实现一个“融合 Softmax（Fused Softmax）” GPU Kernel。
该 Kernel 仅适用于每一行都能放入 GPU SRAM（静态随机存取存储器/片上内存）的矩阵。

你将学到：
- 减少内存读/写操作的重要性
- 如何将多个操作融合（Fuse）进一个 Kernel 以减少 DRAM 访问
- 如何获取并利用 GPU 硬件规格参数
- 在编写 Triton 内核时通常不需要考虑的 GPU 架构特性
- 如何根据 GPU 特有属性和启发式算法（Heuristics）定义元参数
- 流水线并行（Pipeline Parallelism）以及 for 循环在 GPU Kernel 内部的运行方式
- 在使用 mask 时，如何选择填充值（如 -inf 或 0）

推荐阅读顺序：
Step 1 - 朴素实现（Naive Implementation）
Step 2 - 单元测试
Step 3 - Wrapper 封装与硬件参数获取
Step 4 - Kernel 实现
Step 5 - 性能测试 Benchmark

参考官方教程：
https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
"""
import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

######### Step 1 #########
# 首先看一个朴素的、未融合的实现，用来分析内存开销
def naive_softmax(x):
    '''
    适用于形状为 (M, N) 的输入。
    "Safe Softmax" 是指减去最大元素以避免指数运算 (exp) 时的数值溢出。
    Softmax 对平移是不变的：softmax(x) = softmax(x + c)。
    '''
    # 1. read MN (读取所有元素), find max along N (算最大值), write M (存最大值)
    # 内存操作总计：MN (读) + M (写)
    x_max = x.max(dim=1)[0] 

    # 2. read MN + M (读原数据和最大值), subtraction (减法), write MN (存结果)
    # 内存操作总计：(MN + M) (读) + MN (写)
    z = x - x_max[:, None]

    # 3. read MN (读减法结果), write MN (存 exp 结果)
    # exp 实际上计算量很大，但目前我们更关心内存访问频率（Memory-bound 任务的瓶颈）
    numerator = torch.exp(z)

    # 4. read MN (读分子), find sum (求和), write M (存分母)
    denominator = numerator.sum(dim=1)

    # 5. read MN + M (读分子和分母), write MN (写最终结果)
    out = numerator / denominator[:, None]

    # 总结：整个过程做了 8MN + 4M 次内存操作。
    # (读取了 5MN + 2M 个元素；写入了 3MN + 2M 个元素)
    # 在 GPU 上，DRAM（显存）访问比寄存器慢几个数量级。如果能把这些操作合并成一个 Kernel，
    # 只读取 X 一遍并在片上完成所有计算，理论上能提速约 4 倍。
    return out

######### Step 4 #########
@triton.jit 
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,    # 处理非连续张量时的行跨步
    n_rows, n_cols,                         # 矩阵维度
    BLOCK_SIZE: tl.constexpr,               # 大于 n_cols 的最小 2 的幂
    num_stages: tl.constexpr,               # 流水线级数
): 
    # 每个并行实例（program）处理矩阵的一行或多行
    # 通过 pid 确定起始处理哪一行
    row_start = tl.program_id(0) 

    # 获取总的并行 program 数量
    # 如果 rows 很多，我们可以让每个 program 处理多行（使用跨步 row_step）
    row_step = tl.num_programs(0) 
        # 例如：如果有 4 个 programs，program 0 处理第 0, 4, 8... 行
    
    # tl.range 是一个迭代器，与 tl.arange 返回数组不同
    # num_stages 允许 Triton 启用内核内部的流水线并行：
    # 当一部分硬件在计算当前循环的数据时，另一部分硬件已经在加载下一轮循环的数据。
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        
        # 计算当前行的首地址
        # stride 代表从一行移动到下一行需要跳过的元素个数
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # 将整行数据加载到 SRAM
        # 由于 Triton 的 Block 必须是 2 的幂，而行列数未必是，所以需要 padding 和 mask
        col_offsets = tl.arange(0, BLOCK_SIZE) 
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        
        # 加载数据：越界的位置用 -inf 填充
        # 原因：max(-inf, x) = x，exp(-inf) = 0，
        # 这样在后续的计算中（求最大值和求和），越界位置的值不会产生干扰。
        row = tl.load(input_ptrs, mask=mask, other=float('-inf')) 

        # 1. 减去最大值（数值稳定性）
        # tl.max 会在片内（SRAM）高效完成
        row_minus_max = row - tl.max(row, axis=0)

        # 2. 指数运算
        # 注意：Triton 中的 exp 是快速近似计算，精度虽略低但吞吐量极高
        numerator = tl.exp(row_minus_max)

        # 3. 求和并归一化
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # 4. 存回最终结果（回写 DRAM）
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
            # 同样使用 mask，保证只写回 valid（有效）的列内容

######### Step 3 #########
"""
在创建启动 Kernel 的 Wrapper 函数之前，我们需要先获取 GPU 的硬件规格（Specifications）。
这能帮助我们定义“元参数（meta-parameters）”，使 Kernel 能根据具体 GPU 的算力资源
进行自我优化（例如：决定同时跑多少个实例）。
"""
# 获取当前 GPU 的属性字典
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)

# 1. SM (Streaming Multi-processor) 数量
# 每个 SM 就像 GPU 里的一个小处理器集群，可以同时运行多个程序
NUM_SM = properties["multiprocessor_count"] 

# 2. 寄存器（Registers）总量
# 寄存器是 GPU 访问速度最快的存储空间
NUM_REGS = properties["max_num_regs"] 
    # 每个 SM 的寄存器数量是有限的（通常为 65536 个）。
    # 如果一个 Program 用了太多寄存器，那么能并行跑的实例就变少了（并行度降低）。

# 3. 共享内存（SRAM / Shared Memory）总量
# 每个 SM 都有一个专属于该 SM 内部 Program 共享的 SRAM 池。
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] 

# 4. Warp 大小
# Warp 是指令分发的基本单位，NVIDIA 默认是 32 个线程一组，AMD 通常是 64。
WARP_SIZE = properties["warpSize"]

def softmax(x):
    '''
    Wrapper 封装函数，负责：
        1) 为输出张量分配显存
        2) 计算最佳的 Grid/Block 配置并启动 Kernel
    
    注意：本函数不支持自动求导。
    '''
    # 目前仅支持矩阵输入
    assert x.ndim == 2
    n_rows, n_cols = x.shape

    # BLOCK_SIZE 取大于列数的最小 2 的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 简单的启发式算法来决定 num_warps：
    # 给编译器一个建议：列数越多，使用的线程（Warps）就越多。
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # 软件流水线级数 (num_stages)：
    # 允许 GPU 同时执行多个循环动作。例如 stages=2 时，硬件可以一边做当前循环计算，
    # 一边预取下一轮循环的数据。
    # 启发式规则：显存多就用 4 级，显存少用 2 级。
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    # 分配输出空间
    y = torch.empty_like(x)

    # .warmup() 预编译过程非常重要：
    # 它会基于输入属性编译 Kernel，并返回该 Kernel 实际消耗了多少寄存器和共享内存。
    kernel = _softmax_kernel.warmup(
        x, y, 
        x.stride(0), y.stride(0), # 传入行步长（stride），处理非连续 Tensor 的关键
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)  # 预编译时只需要模拟启动一次
    )

    # 获取预编译后的性能元数据
    kernel._init_handles()
    n_regs = kernel.n_regs  # 每个线程使用的寄存器数
    sram_needed_per_program = kernel.metadata.shared # 每个 program 需要的 SRAM 单量

    # 计算 GPU 占有率 (Occupancy)：
    # 每个 SM 能同时跑多少个 Program？
    # 1. 寄存器约束：能塞进多少个 program？
    # 公式：总寄存器 / (每个线程寄存器 * Warp大小 * Warp数量)
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    
    # 2. SRAM 约束：能塞进多少个 program？
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program

    # 实际运行能力 = 两者中较小的那个（瓶颈原理）
    programs_per_sm = min(reg_occupancy, sram_occupancy)

    # 计算总的可启动 program 数量：
    # 不能超过矩阵的总行数（多出来的 program 没活干）
    num_programs = min(NUM_SM * programs_per_sm, n_rows)

    # 最终的 Grid 配置
    grid = (num_programs, 1, 1)

    # 运行最终优化的 Kernel
    kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE,
        num_stages
    )
    return y

######### Step 2 #########
def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    """
    单元测试：
    验证 Triton Kernel 的结果与 PyTorch 标准实现是否一致。
    支持非规则形状（验证 padding/mask 逻辑是否稳健）。
    """
    torch.manual_seed(0)
    assert type(size) is tuple and len(size) == 2
    x = torch.randn(size[0], size[1], device=DEVICE)
    
    # 运行 Triton 版本
    z_tri = softmax(x)
    # 运行 PyTorch 基准版本
    z_ref = torch.softmax(x, axis=1)

    # 断言结果接近
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")

######### Step 5 #########
# 性能测试：对比 Triton 和 PyTorch 的吞吐量
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 横轴：矩阵列数
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=["Triton", "Torch"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={'M': 4096} # 矩阵行数固定为 4096
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    # 设置 GPU Stream 以保证测试不受干扰
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    
    # 吞吐量公式：数据总量 / 运行时间
    # 2 = 1 次读取 (x) + 1 次写入 (y)
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    
    return gbps(ms)

if __name__ == "__main__":
    # 默认运行单元测试
    # 使用非规则尺寸测试鲁棒性
    test_softmax_kernel(size=(1823, 781))

    # 可选：运行 benchmark 命令 `python fused_softmax_cn.py --benchmark`
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)
