import torch
import triton
import triton.language as tl
# ============================================================
# SiluAndMul 融合（Lesson 1）
# ============================================================
# 原始版本（fastvllm/layers/activation.py）：
#   @torch.compile
#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#       x, y = x.chunk(2, -1)          # 拆成 gate / up 两个 view
#       return F.silu(x) * y           # 先 silu 再乘
# 访存分析：silu 写中间量 temp、乘法再读回，共 5 次 [N, intermediate] 读写
# 融合思路：gate/up 是同一行内存的两个区，各 load 一次，silu×up 全在寄存器，一次 store 写完
# 设计：2D grid（行 × 列段），BLOCK_SIZE=512 固定 constexpr（列数固定），行数动态传参；FP32 计算 + FP16 存储
#
# 为什么用 2D grid 而不是把整块拍平成一维：拍平后每个 program 要用 div/mod 才能换算出
# 自己负责哪个 row 的哪一段，既多算又破坏访存合并（gate 和 up 相隔 intermediate，不连续）。
# 实测拍平版在 N=1 时 2.36us vs 2D grid 1.21us，故保持 2D。

@triton.jit
def silu_and_mul_kernel(
    x_ptr,                         # [num_tokens, 2*intermediate]
    y_ptr,                         # [num_tokens, intermediate]
    intermediate: tl.constexpr,    # 4096
    BLOCK_SIZE: tl.constexpr,      # 512
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < intermediate   # intermediate 不整除 BLOCK_SIZE 时挡住越界列

    # 1: 加载 gate（前半段）和 up（后半段）
    # 提示：x 是 [num_tokens, 2*intermediate]，行偏移 = row * 2 * intermediate
    # gate = tl.load(x_ptr + row * (2 * intermediate) + cols)
    gate = tl.load(x_ptr + row * (2*intermediate)+ cols, mask=mask, other=0.0)# 一次读
    # up   = 注意要跳过 intermediate 的偏移
    up = tl.load(x_ptr + row * (2*intermediate) + intermediate+ cols, mask=mask, other=0.0)# 一次读
    # 2: FP32 计算 silu(gate) * up
    # 注意：Triton 没有 tl.silu，silu(x) = x * sigmoid(x)，要手动拼
    gate_f = gate.to(tl.float32)
    result = (gate_f * tl.sigmoid(gate_f)) * up.to(tl.float32)
    # 3: 存回 FP16
    # mask 必须带上：intermediate 不是 BLOCK_SIZE 整数倍时（如 Llama-2-7B 的 11008），
    # 尾部 program 的 cols 会越过 intermediate，不带 mask 就是越界写、踩别人的显存。
    tl.store(y_ptr + row * intermediate + cols, result.to(y_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, f"expect [num_tokens, 2*intermediate], got {tuple(x.shape)}"
    num_tokens, dim = x.shape # gate和up合在一起
    assert dim % 2 == 0, f"dim={dim} must be even (gate | up)"
    # kernel 用 row * stride + cols 寻址，只认行主序连续布局。
    # qkv.split() 之类产出的 view（stride(3072,1)）喂进来不会报错，只会静默算错。
    assert x.is_contiguous(), "silu_and_mul expects a contiguous [N, 2*intermediate] tensor"

    intermediate = dim // 2
    y = torch.empty(num_tokens, intermediate,dtype=x.dtype,device=x.device)
    if num_tokens == 0 or intermediate == 0:
        return y

    # 512 是实测的甜点：再大对带宽没帮助，再小则每个 program 的访存事务太碎
    BLOCK_SIZE = 512
    grid = (num_tokens,triton.cdiv(intermediate,BLOCK_SIZE))
    silu_and_mul_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        intermediate=intermediate,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# ============================================================
# Add + RMSNorm 融合（Lesson 2）
# ============================================================
# 原始版本（fastvllm/layers/layernorm.py add_rms_forward）：
#   @torch.compile
#   def add_rms_forward(self, x, residual):
#       orig_dtype = x.dtype
#       x = x.float().add_(residual.float())
#       residual = x.to(orig_dtype)              # FP16 存回，供下一层残差相加
#       var = x.pow(2).mean(dim=-1, keepdim=True) # 每行一个标量（行规约）
#       x.mul_(torch.rsqrt(var + self.eps))
#       x = x.to(orig_dtype).mul_(self.weight)
#       return x, residual
#
# 融合思路：y = x + residual（FP32）→ var = mean(y²) 行规约 → out = y/√(var+eps)*w
# 设计：1D grid（一个 program = 一整行，因为行规约依赖整行）
#       y 读进寄存器后用两次：先 tl.sum(y²) 算 var，再归一化，无需第二次 load
#       H=1536 < BLOCK_SIZE=2048，超出的列用 mask 挡住
#
# 为什么只有一遍、没有 two-pass 回退：
#   一遍要把整行同时攥在寄存器里（x/r/w/x_fused/x_squared 五份 fp32），H 太大就会 spill。
#   但 spill 的临界点不在 H 本身，而在「每线程扛多少个元素」——加 num_warps 就能把
#   每线程的元素数摊薄，实测（N=1024）：
#       H=16384  w=4  → 243 regs,   0 spill, 199us (1.04x roofline)
#       H=24576  w=4  → 255 regs, 128 spill, 536us (1.86x)   ← 不加 warp 就在这里崩
#       H=24576  w=8  → 172 regs,   0 spill, 298us (1.04x)
#       H=32768  w=4  → 255 regs, 322 spill, 1072us (2.79x)
#       H=32768  w=8  → 248 regs,   0 spill, 398us (1.04x)
#   所以按 BLOCK_SIZE 线性抬 num_warps（见 _pick_num_warps）就能一路覆盖到 H=32768，
#   而真实模型 hidden_size 最大只到 16384（Llama-3.1-405B），全覆盖。
#   two-pass 要到 H > 65536 才重新领先，且它多读一遍 x 和 r（6 个 tensor vs 4 个），
#   在 H=32768 时反而更慢（592us vs 398us）——为不存在的场景付日常的代价，不划算。

@triton.jit
def add_rmsnorm_onepass_kernel(
    # 输入指针
    x_ptr,
    residual_ptr,
    # 输出指针
    out_ptr,
    new_residual_ptr,
    weight_ptr,
    # 形状
    H: tl.constexpr,             # hidden_size；声明为 constexpr 好让 mask 在编译期求值
    eps,                         # 运行时 float（eps 是超参，不该触发重编译）
    # 编译时常量
    BLOCK_SIZE: tl.constexpr,    # >= H 的 2 的幂
):
    """
    输入按 [N, H] 布局（x/residual/out/new_residual 均为 [N, H]，与 layernorm.py
    add_rms_forward 传入的张量对齐），weight 是一维共享向量 [H]。
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H
    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x_fused = x + r
    x_squared = x_fused*x_fused
    mean_val = tl.sum(x_squared)
    mean_val = mean_val / H
    rstd = 1.0 / tl.sqrt(mean_val + eps)
    output = x_fused * rstd * w
    tl.store(new_residual_ptr + row * H + cols, x_fused.to(new_residual_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + row * H+ cols, output.to(out_ptr.dtype.element_ty), mask=mask)


def _pick_num_warps(block_size: int) -> int:
    """按 BLOCK_SIZE 抬 warp 数，保证每线程最多扛 64 个元素（BLOCK_SIZE / 32 / warps）。

    寄存器预算：每个线程要同时放 x/r/w/x_fused/x_squared 五份 fp32，
    64 元素 × 5 ≈ 320 个理论值远低于 255 上限，实测 spill 为 0。
    H ≤ 2048 时用默认 4 个 warp（再多只会让规约变慢）。
    """
    return max(4, min(32, block_size // 2048))


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 输入布局说明：与 layernorm.py add_rms_forward 对齐，x/residual 均为 [N, H]，
    # weight 是一维共享向量 [H]。输出同形 [N, H]。
    assert x.dim() == 2, f"expect [N, H], got {tuple(x.shape)}"
    batch_size, hidden_size = x.shape
    assert residual.shape == (batch_size, hidden_size) and weight.shape[0] == hidden_size
    # 同上：kernel 用 row * H + cols 寻址，非连续布局会静默算错
    assert x.is_contiguous(), "add_rms_norm expects a contiguous [N, H] x"
    assert residual.is_contiguous(), "add_rms_norm expects a contiguous [N, H] residual"
    assert weight.is_contiguous(), "add_rms_norm expects a contiguous [H] weight"

    output = torch.empty_like(x)          # [N, H] 归一化结果
    residual_out = torch.empty_like(x)    # [N, H] 下一层残差
    if batch_size == 0:
        return output, residual_out

    # 1 << (n-1).bit_length() 就是 triton.next_power_of_2，省掉一次函数调用和查表
    BLOCK_SIZE = 1 << (hidden_size - 1).bit_length()
    grid = (batch_size,)

    add_rmsnorm_onepass_kernel[grid](
        x_ptr=x, residual_ptr=residual, out_ptr=output,
        new_residual_ptr=residual_out, weight_ptr=weight,
        H=hidden_size, eps=eps, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=_pick_num_warps(BLOCK_SIZE),
    )
    return output, residual_out
