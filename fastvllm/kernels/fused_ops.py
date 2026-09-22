import torch
import triton
import triton.language as tl

# =============================================================================
# SiluAndMul 融合: y = silu(gate) * up
# -----------------------------------------------------------------------------
# 输入   x      : [num_tokens, 2 * intermediate]  前半段为 gate, 后半段为 up
# 输出   y      : [num_tokens, intermediate]
# 布局   grid   : 2D = (num_tokens, cdiv(intermediate, BLOCK_SIZE))
# 精度   FP32 计算 + FP16 存储
#
# 为什么用 2D grid 而非拍平成一维:
#   拍平后每个 program 需用 div/mod 换算行号与列段, 既多算又破坏访存合并
#   (gate 与 up 相隔 intermediate 字节, 不连续)。故保持 2D。
# =============================================================================


@triton.jit
def silu_and_mul_kernel(
    x_ptr,                               # [num_tokens, 2*intermediate]  gate|up
    y_ptr,                               # [num_tokens, intermediate]
    intermediate: tl.constexpr,          # 一半列数
    BLOCK_SIZE: tl.constexpr,            # 每个 program 处理的列段大小
):
    row = tl.program_id(0)               # 第几行 (token)
    col_block = tl.program_id(1)         # 第几段列
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < intermediate           # 挡掉尾部越过 intermediate 的列

    # gate 在前半段, up 在后半段, 同一行连续内存各 load 一次
    gate = tl.load(x_ptr + row * (2 * intermediate) + cols, mask=mask, other=0.0)
    up = tl.load(x_ptr + row * (2 * intermediate) + intermediate + cols, mask=mask, other=0.0)

    # silu(x) = x * sigmoid(x)。FP32 计算, 融合在寄存器完成, 一次 store
    gate_f = gate.to(tl.float32)
    result = (gate_f * tl.sigmoid(gate_f)) * up.to(tl.float32)

    # 带回 mask 存回, 否则尾部越界写会踩到别人的显存
    tl.store(y_ptr + row * intermediate + cols, result.to(y_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, f"expect [num_tokens, 2*intermediate], got {tuple(x.shape)}"
    num_tokens, dim = x.shape
    assert dim % 2 == 0, f"dim={dim} must be even (gate | up)"
    # kernel 用 row * stride + cols 寻址, 只认行主序连续布局; 非连续 view 会静默算错
    assert x.is_contiguous(), "silu_and_mul expects a contiguous [N, 2*intermediate] tensor"

    intermediate = dim // 2
    y = torch.empty(num_tokens, intermediate, dtype=x.dtype, device=x.device)
    if num_tokens == 0 or intermediate == 0:   # 空输入直接返回
        return y

    BLOCK_SIZE = 512                                # 实测甜点: 列段大小
    grid = (num_tokens, triton.cdiv(intermediate, BLOCK_SIZE))
    silu_and_mul_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        intermediate=intermediate,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# =============================================================================
# Add + RMSNorm 融合: out = (x + residual) / sqrt(mean(y^2) + eps) * w
# -----------------------------------------------------------------------------
# 输入   x / residual : [N, H]      (与 layernorm.add_rms_forward 对齐)
#        weight       : [H]        一维共享缩放向量
# 输出   out          : [N, H]     归一化结果
#        new_residual : [N, H]     x + residual, 供下一层残差相加
# 布局   grid  : 1D = (N,), 一个 program 处理一整行 (行规约依赖整行)
#
# 为什么只有 onepass、没有 two-pass 回退:
#   onepass 需把整行攥在寄存器 (x/r/w/x_fused/x_squared 五份 fp32), H 太大即 spill。
#   但 spill 临界在「每线程扛多少元素」——按 BLOCK_SIZE 抬 num_warps 即可摊薄
#   (见 _pick_num_warps), 真实模型 hidden_size 最大 16384, 全覆盖。two-pass 需
#   H > 65536 才重新领先, 平常还多读一遍 x/r, 为不存在的场景付代价, 不值。
# =============================================================================


@triton.jit
def add_rmsnorm_onepass_kernel(
    x_ptr,                               # [N, H] 输入
    residual_ptr,                        # [N, H] 残差
    out_ptr,                             # [N, H] 归一化输出
    new_residual_ptr,                    # [N, H] 下一层残差
    weight_ptr,                          # [H]    缩放向量
    H: tl.constexpr,                     # hidden_size, 编译期常量便于 mask 求值
    eps,                                 # 小常数, 运行时浮点, 不触发重编译
    BLOCK_SIZE: tl.constexpr,            # >= H 的 2 的幂
):
    row = tl.program_id(0)               # 第几行
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H                      # 超出 H 的列填 0

    # 一次 load 三份: x / residual / weight
    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    x_fused = x + r
    x_squared = x_fused * x_fused
    mean_val = tl.sum(x_squared)         # 行规约; 被 mask 填 0 的列不影响求和
    mean_val = mean_val / H              # 除以真实宽度 H, 不是 BLOCK_SIZE
    rstd = 1.0 / tl.sqrt(mean_val + eps)
    output = x_fused * rstd * w

    # 归一化结果写 out; x + residual (未归一化) 写回, 作为下一层残差
    tl.store(new_residual_ptr + row * H + cols, x_fused.to(new_residual_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + row * H + cols, output.to(out_ptr.dtype.element_ty), mask=mask)


def _pick_num_warps(block_size: int) -> int:
    """按 BLOCK_SIZE 抬 warp 数, 保证每线程最多扛 64 个元素
    (BLOCK_SIZE / 32 / warps <= 64)。H <= 2048 时用默认 4 个 warp。"""
    return max(4, min(32, block_size // 2048))


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"expect [N, H], got {tuple(x.shape)}"
    batch_size, hidden_size = x.shape
    assert residual.shape == (batch_size, hidden_size) and weight.shape[0] == hidden_size
    # 非连续布局会静默算错, 故显式校验
    assert x.is_contiguous(), "add_rms_norm expects a contiguous [N, H] x"
    assert residual.is_contiguous(), "add_rms_norm expects a contiguous [N, H] residual"
    assert weight.is_contiguous(), "add_rms_norm expects a contiguous [H] weight"

    output = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    if batch_size == 0:                  # 空输入直接返回
        return output, residual_out

    BLOCK_SIZE = 1 << (hidden_size - 1).bit_length()   # triton 下一个 2 的幂
    grid = (batch_size,)

    add_rmsnorm_onepass_kernel[grid](
        x_ptr=x, residual_ptr=residual, out_ptr=output,
        new_residual_ptr=residual_out, weight_ptr=weight,
        H=hidden_size, eps=eps, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=_pick_num_warps(BLOCK_SIZE),
    )
    return output, residual_out