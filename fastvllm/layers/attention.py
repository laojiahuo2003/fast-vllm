import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from fastvllm.utils.context import get_context

# E4M3 最大可表示值
FP8_E4M3_MAX = 448.0

# 将计算得到的kv，按照slot_mapping放到KV Cache对应的位置
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)  # 我是第几个program，当前gpu正在处理哪个token
    slot = tl.load(slot_mapping_ptr + idx)  # 加载对应的slot
    if slot == -1: return  # 不需要存入

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    cache_offsets = slot * D + tl.arange(0, D)

    # 直接加载和存储（FP8 模式下传入的是 uint8 view）
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor, is_fp8: bool = False, k_scale: float = 1.0,
                  v_scale_cache: torch.Tensor = None):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    if is_fp8:
        # FP8 量化（在 PyTorch 端完成）
        # V 动态 per-token scale
        v_amax = value.float().abs().amax(dim=(1, 2)).clamp_min(1e-12)
        v_scale = v_amax / FP8_E4M3_MAX

        # 量化到 FP8
        k_fp8 = (key.float() / k_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        v_fp8 = (value.float() / v_scale[:, None, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

        # 存储 V 的 scale（无条件写入，用 torch.where 避免 CUDA Graph 问题）
        # 对于 slot == -1 的位置，写入到 slot 0（不影响结果，因为这些位置不会被读取）
        safe_slots = torch.where(slot_mapping >= 0, slot_mapping, torch.zeros_like(slot_mapping))
        v_scale_cache.scatter_(0, safe_slots, v_scale)

        # Triton kernel 搬运（把 FP8 当 uint8 传递）
        key_to_store = k_fp8.view(torch.uint8)
        value_to_store = v_fp8.view(torch.uint8)
        k_cache_u8 = k_cache.view(torch.uint8)
        v_cache_u8 = v_cache.view(torch.uint8)
    else:
        key_to_store = key
        value_to_store = value
        k_cache_u8 = k_cache
        v_cache_u8 = v_cache

    # 检查 stride
    assert key_to_store.stride(-1) == 1 and value_to_store.stride(-1) == 1
    assert key_to_store.stride(1) == head_dim and value_to_store.stride(1) == head_dim
    assert k_cache_u8.stride(1) == D and v_cache_u8.stride(1) == D
    assert slot_mapping.numel() == N

    store_kvcache_kernel[(N,)](
        key_to_store, key_to_store.stride(0),
        value_to_store, value_to_store.stride(0),
        k_cache_u8, v_cache_u8,
        slot_mapping,
        D
    )


def dequant_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor,
                   v_scale_cache: torch.Tensor, k_scale: float):
    """反量化 FP8 KV Cache 回 BF16（全量，仅用于 prefill）"""
    B, bs = k_cache.shape[0], k_cache.shape[1]
    k = k_cache.float() * k_scale
    v = v_cache.float() * v_scale_cache.view(B, bs, 1, 1)
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


def dequant_kvcache_selective(k_cache: torch.Tensor, v_cache: torch.Tensor,
                               v_scale_cache: torch.Tensor, k_scale: float,
                               block_tables: torch.Tensor):
    """按需反量化 FP8 KV Cache（只反量化 block_tables 中用到的 blocks）

    Args:
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim] FP8
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim] FP8
        v_scale_cache: [num_blocks * block_size] FP32 (1D, 需要 reshape)
        k_scale: float
        block_tables: [batch_size, max_num_blocks_per_seq] int32, 值为 -1 表示 padding

    Returns:
        k_cache_dequant: [num_blocks, block_size, num_kv_heads, head_dim] BF16
        v_cache_dequant: [num_blocks, block_size, num_kv_heads, head_dim] BF16
    """
    # Reshape v_scale_cache: [num_blocks * block_size] -> [num_blocks, block_size]
    num_blocks, block_size = k_cache.shape[0], k_cache.shape[1]
    v_scale_cache_2d = v_scale_cache.view(num_blocks, block_size)

    # 反量化全部（CUDA Graph 不支持动态索引/unique）
    # TODO: 优化为只反量化 active blocks（需要预先收集 active blocks 或使用 gather）
    k_dequant = k_cache.float() * k_scale
    v_dequant = v_cache.float() * v_scale_cache_2d.unsqueeze(-1).unsqueeze(-1)

    return k_dequant.to(torch.bfloat16), v_dequant.to(torch.bfloat16)


# qkv -> o
class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale # 缩放因子，除的那个
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])# 一开始分配空的
        self.is_fp8_kv_cache = False  # 是否启用FP8
        self.k_scale = 1.0  # K 的静态 scale（标量）
        self.v_scale_cache = torch.tensor([])  # V 的动态 per-token scale

    # Q = 【N, num_heads, head_dim】, V = 【N, num_kv_heads, head_dim】
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 存储新的 KV（FP8 模式下会自动量化）
        if self.is_fp8_kv_cache and k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping,
                         is_fp8=True, k_scale=self.k_scale, v_scale_cache=self.v_scale_cache)

            # FP8: 反量化供 Flash Attention 使用（它不支持 FP8 dtype）
            if context.is_prefill:
                # Prefill: 可能需要全量反量化（取决于是否有 prefix cache）
                if context.block_tables is not None:
                    # 有 prefix cache，按需反量化
                    k_cache, v_cache = dequant_kvcache_selective(
                        k_cache, v_cache, self.v_scale_cache, self.k_scale, context.block_tables
                    )
                # 否则不需要反量化（只用新 k, v）
            else:
                # Decode: 按需反量化（只反量化 block_tables 中的 active blocks）
                k_cache, v_cache = dequant_kvcache_selective(
                    k_cache, v_cache, self.v_scale_cache, self.k_scale, context.block_tables
                )
        elif k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
        return o
