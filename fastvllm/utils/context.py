from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_spec_verify: bool = False  # Speculative decoding verify mode (q is [m, heads, dim])
    # Speculative decoding 支持
    use_temp_kv_cache: bool = False  # 是否使用临时 KV cache（不写入全局）
    temp_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None  # (k, v) 临时缓存

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill=False, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, use_temp_kv_cache=False, temp_kv_cache=None, is_spec_verify=False):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        is_spec_verify=is_spec_verify,
        use_temp_kv_cache=use_temp_kv_cache,
        temp_kv_cache=temp_kv_cache,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
