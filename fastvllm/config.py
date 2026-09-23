import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Chunked Prefill 分块旋钮：
    # 调小→每步 prefill 被切小，P99/ITL 更优（延迟优先）；调大→TTFT/吞吐更优（吞吐优先）。
    prefill_chunk_tokens: int = 0
    # FP8 KV Cache 量化：
    # 启用后将 KV Cache 以 FP8 格式存储，可节省约 50% 显存，略微影响精度
    kv_cache_dtype: str = "auto"  # "auto", "fp8", "fp16", "bf16"
    # FP8 K scale（支持 per-layer 校准）：
    # - None: 使用全局 scale=1.0
    # - str: JSON 文件路径（{"num_hidden_layers": N, "k_scale": [s0, s1, ...]}）
    # - list[float]: 直接传入每层 scale
    kvcache_k_scale: str | list[float] | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        # 解析 kv_cache_dtype
        import torch
        if self.kv_cache_dtype == "auto":
            # 默认使用模型的 dtype
            self.kv_cache_dtype = self.hf_config.dtype
        elif self.kv_cache_dtype == "fp8":
            self.kv_cache_dtype = torch.float8_e4m3fn
        elif self.kv_cache_dtype == "fp16":
            self.kv_cache_dtype = torch.float16
        elif self.kv_cache_dtype == "bf16":
            self.kv_cache_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported kv_cache_dtype: {self.kv_cache_dtype}")

        # 解析 kvcache_k_scale
        if self.kvcache_k_scale is not None:
            if isinstance(self.kvcache_k_scale, str):
                # 从 JSON 文件加载
                import json
                with open(self.kvcache_k_scale, "r") as f:
                    data = json.load(f)
                assert data["num_hidden_layers"] == self.hf_config.num_hidden_layers, \
                    f"Scale file has {data['num_hidden_layers']} layers, model has {self.hf_config.num_hidden_layers}"
                self.kvcache_k_scale = data["k_scale"]
            assert len(self.kvcache_k_scale) == self.hf_config.num_hidden_layers, \
                f"k_scale length {len(self.kvcache_k_scale)} != num_hidden_layers {self.hf_config.num_hidden_layers}"
