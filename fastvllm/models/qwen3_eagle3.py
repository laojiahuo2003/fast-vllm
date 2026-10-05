"""
Qwen3 模型 + Eagle3 Speculative Decoding 集成版本（修正版）

根据实际 Eagle3 权重结构重新实现
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

# 导入原始 Qwen3 实现
from fastvllm.models.qwen3 import Qwen3ForCausalLM


@dataclass
class Eagle3Config:
    """Eagle3 Draft Head 配置"""
    # Eagle3 架构参数
    hidden_size: int = 1024
    num_attention_heads: int = 16  # midlayer 的 q_proj 是 2048，head=16
    num_key_value_heads: int = 8
    intermediate_size: int = 3072

    # Vocab 映射
    target_vocab_size: int = 151936  # Qwen3 vocab
    draft_vocab_size: int = 32000     # Eagle3 compressed vocab

    # 特征提取层
    extract_layers: List[int] = None  # 默认 [1, 13, 24]
    num_input_features: int = 3       # 提取3个层

    # 其他参数
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960  # 来自 eagle3 checkpoint 的 config

    def __post_init__(self):
        if self.extract_layers is None:
            self.extract_layers = [1, 13, 24]


class LlamaRMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Eagle3MidLayer(nn.Module):
    """Eagle3 draft 的单层 Transformer

    架构对齐 SGLang: python/sglang/srt/models/llama_eagle3.py
    is_input_layer (layer 0) 会在注意力前拼接 embeds + target features,
    因此 q/k/v_proj 的输入维度是 hidden_size * 2.
    """

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.num_heads = 16
        self.head_dim = 128
        self.num_kv_heads = 8

        # Self-attention：输入层拼接 embeds + features -> 2048
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # MLP
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.hidden_norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

        self.is_input_layer = True  # Eagle3 draft 只有 1 层，即输入层

        # RoPE：draft 的 q/k 权重是带旋转训练的，必须应用（与 target 同参数）
        from fastvllm.layers.rotary_embedding import get_rope
        self.rotary_emb = get_rope(
            self.head_dim,
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

        # draft 自己的增量 KV cache（只有 1 层，占用很小）。
        # SGLang 的 draft 也有独立 cache: target 的 cache 是 28 层的，draft
        # 的 q/k/v 权重不同，不能共用。这里按需倍增分配。
        self.kv_cache = None

    def _ensure_kv_cache(self, needed: int, dtype, device):
        if self.kv_cache is not None and self.kv_cache.shape[1] >= needed:
            return
        cap = max(needed, 1024)
        if self.kv_cache is not None:
            cap = max(cap, self.kv_cache.shape[1] * 2)
        grown = torch.zeros(2, cap, self.num_kv_heads, self.head_dim,
                            dtype=dtype, device=device)
        if self.kv_cache is not None:
            grown[:, :self.kv_cache.shape[1]] = self.kv_cache
        self.kv_cache = grown

    def _attention(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache_len: int = 0,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """flash-attn + 增量 KV cache

        Args:
            hidden_states: [seq_len, hidden*2]（输入层已拼接 embeds + features）
            positions: [seq_len]
            cache_len: decode 时 cache 里已有的 token 数；prefill 时传 0
            is_prefill: True=全序列一次算完并整段写入 cache

        Returns:
            [seq_len, hidden_size]
        """
        from flash_attn import flash_attn_varlen_func

        seq_len = hidden_states.shape[0]
        self._ensure_kv_cache(cache_len + seq_len, hidden_states.dtype, hidden_states.device)

        q = self.q_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(seq_len, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        if is_prefill:
            self.kv_cache[0, :seq_len] = k
            self.kv_cache[1, :seq_len] = v
            k_all, v_all = k, v
            cu_q = torch.tensor([0, seq_len], dtype=torch.int32, device=q.device)
            cu_k = cu_q
            max_q = max_k = seq_len
        else:
            # decode：追加 1 个 token，读整个 cache。
            # causal + q 短于 k 时 flash-attn 按右下角对齐，query 0 能看到全部 k。
            assert seq_len == 1, "decode 每次只处理 1 个 token"
            self.kv_cache[0, cache_len] = k[0]
            self.kv_cache[1, cache_len] = v[0]
            n = cache_len + 1
            k_all = self.kv_cache[0, :n]
            v_all = self.kv_cache[1, :n]
            # q 和 k 的序列长度不同，cu_seqlens 必须各写一份；
            # 共用一个会让 flash-attn 以为 k 也只有 1 个，只attend到最旧的 key
            cu_q = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
            cu_k = torch.tensor([0, n], dtype=torch.int32, device=q.device)
            max_q, max_k = 1, n

        o = flash_attn_varlen_func(
            q, k_all, v_all,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
            softmax_scale=self.head_dim ** -0.5,
            causal=True,
        )
        return self.o_proj(o.flatten(1, -1))

    def forward(
        self,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache_len: int = 0,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            embeds: token embedding [seq_len, hidden_size]
            hidden_states: target 特征经 fc 投影后的 [seq_len, hidden_size]
            positions: position ids [seq_len]
            cache_len / is_prefill: 见 _attention

        Returns:
            残差流（hidden + residual），交给上层 norm 归一化
        """
        if self.is_input_layer:
            # 输入层：residual 取 target features，不融合上层残留
            residual = hidden_states
            hidden_states = self.hidden_norm(hidden_states)
            embeds = self.input_layernorm(embeds)
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        else:
            # 非输入层（多层扩展时）：残差取上层输出
            residual = hidden_states

        hidden_states = self._attention(hidden_states, positions, cache_len, is_prefill)

        # post_attention_layernorm 是融合版：RMSNorm(attn_out + residual)
        # 且返回的新 residual 就是这个和（见 SGLang layernorm.py:804-811）
        residual = residual + hidden_states
        hidden_states = self.post_attention_layernorm(residual)

        hidden_states = self.down_proj(
            nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )

        # 返回残差流，与 SGLang 的 (hidden_states, residual) 一致
        return residual + hidden_states


class Eagle3DraftHead(nn.Module):
    """Eagle3 Draft Head - 根据实际权重结构实现"""

    def __init__(self, eagle3_config: Eagle3Config):
        super().__init__()
        self.config = eagle3_config

        # 特征融合层
        # fc.weight: [1024, 3072] = [hidden, hidden * num_features]
        # 输入: 3个层的 hidden states concat [B, L, 3072]
        # 输出: [B, L, 1024]
        self.fc = nn.Linear(
            eagle3_config.hidden_size * eagle3_config.num_input_features,
            eagle3_config.hidden_size,
            bias=False
        )

        # MidLayer - 单层 Transformer
        self.midlayer = Eagle3MidLayer(eagle3_config)

        # Norm
        self.norm = LlamaRMSNorm(eagle3_config.hidden_size, eagle3_config.rms_norm_eps)

        # LM head: [draft_vocab_size, hidden_size]
        self.lm_head = nn.Linear(
            eagle3_config.hidden_size,
            eagle3_config.draft_vocab_size,
            bias=False
        )

        # Vocab 映射向量（1D）
        # d2t_diff: draft vocab -> target vocab 的差值（需要加上 draft_id 得到 target_id）
        # t2d: target vocab -> draft vocab 的 bool mask
        self.register_buffer('d2t_diff', torch.zeros(eagle3_config.draft_vocab_size, dtype=torch.long))
        self.register_buffer('t2d', torch.zeros(eagle3_config.target_vocab_size, dtype=torch.bool))
        # 实际映射（d2t_diff + draft_id）
        self.register_buffer('d2t', torch.zeros(eagle3_config.draft_vocab_size, dtype=torch.long))

    def _get_embed_tokens(self):
        """Eagle3 没有独立 embedding，返回 target 的 embed_tokens"""
        ref = getattr(self, '_embed_tokens_ref', None)
        if ref is None:
            raise RuntimeError("draft_head._embed_tokens_ref 未设置")
        return ref

    def forward(
        self,
        target_hidden_states,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        cache_len: int = 0,
        is_prefill: bool = True,
    ):
        """
        Eagle3 forward

        Args:
            target_hidden_states: 两种形态
                - dict {layer_id: [seq_len, hidden]}：target 的 aux 特征，
                  会被 concat 成 3*hidden 再过 fc
                - [seq_len, hidden] tensor：draft 自己上一步的输出，
                  维度已是 hidden_size，直接过 midlayer（跳过多层 concat + fc）。
                  对齐 SGLang llama_eagle3.py:229 的 shape 分支
            input_ids: token ids [seq_len]，用于取 token embedding
                (Eagle3 checkpoint 没有自己的 embedding，复用 target 的)
            positions: position ids [seq_len]
            cache_len / is_prefill: 见 Eagle3MidLayer._attention

        Returns:
            (logits, aux_hidden)
              logits: [seq_len, target_vocab_size]
              aux_hidden: [seq_len, hidden_size] draft 自己的残差流输出，
                下一步 decode 时作为特征喂回去 (对齐 SGLang 的
                logits_output.hidden_states)
        """
        if isinstance(target_hidden_states, dict):
            # 1. 融合多层特征: concat 3个层
            ordered_features = [target_hidden_states[layer_id]
                               for layer_id in sorted(target_hidden_states.keys())]
            h = torch.cat(ordered_features, dim=-1)  # [seq_len, 3*hidden]
            # 2. 特征投影
            h = self.fc(h)  # [seq_len, hidden]
        else:
            # draft 自己的输出，已经投影过
            h = target_hidden_states  # [seq_len, hidden]

        # 3. 取 token embedding（Eagle3 输入层需要 embeds + features 拼接）
        embeds = None
        if input_ids is not None:
            embeds = self._get_embed_tokens()(input_ids)

        # 4. MidLayer 处理，返回残差流
        aux_hidden = self.midlayer(embeds, h, positions, cache_len, is_prefill)

        # 5. Norm + LM head -> draft vocab logits
        h = self.norm(aux_hidden)  # [seq_len, 1024]
        draft_logits = self.lm_head(h)  # [seq_len, 32000]

        # 6. 映射到 target vocab：target_logits[d2t[i]] = draft_logits[i]
        seq_len = draft_logits.shape[0]

        # 创建 target vocab size 的 logits（初始化为很小的值）
        fill_value = -65504.0 if draft_logits.dtype == torch.float16 else -1e9
        target_logits = torch.full(
            (seq_len, self.config.target_vocab_size),
            fill_value,
            dtype=draft_logits.dtype,
            device=draft_logits.device
        )
        # scatter: target_logits[p, d2t[i]] = draft_logits[p, i]
        # dim=1 是 vocab 维；d2t 值本身就是 target token id
        target_logits.scatter_(
            1, self.d2t.unsqueeze(0).expand(seq_len, -1), draft_logits
        )

        return target_logits, aux_hidden

    @classmethod
    def from_pretrained(cls, model_path: str, eagle3_config: Eagle3Config):
        """从预训练权重加载 Eagle3 head"""
        import safetensors.torch

        # 创建模型
        model = cls(eagle3_config)

        # 加载权重
        weights_file = Path(model_path) / "model.safetensors"
        if not weights_file.exists():
            raise FileNotFoundError(f"Eagle3 weights not found: {weights_file}")

        state_dict = safetensors.torch.load_file(str(weights_file))

        # 修复权重 key：midlayer.self_attn.* -> midlayer.*
        fixed_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('midlayer.self_attn.'):
                # midlayer.self_attn.q_proj.weight -> midlayer.q_proj.weight
                new_k = k.replace('midlayer.self_attn.', 'midlayer.')
                fixed_state_dict[new_k] = v
            elif k.startswith('midlayer.mlp.'):
                # midlayer.mlp.gate_proj.weight -> midlayer.gate_proj.weight
                new_k = k.replace('midlayer.mlp.', 'midlayer.')
                fixed_state_dict[new_k] = v
            elif k == 'd2t':
                # d2t 存储的是差值，需要加上 draft_id 得到 target_id
                fixed_state_dict['d2t_diff'] = v
            else:
                fixed_state_dict[k] = v

        # 打印权重信息
        print("Loading Eagle3 weights (fixed keys):")
        for k, v in sorted(fixed_state_dict.items()):
            print(f"  {k:50s} {str(v.shape):20s}")

        # 加载权重（strict=False 允许部分缺失）
        missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)

        # 计算实际的 d2t 映射：d2t[i] = d2t_diff[i] + i
        model.d2t = model.d2t_diff + torch.arange(eagle3_config.draft_vocab_size, dtype=torch.long)

        print(f"\n✅ d2t 映射计算完成:")
        print(f"  d2t_diff 范围: [{model.d2t_diff.min().item()}, {model.d2t_diff.max().item()}]")
        print(f"  d2t 映射到的 target tokens: {model.d2t.unique().numel()}")

        if missing:
            print(f"\n⚠️  Missing keys (will use random init): {missing}")
        if unexpected:
            print(f"\n⚠️  Unexpected keys (ignored): {unexpected}")

        print("\n✅ Eagle3 head loaded successfully")

        return model


class Qwen3WithEagle3(nn.Module):
    """
    Qwen3 + Eagle3 集成模型

    在原始 Qwen3 基础上添加 Eagle3 speculative decoding 支持
    """

    def __init__(
        self,
        qwen3_config,
        eagle3_config: Eagle3Config,
        eagle3_path: Optional[str] = None
    ):
        super().__init__()

        self.qwen3_config = qwen3_config
        self.eagle3_config = eagle3_config

        # 主模型：原始 Qwen3
        self.target_model = Qwen3ForCausalLM(qwen3_config)

        # Draft head: Eagle3
        if eagle3_path:
            self.draft_head = Eagle3DraftHead.from_pretrained(
                eagle3_path,
                eagle3_config
            )
        else:
            self.draft_head = Eagle3DraftHead(eagle3_config)

        # Eagle3 checkpoint 没有自己的 embedding，draft 输入层复用 target 的
        self.draft_head._embed_tokens_ref = self.target_model.model.embed_tokens

        # 用于缓存中间特征
        self._cached_features: Dict[int, torch.Tensor] = {}
        self._hooks_registered = False

    def _register_feature_hooks(self):
        """注册 hooks 以提取中间层特征"""
        if self._hooks_registered:
            return

        def make_hook(layer_id: int):
            def hook(module, input, output):
                # SGLang 在目标模型的层循环里捕获 aux hidden state:
                #   aux_hidden_states.append(hidden_states + residual)
                # 这里 layer 返回 (hidden_states, residual)，两者之和即为
                # 该层的完整输出 hidden state（对齐 SGLang 的取值）。
                if isinstance(output, tuple):
                    hidden_states, residual = output
                    self._cached_features[layer_id] = (hidden_states + residual).clone()
                else:
                    self._cached_features[layer_id] = output.clone()
            return hook

        # 在指定层注册 hook (Qwen3ForCausalLM.model.layers)
        for layer_id in self.eagle3_config.extract_layers:
            self.target_model.model.layers[layer_id].register_forward_hook(
                make_hook(layer_id)
            )

        self._hooks_registered = True
        print(f"✅ Registered feature extraction hooks at layers: {self.eagle3_config.extract_layers}")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]
            positions: [batch, seq_len]
            return_features: 是否返回中间特征

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            如果 return_features=True，还返回 cached_features
        """
        # 确保 hooks 已注册
        self._register_feature_hooks()

        # 清空缓存
        self._cached_features.clear()

        # Target model forward（hooks 会自动填充 _cached_features）
        output = self.target_model(input_ids, positions)

        if return_features:
            return output, self._cached_features
        return output

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算 logits（直接调用 target model 的方法）"""
        return self.target_model.compute_logits(hidden_states)

    def _forward_without_cache(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        简化的 forward（使用临时 KV cache）用于 draft 生成

        使用 fast-vllm 的模型，但开启临时 KV cache 模式，不写入全局 cache

        Returns:
            features: 提取的中间层特征
        """
        from fastvllm.utils.context import set_context

        # 清空特征缓存
        self._cached_features.clear()

        # 去掉 batch 维度（flash attention varlen 需要）
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)  # [seq_len]
        if positions.dim() == 2:
            positions = positions.squeeze(0)  # [seq_len]

        seq_len = input_ids.size(0)

        # 设置临时 KV cache 上下文（prefill 模式，但不写入全局 cache）
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=input_ids.device)
        cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32, device=input_ids.device)

        # 关键：use_temp_kv_cache=True，不需要 slot_mapping
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            slot_mapping=None,  # 临时模式不需要
            block_tables=None,
            use_temp_kv_cache=True  # 关键标志
        )

        # Target model forward（hooks 会自动填充 _cached_features）
        # 调用 model() 返回 hidden_states，不调用 compute_logits
        try:
            _ = self.target_model(input_ids, positions)
        except Exception as e:
            # 如果出错，打印详细信息
            print(f"Error in _forward_without_cache: {e}")
            print(f"input_ids shape: {input_ids.shape}")
            print(f"positions shape: {positions.shape}")
            raise

        return self._cached_features.copy()

    def _get_global_kv_cache(self):
        """获取全局 KV cache（从 target model 的第一个 attention 层）"""
        # 直接访问第一个 attention 层，不遍历所有 modules
        first_attn = self.target_model.model.layers[0].self_attn.attn
        if hasattr(first_attn, 'k_cache') and hasattr(first_attn, 'v_cache'):
            return first_attn.k_cache, first_attn.v_cache
        raise RuntimeError("No KV cache found in target model")

    @torch.inference_mode()
    def harvest_features(self, ids: torch.Tensor, pos: torch.Tensor) -> Dict[int, torch.Tensor]:
        """只读地跑一遍全序列，抓 3 层 aux 特征（不写任何 KV cache）

        投机解码每轮只需要 target 特征的最后几行，而 target 的 KV 已由 verify
        增量维护好了，所以首次进入某个序列时用这个函数"收一次庄稼"。
        use_temp_kv_cache=True 保证不碰全局 cache 的槽位。
        """
        from fastvllm.utils.context import set_context, reset_context

        seq_len = ids.shape[0]
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device=ids.device)
        self._cached_features.clear()
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=seq_len, max_seqlen_k=seq_len,
            slot_mapping=None, block_tables=None,
            use_temp_kv_cache=True,
        )
        try:
            self.target_model(ids, pos)
        finally:
            reset_context()
        # 特征会被下一次前向覆盖，必须立刻拷走
        return {lid: f.clone() for lid, f in self._cached_features.items()}

    @torch.inference_mode()
    def target_extend(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        prefix_len: int,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """target 增量 extend：读全局 KV 的前 prefix_len 个位置，只为新 token 算 KV

        走的就是 chunked prefill / prefix-cache 那条路径：cu_seqlens_q 短于
        cu_seqlens_k，flash-attn 右下角对齐，query j 能看到前 prefix_len+j 个 key。
        新 token 的 KV 经 slot_mapping 写进真实 paged cache（被拒的候选占用的槽位
        下一轮直接覆盖，永不回读）。

        Returns:
            (features, hidden_states)
              features: 3 层 aux 特征，行数 == 新 token 数
              hidden_states: [m, hidden_size] 最终残差流
        """
        from fastvllm.utils.context import set_context, reset_context

        m = input_ids.shape[0]
        cu_q = torch.tensor([0, m], dtype=torch.int32, device=input_ids.device)
        cu_k = torch.tensor([0, prefix_len + m], dtype=torch.int32, device=input_ids.device)
        self._cached_features.clear()
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=m, max_seqlen_k=prefix_len + m,
            slot_mapping=slot_mapping, block_tables=block_tables,
        )
        try:
            hidden_states = self.target_model(input_ids, positions)
        finally:
            reset_context()
        features = {lid: f.clone() for lid, f in self._cached_features.items()}
        return features, hidden_states

    @torch.inference_mode()
    def draft_round(
        self,
        features: Dict[int, torch.Tensor],
        feat_start: int,
        tokens: List[int],
        k: int,
        draft_len: int,
    ) -> tuple[List[int], int]:
        """增量生成 k 个候选 token（对齐 SGLang 的 draft extend + decode）

        约定（与 verify 严格一致）：draft 的第 t 行吃 (embed s_{t+1}, f_t)，
        输出是对 s_{t+2} 的预测。所以预测 s_S 的是第 S-2 行，预测 s_{S+1} 的是第
        S-1 行，……，预测 s_{S+k-1} 的是第 S+k-3 行。

        本轮写下的行是 [S-2, S+k-2]：前 k 行 [S-2, S+k-3] 产出候选，
        最后第 S+k-2 行只用来填 KV（输出丢弃）。为什么必须多填一行：
        下一轮开头 S' = S + E（E 是本轮实际吐出的 token 数，最多 k+1），
        它的 teacher-forcing 行是 S'-2，要读 draft KV 的 [0, S'-2]。
        只铺 k 行的话有效范围到 S+k-3，而 S'-2 最大能到 S+k-1，
        中间会有一行没写过 -> 读到垃圾 key，候选质量直接崩。多跑一行
        （代价是 draft 的一步小前向）把有效范围抬到 S+k-2，就永远够用了。

        两条路径的区别仅在 draft KV 是否已铺好：
        - draft_len < S-2：KV 缺前缀，先 teacher-forcing 把第 [0, S-2] 行铺一遍
          （该序列第一次进投机路径，或 draft KV 的使用者刚切换）
        - 否则：只 teacher-forcing 第 S-2 行（要 f_{S-2}，来自上一轮 verify 抓的特征）
        第 S-1 .. S+k-2 行一律 free-running，吃上一步自己的 aux 输出。

        Returns:
            (candidates, draft_len)  —— candidates 长度 k，draft_len 为铺完后
            draft KV 的有效行数
        """
        device = self.draft_head.midlayer.q_proj.weight.device
        S = len(tokens)

        if draft_len < S - 2:
            # KV 缺前缀：draft 第 [0, S-2] 行一次铺完。第 p 行吃
            # (embed s_{p+1}, f_p)，所以输入是 tokens[1:S]，特征取前 S-1 行。
            m = S - 1
            fb = {lid: f[feat_start:feat_start + m] for lid, f in features.items()}
            logits, aux = self.draft_head(
                fb,
                torch.tensor(tokens[1:S], dtype=torch.long, device=device),
                torch.arange(m, dtype=torch.long, device=device),
                cache_len=0, is_prefill=True,
            )
            # 第 S-2 行 -> 预测 s_S（第一个候选）
            candidates = [int(logits[-1].argmax().item())]
            draft_len = m
        else:
            # KV 已铺好：第 S-2 行单独 teacher-forcing（拿 target 的 f_{S-2}）
            idx = S - 2 - feat_start
            fb = {lid: f[idx:idx + 1] for lid, f in features.items()}
            logits, aux = self.draft_head(
                fb,
                torch.tensor([tokens[S - 1]], dtype=torch.long, device=device),
                torch.tensor([S - 2], dtype=torch.long, device=device),
                cache_len=S - 2, is_prefill=False,
            )
            candidates = [int(logits[-1].argmax().item())]
            draft_len = S - 1

        # 第 S-1 .. S+k-3 行 free-running：吃上一个候选 + 自己的 aux 输出
        for j in range(1, k):
            t = S - 2 + j
            step_logits, aux = self.draft_head(
                aux[-1:],
                torch.tensor([candidates[-1]], dtype=torch.long, device=device),
                torch.tensor([t], dtype=torch.long, device=device),
                cache_len=t, is_prefill=False,
            )
            candidates.append(int(step_logits[-1].argmax().item()))
            draft_len = t + 1

        # 第 S+k-2 行：只为把 draft KV 铺满（见上面的说明），预测结果丢弃
        t = S + k - 2
        _, aux = self.draft_head(
            aux[-1:],
            torch.tensor([candidates[-1]], dtype=torch.long, device=device),
            torch.tensor([t], dtype=torch.long, device=device),
            cache_len=t, is_prefill=False,
        )
        draft_len = t + 1
        return candidates, draft_len


