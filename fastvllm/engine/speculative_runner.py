"""
Speculative Decoding with Draft Model (e.g., Eagle3)

核心思路：
1. Draft model 快速生成 K 个候选 token
2. Target model 并行验证这些候选
3. 接受匹配的前缀，拒绝后续
"""
import torch
from typing import List, Tuple
from fastvllm.config import Config
from fastvllm.engine.sequence import Sequence
from fastvllm.models.qwen3 import Qwen3ForCausalLM
from fastvllm.layers.sampler import Sampler
from fastvllm.utils.loader import load_model
from fastvllm.utils.context import set_context, reset_context


class SpeculativeConfig:
    """Speculative decoding 配置"""
    def __init__(
        self,
        draft_model_path: str,
        num_speculative_tokens: int = 4,  # K：每次推测生成几个 token
        enable_adaptive: bool = True,      # 是否自适应调整 K
        min_acceptance_rate: float = 0.5,  # 低于此接受率时降低 K
    ):
        self.draft_model_path = draft_model_path
        self.num_speculative_tokens = num_speculative_tokens
        self.enable_adaptive = enable_adaptive
        self.min_acceptance_rate = min_acceptance_rate
        self.max_speculative_tokens = num_speculative_tokens
        self.min_speculative_tokens = 1


class DraftModel:
    """Draft Model Wrapper - 独立的小模型用于快速生成候选 token"""

    def __init__(self, config: Config, spec_config: SpeculativeConfig, rank: int):
        self.config = config
        self.spec_config = spec_config
        self.rank = rank
        self.block_size = config.kvcache_block_size

        # 加载 draft model
        hf_config = config.hf_config
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, spec_config.draft_model_path)
        self.sampler = Sampler()

        # Draft model 使用独立的 KV cache（更小）
        self._allocate_draft_kv_cache()

        torch.set_default_device("cpu")

        # 统计信息
        self.total_drafts = 0
        self.total_accepted = 0

    def _allocate_draft_kv_cache(self):
        """为 draft model 分配独立的 KV cache"""
        config = self.config
        hf_config = config.hf_config

        # Draft model 通常只需要较少的 blocks（因为只做短期推测）
        # 这里分配原来 1/4 的容量
        num_draft_blocks = max(config.num_kvcache_blocks // 4, 128)

        num_kv_heads = hf_config.num_key_value_heads // config.tensor_parallel_size
        head_dim = getattr(hf_config, "head_dim",
                          hf_config.hidden_size // hf_config.num_attention_heads)

        kv_cache_dtype = config.kv_cache_dtype

        # 创建 draft KV cache
        self.draft_kv_cache = torch.empty(
            2, hf_config.num_hidden_layers, num_draft_blocks,
            self.block_size, num_kv_heads, head_dim,
            dtype=kv_cache_dtype
        )

        # 绑定到 model layers
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.draft_kv_cache[0, layer_id]
                module.v_cache = self.draft_kv_cache[1, layer_id]
                layer_id += 1

    @torch.inference_mode()
    def generate_candidates(
        self,
        seq: Sequence,
        num_tokens: int
    ) -> List[int]:
        """
        用 draft model 生成 num_tokens 个候选 token

        Returns:
            List[int]: 候选 token IDs
        """
        candidates = []
        current_seq = seq.token_ids.copy()

        for _ in range(num_tokens):
            # 准备输入
            input_id = torch.tensor([current_seq[-1]], dtype=torch.int64).cuda()
            position = torch.tensor([len(current_seq) - 1], dtype=torch.int64).cuda()

            # Draft model forward
            # 注意：这里简化了 KV cache 管理，实际需要正确设置 context
            logits = self.model.compute_logits(self.model(input_id, position))

            # 贪心采样（draft 通常用贪心，保持确定性）
            next_token = logits.argmax(dim=-1).item()
            candidates.append(next_token)
            current_seq.append(next_token)

            # 如果遇到 EOS，提前终止
            if next_token == self.config.eos:
                break

        self.total_drafts += len(candidates)
        return candidates

    def get_acceptance_rate(self) -> float:
        """获取当前的接受率"""
        if self.total_drafts == 0:
            return 1.0
        return self.total_accepted / self.total_drafts


class SpeculativeModelRunner:
    """
    带 Speculative Decoding 的 ModelRunner

    在原有 ModelRunner 基础上增加：
    1. Draft model 管理
    2. Draft-then-verify 流程
    3. 自适应 K 值调整
    """

    def __init__(self, base_runner, spec_config: SpeculativeConfig):
        self.base_runner = base_runner
        self.spec_config = spec_config
        self.draft_model = DraftModel(
            base_runner.config,
            spec_config,
            base_runner.rank
        )

        # 自适应参数
        self.current_k = spec_config.num_speculative_tokens
        self.recent_acceptance_rates = []
        self.adaptive_window = 100  # 每 100 次调整一次 K
        self.step_count = 0

    def speculative_decode_single(
        self,
        seq: Sequence
    ) -> Tuple[List[int], int]:
        """
        对单个序列做 speculative decoding

        Returns:
            accepted_tokens: 被接受的 token 列表
            num_accepted: 接受的 token 数量
        """
        # Step 1: Draft model 生成候选
        candidates = self.draft_model.generate_candidates(seq, self.current_k)

        if not candidates:
            # Fallback：直接用 target model 生成 1 个 token
            return self._fallback_generate(seq)

        # Step 2: Target model 验证
        # 将候选 token 附加到序列，然后让 target model 并行计算所有位置的 logits
        accepted_tokens, num_accepted = self._verify_candidates(seq, candidates)

        # 更新统计
        self.draft_model.total_accepted += num_accepted

        # 自适应调整
        if self.spec_config.enable_adaptive:
            self._adaptive_adjust()

        return accepted_tokens, num_accepted

    def _verify_candidates(
        self,
        seq: Sequence,
        candidates: List[int]
    ) -> Tuple[List[int], int]:
        """
        用 target model 验证候选 token

        验证方式：
        1. 将候选 token 附加到序列
        2. Target model forward，获取每个位置的 logits
        3. 比较 target 的预测和 draft 的预测
        4. 从第一个不匹配的位置截断

        标准投机解码：
        - 输入位置 [S-1, S+K-1] 共 K+1 行
        - 第 i 行（i=0..K-1）的 target 预测 predictions[i] 与 candidates[i] 比较
        - 全部匹配时，用 predictions[K]（target 在最后位置的预测）作为 bonus token
        - 不匹配时，用 predictions[first_mismatch] 替换（target 在该位置的正确预测）
        """
        # 保存原始序列状态
        original_len = len(seq)

        # 构造验证序列：原序列最后一个 token + 所有候选
        # 这样 target model 可以为每个候选的位置给出预测
        verify_seq = Sequence(seq.token_ids + candidates)
        verify_seq.block_table = seq.block_table.copy()
        verify_seq.num_scheduled_tokens = len(candidates) + 1  # 最后一个 token + K 个候选
        verify_seq.num_cached_tokens = original_len - 1         # 从序列倒数第一个 token 开始

        # Target model forward（类似 prefill 模式）
        input_ids, positions = self.base_runner.prepare_prefill([verify_seq])
        logits = self.base_runner.run_model(input_ids, positions, is_prefill=True)
        reset_context()

        # logits 有 K+1 行：
        # predictions[0] 是 target 对位置 S-1 的输入给出的预测 → 预测 s_S → 对比 candidates[0]
        # predictions[i] 对比 candidates[i]，i = 0..K-1
        # predictions[K] 是 target 对最后一个候选位置的预测 → bonus token
        predicted_tokens = logits.argmax(dim=-1).tolist()

        accepted = []
        num_accepted = 0
        for i in range(len(candidates)):
            if predicted_tokens[i] == candidates[i]:
                accepted.append(candidates[i])
                num_accepted += 1
            else:
                # 不匹配：接受到此为止，用 target 的预测替换
                accepted.append(predicted_tokens[i])
                break

        # 全部匹配时，用 target 在最后位置的预测作为 bonus token
        if num_accepted == len(candidates):
            accepted.append(predicted_tokens[len(candidates)])

        return accepted, num_accepted

    def _fallback_generate(self, seq: Sequence) -> Tuple[List[int], int]:
        """降级方案：直接用 target model 生成 1 个 token"""
        token_ids = self.base_runner.run1([seq], is_prefill=False)
        return token_ids, 1

    def _adaptive_adjust(self):
        """自适应调整 K 值"""
        self.step_count += 1

        if self.step_count % self.adaptive_window == 0:
            acceptance_rate = self.draft_model.get_acceptance_rate()
            self.recent_acceptance_rates.append(acceptance_rate)

            # 根据接受率调整 K
            if acceptance_rate < self.spec_config.min_acceptance_rate:
                # 接受率太低，减小 K
                self.current_k = max(
                    self.current_k - 1,
                    self.spec_config.min_speculative_tokens
                )
            elif acceptance_rate > 0.8 and self.current_k < self.spec_config.max_speculative_tokens:
                # 接受率很高，增大 K
                self.current_k = min(
                    self.current_k + 1,
                    self.spec_config.max_speculative_tokens
                )
