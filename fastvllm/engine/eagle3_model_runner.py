"""
带 Eagle3 支持的 ModelRunner

新增文件，不修改原有 model_runner.py

投机解码的形状（对齐 SGLang / vLLM）：
  scheduler 每步只准往 seq 里 append 一个 token，所以一个 round 产出的多个 token
  先进 self._pending 队列，之后每个 step 排一个出去。

  每 round：
    draft   ： k 个候选，对应位置 S .. S+k-1
               （增量 KV：第一行 teacher-forcing 吃 target 特征，其余 free-running）
    verify  ： 一次 target extend，跑新 token 的 k+1 行（位置 S-1 .. S+k-1），
               前缀 KV 直接复用 paged cache，只写新行
    接受     ： 前 m 个候选命中 target 的贪心预测；m == k 时再白拿一个 bonus token

  推导好的不变式（k = num_speculative_tokens，E = 本轮实际产出 token 数 ≤ k+1）：
    - verify 写完后 target KV 有效长度 = S + k，下一轮需要 ≥ S + E - 1  <= S + k ✓
    - verify 抓到的特征覆盖位置 [S-1, S+k-1]，下一轮 draft 要 f_(S'-2)，
      S'-2 = S+E-2 落在该区间 ✓（E ≥ 1）
    - draft KV 铺完后 draft_len = S+k-2，下一轮需要 ≥ S'-1 = S+E-1 <= S+k-2 ✓
  所以全程不需要任何"补洞"前向，也没有全序列重算。
"""
from collections import deque
from typing import List, Optional

import torch
import torch.nn.functional as F

from fastvllm.config import Config
from fastvllm.engine.sequence import Sequence
from fastvllm.engine.model_runner import ModelRunner
from fastvllm.models.qwen3_eagle3 import Qwen3WithEagle3, Eagle3Config
from fastvllm.utils.loader import load_model
from fastvllm.utils.context import reset_context


class Eagle3ModelRunner(ModelRunner):
    """
    支持 Eagle3 Speculative Decoding 的 ModelRunner

    策略：重写 model 创建逻辑，其他复用父类
    """

    def __init__(self, config: Config, rank: int, event):
        # 保存配置
        self.config = config
        self.rank = rank
        self.event = event
        hf_config = config.hf_config

        # 初始化分布式（即使单 GPU 也要初始化，因为 embed/head 层会用到）
        import torch.distributed as dist
        self.world_size = config.tensor_parallel_size
        dist.init_process_group("nccl", "tcp://localhost:2333",
                               world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # 创建 Qwen3 + Eagle3 模型（在父类初始化之前）
        print("Creating Qwen3WithEagle3 model...", flush=True)
        eagle3_config = Eagle3Config(
            hidden_size=hf_config.hidden_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            intermediate_size=hf_config.intermediate_size,
            target_vocab_size=hf_config.vocab_size,
            extract_layers=getattr(config, 'eagle3_extract_layers', [1, 13, 24]),
        )

        # 暂存父类会创建的 model
        original_model_path = config.model

        # 临时修改 dtype 和 device
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        self.model = Qwen3WithEagle3(
            qwen3_config=hf_config,
            eagle3_config=eagle3_config,
            eagle3_path=getattr(config, 'eagle3_model_path', None)
        )
        print("✅ Qwen3WithEagle3 model created", flush=True)

        # 加载 target model 权重
        print("Loading target model weights...", flush=True)
        load_model(self.model.target_model, original_model_path)
        print("✅ Target model weights loaded", flush=True)

        # 注意：这里不恢复默认 device，因为后面的 warmup / capture_cudagraph
        # 还会新建 tensor（原始 ModelRunner 也是最后才恢复，见 model_runner.py:38）

        # 初始化其他属性（父类会用到）
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size

        # 手动初始化父类的其他部分（跳过 model 创建）
        from fastvllm.layers.sampler import Sampler
        self.sampler = Sampler()

        # ---- 投机解码状态 ----
        # 必须建在 warmup 之前：warmup_model() 会调 self.run()，那条路也会碰这些字段
        # 每步只能吐一个 token，多出来的先排在这里
        self._pending: dict[int, deque] = {}
        # seq_id -> {"kv_len", "feat_start", "feats", "draft_len"}
        self._spec_state: dict[int, dict] = {}
        # seq_id -> prefill 时 hooks 抓到的 aux 特征（省掉 _seed_state 里
        # 那次全序列 harvest_features 重跑）
        self._prefill_feats: dict[int, dict] = {}
        # draft KV 是全模型唯一一份缓冲，记录当前归谁所有
        self._draft_owner: Optional[int] = None
        self.block_manager = None
        self.spec_stats = {
            'total_drafted': 0,
            'total_accepted': 0,
            'total_steps': 0,
            'rounds': 0,
        }

        # Warmup 和 KV cache
        print("Warming up model...", flush=True)
        self.warmup_model()
        print("Allocating KV cache...", flush=True)
        self.allocate_kv_cache()
        print("✅ KV cache allocated", flush=True)

        if not self.enforce_eager:
            print("Capturing CUDA graph...", flush=True)
            self.capture_cudagraph()
            print("✅ CUDA graph captured", flush=True)

        # 最后恢复默认 dtype 和 device（对齐原始 ModelRunner）
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多进程支持
        if self.world_size > 1:
            if rank == 0:
                from multiprocessing.shared_memory import SharedMemory
                self.shm = SharedMemory(name="fastvllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                from multiprocessing.shared_memory import SharedMemory
                self.shm = SharedMemory(name="fastvllm")
                self.loop()

    @torch.inference_mode()
    def capture_cudagraph(self):
        """捕获常规 decode 和投机解码验证步 (Verify) 的 CUDA Graphs"""
        super().capture_cudagraph()

        # 捕获专用的投机解码验证步 (Verify) CUDA Graph (针对 M = k + 1 个 token)
        k = getattr(self.config, 'num_speculative_tokens', 3)
        M = k + 1
        device = self.model.target_model.lm_head.weight.device
        hf_config = self.config.hf_config
        bs = self.block_size
        max_num_blocks = (self.config.max_model_len + bs - 1) // bs

        self.verify_m = M
        self.model._register_feature_hooks()
        self.static_verify_ids = torch.zeros(M, dtype=torch.int64, device=device)
        self.static_verify_pos = torch.zeros(M, dtype=torch.int64, device=device)
        self.static_verify_slot = torch.zeros(M, dtype=torch.int32, device=device)
        self.static_verify_context_lens = torch.zeros(1, dtype=torch.int32, device=device)
        self.static_verify_block_tables = torch.zeros(1, max_num_blocks, dtype=torch.int32, device=device)
        self.static_verify_hidden = torch.zeros(M, hf_config.hidden_size, dtype=hf_config.dtype, device=device)
        self.static_verify_logits = torch.zeros(M, hf_config.vocab_size, dtype=hf_config.dtype, device=device)
        self.static_verify_feats = {
            lid: torch.zeros(M, hf_config.hidden_size, dtype=hf_config.dtype, device=device)
            for lid in self.model.eagle3_config.extract_layers
        }

        # 模拟初始合法值用于 warmup
        self.static_verify_block_tables.fill_(-1)
        self.static_verify_block_tables[0, 0] = 0
        self.static_verify_context_lens[0] = M
        self.static_verify_slot.copy_(torch.arange(M, dtype=torch.int32, device=device))
        self.static_verify_pos.copy_(torch.arange(M, dtype=torch.int64, device=device))

        from fastvllm.utils.context import set_context, reset_context
        set_context(
            is_prefill=False,
            slot_mapping=self.static_verify_slot,
            context_lens=self.static_verify_context_lens,
            block_tables=self.static_verify_block_tables,
            is_spec_verify=True,
        )

        # Warmup
        h = self.model.target_model(self.static_verify_ids, self.static_verify_pos)
        self.static_verify_hidden.copy_(h)
        self.static_verify_logits.copy_(F.linear(self.static_verify_hidden, self.model.target_model.lm_head.weight))
        for lid in self.model.eagle3_config.extract_layers:
            self.static_verify_feats[lid].copy_(self.model._cached_features[lid])
        torch.cuda.synchronize()

        # Capture
        self.spec_verify_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.spec_verify_graph, pool=self.graph_pool):
            h = self.model.target_model(self.static_verify_ids, self.static_verify_pos)
            self.static_verify_hidden.copy_(h)
            self.static_verify_logits.copy_(F.linear(self.static_verify_hidden, self.model.target_model.lm_head.weight))
            for lid in self.model.eagle3_config.extract_layers:
                self.static_verify_feats[lid].copy_(self.model._cached_features[lid])
        torch.cuda.synchronize()
        reset_context()
        print(f"✅ Speculative Verify CUDA Graph (M={M}) captured successfully", flush=True)

    def run(self, seqs: List[Sequence]) -> List[Optional[int]]:
        """
        重写 run 方法，添加 speculative decoding 支持

        单序列 decode：Eagle3 投机；多序列或 prefill：走原方法
        """
        result: List[Optional[int]] = [None] * len(seqs)

        decode_idx = [i for i, s in enumerate(seqs) if not s.is_prefill]
        prefill_idx = [i for i, s in enumerate(seqs) if s.is_prefill]

        # Prefill 用原方法
        if prefill_idx:
            prefill_seqs = [seqs[i] for i in prefill_idx]
            tokens = self._run_group(prefill_seqs, is_prefill=True)
            for idx, token in zip(prefill_idx, tokens):
                result[idx] = token
            self._stash_prefill_features(prefill_seqs)

        # Decode：单序列且贪心采样时用 speculative，多序列或有温度时用原方法
        if decode_idx:
            seq0 = seqs[decode_idx[0]]
            if (
                len(decode_idx) == 1
                and hasattr(self.model, 'draft_head')
                and getattr(seq0, 'temperature', 0.0) == 0
            ):
                token = self._spec_step(seq0)
                result[decode_idx[0]] = token
            else:
                decode_seqs = [seqs[i] for i in decode_idx]
                tokens = self._run_group(decode_seqs, is_prefill=False)
                for idx, token in zip(decode_idx, tokens):
                    result[idx] = token

        return result

    def _run_group(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        """原始的 run 逻辑（从父类复制）"""
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill
            else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # ------------------------------------------------------------------
    # 投机解码
    # ------------------------------------------------------------------

    def cleanup_seq(self, seq_id: int):
        """清理指定序列的投机解码状态，释放 GPU 特征张量显存"""
        self._pending.pop(seq_id, None)
        self._spec_state.pop(seq_id, None)
        self._prefill_feats.pop(seq_id, None)
        if self._draft_owner == seq_id:
            self._draft_owner = None

    def _stash_prefill_features(self, seqs: List[Sequence]):
        """把 prefill 时 hooks 已经抓到的 aux 特征按 seq 存起来

        引擎自己的 prefill 走的也是 `Qwen3WithEagle3.forward`，3 层特征
        本来就被抓进 `_cached_features` 了，`_seed_state` 再调一次
        `harvest_features` 等于把整条 prompt 重跑一遍。这里按 seq 切开存下，
        第一步直接拿来用。

        只有"这条 seq 的整段 prompt 都是这一步从位置 0 算的"时才有效：
        分块 prefill 或命中 prefix cache 时 hooks 只覆盖尾部一段，而 draft 的
        teacher-forcing 行要的是 [0, S-1] 全段，切片会缺头，不能用。
        block_table 为空的是 warmup_model 造的假序列，直接跳过，否则它们的
        seq_id 永远不会被 cleanup_seq 收走，白占显存。
        （prompt 的 token 序列之后不会变，所以存下的特征一直有效。）
        """
        feats = getattr(self.model, '_cached_features', None)
        if not feats:
            return
        offset = 0
        for seq in seqs:
            n = seq.num_scheduled_tokens
            if seq.block_table and seq.num_cached_tokens == 0 and n == seq.num_tokens:
                self._prefill_feats[seq.seq_id] = {
                    lid: f[offset:offset + n].clone() for lid, f in feats.items()
                }
            offset += n

    @torch.inference_mode()
    def _spec_step(self, seq: Sequence) -> Optional[int]:
        """一步投机解码：队列里有存货就直接吐，否则跑一个完整 round"""
        queue = self._pending.get(seq.seq_id)
        if queue:
            tok = queue.popleft()
            if not queue:
                self._pending.pop(seq.seq_id, None)
            # 如果是该序列最后一步，及时清理投机状态
            eos = getattr(self.config, 'eos', None)
            if (eos is not None and not seq.ignore_eos and tok == eos) or (seq.num_completion_tokens + 1 >= seq.max_tokens):
                self.cleanup_seq(seq.seq_id)
            return tok

        state = self._spec_state.get(seq.seq_id)
        if state is None:
            state = self._seed_state(seq)

        emitted = self._spec_round(seq, state)

        # 截断 EOS 之后的多余 token（避免队列里残留无用候选）
        eos = getattr(self.config, 'eos', None)
        if eos is not None and not seq.ignore_eos:
            for i, tok in enumerate(emitted):
                if tok == eos:
                    emitted = emitted[:i + 1]
                    break

        # max_tokens / ignore_eos 兜底：别超发
        remaining = max(1, seq.max_tokens - seq.num_completion_tokens)
        if remaining < len(emitted):
            emitted = emitted[:remaining]

        if not emitted:
            self.cleanup_seq(seq.seq_id)
            return eos if eos is not None else 0

        if len(emitted) > 1:
            self._pending[seq.seq_id] = deque(emitted[1:])
        else:
            self._pending.pop(seq.seq_id, None)

        tok = emitted[0]
        if (eos is not None and not seq.ignore_eos and tok == eos) or (seq.num_completion_tokens + 1 >= seq.max_tokens):
            self.cleanup_seq(seq.seq_id)

        return tok

    @torch.inference_mode()
    def _seed_state(self, seq: Sequence) -> dict:
        """序列第一次进投机路径：抓一次 target 特征，初始化增量状态

        target 的 KV 已经是好的（engine 自己的 prefill 写好了 prompt 段），
        这里只补"特征"这一半：draft 的 teacher-forcing 行需要 f_(S-2)。
        特征优先用 prefill 时 hooks 顺手抓的那份（_stash_prefill_features），
        没留下才单独跑一遍只读前向——分块 prefill / prefix cache / 被抢占过
        都会走到这条兜底路径。
        特征是全序列的，所以 feat_start = 0，之后每轮往后平移。
        """
        tokens = seq.token_ids
        feats = self._prefill_feats.pop(seq.seq_id, None)
        if feats is None:
            device = self.model.target_model.lm_head.weight.device
            ids = torch.tensor(tokens, dtype=torch.long, device=device)
            pos = torch.arange(len(tokens), dtype=torch.long, device=device)
            feats = self.model.harvest_features(ids, pos)
        self._draft_owner = seq.seq_id
        state = {
            'kv_len': seq.num_cached_tokens,
            'feat_start': 0,
            'feats': feats,
            'draft_len': 0,   # 0 = draft KV 一份都没铺，draft_round 会整段铺
        }
        self._spec_state[seq.seq_id] = state
        return state

    def _grow_block_table(self, seq: Sequence, num_positions: int) -> bool:
        """保证 block_table 覆盖前 num_positions 个位置（投机一次写 k+1 个新 token）"""
        if self.block_manager is None:
            return False
        bs = self.block_size
        need = (num_positions + bs - 1) // bs
        while len(seq.block_table) < need:
            if not self.block_manager.append_block(seq):
                return False
        return True

    @torch.inference_mode()
    def _spec_round(self, seq: Sequence, state: dict) -> List[int]:
        """一个完整 round：draft k 个 -> verify k+1 行 -> 返回本轮全部产出"""
        k = getattr(self.config, 'num_speculative_tokens', 3)
        tokens = seq.token_ids
        S = len(tokens)
        device = self.model.target_model.lm_head.weight.device

        if self._draft_owner != seq.seq_id:
            # draft 的 KV 只有一份缓冲区，被别的 seq 用过就整段重铺
            state['draft_len'] = 0
            self._draft_owner = seq.seq_id

        # ---- Step 1: draft ----
        candidates, draft_len = self.model.draft_round(
            state['feats'], state['feat_start'], tokens, k, state['draft_len'])
        state['draft_len'] = draft_len

        # ---- Step 2: verify（增量 extend，只算新行）----
        # 行 [S-1, S+k-1] 共 k+1 行；第 i 行在位置 S-1+i，预测 s_{S+i}，
        # 正好对着 candidates[i]（candidates[i] 就是 s_{S+i} 的预测）。
        start = S - 1
        end = S + k
        if not self._grow_block_table(seq, end):
            # block 不够就退回普通 decode，绝不超发
            return [self._run_group([seq], is_prefill=False)[0]]

        bs = self.block_size
        if (
            getattr(self, 'spec_verify_graph', None) is not None
            and len(candidates) + 1 == self.verify_m
        ):
            # === CUDA Graph Verify Path (极速硬件重放，消除 92% 验证延迟) ===
            slot_list = [seq.block_table[p // bs] * bs + p % bs for p in range(start, end)]
            self.static_verify_slot.copy_(torch.tensor(slot_list, dtype=torch.int32, device=device))
            self.static_verify_context_lens[0] = end
            self.static_verify_block_tables[0].fill_(-1)
            cur_blocks = min(len(seq.block_table), self.static_verify_block_tables.shape[1])
            self.static_verify_block_tables[0, :cur_blocks].copy_(
                torch.tensor(seq.block_table[:cur_blocks], dtype=torch.int32, device=device)
            )
            self.static_verify_ids.copy_(
                torch.tensor((tokens + candidates)[start:end], dtype=torch.long, device=device)
            )
            self.static_verify_pos.copy_(
                torch.arange(start, end, dtype=torch.long, device=device)
            )

            self.spec_verify_graph.replay()

            predictions = self.static_verify_logits.argmax(dim=-1).tolist()
            feats = {lid: self.static_verify_feats[lid].clone() for lid in self.model.eagle3_config.extract_layers}
        else:
            # === Eager Fallback Path ===
            slot_mapping = torch.tensor(
                [seq.block_table[p // bs] * bs + p % bs for p in range(start, end)],
                dtype=torch.int32, device=device)
            block_tables = torch.tensor(
                [seq.block_table + [-1] * ((end + bs - 1) // bs - len(seq.block_table))],
                dtype=torch.int32, device=device)
            new_ids = torch.tensor(
                (tokens + candidates)[start:end], dtype=torch.long, device=device)
            new_pos = torch.arange(start, end, dtype=torch.long, device=device)

            feats, hidden = self.model.target_extend(
                new_ids, new_pos, prefix_len=start,
                slot_mapping=slot_mapping, block_tables=block_tables)

            logits = F.linear(hidden, self.model.target_model.lm_head.weight)
            predictions = logits.argmax(dim=-1).tolist()

        # ---- Step 3: 接受前缀 ----
        m = 0
        while m < k and predictions[m] == candidates[m]:
            m += 1

        emitted = list(candidates[:m])
        emitted.append(predictions[k] if m == k else predictions[m])

        # ---- 记账 ----
        # target KV 铺到 end；特征覆盖 [start, end-1]，下一轮 boundary 行要
        # f_(S'-2)，S'-2 = start + E - 1 <= start + k = end - 1 ✓
        state['kv_len'] = end
        state['feat_start'] = start
        state['feats'] = feats

        self.spec_stats['total_drafted'] += len(candidates)
        self.spec_stats['total_accepted'] += m
        # 这一步吐给 engine 的 token 数（含最后兜底那一个），也就是
        # 原本需要这么多步普通 decode 才能拿到的量
        self.spec_stats['total_steps'] += len(emitted)
        self.spec_stats['rounds'] += 1
        return emitted

    def get_spec_stats(self) -> dict:
        """获取 speculative decoding 统计信息

        avg_accepted_per_step = total_steps / rounds：每轮 verify 实际吐出的
        token 数。普通 decode 一步只吐 1 个，所以它是"省下来的步数比"的上界；
        真实加速比还要扣掉一次 verify 多跑 k 行的开销，这里不估。
        """
        drafted = self.spec_stats['total_drafted']
        accepted = self.spec_stats['total_accepted']
        rounds = self.spec_stats['rounds']
        steps = self.spec_stats['total_steps']
        if drafted > 0:
            acceptance_rate = accepted / drafted
        else:
            acceptance_rate = 0.0

        avg_accepted = (steps / rounds) if rounds > 0 else 0.0

        return {
            'total_drafted': drafted,
            'total_accepted': accepted,
            'total_steps': steps,
            'rounds': rounds,
            'acceptance_rate': acceptance_rate,
            'avg_accepted_per_step': avg_accepted,
            'speedup': avg_accepted,
            'upper_bound_speedup': avg_accepted,
        }
