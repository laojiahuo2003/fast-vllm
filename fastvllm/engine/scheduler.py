from collections import deque

from fastvllm.config import Config
from fastvllm.engine.sequence import Sequence, SequenceStatus
from fastvllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 最大并发序列数量
        self.max_num_batched_tokens = config.max_num_batched_tokens# 每一轮最多凑多少token
        self.eos = config.eos# 结束符号
        self.block_size = config.kvcache_block_size# block大小
        self.prefill_chunk_tokens = config.prefill_chunk_tokens # per-request
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
    # 同一个Step中，Decode严格优先 + Token Budget
    def schedule(self) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0
        num_scheduled_decodes = 0# 本次有多少decode的序列
        # 阶段一：Decode优先
        # docode对延迟敏感，必须先拿到Token Budget。并且每个seq每步只有1个token
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            if num_batched_tokens >= self.max_num_batched_tokens:
                break # 预算已经满了
            seq = self.running.popleft()
            # decode 追加 1 token 最多需要 1 个新 block；free 池不足时先抢占其他 running 释放，
            # 实在凑不够（连自己都释放不动）就放弃本 seq 这一步的 decode，绝不空池崩溃。
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # 调度成功
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                if not self.block_manager.may_append(seq):
                    # 兜底：can_append 通过后 free 又被抢占耗尽（理论上不应发生），
                    # 放回 running 下个 step 再试，不崩溃。
                    self.running.appendleft(seq)
                    continue
                scheduled_seqs.append(seq)
                num_scheduled_decodes += 1
                num_batched_tokens += 1
        if num_scheduled_decodes:
            self.running.extendleft(reversed(scheduled_seqs[:num_scheduled_decodes]))
        # 阶段二：用剩余Token Budget去调度Chunked Prefill
        remaining = self.max_num_batched_tokens - num_batched_tokens
        # 优化：限制扫描深度，避免O(n²)遍历
        MAX_SCAN_DEPTH = 32  # 最多检查前32个waiting请求
        waiting_idx = 0
        scanned = 0

        # 动态调整chunk size：如果剩余预算充足且并发度低，使用更大的chunk
        effective_chunk_size = self.prefill_chunk_tokens
        if self.prefill_chunk_tokens > 0 and remaining > self.prefill_chunk_tokens * 4:
            # 剩余预算充足时，允许更大的chunk以提高GPU利用率
            effective_chunk_size = min(remaining // 2, self.prefill_chunk_tokens * 4)

        while (waiting_idx < len(self.waiting) and len(scheduled_seqs)<self.max_num_seqs and remaining>0 and scanned < MAX_SCAN_DEPTH):
            seq = self.waiting[waiting_idx]
            scanned += 1

             # 是否已经分配过 kvcache block？(没分配就是第一次prefill)
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)# 可以复用多少缓存
                if num_cached_blocks == -1:
                    # 空闲 block 不够这个 seq，跳过它，试试下一个 seq
                    waiting_idx += 1
                    continue
                # 还剩多少 token 需要真正计算（已缓存的 block 直接复用）
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 被 partial 抢占后 block_table 可能缺尾部块，先补齐（重新分配）
                if not self.block_manager.ensure_tail_blocks(seq):
                    waiting_idx += 1  # 空余 block 不够，跳过这个 seq
                    continue
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # per-request 分块上限：单个请求单步最多推进 prefill_chunk_tokens
            request_quota = remaining
            if effective_chunk_size > 0:
                request_quota = min(request_quota, effective_chunk_size)# 分块大小设置
            # 真正分配 block（只在第一次进入 prefill 时建立 block_table）
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            # 该 seq 本次 step 处理的 token 数
            seq.num_scheduled_tokens = min(num_tokens, request_quota)# 序列计划处理token数
            seq.is_prefill = True
            num_batched_tokens += seq.num_scheduled_tokens
            remaining -= seq.num_scheduled_tokens  # 在本 step 内扣减配额
            scheduled_seqs.append(seq)
            # 单请求处理完 → 转入 running；否则留在 waiting 下个 step 继续 chunk
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                del self.waiting[waiting_idx]  # 移除；不递增索引（后一个 seq 顶上）
                self.running.append(seq)
            else:
                waiting_idx += 1  # 未完：留在 waiting，跳到下一个 seq
            if remaining <= 0:
                break

        assert scheduled_seqs
        return scheduled_seqs
                

    # recompute-mode：只释放尾部 1 个 block（保留前面 KV），重算时只算尾部，避免整段 prompt 重算。
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate_tail(seq, 1)
        self.waiting.appendleft(seq)
    
    # 每步结束后更新状态。is_prefill 逐条取 seq.is_prefill，兼容同一步内 decode+prefill 混合。
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # prefill 阶段还没结束（chunk 未完），留在 waiting 等下个 step 继续
            if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue# 改成按照seq来判断阶段，而不是整个调度序列都是同一个状态
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
