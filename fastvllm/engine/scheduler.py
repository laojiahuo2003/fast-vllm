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
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
    # 调度
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill：存在等待队列，并且可以处理的seq数量还没到上限
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            # 本轮还有多少token预算
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            # 如果是序列第一次调度（新seq）
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)# 可复用多少block
                if num_cached_blocks == -1:
                    break
                # 需要新算的token数量
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)# 该序列本轮调度的token数量
            num_batched_tokens += seq.num_scheduled_tokens# 本step调度的token数量
            # 如果该序列已经prefill完成
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
        # 出了循环，返回seqs和true
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode：只有prefill都完成了，才开始decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # seq继续增加一个token，如果block不够用，进入
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 如果还有其他seq在decode阶段，从队尾拿一个释放他的空间
                    self.preempt(self.running.pop())
                else:
                    # 否则释放自己
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False
    # 释放seq占用的block，把seq放回waiting队列
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
    # 后处理seqs
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
