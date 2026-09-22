from collections import deque
import xxhash
import numpy as np

from fastvllm.engine.sequence import Sequence

# 物理显存块元数据
class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0# 引用计数
        self.hash = -1# 内容哈希
        self.token_ids = []# 内容token_ids的副本
    # 当一个块被完整填满并完成前向计算后，调用此方法将其固化为只读的缓存块。
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids
    # 从空闲池捞出一个块重新分配时调用。
    def reset(self):
        self.ref_count = 1# reset的时候说明有人要用这个块，就直接置1
        self.hash = -1
        self.token_ids = []

# 物理显存块管理器
class BlockManager:
    # 一共有多少物理block，每个物理块可容纳的Token数量
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size# 每个物理块可容纳的Token数量
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]# 物理块列表（是元数据对象，不是创建 GPU KV Cache Tensor）
        self.hash_to_block_id: dict[int, int] = dict()# 前缀缓存的核心索引字典（hash → block_id）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
    #前一个block的hash完全相同 + 当前块的 token_ids 完全相同 = 算出来的数字相同
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        # 如果有前缀，那就加起来链式hash
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()# 返回最终的整数hash
    # 分配一个空闲的物理块，返回ID
    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise IndexError("KV cache block pool exhausted")  # 最后防线：不应被调到空池
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash] # 只有真的用到这个块，才把他清除映射（惰性淘汰）
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id
    # 删除一个物理块，将其放回空闲池
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0# 只有引用计数为0的块才能被释放
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
    # 这个 Sequence 如果现在要运行，需要多少新 Block？有多少旧 Block 可以复用？显存够不够？
    def can_allocate(self, seq: Sequence) -> int:
        # 如果seq已经缓存过prefix信息，直接使用
        if hasattr(seq, '_cached_prefix_blocks') and seq._cached_prefix_blocks is not None:
            num_cached_blocks = seq._cached_prefix_blocks
            num_new_blocks = seq.num_blocks - (num_cached_blocks if seq._cached_prefix_blocks in self.used_block_ids else 0)
            if len(self.free_block_ids) < num_new_blocks:
                return -1
            return num_cached_blocks

        h = -1
        num_cached_blocks = 0# 有多少block可以从prefix cache中复用
        num_new_blocks = seq.num_blocks# 序列目前总共需要多少新block
        for i in range(seq.num_blocks - 1):# 序列最后的块还没满，所以不缓存，只缓存完整的block
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 如果hash不存在，或者token_ids不匹配，就跳出循环
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break# 一旦不匹配就跳出循环
            num_cached_blocks += 1# 当前block可以复用，所以加1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1 # 如果这个block在使用队列中，那么就不需要再分配新block（修正seq的内部状态）

        # 缓存结果，避免重复计算
        seq._cached_prefix_blocks = num_cached_blocks

        if len(self.free_block_ids) < num_new_blocks:
            return -1# 空闲block不够，返回-1表示不能分配
        return num_cached_blocks# 可以复用多少个完整 Prefix Block
    # 建立该seq的block table，传入seq和可以复用的block
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        # 可以复用部分
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 需要新分配部分
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size # 多少token来自缓存

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # recompute-mode重计算 抢占：只释放 seq 尾部若干 Block 回 free，保留前面已算好的 KV。
    # num_cached_tokens 回退到保留 Block 覆盖的 token 数，重新调度时只需重算尾部那段。
    def deallocate_tail(self, seq:Sequence, num_blocks_to_free: int =1):
        for _ in range(num_blocks_to_free):
            if not seq.block_table:
                break
            # 释放序列最后一块block
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count-=1
            if block.ref_count==0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = len(seq.block_table) * self.block_size # 重新计算序列已缓存的token数量
        
    # 给被抢占的seq补齐尾部缺失的Block；空余不足直接返回False
    def ensure_tail_blocks(self,seq: Sequence)->bool:
        # seq.num_blocks是根据总token算出来的总共block数量
        missing = seq.num_blocks - len(seq.block_table)
        if missing <=0:
            return True
        if len(self.free_block_ids)<missing:
            return False
        for _ in range(missing):
            seq.block_table.append(self._allocate_block())
        return True

    # seq追加token的时候，是否有足够的空闲block？
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)# 追加一个token是不是会跨一个新block
    # 如果需要新block就分配一个；free 池不足时不分配，返回 False。
    def may_append(self, seq: Sequence) -> bool:
        if len(seq) % self.block_size == 1:
            if not self.free_block_ids:
                return False
            seq.block_table.append(self._allocate_block())
        return True
    # 把完整算好的block放入hash表
    def hash_blocks(self, seq: Sequence):
        # 开始block
        start = seq.num_cached_tokens // self.block_size
        # 结束block
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
