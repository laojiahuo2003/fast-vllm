# SGLang Eagle3 实现分析

## 核心发现

### 1. SGLang 的增量计算机制

**关键点：SGLang 使用 `out_cache_loc` 来指定每一步写入 KV cache 的位置**

```python
# 在 prepare_for_draft 中预先分配所有步骤的 cache 位置
batch.out_cache_loc = torch.empty(
    (bs * topk * num_steps,),  # 预分配所有步骤
    dtype=torch.int64,
    device=batch.device,
)
assign_draft_cache_locs_contiguous(...)  # 分配连续的 slots

# 在 draft_forward 中逐步使用
out_cache_loc = per_step_draft_out_cache_loc(
    out_cache_loc,
    forward_batch.batch_size,
    self.topk,
    self.speculative_num_steps,
)

# 每一步使用不同的 cache_loc
for i in range(self.speculative_num_steps):
    forward_batch.out_cache_loc = out_cache_loc[i]  # 第 i 步的位置
    logits_output = self.draft_runner.forward(forward_batch)
```

**工作原理**：
- 第 0 步：写入 slots [seq_len, seq_len+1, ...]
- 第 1 步：写入 slots [seq_len+num_steps, seq_len+num_steps+1, ...]  
- 第 2 步：写入 slots [seq_len+2*num_steps, ...]
- 每步 forward 都**读取之前写入的 cache**，只计算新 token

### 2. ForwardBatch 的关键字段

```python
class ForwardBatch:
    input_ids: torch.Tensor       # 当前步要计算的 token(s)
    positions: torch.Tensor       # 对应的 position IDs
    out_cache_loc: torch.Tensor   # 写入 KV cache 的物理位置
    seq_lens: torch.Tensor        # 每个请求的当前序列长度
    spec_info: EagleDraftInput    # spec 相关信息（hidden_states 等）
```

**每步的变化**：
```python
for i in range(num_steps):
    # 只计算 topk 个候选 token
    forward_batch.input_ids = topk_index.flatten()  # [batch * topk]
    forward_batch.out_cache_loc = out_cache_loc[i]  # 第 i 步的位置
    forward_batch.positions.add_(1)                 # position 递增
    
    # forward 会：
    # 1. 读取 cache 中 [0, position) 的 KV
    # 2. 计算当前 token 的 KV
    # 3. 写入到 out_cache_loc[i] 指定的位置
    logits = self.draft_runner.forward(forward_batch)
```

### 3. 与 fast-vllm 的对比

| 维度 | SGLang | fast-vllm (当前实现) |
|------|--------|---------------------|
| Cache 管理 | 全局 KV cache，通过 `out_cache_loc` 指定写入位置 | 试图创建独立的 draft_kv_cache |
| 增量计算 | 每步只 forward 1 个 token，读取之前的 cache | 每步 forward 整个序列（重复计算）|
| slot_mapping | 预先分配所有步骤的连续 slots | 每步重新计算，没有复用 |
| Context 模式 | decode 模式（batch_size=topk, seq_len递增）| 试图用 temp_kv_cache |

## Fast-vllm 需要的修改

### 问题诊断
当前 fast-vllm 的 draft_generate 性能差的根本原因：

```python
# 错误做法（当前）
for step in range(num_tokens):
    # 每步都重新计算整个序列
    features = self._forward_without_cache(
        entire_sequence,  # 越来越长！
        positions
    )
```

这就像每次写信都要重抄前面所有内容，当然慢。

### 正确做法（学习 SGLang）

```python
# 第 0 步：prefill，写入初始序列的 KV
slot_mapping = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]
set_context(is_prefill=True, slot_mapping=slot_mapping, ...)
features = self.target_model(initial_ids, initial_pos)

# 第 1 步：decode，只计算 1 个新 token
new_slot = seq_len  # 写入到下一个 slot
slot_mapping = torch.tensor([new_slot])
context_lens = torch.tensor([seq_len+1])  # 读取前面 seq_len+1 个 KV
set_context(is_prefill=False, slot_mapping=slot_mapping, 
            context_lens=context_lens, block_tables=...)
features = self.target_model(new_token_tensor, new_pos_tensor)

# 第 2 步：decode，继续
new_slot = seq_len + 1
slot_mapping = torch.tensor([new_slot])
context_lens = torch.tensor([seq_len+2])
...
```

**关键变化**：
1. **第 0 步用 prefill 模式** 写入初始序列
2. **后续步用 decode 模式** 每步只计算 1 个 token
3. **slot_mapping 递增** 写入连续的位置
4. **context_lens 递增** 告诉 attention 读取多少 KV
5. **block_tables** 指向全局 KV cache 的物理位置

### 为什么不需要独立的 draft_kv_cache？

SGLang 直接使用全局 KV cache，通过 `req_to_token` 的空闲区域：

```python
# 全局 cache 布局（以一个请求为例）
# [0 ... seq_len-1]: 已提交的 prefix KV
# [seq_len ... seq_len+num_steps*topk]: draft 候选的 KV
#   - 如果接受：移动到 prefix 区域
#   - 如果拒绝：标记为空闲，下次覆盖

# 不需要额外分配，只是借用全局 cache 的一部分
```

Fast-vllm 可以做同样的事：
- 全局 KV cache 已经存在（在 model_runner 中）
- Draft 时写入 `[seq_len, seq_len+num_draft]` 这段
- 验证后决定保留还是丢弃

## 实现路线图

### 短期：最小修改实现增量计算

1. **修改 draft_generate**：
   - 第 0 步：prefill 模式，写入初始序列
   - 后续步：decode 模式，每步 1 token
   - 使用全局 KV cache，不创建独立 cache

2. **修改 Context**：
   - 确保 decode 模式下 `context_lens` 和 `block_tables` 正确
   - slot_mapping 递增指向连续位置

3. **修改 Attention**：
   - decode 模式：读取 cache 中的 KV，计算新 token
   - 不需要 temp_kv_cache 参数

### 中期：优化 slot 分配

像 SGLang 一样预先分配：
```python
draft_slots = allocate_draft_slots(seq_len, num_tokens)
# [seq_len, seq_len+1, ..., seq_len+num_tokens-1]

for i, slot in enumerate(draft_slots):
    slot_mapping = torch.tensor([slot])
    ...
```

### 长期：CUDA Graph 加速

SGLang 用 CUDA graph 优化 draft forward，但这是锦上添花。
先实现正确的增量计算，性能就会提升 10x+。

## 总结

**核心洞察**：
- SGLang 没有 "临时 KV cache"，只有 "全局 KV cache 的一段临时区域"
- 增量计算的关键是每步只 forward 1 个 token，读取之前写入的 KV
- slot_mapping 和 context_lens 是控制读写位置的关键

**Fast-vllm 要做的**：
1. 放弃 `_forward_without_cache`（它绕过了 KV cache）
2. 第 0 步用 prefill 写入初始序列到全局 cache
3. 后续步用 decode 读取 cache 并写入新 token
4. slot_mapping 指向全局 cache 的 `[seq_len, seq_len+num_draft]` 区域
