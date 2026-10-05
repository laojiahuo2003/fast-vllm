# Eagle3 优化进展报告

## ✅ 完成的优化

### 1. 支持贪心采样（temperature=0）

**修改文件**:
- `fastvllm/sampling_params.py` - 移除 temperature > 0 的限制
- `fastvllm/layers/sampler.py` - 实现真正的贪心采样逻辑

**实现**:
```python
# sampler.py
if greedy_mask.all():
    return logits.argmax(dim=-1)  # 纯贪心
```

### 2. 去除 HuggingFace 依赖

**修改文件**:
- `fastvllm/models/qwen3_eagle3.py` - 重写 `draft_generate` 使用纯 fast-vllm
- `fastvllm/engine/eagle3_model_runner.py` - 移除 hf_model_path 参数

**实现**:
```python
# 使用 _forward_without_cache（临时 KV cache 模式）
features = self._forward_without_cache(input_ids, positions)
draft_logits = self.draft_head(features)
next_token = draft_logits[0, -1].argmax().item()
```

### 3. 不绕过优化

**实现细节**:
- 使用 `_forward_without_cache` 方法
- 通过 `use_temp_kv_cache=True` 启用临时 KV cache
- 直接使用 fast-vllm 的 target model 提取特征
- Eagle3 draft head 在提取的特征上生成候选

## ⚠️ 发现的新问题

### 性能问题

**观察**:
- Draft generation 非常慢（每步 ~3秒）
- 接受率仍然很低（~25%，每次只接受1个token）
- 总体比baseline还慢

**根本原因**:

1. **重复计算**: 每次 draft 都要做完整的 forward pass
   ```python
   for step in range(num_tokens):
       # 每步都要重新计算整个序列的 hidden states
       features = self._forward_without_cache(
           entire_sequence,  # 越来越长
           positions
       )
   ```
   这就是为什么 HuggingFace 版本使用 `past_key_values` - 避免重复计算。

2. **没有 KV cache 复用**: `use_temp_kv_cache=True` 意味着不写入也不读取 KV cache
   - 第1步计算：tokens [1..8]
   - 第2步计算：tokens [1..9]（重新计算 1..8）
   - 第3步计算：tokens [1..10]（重新计算 1..9）
   - ...

3. **接受率低**: draft model 质量问题
   - d2t 映射可能不正确
   - 或者 Eagle3 权重本身质量不高

## 🔧 需要的进一步优化

### 优先级 1: 启用增量计算

**方案A**: 使用全局 KV cache（但标记为 draft）
```python
# 为 draft 分配独立的 KV cache slots
draft_slots = allocate_temp_slots(seq_len + num_draft)

# 第1步：写入全局 cache
set_context(slot_mapping=draft_slots[:8])
features = self.target_model(tokens[:8])

# 第2步：读取 cache，只计算新token
set_context(slot_mapping=draft_slots[:9])
features = self.target_model(tokens[8:9])

# 最后：清理 draft slots（如果不接受）
```

**方案B**: 实现 draft-specific KV cache
```python
# 专门为 draft 维护一个临时 cache
draft_cache = DraftKVCache()

for step in range(num_tokens):
    # 只计算新token，读取 draft_cache
    features = self.target_model(
        new_token,
        use_cache=draft_cache
    )
```

### 优先级 2: 检查 d2t 映射

当前实现：
```python
model.d2t = model.d2t_diff + torch.arange(draft_vocab_size)
```

可能的问题：
- `d2t_diff` 的含义可能理解错误
- 需要检查 Eagle3 原始实现

### 优先级 3: 预先计算而非增量

**想法**: 一次性计算所有候选的特征
```python
# 生成 K 个候选的 K 个序列
sequences = [
    tokens + [cand1],
    tokens + [cand2],
    tokens + [cand3],
    tokens + [cand4],
]

# 批量计算（共享前缀）
features_batch = self.target_model(sequences)
```

## 📊 当前测试结果

**测试配置**:
- 模型: Qwen3-0.6B + Eagle3
- 采样: temperature=0.0 (贪心)
- 候选数: 4

**观测**:
- ✅ 功能正常：draft generation 工作
- ✅ 无 HuggingFace 依赖
- ✅ 不绕过优化（使用 _forward_without_cache）
- ❌ 性能很差：每步 ~3秒
- ❌ 接受率低：~25%

**速度分析**:
```
Baseline: 50 tokens / 16s = 3.1 tok/s
Eagle3:   30 tokens / 45s = 0.67 tok/s (慢5倍)
```

## 🎯 建议的下一步

### 短期（修复性能）

1. **实现增量计算**: 采用方案A或B，避免重复计算
2. **检查 d2t 映射**: 对比 SGLang 的实现
3. **减少候选数**: 先用 `num_speculative_tokens=2` 测试

### 中期（提升质量）

1. **验证 Eagle3 权重**: 确保正确加载
2. **对比 SGLang**: 逐函数对比，找出差异
3. **添加日志**: 打印中间值，调试接受率

### 长期（优化架构）

1. **批量 draft**: 一次生成多个候选
2. **树状验证**: 提高接受率
3. **自适应策略**: 根据接受率动态调整

## 总结

✅ **已完成**: 
- 贪心采样支持
- 去除 HuggingFace 依赖  
- 使用 fast-vllm 原生实现（不绕过优化）

⚠️ **当前问题**:
- 性能反而变慢（增量计算缺失）
- 接受率仍然很低（d2t 映射或权重问题）

🔧 **核心瓶颈**: 
每次 draft 都重新计算整个序列，需要实现增量计算机制。

---

**现状**: Eagle3 功能完整，但性能未达预期。下一步应该优先解决增量计算问题。
