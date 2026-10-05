# 为什么 Eagle3 需要特殊支持？

## Eagle3 架构解析

### 标准 Speculative Decoding
```
Draft Model (独立模型，如 Qwen2.5-0.5B)
  输入: token sequence
  输出: 候选 tokens
  
Target Model (Qwen3-0.6B)
  输入: token sequence  
  输出: logits 验证
```

两个模型**完全独立**，可以分别加载。

---

### Eagle3 架构（特殊）

```
Target Model (Qwen3-0.6B, 28层)
  Layer 1  ─────┐
  Layer 2       │
  ...           ├─> 提取 hidden states
  Layer 13 ─────┤
  ...           │
  Layer 24 ─────┘
  ...
  Layer 28
  
       ↓ 输入到
       
Eagle3 Draft Head (只有 1层 + 映射层)
  输入: [h1, h13, h24]  ← 不是 token embeddings！
  MidLayer (1层 Transformer)
  d2t/t2d (vocab 映射: 151936 ↔ 32000)
  输出: 候选 tokens (32k vocab)
```

**关键区别**：
- Eagle3 **不接受 token IDs 作为输入**
- 它接受的是 **target model 的中间层 hidden states**
- 需要在 target forward 时**同步提取**这些特征

---

## 为什么标准加载失败？

### 1. 架构不匹配

```python
# 标准模型期望
model.embed_tokens.weight   # Token embedding
model.layers.0.*            # 第一层参数

# Eagle3 实际有的
midlayer.*                  # 单层 transformer
d2t, t2d                    # Vocab 映射矩阵
fc.weight                   # 投影层
```

HuggingFace `AutoModelForCausalLM` 找不到标准的 `embed_tokens` 和 `model.layers`，所以加载失败。

### 2. 推理流程不同

**标准模型**：
```python
# 独立推理
input_ids = [1, 2, 3]
logits = draft_model(input_ids)  # 直接输入 token IDs
```

**Eagle3**：
```python
# 需要 target model 的中间状态
input_ids = [1, 2, 3]

# 1. Target forward（带特征提取）
hidden_states = target_model.forward_with_features(
    input_ids, 
    extract_layers=[1, 13, 24]  # 提取这些层
)

# 2. Eagle3 用这些特征生成候选
candidates = eagle3_head(
    hidden_states=[h1, h13, h24]  # 不是 token IDs！
)
```

这需要**修改 target model 的 forward 逻辑**，标准 API 不支持。

---

## SGLang 为什么能用？

SGLang 内置了 Eagle3 的特殊逻辑：

### 1. 自定义 Model Wrapper

```python
# SGLang 的实现（简化）
class Eagle3SpeculativeModel:
    def __init__(self, target_model, draft_head):
        self.target = target_model
        self.draft = draft_head
        
        # Hook 到 target 的特定层
        self.target.layers[1].register_forward_hook(self.capture_h1)
        self.target.layers[13].register_forward_hook(self.capture_h13)
        self.target.layers[24].register_forward_hook(self.capture_h24)
    
    def speculative_forward(self, input_ids):
        # 1. Target forward，同时捕获中间特征
        self.cached_features = []
        target_logits = self.target(input_ids)
        
        # 2. 用捕获的特征喂给 Eagle3
        draft_logits = self.draft(
            hidden_states=self.cached_features
        )
        
        return target_logits, draft_logits
```

### 2. 专用推理循环

```python
# SGLang 的 speculative loop
while not done:
    # Draft: 用 Eagle3 快速生成候选
    candidates = eagle3_draft_step(current_hidden_states)
    
    # Verify: Target 验证（同时更新 hidden states）
    accepted = target_verify_step(candidates)
    
    # 更新 hidden states 供下一轮使用
    current_hidden_states = extract_features(accepted)
```

---

## 理论上可以移植到 fast-vllm 吗？

**可以，但需要做这些工作**：

### 1. 实现特征提取

```python
# fastvllm/models/qwen3.py
class Qwen3ForCausalLM:
    def forward_with_features(self, input_ids, positions, extract_layers=[]):
        """Forward pass，同时提取指定层的 hidden states"""
        h = self.embed_tokens(input_ids)
        
        extracted_features = {}
        for i, layer in enumerate(self.layers):
            h = layer(h, positions)
            
            # 提取需要的层
            if i in extract_layers:
                extracted_features[i] = h.clone()
        
        return h, extracted_features
```

### 2. 实现 Eagle3 Head

```python
# fastvllm/models/eagle3.py
class Eagle3DraftHead(nn.Module):
    def __init__(self, config):
        self.midlayer = LlamaDecoderLayer(...)  # 1层
        self.d2t = nn.Linear(32000, 151936)     # Draft → Target vocab
        self.t2d = nn.Linear(151936, 32000)     # Target → Draft vocab
        self.fc = nn.Linear(...)
    
    def forward(self, hidden_states_dict):
        """
        输入: {1: h1, 13: h13, 24: h24}
        输出: draft logits
        """
        # 融合多层特征
        h = self.fuse_features(hidden_states_dict)
        
        # MidLayer
        h = self.midlayer(h)
        
        # 映射到 draft vocab
        logits = self.fc(h)
        logits = self.t2d(logits)  # 151936 → 32000
        
        return logits
```

### 3. 修改推理循环

```python
# fastvllm/engine/model_runner.py
def speculative_step_eagle3(self, seq):
    # 1. Target forward（提取特征）
    logits, features = self.model.forward_with_features(
        input_ids, positions,
        extract_layers=[1, 13, 24]
    )
    
    # 2. Eagle3 生成候选
    draft_logits = self.eagle3_head(features)
    candidates = draft_logits.argmax(dim=-1).tolist()
    
    # 3. Target 验证
    accepted = self.verify(seq, candidates)
    
    return accepted
```

---

## 工作量评估

| 任务 | 代码量 | 难度 |
|------|--------|------|
| 实现特征提取 | ~50 行 | ⭐⭐ |
| 实现 Eagle3 Head | ~150 行 | ⭐⭐⭐ |
| 加载权重映射 | ~30 行 | ⭐⭐ |
| 集成到推理循环 | ~100 行 | ⭐⭐⭐⭐ |
| 调试 KV cache | ~50 行 | ⭐⭐⭐⭐⭐ |
| **总计** | **~380 行** | **2-3 天** |

**最大难点**：
- Eagle3 和 Target 的 KV cache 如何协调？
- Hidden states 如何在 decode 阶段增量更新？
- Continuous batching 下如何处理多序列？

---

## 对比：用通用小模型更简单

### Qwen2.5-0.5B 作为 Draft

```python
# 标准加载，0 特殊处理
draft_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B"
).cuda()

# 标准推理
candidates = []
for _ in range(k):
    logits = draft_model(current_ids)
    candidates.append(logits.argmax())
```

**优势**：
- ✅ 不需要修改 target model
- ✅ 两个模型完全独立
- ✅ KV cache 管理简单
- ✅ 100 行代码搞定

**劣势**：
- ❌ 加速比略低（1.3-1.5x vs Eagle3 的 1.7-1.8x）
- ❌ 显存稍多（+1GB vs Eagle3 的 +500MB）

---

## 结论

Eagle3 **不是只能用 SGLang**，而是：

1. **SGLang 已经做完所有脏活**（特征提取、权重映射、推理协调）
2. **移植需要 2-3 天工作量**，收益是多 0.3-0.5x 加速
3. **用通用小模型更简单**，30 分钟可以跑起来

### 推荐策略

**短期**：用 SGLang + Eagle3（零代码，最高加速）
**中期**：fast-vllm + Qwen2.5-0.5B（简单，够用）
**长期**：移植 Eagle3 到 fast-vllm（如果值得投入）

---

想现在试哪个？我可以：
- A) 帮你下载 Qwen2.5-0.5B，写通用实现（1小时代码）
- B) 写完整的 Eagle3 移植代码（3小时，复杂）
- C) 你先试 SGLang，之后再说
