# 通用 Speculative Decoding 方案

## 问题分析

Eagle3 是 SGLang 专用的 draft head，不是独立模型。但 speculative decoding 的思路是通用的。

## 替代方案：用小模型做 Draft

### 方案 1：Qwen2.5-0.5B 做 draft（推荐）

```python
# Target: Qwen3-0.6B (你现在在用的)
# Draft:  Qwen2.5-0.5B (更小，架构兼容)

# 下载 draft model
huggingface-cli download Qwen/Qwen2.5-0.5B \
  --local-dir ~/huggingface/Qwen2.5-0.5B/
```

优势：
- ✅ 完整的自回归模型，可以直接加载
- ✅ 和 Qwen3 架构相似，tokenizer 兼容
- ✅ 只有 500M，显存占用小
- ✅ 预期加速 1.3-1.5x（比 Eagle3 低，但实现简单）

### 方案 2：MiniCPM-1B 做 draft

```bash
huggingface-cli download openbmb/MiniCPM-1B-sft-bf16 \
  --local-dir ~/huggingface/MiniCPM-1B/
```

优势：
- ✅ 1B 参数，质量更好
- ✅ 中文友好
- ⚠️ 需要检查 tokenizer 是否兼容

## 实现步骤

### 1. 修改 fast-vllm 架构

为了支持 speculative decoding，需要：

1. **加载两个模型**：target + draft
2. **Draft 生成候选**：用小模型快速生成 K 个 token
3. **Target 验证**：并行验证这 K 个 token
4. **接受匹配的前缀**

### 2. 核心代码修改

#### `fastvllm/config.py`

```python
@dataclass
class Config:
    # ... 现有字段 ...
    
    # Speculative Decoding
    enable_speculative: bool = False
    draft_model_path: str | None = None
    num_speculative_tokens: int = 4
```

#### `fastvllm/engine/model_runner.py`

在 `__init__` 中加载 draft model：

```python
def __init__(self, config: Config, rank: int, event):
    # ... 原有初始化 ...
    
    # 如果启用 speculative decoding
    if config.enable_speculative and rank == 0:
        self._init_draft_model(config)
```

新增方法：

```python
def _init_draft_model(self, config: Config):
    """初始化 draft model"""
    from fastvllm.models.qwen3 import Qwen3ForCausalLM
    
    # 加载 draft model config
    draft_hf_config = AutoConfig.from_pretrained(config.draft_model_path)
    draft_hf_config.dtype = config.hf_config.dtype
    
    # 创建 draft model
    self.draft_model = Qwen3ForCausalLM(draft_hf_config)
    load_model(self.draft_model, config.draft_model_path)
    
    # Draft model 用独立的小 KV cache
    self._allocate_draft_kv_cache()
    
    self.num_spec_tokens = config.num_speculative_tokens

def _allocate_draft_kv_cache(self):
    """为 draft model 分配 KV cache（容量小于 target）"""
    # 只需要容纳 num_spec_tokens 的 cache
    # 实现略
    pass

@torch.inference_mode()
def draft_generate(self, seq: Sequence, k: int) -> list[int]:
    """用 draft model 生成 k 个候选 token"""
    candidates = []
    input_ids = seq.token_ids[-1:]  # 只需要最后一个 token
    
    for _ in range(k):
        # Draft forward
        logits = self.draft_model.compute_logits(
            self.draft_model(
                torch.tensor([input_ids[-1]], dtype=torch.int64).cuda(),
                torch.tensor([len(seq) + len(candidates)], dtype=torch.int64).cuda()
            )
        )
        
        # 贪心采样（draft 通常用贪心）
        next_token = logits.argmax(dim=-1).item()
        candidates.append(next_token)
        input_ids.append(next_token)
        
        if next_token == self.config.eos:
            break
    
    return candidates

def verify_candidates(self, seq: Sequence, candidates: list[int]) -> tuple[list[int], int]:
    """用 target model 验证候选 token"""
    # 构造验证序列：原序列 + 候选 token（去掉最后一个）
    verify_tokens = candidates[:-1]
    
    # Prefill 模式并行计算所有位置的 logits
    input_ids = torch.tensor(verify_tokens, dtype=torch.int64).cuda()
    positions = torch.arange(
        len(seq), len(seq) + len(verify_tokens), 
        dtype=torch.int64
    ).cuda()
    
    # 准备 context（简化版）
    # ... 设置 slot_mapping 等 ...
    
    logits = self.run_model(input_ids, positions, is_prefill=True)
    predicted = logits.argmax(dim=-1).tolist()
    
    # 逐个比较
    accepted = []
    for pred, cand in zip(predicted, candidates):
        if pred == cand:
            accepted.append(cand)
        else:
            # 不匹配，用 target 的预测
            accepted.append(pred)
            break
    
    # 如果全匹配，接受最后一个候选
    if len(accepted) == len(predicted):
        accepted.append(candidates[-1])
    
    return accepted, len(accepted)
```

#### 在 `run()` 中使用 speculative

```python
def run(self, seqs: list[Sequence]) -> list[int | None]:
    result = [None] * len(seqs)
    
    # Decode 序列
    decode_seqs = [s for s in seqs if not s.is_prefill]
    
    if decode_seqs and hasattr(self, 'draft_model'):
        # 单序列用 speculative
        if len(decode_seqs) == 1:
            seq = decode_seqs[0]
            
            # Draft
            candidates = self.draft_generate(seq, self.num_spec_tokens)
            
            # Verify
            accepted, num_accepted = self.verify_candidates(seq, candidates)
            
            # 返回第一个 token（其余的 scheduler 会处理）
            result[0] = accepted[0]
            
            # 统计
            self.spec_stats['drafted'] += len(candidates)
            self.spec_stats['accepted'] += num_accepted
        else:
            # 批量解码用原方法
            result = self._original_run(seqs)
    else:
        result = self._original_run(seqs)
    
    return result
```

## 使用方法

```python
from fastvllm import LLM, SamplingParams

llm = LLM(
    "~/huggingface/Qwen3-0.6B/",
    enforce_eager=True,
    enable_speculative=True,
    draft_model_path="~/huggingface/Qwen2.5-0.5B/",
    num_speculative_tokens=4,
)

outputs = llm.generate(["Hello"], SamplingParams(max_tokens=100))
```

## 性能预期

| Draft Model | 加速比 | 显存增加 |
|-------------|--------|----------|
| Qwen2.5-0.5B | 1.3-1.5x | +600MB |
| MiniCPM-1B | 1.4-1.6x | +1.2GB |
| (SGLang Eagle3) | 1.7-1.8x | +500MB |

## 实现优先级

1. **现在**：用 SGLang + Eagle3 验证效果（0 代码，10分钟）
2. **如果加速明显**：再花时间移植到 fast-vllm
3. **移植时**：先用 Qwen2.5-0.5B（架构简单），再考虑优化

## 下一步

你想：
- A) 先试 SGLang + Eagle3，看看实际加速效果？
- B) 直接在 fast-vllm 上实现通用方案（Qwen2.5-0.5B）？
- C) 我帮你写个完整的 PR，一次性搞定？
