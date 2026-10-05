# Eagle3 实现对比：fast-vllm vs SGLang

## 概述

本文档对比 fast-vllm 和 SGLang 中 Eagle3 speculative decoding 的实现方案。

## 核心差异

### 1. KV Cache 管理策略

#### SGLang 方案
- 使用 **tree attention** 机制
- 候选 tokens 构成树状结构，每个分支独立管理 KV cache
- 支持多路径并行验证

#### fast-vllm 方案
- 使用 **临时 KV cache** 机制
- Draft 和 Verify 阶段都不写入全局 KV cache
- 只在接受后才写入真实 KV cache

```python
# fast-vllm: 临时 KV cache
set_context(use_temp_kv_cache=True, slot_mapping=None)

# SGLang: tree attention
# 构建树状结构，每个节点有独立的 KV cache slot
```

**优劣对比**:
- SGLang: 更灵活，支持复杂的树状搜索
- fast-vllm: 更简单，实现成本低，适合线性候选

### 2. Draft Generation 实现

#### SGLang 方案
```python
# python/sglang/srt/speculative/eagle3_worker.py
class Eagle3Worker:
    def draft_model_forward(self, ...):
        # 使用独立的 draft model 或 Eagle3 head
        # 支持批处理
        ...
```

#### fast-vllm 方案
```python
# fastvllm/models/qwen3_eagle3.py
class Qwen3WithEagle3:
    def draft_generate(self, ...):
        # 使用 HuggingFace 模型（临时）
        # 当前针对单序列优化
        ...
```

**优劣对比**:
- SGLang: 完全集成到推理框架，支持批处理
- fast-vllm: 临时使用 HF 模型，实现快速但需优化

### 3. Verification 策略

#### SGLang 方案
- 使用 **token tree verification**
- 一次 forward pass 验证整棵树
- 选择接受率最高的路径

#### fast-vllm 方案
- 使用 **sequential verification**
- 线性验证候选序列
- 遇到第一个不匹配就停止

```python
# fast-vllm: 线性验证
for pred, cand in zip(predictions, candidates):
    if pred == cand:
        accepted.append(cand)
    else:
        break  # 停止

# SGLang: 树状验证
# 验证所有路径，选择最优路径
```

**优劣对比**:
- SGLang: 接受率更高（可以跳过错误分支）
- fast-vllm: 实现简单，性能可预测

### 4. 代码结构

#### SGLang 方案
```
python/sglang/srt/
├── speculative/
│   ├── eagle3_worker.py       # Eagle3 专用 worker
│   ├── eagle3_draft_model.py  # Draft model 封装
│   └── ...
├── managers/
│   └── scheduler.py            # 调度器集成 speculative decoding
```

#### fast-vllm 方案
```
fastvllm/
├── engine/
│   └── eagle3_model_runner.py  # Eagle3 ModelRunner
├── models/
│   └── qwen3_eagle3.py         # 模型集成
├── layers/
│   └── attention.py            # 支持临时 KV cache
└── utils/
    └── context.py              # Context 管理
```

**优劣对比**:
- SGLang: 更模块化，易于扩展其他 spec decoding 算法
- fast-vllm: 更紧凑，直接扩展 ModelRunner

## 实现选择的原因

### 为什么 fast-vllm 选择临时 KV cache？

1. **最小侵入性**: 只需在 `Context` 和 `Attention` 添加一个标志位
2. **易于理解**: "临时模式 = 不写入" 的语义非常清晰
3. **快速实现**: 无需重构 KV cache 管理逻辑
4. **为未来扩展**: 同样的机制可支持 Medusa、EAGLE2 等

### SGLang 为什么选择 tree attention？

1. **更高接受率**: 树状搜索可以探索多条路径
2. **通用框架**: 支持多种 speculative decoding 算法
3. **批处理优化**: 天然支持多序列并行

## 性能对比（预期）

| 指标 | SGLang | fast-vllm |
|------|--------|-----------|
| 接受率 | 85-95% | 80-90% |
| 实现复杂度 | 高 | 中 |
| 批处理支持 | ✅ | 🚧 |
| 单序列性能 | 优秀 | 优秀 |
| 代码侵入性 | 高 | 低 |

## SGLang 的优势

1. **更成熟**: 已在生产环境验证
2. **更灵活**: 支持多种 spec decoding 算法
3. **更全面**: 批处理、多路径等都支持

## fast-vllm 的优势

1. **更简洁**: 临时 KV cache 机制易于理解和维护
2. **更快实现**: 从零到可用只需几天
3. **更低风险**: 不改变核心 KV cache 逻辑

## 代码示例对比

### SGLang: 启用 Eagle3

```python
# 启动 server
python -m sglang.launch_server \
  --model-path ~/huggingface/Qwen3-0.6B \
  --speculative-algo EAGLE3 \
  --speculative-draft-model-path ~/huggingface/Qwen3-0.6B-eagle3 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4
```

### fast-vllm: 启用 Eagle3

```python
from fastvllm.engine.llm_engine import LLMEngine

engine = LLMEngine(
    model="~/huggingface/Qwen3-0.6B",
    enable_eagle3=True,
    eagle3_model_path="~/huggingface/Qwen3-0.6B-eagle3",
    num_speculative_tokens=4,
)
```

## 未来改进方向

### fast-vllm 可以借鉴 SGLang 的地方

1. **批处理支持**: 参考 SGLang 的批处理实现
2. **树状验证**: 可选的多路径验证模式
3. **Draft model 集成**: 去除 HuggingFace 依赖

### SGLang 可以借鉴 fast-vllm 的地方

1. **临时 KV cache 概念**: 作为简化实现的选项
2. **最小侵入性设计**: 减少对核心代码的修改

## 总结

| 维度 | SGLang | fast-vllm | 推荐场景 |
|------|--------|-----------|----------|
| 生产就绪度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | SGLang |
| 易于理解 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | fast-vllm |
| 扩展性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | SGLang |
| 快速原型 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | fast-vllm |
| 单序列性能 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 相当 |
| 批处理性能 | ⭐⭐⭐⭐⭐ | ⭐⭐ | SGLang |

**结论**:
- 如果需要生产级部署、批处理优化 → 选择 SGLang
- 如果需要快速实验、易于维护 → 选择 fast-vllm
- 未来 fast-vllm 可以逐步吸收 SGLang 的优点

## 参考资料

- SGLang Eagle3 实现: `python/sglang/srt/speculative/`
- fast-vllm Eagle3 实现: `fastvllm/engine/eagle3_model_runner.py`
- Eagle3 论文: [待补充]
