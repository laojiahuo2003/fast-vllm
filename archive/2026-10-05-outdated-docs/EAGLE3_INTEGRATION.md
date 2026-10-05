# Eagle3 Speculative Decoding 集成文档

## 概述

Eagle3 是一种高效的 speculative decoding 方法，通过训练一个轻量级的 draft head 来预测 target model 的输出，从而加速生成过程。

本实现将 Eagle3 完全集成到 fast-vllm 中，利用临时 KV cache 机制实现了高效的 draft generation 和 verification。

## 核心特性

- **临时 KV Cache**: Draft 阶段不写入全局 KV cache，避免浪费存储和拒绝后的回滚
- **完全集成**: Eagle3 作为 ModelRunner 的一个变体，无缝集成到 LLMEngine
- **自动配置**: 通过简单的配置参数即可启用 Eagle3
- **性能监控**: 内置统计信息，实时监控接受率和加速比

## 使用方法

### 基本用法

```python
from fastvllm.engine.llm_engine import LLMEngine
from fastvllm.sampling_params import SamplingParams

# 初始化引擎（启用 Eagle3）
engine = LLMEngine(
    model="/path/to/qwen3-model",
    enable_eagle3=True,
    eagle3_model_path="/path/to/eagle3-draft-model",
    eagle3_extract_layers=[1, 13, 24],  # 提取的中间层
    num_speculative_tokens=4,            # 每次生成的候选数量
)

# 生成
sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
outputs = engine.generate(["Your prompt here"], sampling_params)

# 查看 Eagle3 统计
stats = engine.model_runner.get_spec_stats()
print(f"接受率: {stats['acceptance_rate']:.2%}")
print(f"加速比: {stats['speedup']:.2f}x")
```

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_eagle3` | bool | False | 是否启用 Eagle3 |
| `eagle3_model_path` | str | None | Eagle3 draft model 的路径 |
| `eagle3_extract_layers` | List[int] | [1, 13, 24] | 提取哪些中间层的特征 |
| `num_speculative_tokens` | int | 4 | 每次生成多少个候选 token |

## 性能指标

Eagle3 的性能主要由以下指标衡量：

- **接受率 (Acceptance Rate)**: 被 target model 接受的候选 token 比例
  - 越高越好，理想情况 > 80%
  
- **平均每步接受 (Avg Accepted per Step)**: 平均每次 draft-verify 循环接受的 token 数量
  - 理想情况接近 `num_speculative_tokens`
  
- **理论加速比 (Speedup)**: 相对于普通生成的加速倍数
  - 计算公式: `(1 + avg_accepted) / (1 + overhead_ratio)`

## 架构说明

### 临时 KV Cache 机制

传统的 speculative decoding 面临一个问题：draft 生成的候选 tokens 可能被拒绝，如果写入了 KV cache，需要复杂的回滚机制。

本实现通过**临时 KV cache 模式**解决了这个问题：

```python
# Draft 阶段：设置临时模式
set_context(
    is_prefill=True,
    use_temp_kv_cache=True,  # 关键标志
    slot_mapping=None,        # 不需要 slot mapping
)

# Attention 层自动检测临时模式
if context.use_temp_kv_cache:
    # 不写入全局 KV cache，直接用当前的 k, v 做 attention
    o = flash_attn_varlen_func(q, k, v, ...)
    return o
```

### 文件结构

```
fastvllm/
├── engine/
│   ├── eagle3_model_runner.py    # Eagle3 ModelRunner 实现
│   └── llm_engine.py              # 自动选择 ModelRunner
├── models/
│   └── qwen3_eagle3.py            # Qwen3 + Eagle3 集成模型
├── layers/
│   └── attention.py               # 支持临时 KV cache 的 Attention
└── utils/
    └── context.py                 # Context 管理（包含 use_temp_kv_cache）
```

## 已知限制

1. **单序列支持**: 当前实现针对单序列生成优化，批处理支持待完善
2. **HuggingFace 依赖**: Draft 阶段临时使用 HuggingFace 模型（未来可优化为纯 fast-vllm 实现）
3. **模型兼容性**: 目前仅支持 Qwen3，其他模型需要适配 Eagle3 head

## 测试

运行完整测试：

```bash
python test_eagle3_final.py
```

这将对比有/无 Eagle3 的性能，并输出详细统计。

## 参考

- Eagle3 论文: [待补充]
- 原始实现: Nicolassuez/Qwen3-0.6B-eagle3
- SGLang 实现: [参考了 SGLang 的 Eagle3 集成]

## 贡献者

- 初始实现: [Your Name]
- 临时 KV cache 设计: 基于 fast-vllm 的架构扩展
