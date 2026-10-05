# Eagle3 快速入门

## 5 分钟上手 Eagle3 Speculative Decoding

### 1. 安装依赖

```bash
pip install transformers safetensors
```

### 2. 准备模型

下载 Qwen3 和对应的 Eagle3 draft model：

```bash
# Target model
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B

# Eagle3 draft model
huggingface-cli download Nicolassuez/Qwen3-0.6B-eagle3 --local-dir ~/huggingface/Qwen3-0.6B-eagle3
```

### 3. 运行示例

```python
from fastvllm.engine.llm_engine import LLMEngine
from fastvllm.sampling_params import SamplingParams

# 初始化引擎（启用 Eagle3）
engine = LLMEngine(
    model="/path/to/Qwen3-0.6B",
    enable_eagle3=True,
    eagle3_model_path="/path/to/Qwen3-0.6B-eagle3",
)

# 生成
sampling_params = SamplingParams(temperature=0.01, max_tokens=100)
outputs = engine.generate(["What is AI?"], sampling_params)

print(outputs[0]['text'])

# 查看加速效果
stats = engine.model_runner.get_spec_stats()
print(f"加速比: {stats['speedup']:.2f}x")
```

### 4. 运行测试

```bash
cd /path/to/fast-vllm
python test_eagle3_final.py
```

这将对比有/无 Eagle3 的性能。

## 核心概念

### 什么是 Speculative Decoding？

传统生成：每次只生成 1 个 token
```
Target: 生成 token 1 → 生成 token 2 → 生成 token 3 → ...
```

Speculative Decoding：draft model 一次生成 K 个候选，target model 并行验证
```
Draft:  生成 [token 1, 2, 3, 4]
Target: 并行验证 [✓, ✓, ✗, -]  → 接受前 2 个
```

### Eagle3 的优势

- **无需独立 draft model**: 只需训练一个轻量级 head（~10MB）
- **高接受率**: 利用 target model 的中间层特征，接受率 > 80%
- **低开销**: draft head 很小，几乎不增加显存

## 性能调优

### 调整候选数量

```python
engine = LLMEngine(
    ...,
    num_speculative_tokens=4,  # 默认 4，范围 2-8
)
```

- 越大：单次生成越多候选，但 overhead 也越大
- 越小：overhead 小，但加速效果有限
- 建议：4-6

### 选择提取层

```python
engine = LLMEngine(
    ...,
    eagle3_extract_layers=[1, 13, 24],  # 浅层、中层、深层
)
```

对于 N 层模型：
- 浅层：~N/4
- 中层：~N/2
- 深层：~N-1

## 常见问题

### Q: 为什么接受率很低？

A: 可能原因：
1. Eagle3 模型和 target model 不匹配
2. `temperature` 太高（建议 < 0.5）
3. 任务难度大（接受率会自然降低）

### Q: 为什么没有加速？

A: 可能原因：
1. 序列太短（< 20 tokens），overhead 占比太大
2. batch size > 1（当前实现针对单序列优化）
3. 接受率太低（< 50%）

### Q: 可以用在其他模型上吗？

A: 可以，但需要：
1. 为该模型训练 Eagle3 draft head
2. 适配模型的中间层提取代码

## 更多信息

- 完整文档：[EAGLE3_INTEGRATION.md](EAGLE3_INTEGRATION.md)
- 技术总结：[EAGLE3_SUMMARY.md](EAGLE3_SUMMARY.md)
- 测试脚本：`test_eagle3_final.py`
