# Eagle3 集成完整指南

## 📁 新增文件清单

为了不修改原有代码，新增了以下文件：

```
fastvllm/
├── models/
│   ├── qwen3.py                      # 原有文件，保持不变 ✅
│   └── qwen3_eagle3.py               # 新增：Qwen3 + Eagle3 集成模型
├── engine/
│   ├── model_runner.py               # 原有文件，保持不变 ✅
│   └── eagle3_model_runner.py        # 新增：支持 Eagle3 的 ModelRunner
└── eagle3_llm.py                      # 新增：Eagle3 版 LLM Engine

example_eagle3.py                      # 新增：使用示例
```

**原有文件完全不变**，你可以随时切换回原版：
- 用原版：`from fastvllm import LLM`
- 用 Eagle3：`from fastvllm.eagle3_llm import LLMWithEagle3`

---

## 🚀 快速开始

### 1. 确保 Eagle3 模型已下载

```bash
ls ~/huggingface/Qwen3-0.6B-eagle3/
# 应该看到: config.json  model.safetensors
```

### 2. 运行示例

```bash
cd /home/uos/code/fast-vllm
uv run python example_eagle3.py
```

### 3. 在你的代码中使用

```python
from fastvllm.eagle3_llm import LLMWithEagle3
from fastvllm.sampling_params import SamplingParams

# 创建 LLM（带 Eagle3）
llm = LLMWithEagle3(
    model="~/huggingface/Qwen3-0.6B/",
    eagle3_model_path="~/huggingface/Qwen3-0.6B-eagle3/",
    num_speculative_tokens=4,  # 每次推测 4 个 token
    enforce_eager=True,
)

# 使用（API 和原版完全一样）
outputs = llm.generate(
    prompts=["Hello, world!"],
    sampling_params=SamplingParams(temperature=0, max_tokens=100)
)

# 查看加速统计
stats = llm.get_spec_stats()
print(f"接受率: {stats['acceptance_rate']:.2%}")
print(f"加速比: {stats['avg_speedup']:.2f}x")
```

---

## 🔧 技术实现

### 架构设计

```
┌─────────────────────────────────────────────────┐
│  LLMWithEagle3 (eagle3_llm.py)                  │
│  - 向后兼容的 API                                │
│  - 管理统计信息                                  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Eagle3ModelRunner (eagle3_model_runner.py)    │
│  - 继承 ModelRunner                             │
│  - 添加 speculative decoding 逻辑               │
│  - 单序列 decode 时自动切换到 Eagle3           │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Qwen3WithEagle3 (qwen3_eagle3.py)             │
│  - 包装原始 Qwen3ForCausalLM                    │
│  - 集成 Eagle3DraftHead                         │
│  - 通过 hooks 提取中间层特征                    │
└─────────────────────────────────────────────────┘
```

### 关键特性

1. **非侵入式**：原有文件一行不改
2. **自动切换**：
   - 单序列 decode → Eagle3 speculative
   - 多序列 / prefill → 原始方法
3. **完全兼容**：API 和原版 fast-vllm 一致

---

## 📊 性能预期

根据 Eagle3 官方数据：
- Baseline: 553 tok/s
- Eagle3: 955 tok/s (**+73% 加速**)

在你的 RTX 3080 上预期：
- Baseline: ~400-500 tok/s
- Eagle3: ~700-850 tok/s (**1.7-1.8x**)

**影响因素**：
- ✅ Temperature=0（贪心）效果最好
- ✅ 单序列推理加速最明显
- ⚠️ 批量推理收益降低

---

## ⚙️ 调优参数

### num_speculative_tokens

```python
llm = LLMWithEagle3(
    model="...",
    num_speculative_tokens=3,  # 保守：更稳定
    # num_speculative_tokens=4,  # 平衡（默认）
    # num_speculative_tokens=5,  # 激进：可能反而变慢
)
```

**建议**：
- 先用默认值 4
- 如果接受率 < 50%，降到 3
- 如果接受率 > 80%，可以试 5

### eagle3_extract_layers

```python
llm = LLMWithEagle3(
    model="...",
    eagle3_extract_layers=[1, 13, 24],  # 默认：早/中/后期
    # eagle3_extract_layers=[0, 14, 27],  # 变体：首/中/尾
)
```

**建议**：保持默认值，除非你想实验不同的特征提取策略。

---

## 🐛 已知限制

### 1. 批量推理不加速

**原因**：当前实现只对单序列 decode 启用 Eagle3。

**解决**：保持 batch=1，或等后续版本支持。

### 2. KV Cache 管理简化

**当前**：verify 阶段的 KV cache 管理是简化版。

**影响**：可能在长序列时略有性能损失。

**TODO**：完善 `_verify_candidates` 中的 slot_mapping 设置。

### 3. 权重加载可能不完整

**原因**：Eagle3 的权重 key 映射可能需要调整。

**表现**：如果看到 "Missing keys" 警告。

**解决**：检查 `_load_eagle3_weights` 的 key 映射逻辑。

---

## 🔍 调试技巧

### 检查是否真的用了 Eagle3

```python
# 生成后查看统计
stats = llm.get_spec_stats()
if stats['total_steps'] == 0:
    print("⚠️ Eagle3 没有被使用！检查配置。")
else:
    print(f"✅ Eagle3 已启用，接受率: {stats['acceptance_rate']:.2%}")
```

### 对比 baseline

```python
# Baseline（无 Eagle3）
from fastvllm import LLM
llm_base = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=True)
outputs_base = llm_base.generate(prompts, sampling_params)

# Eagle3
from fastvllm.eagle3_llm import LLMWithEagle3
llm_eagle3 = LLMWithEagle3(
    "~/huggingface/Qwen3-0.6B/",
    eagle3_model_path="~/huggingface/Qwen3-0.6B-eagle3/"
)
outputs_eagle3 = llm_eagle3.generate(prompts, sampling_params)

# 对比输出（应该完全一致）
assert outputs_base[0]['text'] == outputs_eagle3[0]['text']
```

---

## 📝 代码质量说明

当前版本是 **功能原型**（Proof of Concept），目的是：
- ✅ 验证 Eagle3 可以集成到 fast-vllm
- ✅ 不修改原有代码
- ✅ 演示加速效果

**生产就绪需要**：
- [ ] 完善 KV cache 管理
- [ ] 支持批量 speculative
- [ ] 权重加载的健壮性
- [ ] 错误处理和 fallback
- [ ] 单元测试

**估计工作量**：再投入 1-2 天可以做到生产级别。

---

## 🎯 下一步

### 立即验证（5分钟）

```bash
cd /home/uos/code/fast-vllm
uv run python example_eagle3.py
```

看看：
- 是否能正常运行？
- 接受率多少？
- 加速比如何？

### 如果效果好（值得继续）

1. **完善 KV cache 管理**（最重要）
2. 支持批量 speculative
3. 写测试验证正确性

### 如果效果不好（接受率 < 40%）

可能原因：
- Eagle3 权重没正确加载
- 特征提取有问题
- 或者这个 Eagle3 head 本身就不够好

---

## 💡 使用建议

**短期**：
- 用新的 `LLMWithEagle3` 测试你的场景
- 保留原版 `LLM` 作为 baseline
- 对比性能和正确性

**中期**：
- 如果加速明显，投入时间完善
- 如果效果一般，考虑其他优化（FP8, etc.）

**长期**：
- 等 SGLang 或 vLLM 官方的 Eagle 支持成熟
- 或者自己维护这个 fork

---

想现在试试吗？我可以帮你运行 `example_eagle3.py` 看看效果。
