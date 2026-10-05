# Fast-vLLM Eagle3 Speculative Decoding

✅ **Eagle3 已完全集成到 fast-vllm 的 ModelRunner**

通过引入**临时 KV cache 机制**，实现了高效的 speculative decoding，无需修改核心 KV cache 管理逻辑。

## 快速开始

```python
from fastvllm.engine.llm_engine import LLMEngine
from fastvllm.sampling_params import SamplingParams

# 启用 Eagle3
engine = LLMEngine(
    model="/path/to/qwen3",
    enable_eagle3=True,
    eagle3_model_path="/path/to/eagle3-draft",
)

# 生成（自动使用 Eagle3 加速）
outputs = engine.generate(["Your prompt"], SamplingParams(max_tokens=100))

# 查看统计
stats = engine.model_runner.get_spec_stats()
print(f"接受率: {stats['acceptance_rate']:.2%}")
print(f"加速比: {stats['speedup']:.2f}x")
```

## 核心特性

### 1. 临时 KV Cache 机制

传统问题：draft 生成的候选 tokens 可能被拒绝，如何避免写入全局 KV cache？

解决方案：
```python
# Draft 阶段：设置临时模式
set_context(use_temp_kv_cache=True, slot_mapping=None)

# Attention 自动处理
if context.use_temp_kv_cache:
    # 不写入全局 cache，直接计算 attention
    return flash_attn_varlen_func(q, k, v, ...)
```

### 2. 完全集成的架构

```
LLMEngine
  ├── 检测 enable_eagle3=True
  ├── 自动选择 Eagle3ModelRunner（而非普通 ModelRunner）
  └── Eagle3ModelRunner
        ├── Draft: 生成 K 个候选（使用临时 KV cache）
        ├── Verify: 并行验证（使用临时 KV cache）
        └── Accept: 写入真实 KV cache
```

### 3. 性能监控

内置统计信息：
- **接受率**: 候选 tokens 被接受的比例
- **平均每步接受**: 每次 draft-verify 循环接受的 token 数
- **理论加速比**: 相对于普通生成的加速倍数

## 文档

- 📖 [快速入门](EAGLE3_QUICKSTART.md) - 5 分钟上手
- 📚 [完整文档](EAGLE3_INTEGRATION.md) - API、配置和架构
- 📝 [技术总结](EAGLE3_SUMMARY.md) - 实现细节和设计决策

## 测试

```bash
# 简单测试
python test_eagle3_simple.py

# 完整性能对比（有/无 Eagle3）
python test_eagle3_final.py

# 单独测试（之前的独立测试）
python example_eagle3_single.py
```

## 核心实现

| 文件 | 说明 |
|------|------|
| `fastvllm/utils/context.py` | 添加 `use_temp_kv_cache` 标志 |
| `fastvllm/layers/attention.py` | Attention 支持临时 KV cache |
| `fastvllm/models/qwen3_eagle3.py` | Qwen3 + Eagle3 集成模型 |
| `fastvllm/engine/eagle3_model_runner.py` | Eagle3 ModelRunner 实现 |
| `fastvllm/engine/llm_engine.py` | 自动选择 ModelRunner |

## 性能预期

根据 Eagle3 论文：

| 指标 | 典型值 | 说明 |
|------|--------|------|
| 接受率 | 80-95% | 越高越好 |
| 平均每步接受 | 3-4 tokens | 接近 `num_speculative_tokens` |
| 理论加速比 | 1.5-2.5x | 取决于接受率和 overhead |

## 工作原理

### 传统生成（无 Eagle3）
```
Step 1: 生成 token 1 (写入 KV cache)
Step 2: 生成 token 2 (写入 KV cache)
Step 3: 生成 token 3 (写入 KV cache)
...
总共 N 步
```

### Speculative Decoding（有 Eagle3）
```
Step 1: 
  - Draft: 生成候选 [t1, t2, t3, t4] (临时 KV cache，不写入)
  - Verify: 并行验证 (临时 KV cache，不写入)
  - Accept: [t1, t2] (只写入接受的 2 个到真实 KV cache)

Step 2:
  - Draft: 从 t2 继续生成 [t3', t4', t5', t6']
  - Verify: 并行验证
  - Accept: [t3', t4', t5'] (接受 3 个)
...
总共 N/3 步（如果平均接受 3 个）
```

## 配置参数

```python
LLMEngine(
    model="/path/to/model",
    
    # Eagle3 配置
    enable_eagle3=True,                      # 是否启用
    eagle3_model_path="/path/to/eagle3",    # draft model 路径
    eagle3_extract_layers=[1, 13, 24],      # 提取哪些中间层
    num_speculative_tokens=4,                # 每次生成多少候选
)
```

## 兼容性

- ✅ **支持**: Qwen3 系列
- 🚧 **待支持**: Llama, Mistral, Qwen2 等（需要适配 Eagle3 head）
- ⚠️ **限制**: 当前针对单序列优化，批处理支持待完善

## 依赖

```bash
pip install transformers safetensors
```

## 设计亮点

1. **最小侵入性**: 只在 `Context` 和 `Attention` 添加了少量代码
2. **完全集成**: Eagle3 作为 ModelRunner 变体，与现有架构无缝配合
3. **可扩展**: 为未来支持 Medusa、EAGLE2 等算法奠定基础

## 致谢

- 临时 KV cache 设计：基于 fast-vllm 架构扩展
- Eagle3 实现参考：SGLang、原始 Eagle3 论文
- 模型权重：Nicolassuez/Qwen3-0.6B-eagle3

## License

与 fast-vllm 主项目保持一致。

---

**问题反馈**: [提交 Issue](https://github.com/your-repo/fast-vllm/issues)  
**贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)
