# FP8 KV Cache 量化实现

将 KV Cache 从 BF16 量化到 FP8，**显存提升 63.5%**，**量化精度优秀（K 2.38%, V 2.25%）**。

⚠️ **性能警告**: 由于 Flash Attention 不支持 FP8，性能下降 93.7%。仅适用于显存严重不足的场景。

---

## 快速开始

```python
from fastvllm import LLM

# 基础用法：FP8 + 全局 K scale
llm = LLM(
    model="~/huggingface/Qwen3-0.6B/",
    kv_cache_dtype="fp8",
)

# 推荐用法：FP8 + Per-layer K scale（更高精度）
llm = LLM(
    model="~/huggingface/Qwen3-0.6B/",
    kv_cache_dtype="fp8",
    kvcache_k_scale="fp8_k_scales.json",  # 需先校准
)

output = llm.generate("Hello", max_tokens=100)
```

---

## Per-layer K Scale 校准（推荐）

**一次性操作**，显著提升 K 量化精度（2.38% vs 3.16%，改善 24.7%）：

```bash
# 生成 fp8_k_scales.json
./.venv/bin/python calibrate_fp8_kv.py \
    --model ~/huggingface/Qwen3-0.6B/ \
    --out fp8_k_scales.json
```

详见 [Per-layer K Scale 文档](docs/fp8_per_layer_scale.md)

---

## 测试结果总览

| 指标 | BF16 | FP8 (Global) | FP8 (Per-layer) | 变化 |
|------|------|--------------|-----------------|------|
| 显存 (blocks) | 241 | 394 | 394 | **+63.5%** ✅ |
| K 量化误差 | - | 3.16% | **2.38%** | **-24.7%** ✅ |
| V 量化误差 | - | 2.25% | 2.25% | ✅ |
| 吞吐量 (tok/s) | 5381 | 340 | 340 | **-93.7%** ⚠️ |

**环境**: RTX 3080 10GB, Qwen3-0.6B

---

## 测试命令

```bash
# 校准 per-layer K scales（推荐，一次性）
./.venv/bin/python calibrate_fp8_kv.py

# 量化精度测试（真实前向传播）
./.venv/bin/python test_fp8_calibration.py

# 端到端测试
./.venv/bin/python test_fp8_end_to_end.py
./.venv/bin/python test_fp8_end_to_end.py --per-layer

# 性能 benchmark (需要 5 分钟)
./.venv/bin/python bench_fp8_kvcache.py --both
```

---

## 文档导航

1. **[docs/fp8_per_layer_scale.md](docs/fp8_per_layer_scale.md)** ⭐ 新功能
   - Per-layer K scale 实现细节
   - 量化精度对比（3.16% → 2.38%）
   - 使用方法和校准流程

2. **[FP8_FINAL_SUMMARY.md](FP8_FINAL_SUMMARY.md)** ⭐ 推荐阅读
   - 完整测试结果（显存/精度/性能）
   - 核心实现代码
   - 优化方向和适用场景

3. **[docs/fp8_kvcache_summary.md](docs/fp8_kvcache_summary.md)**
   - 快速参考（5 分钟）
   - 核心改动（4 个文件）
   - 面试话术

4. **[docs/nano_vllm_insights.md](docs/nano_vllm_insights.md)**
   - nano-vllm 优秀思路对比
   - Per-layer scale 的来源和分析

5. **[docs/fp8_precision_report.md](docs/fp8_precision_report.md)**
   - 详细精度测试报告（数值层面）
   - K/V 量化方案对比

---

## 核心改动

修改了 5 个文件，新增 3 个测试脚本：

**核心实现**：
1. **fastvllm/config.py** - 添加 `kv_cache_dtype` 和 `kvcache_k_scale` 参数
2. **fastvllm/engine/model_runner.py** - 分配 FP8 KV Cache + V scale + K scale tensor
3. **fastvllm/layers/attention.py** - 量化/反量化逻辑
4. **fastvllm/models/qwen3.py** - 传递 layer_idx 给 Attention
5. **fastvllm/models/modeling_utils.py** - 传递 k_scale_cache 到 Attention

**测试和校准**：
1. **calibrate_fp8_kv.py** - Per-layer K scale 校准脚本
2. **test_fp8_calibration.py** - 真实前向传播量化精度测试
3. **test_fp8_end_to_end.py** - 端到端文本生成测试
4. **bench_fp8_kvcache.py** - 性能 benchmark

---

## 性能瓶颈

**根本原因**: Flash Attention 不支持 FP8 dtype

- 每次 forward 必须反量化整个 KV Cache（394 blocks = 100k tokens）
- 反量化时间占 94.5%

**优化方向**:
- 短期: 按需反量化（预期延迟 10ms → 15ms）
- 长期: FP8 Native Attention（需要 Hopper GPU）

---

## 何时使用 FP8？

### ✅ 适用场景
- 显存严重不足，必须量化才能运行
- Prefill-heavy 工作负载
- 对延迟要求不严格

### ❌ 不适用场景
- Decode-heavy 低延迟场景
- GPU 显存充足
- 实时推理服务

---

## 技术亮点

1. **Per-layer K Scale**: 每层独立校准 K scale（改善 24.7%）
2. **量化精度优秀**: K 平均相对误差 2.38%，V 2.25%（远好于理论 12.5%）
3. **K 静态 + V 动态**: K 用 per-layer scale，V 用 per-token scale
4. **CUDA Graph 兼容**: 使用 `torch.where` + `scatter_` 避免条件分支
5. **软件实现**: PyTorch 量化，兼容所有 GPU（不需要 Hopper）

---

*实现参考: [nano-vllm](https://github.com/user/nano-vllm)*
