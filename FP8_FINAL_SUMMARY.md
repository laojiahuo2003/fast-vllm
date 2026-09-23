# FP8 KV Cache 实现总结

## 一句话总结

FP8 KV Cache 量化在 RTX 3080 上**显存提升 63.5%**，**量化精度优秀（误差 2.33%）**，但**性能下降 93.7%**（根本原因：Flash Attention 不支持 FP8，需反量化整个 cache）。

---

## 测试环境

- GPU: RTX 3080 10GB (sm_86, Ampere)
- 模型: Qwen3-0.6B
- 工作负载: 64 个在线短请求 + 周期性注入 2048 token 长 prefill

---

## 完整测试结果

### 1. 显存占用 ✅

```bash
./.venv/bin/python bench_fp8_kvcache.py --both
```

| 配置 | KV Cache Blocks | 提升 |
|------|-----------------|------|
| BF16 baseline | 241 | - |
| FP8 quantized | 394 | **+63.5%** |

- **理论最大值**: +100% (FP8 是 BF16 的一半)
- **实际值**: +63.5% (V 的 per-token FP32 scale 占用额外空间)
- **结论**: 显存优化符合预期

---

### 2. 量化精度 ✅

```bash
./.venv/bin/python test_fp8_precision.py
```

| 指标 | K (静态 scale) | V (动态 scale) | 平均 |
|------|----------------|----------------|------|
| **平均相对误差** | 2.41% | 2.25% | **2.33%** |
| **余弦相似度** | 0.99964 | 0.99965 | **0.9996** |

**详细精度数据**:

**K (Static scale = 1.0)**:
- 最大绝对误差: 0.25
- 平均绝对误差: 0.018
- 最大相对误差: 100% (接近 0 的值)
- 平均相对误差: **2.41%**

**V (Dynamic per-token scale)**:
- 最大绝对误差: 0.14
- 平均绝对误差: 0.018
- 最大相对误差: 29%
- 平均相对误差: **2.25%**

**结论**:
- FP8 E4M3 理论量化步长: ~12.5% (2^-3)
- 实测误差: 2.33% (远好于理论值)
- 动态 scale 效果优于静态 scale
- 对 LLM 推理质量影响微乎其微

---

### 3. 端到端性能 ⚠️

```bash
./.venv/bin/python bench_fp8_kvcache.py --both
```

| 指标 | BF16 | FP8 | 变化 |
|------|------|-----|------|
| **Step P50 延迟** | 10.7 ms | 183.6 ms | **+1615%** |
| **Step P99 延迟** | 17.1 ms | 220.1 ms | +1188% |
| **总吞吐量** | 5381 tok/s | 340 tok/s | **-93.7%** |

**性能瓶颈根本原因**:

Flash Attention 不支持 FP8 dtype → 每次 forward 必须反量化整个 KV Cache

```python
# attention.py:125 - 性能瓶颈位置
if self.is_fp8_kv_cache and k_cache.numel() and v_cache.numel():
    # 反量化整个 cache (394 blocks = 100,864 tokens)
    k_cache, v_cache = dequant_kvcache(k_cache, v_cache, ...)
```

**反量化开销分析**:
- 每个 decode step 反量化: 394 blocks × 256 tokens = **100,864 tokens**
- 反量化路径: FP8 → FP32 → ×scale → BF16 (4 次内存访问)
- 反量化时间: ~173ms / step
- Attention 计算时间: ~10ms / step
- **反量化占比**: 94.5%

---

## 核心实现

### 配置层 (fastvllm/config.py)

```python
kv_cache_dtype: str = "auto"  # "auto" | "bf16" | "fp8"

def __post_init__(self):
    if self.kv_cache_dtype == "fp8":
        self.kv_cache_dtype = torch.float8_e4m3fn
    elif self.kv_cache_dtype == "auto":
        self.kv_cache_dtype = torch.bfloat16
```

### 量化层 (fastvllm/layers/attention.py)

```python
FP8_E4M3_MAX = 448.0

def store_kvcache(..., is_fp8=False, k_scale=1.0, v_scale_cache=None):
    if is_fp8:
        # K: 静态 scale 量化
        k_fp8 = (key.float() / k_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        
        # V: 动态 per-token scale
        v_amax = value.float().abs().amax(dim=(1, 2)).clamp_min(1e-12)
        v_scale = v_amax / FP8_E4M3_MAX
        v_fp8 = (value.float() / v_scale[:, None, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        
        # 存储 V scale (CUDA Graph safe)
        safe_slots = torch.where(slot_mapping >= 0, slot_mapping, torch.zeros_like(slot_mapping))
        v_scale_cache.scatter_(0, safe_slots, v_scale)
        
        # Triton kernel 只负责搬运字节
        key_to_store = k_fp8.view(torch.uint8)
        value_to_store = v_fp8.view(torch.uint8)
```

### 反量化层 (fastvllm/layers/attention.py)

```python
def dequant_kvcache(k_cache, v_cache, v_scale_cache, k_scale):
    """FP8 → BF16"""
    B, bs = k_cache.shape[0], k_cache.shape[1]
    k = k_cache.float() * k_scale
    v = v_cache.float() * v_scale_cache.view(B, bs, 1, 1)
    return k.to(torch.bfloat16), v.to(torch.bfloat16)
```

---

## 优化方向

### 方案 1: 按需反量化 (短期，RTX 3080 可用)

只反量化 `block_tables` 中实际用到的 blocks，而非全量反量化：

```python
# 当前实现 (反量化全部)
k_cache, v_cache = dequant_kvcache(k_cache, v_cache, ...)  # 394 blocks

# 优化后 (按需反量化)
active_blocks = get_active_blocks(block_tables)  # ~1-4 blocks per seq
k_cache, v_cache = dequant_kvcache_selective(k_cache, v_cache, active_blocks)
```

**预期收益**:
- Decode: 反量化从 100k tokens → ~64 tokens (64 seqs × 1 block)
- 延迟: 183ms → ~15ms (仍有 ~4ms 反量化开销)
- 吞吐: 340 tok/s → ~4500 tok/s

### 方案 2: FP8 Native Attention (长期，需 Hopper GPU)

使用支持 FP8 Tensor Core 的 attention kernel:
- Flash Attention 3.x (H100)
- 自定义 Triton FP8 attention kernel

**预期收益**:
- 无反量化开销
- 性能接近 baseline (~5% 内)

---

## 适用场景

### ✅ 推荐使用 FP8

- **显存严重不足**: 必须用量化才能运行
- **Prefill-heavy 工作负载**: 反量化开销摊销在长 context
- **低吞吐容忍**: 对延迟要求不严格

### ❌ 不推荐使用 FP8

- **Decode-heavy 低延迟场景**: 93.7% 吞吐下降无法接受
- **GPU 显存充足**: 没有必要牺牲性能换显存
- **实时推理**: 17x 延迟增加不可接受

---

## 测试命令

```bash
# 1. 简单配置测试
./.venv/bin/python test_fp8_simple.py

# 2. 量化精度测试
./.venv/bin/python test_fp8_precision.py

# 3. 端到端性能 benchmark
./.venv/bin/python bench_fp8_kvcache.py --both
```

---

## 相关文档

- [快速参考](docs/fp8_kvcache_summary.md) - 实现要点和面试话术
- [精度报告](docs/fp8_precision_report.md) - 详细的量化精度测试
- [原始文档](docs/fp8_kvcache.md) - 完整技术文档

---

*测试时间: 2026-09-22*  
*GPU: RTX 3080 10GB*  
*模型: Qwen3-0.6B*
