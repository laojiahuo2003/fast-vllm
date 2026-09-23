# FP8 KV Cache 实现结果总结

## 实现完成 ✓

FP8 KV Cache 量化功能已完整实现，核心改动涉及 4 个文件：
- `fastvllm/config.py` - 配置解析
- `fastvllm/engine/model_runner.py` - 显存分配
- `fastvllm/layers/attention.py` - 量化/反量化逻辑
- 测试与 benchmark 脚本

## Benchmark 结果（RTX 3080, Qwen3-0.6B）

| 指标 | BF16 (baseline) | FP8 | 变化 |
|------|-----------------|-----|------|
| **KV Cache blocks** | 241 | 394 | **+63.5%** ✅ |
| **Step P50 延迟** | 10.7 ms | 183.6 ms | **+1615%** ⚠️ |
| **Step P99 延迟** | 17.1 ms | 220.1 ms | **+1188%** ⚠️ |
| **吞吐量** | 5381 tok/s | 340 tok/s | **-93.7%** ⚠️ |
| **并发 (avg/peak)** | 860 / 1656 | 860 / 1656 | 相同 |

**测试条件**：
- 64 个在线 decode 请求（2000 tokens each）
- 每 5 steps 注入 8 个长 prefill（2048 tokens）
- 运行 1000 steps

## 关键发现

### ✅ 成功的部分

1. **显存优化达到预期**：+63.5% blocks（理论最大 +100%，实际受 V scale 开销影响）
2. **量化精度优秀**：平均相对误差 **2.33%**，余弦相似度 **0.9996**
   - K (静态 scale=1.0): 相对误差 2.41%, 余弦相似度 0.9996
   - V (动态 per-token scale): 相对误差 2.25%, 余弦相似度 0.9996
   - FP8 E4M3 理论量化步长 ~12.5%，实际误差远低于此
3. **CUDA Graph 兼容**：使用 `torch.where` + `scatter_` 避免条件分支
4. **功能完整**：支持 prefill + decode 混合工作负载

### ⚠️ 性能瓶颈

**根本原因**：Flash Attention 不支持 FP8 dtype 输入

- 每次 forward 必须反量化**整个 KV Cache**（394 blocks × 256 tokens = 100,864 tokens）
- 反量化路径：FP8 → FP32 → ×scale → BF16（4 次内存访问）
- 反量化开销远超 attention 计算本身

**代码位置**：[attention.py:125](fastvllm/layers/attention.py#L125)
```python
# 每个 forward 都执行
k_cache, v_cache = dequant_kvcache(k_cache, v_cache, self.v_scale_cache, self.k_scale)
```

## 优化方向

### 方案 1：按需反量化（短期，RTX 3080 可用）
只反量化 `block_tables` 中实际用到的 blocks：
- Prefill: 反量化 prefix cache 部分
- Decode: 反量化每个 sequence 的 active blocks

**预期收益**：
- Decode 延迟从 183ms 降至 ~15ms（反量化从 100k tokens 降至 ~64 tokens）
- 仍有 ~4ms 反量化开销（vs baseline 10.7ms）

### 方案 2：FP8 Native Attention（长期，需要 Hopper GPU）
使用支持 FP8 Tensor Core 的 attention kernel：
- Flash Attention 3.x（H100）
- 自定义 Triton FP8 attention kernel

**预期收益**：
- 无反量化开销
- 性能接近 baseline（~5% 内）

## 适用场景

### ✅ 推荐使用 FP8
- **显存严重不足**：必须用量化才能运行的模型
- **Prefill-heavy 工作负载**：反量化开销摊销在长 context 上
- **低吞吐容忍**：延迟要求不严格的场景

### ❌ 不推荐使用 FP8
- **显存充足**：BF16 性能更好
- **Decode-heavy 工作负载**：17x 延迟无法接受
- **低延迟要求**：在线服务场景

## 技术细节

### 量化方案
- **K (Query-Key)**：静态 scale = 1.0（数值分布稳定）
- **V (Value)**：per-token 动态 scale（数值范围变化大）
- **FP8 格式**：E4M3（1 sign + 4 exp + 3 mantissa，范围 ±448）

### 实现亮点
1. **PyTorch 量化**：软件实现，兼容所有 GPU（无需 Hopper FP8 指令）
2. **Triton kernel 简化**：只负责字节搬运，量化在外部完成
3. **CUDA Graph safe**：无条件写入，无 CPU-GPU 同步

### 代码结构
```python
# 写入时：量化 + 存储
store_kvcache(k, v, k_cache, v_cache, slot_mapping,
              is_fp8=True, k_scale=1.0, v_scale_cache=self.v_scale_cache)

# 读取前：反量化（性能瓶颈）
k_cache, v_cache = dequant_kvcache(k_cache, v_cache, 
                                   self.v_scale_cache, self.k_scale)

# Flash Attention：在 BF16 上计算
o = flash_attn_with_kvcache(q, k_cache, v_cache, ...)
```

## 结论

FP8 KV Cache 实现**功能完整**，显存优化**达到预期**（+63.5%），但性能**严重下降**（-93.7%）。

**核心问题**：Flash Attention 不支持 FP8，导致反量化开销占主导。

**下一步**：
1. 实现按需反量化（预期恢复至 ~15ms，仍比 baseline 慢 40%）
2. 等待 Hopper GPU + Flash Attention 3.x 的 FP8 native 支持

**当前建议**：如果显存充足，继续使用 BF16。
