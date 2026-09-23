#!/usr/bin/env python3
"""测试 FP8 KV Cache 的量化精度 - 使用 LLM 接口"""

import os
import torch
import numpy as np
from fastvllm import LLM

def test_kv_cache_accuracy():
    """对比 FP8 vs BF16 在 KV Cache 层面的数值精度"""
    print("=" * 60)
    print("FP8 KV Cache Accuracy Test")
    print("=" * 60)

    MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # 测试序列：128 tokens prefill
    PREFILL_LEN = 128
    torch.manual_seed(42)
    input_ids = torch.randint(0, 30000, (PREFILL_LEN,)).tolist()

    print(f"\nTest: Store {PREFILL_LEN} tokens to KV Cache, then read back")

    # ===== BF16 Baseline =====
    print("\n[1/2] Running BF16 baseline...")
    llm_bf16 = LLM(MODEL_PATH, kv_cache_dtype="auto", enforce_eager=True)

    # 获取第一层的 KV cache
    first_attn_layer = None
    for module in llm_bf16.model_runner.model.modules():
        if hasattr(module, 'k_cache'):
            first_attn_layer = module
            break

    # Prefill 一次以填充 KV cache
    from fastvllm import SamplingParams
    _ = llm_bf16.generate([input_ids], SamplingParams(max_tokens=1))

    # 读取 KV cache（BF16）
    k_cache_bf16 = first_attn_layer.k_cache[:1, :PREFILL_LEN].clone()  # [1, 128, num_heads, head_dim]
    v_cache_bf16 = first_attn_layer.v_cache[:1, :PREFILL_LEN].clone()

    print(f"  K cache shape: {k_cache_bf16.shape}, dtype: {k_cache_bf16.dtype}")
    print(f"  V cache shape: {v_cache_bf16.shape}, dtype: {v_cache_bf16.dtype}")

    llm_bf16.exit()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # 等待显存完全释放
    import time
    time.sleep(3)

    # ===== FP8 =====
    print("\n[2/2] Running FP8 quantized...")
    llm_fp8 = LLM(MODEL_PATH, kv_cache_dtype="fp8", enforce_eager=True)

    # 获取第一层的 KV cache 和 scale
    first_attn_layer_fp8 = None
    for module in llm_fp8.model_runner.model.modules():
        if hasattr(module, 'k_cache'):
            first_attn_layer_fp8 = module
            break

    # Prefill 一次以填充 KV cache（FP8）
    _ = llm_fp8.generate([input_ids], SamplingParams(max_tokens=1))

    # 读取 FP8 KV cache（已经反量化为 BF16）
    k_cache_fp8 = first_attn_layer_fp8.k_cache[:1, :PREFILL_LEN].clone()
    v_cache_fp8 = first_attn_layer_fp8.v_cache[:1, :PREFILL_LEN].clone()

    print(f"  K cache shape: {k_cache_fp8.shape}, dtype: {k_cache_fp8.dtype}")
    print(f"  V cache shape: {v_cache_fp8.shape}, dtype: {v_cache_fp8.dtype}")

    # 获取量化参数
    k_scale = first_attn_layer_fp8.k_scale
    v_scale_cache = first_attn_layer_fp8.v_scale_cache[:PREFILL_LEN].clone()

    print(f"  K scale (static): {k_scale}")
    print(f"  V scale (per-token): min={v_scale_cache.min():.6f}, max={v_scale_cache.max():.6f}, mean={v_scale_cache.mean():.6f}")

    llm_fp8.exit()

    # ===== 对比精度 =====
    print("\n" + "=" * 60)
    print("ACCURACY METRICS")
    print("=" * 60)

    def compute_error(bf16_t, fp8_t, name):
        bf16_flat = bf16_t.float().flatten()
        fp8_flat = fp8_t.float().flatten()

        abs_diff = (bf16_flat - fp8_flat).abs()
        rel_diff = abs_diff / (bf16_flat.abs() + 1e-8)

        cosine_sim = torch.nn.functional.cosine_similarity(
            bf16_flat.unsqueeze(0), fp8_flat.unsqueeze(0)
        ).item()

        print(f"\n[{name}]")
        print(f"  Max absolute error:  {abs_diff.max().item():.6f}")
        print(f"  Mean absolute error: {abs_diff.mean().item():.6f}")
        print(f"  Max relative error:  {rel_diff.max().item() * 100:.2f}%")
        print(f"  Mean relative error: {rel_diff.mean().item() * 100:.2f}%")
        print(f"  Cosine similarity:   {cosine_sim:.8f}")

        return {
            'max_abs': abs_diff.max().item(),
            'mean_abs': abs_diff.mean().item(),
            'max_rel': rel_diff.max().item() * 100,
            'mean_rel': rel_diff.mean().item() * 100,
            'cosine_sim': cosine_sim,
        }

    k_metrics = compute_error(k_cache_bf16, k_cache_fp8, "K Cache")
    v_metrics = compute_error(v_cache_bf16, v_cache_fp8, "V Cache")

    # ===== 结论 =====
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    avg_mean_rel = (k_metrics['mean_rel'] + v_metrics['mean_rel']) / 2
    avg_cosine = (k_metrics['cosine_sim'] + v_metrics['cosine_sim']) / 2

    if avg_mean_rel < 1.0 and avg_cosine > 0.9999:
        print("✓ FP8 quantization preserves very high accuracy")
        print(f"  Avg mean relative error: {avg_mean_rel:.2f}%")
        print(f"  Avg cosine similarity:   {avg_cosine:.8f}")
    elif avg_mean_rel < 5.0 and avg_cosine > 0.999:
        print("✓ FP8 quantization has acceptable accuracy")
        print(f"  Avg mean relative error: {avg_mean_rel:.2f}%")
        print(f"  Avg cosine similarity:   {avg_cosine:.8f}")
    else:
        print("⚠ FP8 quantization has noticeable accuracy loss")
        print(f"  Avg mean relative error: {avg_mean_rel:.2f}%")
        print(f"  Avg cosine similarity:   {avg_cosine:.8f}")

    print("\nNote: Despite quantization error, Attention's softmax normalization")
    print("      typically makes the final output relatively insensitive to KV cache noise.")
    print("=" * 60)

if __name__ == "__main__":
    test_kv_cache_accuracy()
