#!/usr/bin/env python3
"""测试 FP8 KV Cache 的量化精度 - 简化版，直接对比量化前后的 K/V"""

import os
import torch
from fastvllm import LLM, SamplingParams

def test_fp8_quantization_precision():
    """直接测试 FP8 量化函数的精度"""
    print("=" * 60)
    print("FP8 Quantization Precision Test")
    print("=" * 60)

    # 模拟真实的 K 和 V tensor
    torch.manual_seed(42)
    batch_size = 128
    num_heads = 8
    head_dim = 128

    # 生成测试数据（模拟 attention 层的输出）
    k_bf16 = torch.randn(batch_size, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    v_bf16 = torch.randn(batch_size, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')

    print(f"\nTest data:")
    print(f"  K shape: {k_bf16.shape}, dtype: {k_bf16.dtype}")
    print(f"  V shape: {v_bf16.shape}, dtype: {v_bf16.dtype}")
    print(f"  K range: [{k_bf16.min():.4f}, {k_bf16.max():.4f}]")
    print(f"  V range: [{v_bf16.min():.4f}, {v_bf16.max():.4f}]")

    # ===== 量化 + 反量化 =====
    print("\n" + "=" * 60)
    print("QUANTIZATION")
    print("=" * 60)

    FP8_E4M3_MAX = 448.0

    # K: 静态 scale
    k_scale = 1.0
    k_fp8 = (k_bf16.float() / k_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    k_dequant = (k_fp8.float() * k_scale).to(torch.bfloat16)

    print(f"\n[K - Static scale = {k_scale}]")
    print(f"  FP8 dtype: {k_fp8.dtype}")
    print(f"  Dequantized range: [{k_dequant.min():.4f}, {k_dequant.max():.4f}]")

    # V: 动态 per-token scale
    v_amax = v_bf16.float().abs().amax(dim=(1, 2)).clamp_min(1e-12)
    v_scale = v_amax / FP8_E4M3_MAX
    v_fp8 = (v_bf16.float() / v_scale[:, None, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    v_dequant = (v_fp8.float() * v_scale[:, None, None]).to(torch.bfloat16)

    print(f"\n[V - Dynamic per-token scale]")
    print(f"  FP8 dtype: {v_fp8.dtype}")
    print(f"  Scale range: [{v_scale.min():.6f}, {v_scale.max():.6f}]")
    print(f"  Dequantized range: [{v_dequant.min():.4f}, {v_dequant.max():.4f}]")

    # ===== 精度对比 =====
    print("\n" + "=" * 60)
    print("ACCURACY METRICS")
    print("=" * 60)

    def compute_metrics(original, dequantized, name):
        orig_flat = original.float().flatten()
        deq_flat = dequantized.float().flatten()

        abs_diff = (orig_flat - deq_flat).abs()
        rel_diff = abs_diff / (orig_flat.abs() + 1e-8)

        cosine_sim = torch.nn.functional.cosine_similarity(
            orig_flat.unsqueeze(0), deq_flat.unsqueeze(0)
        ).item()

        print(f"\n[{name}]")
        print(f"  Max absolute error:  {abs_diff.max().item():.6f}")
        print(f"  Mean absolute error: {abs_diff.mean().item():.6f}")
        print(f"  Max relative error:  {rel_diff.max().item() * 100:.2f}%")
        print(f"  Mean relative error: {rel_diff.mean().item() * 100:.2f}%")
        print(f"  Cosine similarity:   {cosine_sim:.8f}")

        return {
            'mean_rel': rel_diff.mean().item() * 100,
            'cosine_sim': cosine_sim,
        }

    k_metrics = compute_metrics(k_bf16, k_dequant, "K (Static scale)")
    v_metrics = compute_metrics(v_bf16, v_dequant, "V (Dynamic scale)")

    # ===== 结论 =====
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    avg_mean_rel = (k_metrics['mean_rel'] + v_metrics['mean_rel']) / 2
    avg_cosine = (k_metrics['cosine_sim'] + v_metrics['cosine_sim']) / 2

    print(f"\nAverage metrics:")
    print(f"  Mean relative error: {avg_mean_rel:.3f}%")
    print(f"  Cosine similarity:   {avg_cosine:.8f}")

    if avg_mean_rel < 1.0 and avg_cosine > 0.9999:
        print("\n✓ FP8 quantization preserves very high accuracy")
    elif avg_mean_rel < 5.0 and avg_cosine > 0.999:
        print("\n✓ FP8 quantization has acceptable accuracy")
    else:
        print("\n⚠ FP8 quantization has noticeable accuracy loss")

    print("\nNote: FP8 E4M3 format:")
    print("  - 3-bit mantissa → ~12.5% quantization step")
    print("  - Range: ±448")
    print("  - Dynamic V scaling improves precision for varying magnitudes")
    print("=" * 60)

if __name__ == "__main__":
    test_fp8_quantization_precision()
