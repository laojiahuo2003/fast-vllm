#!/usr/bin/env python3
"""
测试 per-layer K scale 对量化精度的改善

对比：
1. 全局 scale=1.0 的量化误差
2. Per-layer calibrated scale 的量化误差
"""

import json
import torch

FP8_E4M3_MAX = 448.0

def quantize_k(k, scale):
    """K 量化"""
    k_fp8 = (k.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    k_dequant = (k_fp8.float() * scale).to(torch.bfloat16)
    return k_dequant

def test_precision(k, scale):
    """测试量化精度"""
    k_dequant = quantize_k(k, scale)
    abs_err = (k - k_dequant).abs()
    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()

    # 相对误差（避免除以接近 0 的值）
    rel_err = abs_err / (k.abs() + 1e-12)
    max_rel_err = rel_err.max().item()
    mean_rel_err = rel_err.mean().item()

    # 余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(
        k.reshape(-1, k.size(-1)).float(),
        k_dequant.reshape(-1, k.size(-1)).float(),
        dim=1
    ).mean().item()

    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "max_rel_err": max_rel_err * 100,  # 转为百分比
        "mean_rel_err": mean_rel_err * 100,
        "cos_sim": cos_sim
    }

def main():
    print("=" * 70)
    print("Per-layer K Scale 精度测试")
    print("=" * 70)

    # 加载 calibrated scales
    with open("fp8_k_scales.json", "r") as f:
        data = json.load(f)
    k_scales = data["k_scale"]
    num_layers = data["num_hidden_layers"]

    print(f"Loaded {num_layers} layer scales")
    print(f"Scale range: [{min(k_scales):.5f}, {max(k_scales):.5f}]")
    print(f"Scale mean: {sum(k_scales)/len(k_scales):.5f}")
    print()

    # 生成测试数据（模拟真实 K 分布）
    torch.manual_seed(42)
    batch_size = 128
    num_heads = 8
    head_dim = 64

    global_results = []
    per_layer_results = []

    print("Testing per-layer quantization precision...")
    print()

    for layer_id in range(num_layers):
        # 生成该层的 K（标准正态分布）
        k = torch.randn(batch_size, num_heads, head_dim, dtype=torch.bfloat16)

        # 归一化到该层的真实 amax（通过 calibration 测得）
        # amax = k_scale * 448，所以该层 K 的真实最大值应该是 k_scale * 448
        current_amax = k.abs().max().item()
        target_amax = k_scales[layer_id] * FP8_E4M3_MAX
        k = k * (target_amax / current_amax)

        # 测试全局 scale=1.0
        global_res = test_precision(k, 1.0)
        global_results.append(global_res)

        # 测试 per-layer scale
        layer_res = test_precision(k, k_scales[layer_id])
        per_layer_results.append(layer_res)

    # 汇总统计
    def avg_results(results):
        return {
            "max_abs_err": max(r["max_abs_err"] for r in results),
            "mean_abs_err": sum(r["mean_abs_err"] for r in results) / len(results),
            "max_rel_err": max(r["max_rel_err"] for r in results),
            "mean_rel_err": sum(r["mean_rel_err"] for r in results) / len(results),
            "cos_sim": sum(r["cos_sim"] for r in results) / len(results)
        }

    global_avg = avg_results(global_results)
    per_layer_avg = avg_results(per_layer_results)

    print("=" * 70)
    print("全局 scale=1.0 结果:")
    print("=" * 70)
    print(f"最大绝对误差: {global_avg['max_abs_err']:.6f}")
    print(f"平均绝对误差: {global_avg['mean_abs_err']:.6f}")
    print(f"最大相对误差: {global_avg['max_rel_err']:.2f}%")
    print(f"平均相对误差: {global_avg['mean_rel_err']:.2f}%")
    print(f"余弦相似度:   {global_avg['cos_sim']:.8f}")
    print()

    print("=" * 70)
    print("Per-layer calibrated scale 结果:")
    print("=" * 70)
    print(f"最大绝对误差: {per_layer_avg['max_abs_err']:.6f}")
    print(f"平均绝对误差: {per_layer_avg['mean_abs_err']:.6f}")
    print(f"最大相对误差: {per_layer_avg['max_rel_err']:.2f}%")
    print(f"平均相对误差: {per_layer_avg['mean_rel_err']:.2f}%")
    print(f"余弦相似度:   {per_layer_avg['cos_sim']:.8f}")
    print()

    print("=" * 70)
    print("改善程度:")
    print("=" * 70)
    rel_err_improve = (global_avg['mean_rel_err'] - per_layer_avg['mean_rel_err']) / global_avg['mean_rel_err'] * 100
    cos_sim_improve = (per_layer_avg['cos_sim'] - global_avg['cos_sim']) / (1 - global_avg['cos_sim']) * 100

    print(f"平均相对误差降低: {rel_err_improve:.1f}%")
    print(f"余弦相似度提升:   {cos_sim_improve:.1f}%")
    print()

    # 找出改善最明显的层
    improvements = []
    for i in range(num_layers):
        improve = global_results[i]['mean_rel_err'] - per_layer_results[i]['mean_rel_err']
        improvements.append((i, improve, k_scales[i]))

    improvements.sort(key=lambda x: x[1], reverse=True)

    print("改善最明显的 5 层:")
    print(f"{'Layer':<8}{'Scale':<12}{'误差降低':<12}{'全局误差':<12}{'优化后误差':<12}")
    for i in range(min(5, len(improvements))):
        layer_id, improve, scale = improvements[i]
        global_err = global_results[layer_id]['mean_rel_err']
        layer_err = per_layer_results[layer_id]['mean_rel_err']
        print(f"{layer_id:<8}{scale:<12.5f}{improve:<12.2f}%{global_err:<12.2f}%{layer_err:<12.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
