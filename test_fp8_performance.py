#!/usr/bin/env python3
"""简化的 FP8 性能测试：对比 per-layer scale 前后的延迟"""

import argparse
import os
import time
import torch
from fastvllm import LLM

def benchmark_simple(kv_cache_dtype, kvcache_k_scale=None):
    """简单性能测试"""
    model = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    llm = LLM(
        model=model,
        max_num_seqs=64,
        max_seq_len=2048,
        block_size=256,
        kv_cache_dtype=kv_cache_dtype,
        kvcache_k_scale=kvcache_k_scale,
    )

    # Warmup
    for _ in range(3):
        list(llm.generate("Hello", sampling_params={"max_tokens": 10}))

    # 测试短序列生成
    prompts = ["什么是机器学习？"] * 10

    torch.cuda.synchronize()
    start = time.time()

    for p in prompts:
        list(llm.generate(p, sampling_params={"max_tokens": 20}))

    torch.cuda.synchronize()
    elapsed = time.time() - start

    avg_latency = elapsed / len(prompts) * 1000  # ms

    return {
        "kv_cache_dtype": kv_cache_dtype,
        "per_layer_scale": kvcache_k_scale is not None,
        "avg_latency_ms": avg_latency,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--fp8-global", action="store_true")
    parser.add_argument("--fp8-per-layer", action="store_true")
    args = parser.parse_args()

    results = []

    if args.baseline:
        print("\n[1/3] 测试 BF16 baseline...")
        r = benchmark_simple("auto")
        results.append(r)
        print(f"  平均延迟: {r['avg_latency_ms']:.1f} ms")

    if args.fp8_global:
        print("\n[2/3] 测试 FP8 (全局 K scale = 1.0)...")
        r = benchmark_simple("fp8")
        results.append(r)
        print(f"  平均延迟: {r['avg_latency_ms']:.1f} ms")

    if args.fp8_per_layer:
        print("\n[3/3] 测试 FP8 (Per-layer K scale)...")
        r = benchmark_simple("fp8", kvcache_k_scale="fp8_k_scales.json")
        results.append(r)
        print(f"  平均延迟: {r['avg_latency_ms']:.1f} ms")

    # 总结
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("性能对比")
        print("=" * 60)
        for r in results:
            label = r['kv_cache_dtype']
            if r['per_layer_scale']:
                label += " + per-layer"
            print(f"{label:20s}: {r['avg_latency_ms']:6.1f} ms")


if __name__ == "__main__":
    main()
