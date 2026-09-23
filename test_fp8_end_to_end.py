#!/usr/bin/env python3
"""
端到端测试：对比使用 per-layer K scale 前后的实际效果

测试内容：
1. FP8 (global scale=1.0)
2. FP8 (per-layer calibrated scale)

对比：
- 显存占用
- 量化精度
- logits 输出差异
"""

from fastvllm import LLM
from fastvllm.sampling_params import SamplingParams
import torch

def test_fp8_scale(use_calibrated_scale=False):
    """测试 FP8 KV Cache"""
    print("=" * 70)
    if use_calibrated_scale:
        print("测试配置: FP8 + Per-layer K Scale")
    else:
        print("测试配置: FP8 + Global K Scale (1.0)")
    print("=" * 70)

    llm = LLM(
        model="/home/uos/huggingface/Qwen3-0.6B/",
        kv_cache_dtype="fp8",
        kvcache_k_scale="fp8_k_scales.json" if use_calibrated_scale else None,
        max_num_seqs=64,
        enforce_eager=True,
    )

    print(f"KV Cache blocks: {len(llm.scheduler.block_manager.blocks)}")

    # 生成测试
    prompts = [
        "什么是机器学习？",
        "请解释深度学习。",
        "Transformer 架构的核心是什么？",
    ]

    sampling_params = SamplingParams(max_tokens=20, temperature=0.01)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for prompt, output in zip(prompts, outputs):
        print(f"输入: {prompt}")
        print(f"输出: {output['text']}")
        print()

    return {
        "num_blocks": len(llm.scheduler.block_manager.blocks),
        "outputs": [o['text'] for o in outputs]
    }

def main():
    import sys

    if "--per-layer" in sys.argv:
        # 测试 per-layer scale
        print("\n" + "=" * 70)
        print("测试配置: FP8 + Per-layer K Scale")
        print("=" * 70)
        result = test_fp8_scale(use_calibrated_scale=True)
        print(f"\nKV Cache blocks: {result['num_blocks']}")

    else:
        # 测试全局 scale
        print("\n" + "=" * 70)
        print("FP8 Per-layer K Scale 端到端测试")
        print("=" * 70)
        print()
        print("=" * 70)
        print("测试配置: FP8 + Global K Scale (1.0)")
        print("=" * 70)
        result = test_fp8_scale(use_calibrated_scale=False)
        print(f"\nKV Cache blocks: {result['num_blocks']}")

        print("\n" + "=" * 70)
        print("下一步：测试 per-layer scale")
        print("=" * 70)
        print("运行: ./.venv/bin/python test_fp8_end_to_end.py --per-layer")
        print("\n注意: 需要分开运行以避免进程组重复初始化错误")

if __name__ == "__main__":
    main()
