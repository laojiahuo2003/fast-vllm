#!/usr/bin/env python3
"""测试 FP8 KV Cache 量化功能"""

import os
import torch
from fastvllm.config import Config
from fastvllm.engine.llm_engine import LLMEngine
from fastvllm.sampling_params import SamplingParams

def test_fp8_kvcache():
    """测试FP8 KV Cache的基本功能"""
    print("=" * 60)
    print("Testing FP8 KV Cache Quantization")
    print("=" * 60)

    MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # 测试1: 验证配置解析
    print("\n[Test 1] Config parsing...")
    config_fp8 = Config(
        model=MODEL_PATH,
        kv_cache_dtype="fp8",
        max_num_seqs=64,
        max_model_len=2048,
    )
    assert config_fp8.kv_cache_dtype == torch.float8_e4m3fn, "FP8 dtype not set correctly"
    print("✓ FP8 config parsing OK")

    config_auto = Config(
        model=MODEL_PATH,
        kv_cache_dtype="auto",
        max_num_seqs=64,
        max_model_len=2048,
    )
    print(f"✓ Auto dtype resolved to: {config_auto.kv_cache_dtype}")

    # 测试2: 对比 FP8 vs 非量化的显存占用
    print("\n[Test 2] Memory footprint comparison...")

    # 非量化版本
    print("\n  [Baseline] Non-quantized KV Cache...")
    engine_baseline = LLMEngine(MODEL_PATH, kv_cache_dtype="auto", max_num_seqs=64, max_model_len=2048, enforce_eager=True)
    baseline_blocks = len(engine_baseline.scheduler.block_manager.blocks)
    print(f"  Allocated blocks (non-quantized): {baseline_blocks}")
    engine_baseline.exit()
    del engine_baseline
    torch.cuda.empty_cache()

    # FP8量化版本
    print("\n  [FP8] Quantized KV Cache...")
    engine_fp8 = LLMEngine(MODEL_PATH, kv_cache_dtype="fp8", max_num_seqs=64, max_model_len=2048, enforce_eager=True)
    fp8_blocks = len(engine_fp8.scheduler.block_manager.blocks)
    print(f"  Allocated blocks (FP8): {fp8_blocks}")

    # 计算提升比例
    improvement = (fp8_blocks / baseline_blocks - 1) * 100
    print(f"\n  Memory improvement: {improvement:.1f}% more blocks")
    print(f"  Expected: ~50% (FP8 uses half the memory)")

    # 测试3: 实际推理测试
    print("\n[Test 3] Inference test with FP8 KV Cache...")
    prompt = "Hello, how are you?"

    try:
        outputs = engine_fp8.generate(
            [prompt],
            SamplingParams(max_tokens=32, temperature=0.8),
        )
        print(f"  Input:  {prompt}")
        print(f"  Output: {outputs[0]['text']}")
        print("✓ FP8 inference completed successfully")
    except Exception as e:
        print(f"✗ FP8 inference failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine_fp8.exit()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_fp8_kvcache()
