#!/usr/bin/env python3
"""简化的FP8 KV Cache测试 - 只测试配置和内存分配"""

import os
import torch
from fastvllm.config import Config

def main():
    print("=" * 60)
    print("FP8 KV Cache - Configuration Test")
    print("=" * 60)

    MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # Test 1: FP8 配置
    print("\n[1] Testing FP8 config...")
    try:
        config_fp8 = Config(
            model=MODEL_PATH,
            kv_cache_dtype="fp8",
            max_num_seqs=32,
            max_model_len=2048,
        )
        assert config_fp8.kv_cache_dtype == torch.float8_e4m3fn
        print(f"✓ FP8 dtype: {config_fp8.kv_cache_dtype}")
        print(f"✓ FP8 blocks: {config_fp8.num_kvcache_blocks}")
    except Exception as e:
        print(f"✗ FP8 config failed: {e}")
        return

    # Test 2: Auto 配置 (baseline)
    print("\n[2] Testing auto/FP16 config...")
    try:
        config_fp16 = Config(
            model=MODEL_PATH,
            kv_cache_dtype="auto",
            max_num_seqs=32,
            max_model_len=2048,
        )
        print(f"✓ Auto dtype: {config_fp16.kv_cache_dtype}")
        print(f"✓ FP16 blocks: {config_fp16.num_kvcache_blocks}")
    except Exception as e:
        print(f"✗ Auto config failed: {e}")
        return

    # Test 3: 计算内存节省
    print("\n[3] Memory savings calculation...")
    fp8_blocks = config_fp8.num_kvcache_blocks
    fp16_blocks = config_fp16.num_kvcache_blocks
    improvement = (fp8_blocks / fp16_blocks - 1) * 100

    print(f"  FP16 blocks: {fp16_blocks}")
    print(f"  FP8 blocks:  {fp8_blocks}")
    print(f"  Improvement: +{improvement:.1f}% more blocks")
    print(f"  Expected:    ~+100% (FP8 uses half the memory)")

    if improvement > 80:
        print("\n✓ Memory savings verified!")
    else:
        print(f"\n⚠ Warning: improvement ({improvement:.1f}%) lower than expected")

    print("\n" + "=" * 60)
    print("Configuration tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
