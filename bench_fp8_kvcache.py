#!/usr/bin/env python3
"""
FP8 KV Cache Benchmark - 对比量化前后的性能和显存占用

用法:
    ./.venv/bin/python bench_fp8_kvcache.py --baseline  # 测试 FP16 baseline
    ./.venv/bin/python bench_fp8_kvcache.py --fp8       # 测试 FP8 量化
    ./.venv/bin/python bench_fp8_kvcache.py --both      # 对比测试
"""

import os
import sys
import time
import argparse
import random
from random import seed

from fastvllm import LLM, SamplingParams

def percentile(sorted_vals, p):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    return sorted_vals[min(n - 1, int(p / 100.0 * n))]

def run_benchmark(kv_dtype: str, max_steps: int = 1000):
    """运行一次benchmark"""
    seed(0)

    MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    print(f"\n{'='*60}")
    print(f"[{kv_dtype.upper()}] KV Cache Benchmark")
    print(f"{'='*60}")

    # 创建引擎
    llm = LLM(
        MODEL_PATH,
        kv_cache_dtype=kv_dtype,
        max_num_batched_tokens=8192,
        max_model_len=4096,
        max_num_seqs=64,
        enforce_eager=False,
    )

    print(f"KV Cache dtype   : {kv_dtype}")
    print(f"KV Cache blocks  : {len(llm.scheduler.block_manager.blocks)}")

    # 启动持续在线的短 decode 请求
    NUM_ONLINE_SEQS = 64
    for _ in range(NUM_ONLINE_SEQS):
        prompt = [random.randint(0, 30000) for _ in range(16)]
        llm.add_request(prompt, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2000))

    # 预热
    llm.step()

    # 周期性注入长 prefill
    INJECT_STEP_GAP = 5
    INJECT_BATCH = 8
    INJECT_LEN = 2048

    step_times = []
    total_prefill = 0
    total_decode = 0
    per_token_dts = []
    last_ts = {}
    last_compl = {}
    concurrency = []

    t_start = time.perf_counter()

    while not llm.is_finished():
        concurrency.append(len(llm.scheduler.running) + len(llm.scheduler.waiting))

        now = time.perf_counter()
        before = {}
        for s in list(llm.scheduler.running) + list(llm.scheduler.waiting):
            before[s.seq_id] = s.num_completion_tokens

        output = llm.step()

        st = (time.perf_counter() - now) * 1000
        step_times.append(st)

        _o, prefill_tokens, decode_tokens = output
        total_prefill += prefill_tokens
        total_decode += decode_tokens

        # Decode per-token 延迟
        alive = []
        for s in list(llm.scheduler.running) + list(llm.scheduler.waiting):
            alive.append(s.seq_id)
            c = s.num_completion_tokens
            prev_c = last_compl.get(s.seq_id, 0)
            prev_t = last_ts.get(s.seq_id, None)
            if c > prev_c and prev_t is not None:
                per_token_dts.append((now - prev_t) * 1000)
            if c > prev_c:
                last_ts[s.seq_id] = now
            last_compl[s.seq_id] = c

        for sid in list(last_ts):
            if sid not in alive:
                last_ts.pop(sid)
                last_compl.pop(sid)

        # 周期性注入
        if len(step_times) % INJECT_STEP_GAP == 0:
            for _ in range(INJECT_BATCH):
                llm.add_request(
                    [random.randint(0, 30000) for _ in range(INJECT_LEN)],
                    SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=random.randint(10, 40)),
                )

        if len(step_times) >= max_steps:
            break

    elapsed = time.perf_counter() - t_start

    # 统计
    step_times.sort()
    per_token_dts.sort()

    print(f"steps            : {len(step_times)}")
    print(f"concurrency      : avg={sum(concurrency)/len(concurrency):.1f} peak={max(concurrency)}")
    print(f"-- Step latency (ms) --")
    print(f"  mean           : {sum(step_times)/len(step_times):8.2f}")
    print(f"  P50            : {percentile(step_times, 50):8.2f}")
    print(f"  P90            : {percentile(step_times, 90):8.2f}")
    print(f"  P99            : {percentile(step_times, 99):8.2f}")
    print(f"  max            : {max(step_times):8.2f}")

    if per_token_dts:
        print(f"-- Decode per-token latency (ms) --")
        print(f"  samples        : {len(per_token_dts)}")
        print(f"  P50            : {percentile(per_token_dts, 50):8.2f}")
        print(f"  P99            : {percentile(per_token_dts, 99):8.2f}")

    total_tokens = total_prefill + total_decode
    throughput = total_tokens / elapsed
    prefill_throughput = total_prefill / elapsed
    decode_throughput = total_decode / elapsed

    print(f"-- Throughput (tok/s) --")
    print(f"  total          : {throughput:9.2f}")
    print(f"  prefill        : {prefill_throughput:9.2f}")
    print(f"  decode         : {decode_throughput:9.2f}")
    print(f"{'='*60}\n")

    llm.exit()

    return {
        'kv_dtype': kv_dtype,
        'num_blocks': len(llm.scheduler.block_manager.blocks),
        'step_p50': percentile(step_times, 50),
        'step_p90': percentile(step_times, 90),
        'step_p99': percentile(step_times, 99),
        'decode_p50': percentile(per_token_dts, 50) if per_token_dts else 0,
        'throughput': throughput,
        'prefill_throughput': prefill_throughput,
        'decode_throughput': decode_throughput,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true', help='Run FP16 baseline')
    parser.add_argument('--fp8', action='store_true', help='Run FP8 quantized')
    parser.add_argument('--both', action='store_true', help='Run both and compare')
    parser.add_argument('--max-steps', type=int, default=1000, help='Number of steps')
    args = parser.parse_args()

    if not (args.baseline or args.fp8 or args.both):
        print("请指定 --baseline, --fp8 或 --both")
        sys.exit(1)

    results = []

    if args.both or args.baseline:
        results.append(run_benchmark('auto', args.max_steps))
        # 强制清理显存，为下一次分配做准备
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import time
        time.sleep(2)  # 等待 CUDA 完全释放

    if args.both or args.fp8:
        results.append(run_benchmark('fp8', args.max_steps))

    # 对比
    if len(results) == 2:
        baseline, fp8 = results
        print("\n" + "="*60)
        print("COMPARISON: FP8 vs Baseline")
        print("="*60)
        print(f"Memory (blocks)  : {baseline['num_blocks']} → {fp8['num_blocks']} "
              f"(+{(fp8['num_blocks']/baseline['num_blocks']-1)*100:.1f}%)")
        print(f"Step P50 (ms)    : {baseline['step_p50']:.2f} → {fp8['step_p50']:.2f} "
              f"({(fp8['step_p50']/baseline['step_p50']-1)*100:+.1f}%)")
        print(f"Step P99 (ms)    : {baseline['step_p99']:.2f} → {fp8['step_p99']:.2f} "
              f"({(fp8['step_p99']/baseline['step_p99']-1)*100:+.1f}%)")
        print(f"Throughput       : {baseline['throughput']:.2f} → {fp8['throughput']:.2f} tok/s "
              f"({(fp8['throughput']/baseline['throughput']-1)*100:+.1f}%)")
        print("="*60)

if __name__ == '__main__':
    main()
