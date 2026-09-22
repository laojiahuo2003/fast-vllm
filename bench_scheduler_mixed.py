"""
统一调度对比 Benchmark：fast-vllm (PD 混排) vs nano-vllm-baseline (两阶段串行)

三条指标：
  1. Step 级延迟    : P50 / P90 / P99 / max 单步延迟 (ms)
  2. Decode per-token 延迟 : 每个 decode 相邻两次采样 token 的间隔, 取 P50 / P99 (ms)
  3. 吞吐           : total / prefill / decode (tok/s)

负载：持续在线短 decode 请求 + 周期性注入长 prefill 请求（模拟 decode 在线 + prefill 到达）。

用法（两边各跑一次，负载参数保持一致）:
    # 在 fast-vllm 的 venv 里
    ./.venv/bin/python bench_scheduler_mixed.py --engine fastvllm \
        --budget 8192 --seqs 64 --max-steps 3000
    # 在 nanovllm-baseline 的 venv 里
    ./.venv/bin/python bench_scheduler_mixed.py --engine nanovllm \
        --budget 8192 --seqs 64 --max-steps 3000

可选参数：
    --engine fastvllm|nanovllm  目标引擎（默认 fastvllm）
    --budget <n>   覆盖 max_num_batched_tokens（每步总 token 预算）
    --chunk <n>    覆盖 prefill_chunk_tokens（仅 fastvllm 启用分块）
    --seqs <n>     在线 decode 请求数（并发档位）
    --max-steps <n> 采样的 step 数（样本量，建议 >=2000）
    --long-min/--long-max 长 prefill 的 token 长度范围
    --model <path> 模型路径
    --eager        强制 eager（消除 CUDA Graph 变量，控制变量用）
"""

import os
import sys
import time
import argparse
import random
from random import seed


def load_engine(engine: str):
    """按 --engine 把目标包根目录注入 sys.path 并返回 (LLM, SamplingParams)。"""
    pkg_map = {
        "fastvllm": ("fastvllm", "/home/uos/code/fast-vllm"),
        "nanovllm": ("nanovllm", "/home/uos/code/nano-vllm-baseline"),
    }
    pkg, root = pkg_map[engine]
    if root not in sys.path:
        sys.path.insert(0, root)
    mod = __import__(pkg, fromlist=["LLM", "SamplingParams"])
    return mod.LLM, mod.SamplingParams


def percentile(sorted_vals, p):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    return sorted_vals[min(n - 1, int(p / 100.0 * n))]


def main():
    seed(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["fastvllm", "nanovllm"], default="fastvllm")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--long-min", type=int, default=1200)
    ap.add_argument("--long-max", type=int, default=4096)
    ap.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    LLM, SamplingParams = load_engine(args.engine)

    kwargs = dict(enforce_eager=args.eager, max_model_len=4096)
    if args.budget is not None:
        kwargs["max_num_batched_tokens"] = args.budget
    if args.chunk > 0 and args.engine == "fastvllm":
        kwargs["prefill_chunk_tokens"] = args.chunk

    llm = LLM(args.model, **kwargs)

    # ── 持续在线的短 decode 请求 ──────────────────────────────
    for _ in range(args.seqs):
        prompt = [random.randint(0, 30000) for _ in range(16)]
        llm.add_request(prompt, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2000))

    # 预热一个 step, 排除 CUDA Graph / 初始化固定开销
    llm.step()

    # ── 周期性到达的长 prefill（最常见混合场景）────────────────
    # 每 INJECT_STEP_GAP 步，固定注入 INJECT_BATCH 个长 prefill 请求。
    # 长度经 seed(0) 固定 → 两边引擎收到完全相同的请求序列，公平可复现。
    INJECT_STEP_GAP = 5        # 每隔多少步注入一批
    INJECT_BATCH = 8           # 每批注入的请求数
    INJECT_LEN = 2048          # 长 prefill 固定的 prompt 长度

    step_times = []
    total_prefill = 0
    total_decode = 0
    per_token_dts = []          # decode 相邻两次采样的墙钟间隔
    last_ts = {}                # seq_id -> 上一次采样完成时刻
    last_compl = {}             # seq_id -> 上一次的 num_completion_tokens
    concurrency = []
    t_start = time.perf_counter()

    while not llm.is_finished():
        concurrency.append(len(llm.scheduler.running) + len(llm.scheduler.waiting))

        now = time.perf_counter()
        # step 之前：记录当前所有活跃 seq 的 completion 数（用于识别本步采样了哪个 decode）
        before = {}
        for s in list(llm.scheduler.running) + list(llm.scheduler.waiting):
            before[s.seq_id] = s.num_completion_tokens

        output = llm.step()

        st = (time.perf_counter() - now) * 1000
        step_times.append(st)

        # 吞吐解析（两引擎 step() 返回值不同）
        if args.engine == "fastvllm":
            _o, prefill_tokens, decode_tokens = output
        else:
            _o, num_tokens = output
            prefill_tokens = num_tokens if num_tokens > 0 else 0
            decode_tokens = -num_tokens if num_tokens < 0 else 0
        total_prefill += prefill_tokens
        total_decode += decode_tokens

        # Decode per-token 延迟：比较 completion 数是否增加
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
        # 清理已不活跃（完成/离开队列）的 seq，避免内存膨胀
        for sid in list(last_ts):
            if sid not in alive:
                last_ts.pop(sid)
                last_compl.pop(sid)

        # 周期性注入固定数量的长 prefill（确定性，两边一致）
        if len(step_times) % INJECT_STEP_GAP == 0:
            for _ in range(INJECT_BATCH):
                llm.add_request(
                    [random.randint(0, 30000) for _ in range(INJECT_LEN)],
                    SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=random.randint(10, 40)),
                )

        if len(step_times) >= args.max_steps:
            break

    t_total = time.perf_counter() - t_start

    # ── 汇总 ──────────────────────────────────────────────────
    st = sorted(step_times)
    pd = sorted(per_token_dts)
    n = len(st)
    avg_conc = sum(concurrency) / len(concurrency) if concurrency else 0
    peak_conc = max(concurrency) if concurrency else 0

    print("=" * 64)
    print(f"[{args.engine}] Mixed-Workload Scheduler Benchmark")
    print("=" * 64)
    print(f"steps            : {n}")
    print(f"concurrency      : avg={avg_conc:.1f} peak={peak_conc}")
    print("-- Step latency (ms) --")
    print(f"  mean           : {sum(step_times)/n:8.2f}")
    print(f"  P50            : {percentile(st, 50):8.2f}")
    print(f"  P90            : {percentile(st, 90):8.2f}")
    print(f"  P99            : {percentile(st, 99):8.2f}")
    print(f"  max            : {st[-1]:8.2f}")
    print("-- Decode per-token latency (ms) --")
    if pd:
        print(f"  samples        : {len(pd)}")
        print(f"  P50            : {percentile(pd, 50):8.2f}")
        print(f"  P99            : {percentile(pd, 99):8.2f}")
    else:
        print("  (no decode sample collected)")
    print("-- Throughput (tok/s) --")
    print(f"  total          : {(total_prefill + total_decode) / t_total:9.2f}")
    print(f"  prefill        : {total_prefill / t_total:9.2f}")
    print(f"  decode         : {total_decode / t_total:9.2f}")
    print("=" * 64)


if __name__ == "__main__":
    main()