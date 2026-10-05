import sys
import time
import subprocess
import json

JSON_PROMPT = (
    'You are a helpful AI assistant with tool calling capabilities.\n\n'
    'User: Can you check the current weather in San Francisco and Tokyo?\n\n'
    'Assistant: Certainly! I will invoke the weather API for both locations.\n\n'
    'Tool call: {"tool_calls": [{"id": "call_001", "type": "function", "function": '
    '{"name": "get_weather", "arguments": {"city": "San Francisco", "units": "celsius"}}}, '
    '{"id": "call_002", "type": "function", "function": {"name": "get_weather", "arguments": '
)

if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    mode = sys.argv[2] # "base_eager", "base_graph", "spec_k1_graph", "spec_k3_graph"
    n_tokens = int(sys.argv[3])
    
    from fastvllm.engine.llm_engine import LLMEngine
    from fastvllm.sampling_params import SamplingParams
    
    MODEL = "/home/uos/huggingface/Qwen3-0.6B"
    EAGLE3 = "/home/uos/huggingface/Qwen3-0.6B-eagle3"
    
    is_spec = "spec" in mode
    is_graph = "graph" in mode
    k = 1 if "k1" in mode else 3
    
    kwargs = dict(
        model=MODEL,
        max_num_batched_tokens=4096,
        max_model_len=1024,
        gpu_memory_utilization=0.6,
        enforce_eager=not is_graph,
    )
    if is_spec:
        kwargs.update(
            enable_eagle3=True,
            eagle3_model_path=EAGLE3,
            num_speculative_tokens=k,
        )
        
    eng = LLMEngine(**kwargs)
    # warmup
    eng.generate([JSON_PROMPT], SamplingParams(temperature=0.0, max_tokens=10), use_tqdm=False)
    
    t0 = time.perf_counter()
    out = eng.generate([JSON_PROMPT], SamplingParams(temperature=0.0, max_tokens=n_tokens), use_tqdm=False)
    dt = time.perf_counter() - t0
    tokens = len(out[0]["token_ids"])
    tps = tokens / dt
    stats = eng.model_runner.get_spec_stats() if is_spec else None
    eng.exit()
    
    res = {"mode": mode, "tokens": tokens, "dt": dt, "tps": tps, "stats": stats}
    print("<<<RES>>>" + json.dumps(res) + "<<<END>>>")
    sys.exit(0)

def run(mode, n=80):
    cmd = [sys.executable, __file__, "--worker", mode, str(n)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = p.stdout
    if "<<<RES>>>" not in out:
        print(f"Error {mode}:", p.stderr, out)
        return None
    return json.loads(out.split("<<<RES>>>")[1].split("<<<END>>>")[0])

def main():
    print("=" * 80)
    print("实测验证：Baseline (+CUDA Graph) vs 投机解码 (+CUDA Graph)")
    print("=" * 80)
    for m in ["base_eager", "base_graph", "spec_k1_graph", "spec_k3_graph"]:
        res = run(m, n=80)
        if res:
            st = res.get("stats")
            extra = f"(acc: {st['acceptance_rate']:.1%}, tpr: {st['avg_accepted_per_step']:.2f})" if st else ""
            print(f"{res['mode']:<18} | 吞吐: {res['tps']:>6.2f} tok/s | 耗时: {res['dt']:.3f}s {extra}")

if __name__ == "__main__":
    main()
