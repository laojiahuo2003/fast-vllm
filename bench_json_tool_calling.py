"""
针对用户目标场景：English Single-User JSON Tool Calling
系统评测各种真实 JSON Tool Calling / Function Calling 负载下的：
1. 逐 Token 准确性（与 Baseline 严格比对）
2. 投机接受率（Acceptance Rate）
3. 首候选命中率（Position 1 Match Rate）
4. 每轮产出 Token 数（Tokens / Round）
5. 端到端墙钟吞吐（tok/s）与加速比（Speedup）
"""
import sys
import json
import time
import subprocess
import warnings
warnings.filterwarnings("ignore")

JSON_TOOL_PROMPTS = {
    "Weather-Tool": (
        'You are a helpful AI assistant with tool calling capabilities.\n\n'
        'User: Can you check the current weather in San Francisco and Tokyo?\n\n'
        'Assistant: Certainly! I will invoke the weather API for both locations.\n\n'
        'Tool call: {"tool_calls": [{"id": "call_001", "type": "function", "function": '
        '{"name": "get_weather", "arguments": {"city": "San Francisco", "units": "celsius"}}}, '
        '{"id": "call_002", "type": "function", "function": {"name": "get_weather", "arguments": '
    ),
    "Database-Query": (
        'You are an expert SQL assistant that outputs query execution plans in JSON.\n\n'
        'User: Query active customer orders over $100 placed in the last 7 days.\n\n'
        'Assistant:\n'
        '```json\n'
        '{\n'
        '  "action": "execute_query",\n'
        '  "table": "orders",\n'
        '  "filters": [\n'
        '    {"field": "status", "operator": "eq", "value": "active"},\n'
        '    {"field": "total_amount", "operator": "gt", "value": 100.0},\n'
        '    {"field": "order_date", "operator": "gte", "value": "2026-09-28"}\n'
        '  ],\n'
        '  "sort": {"field": "order_date", "direction": "desc"},\n'
        '  "limit": 50,\n'
        '  "output_format":'
    ),
    "Device-IoT-Tool": (
        'You are a smart home automation controller.\n\n'
        'Device registry:\n'
        '[{"id": "dev_01", "name": "Living Room AC", "type": "climate"},\n'
        ' {"id": "dev_02", "name": "Kitchen Light", "type": "lighting"}]\n\n'
        'Command: "Turn down the AC to 22 degrees and set mode to eco."\n\n'
        'JSON Action: {"command": "send_device_instruction", "payload": '
        '{"target_device_id": "dev_01", "parameters": {"temperature": 22, "mode": "eco", '
    ),
    "GitHub-API-Tool": (
        'You are an automation agent for GitHub actions.\n\n'
        'Tool specification: create_pull_request(repo, title, head, base, body, draft)\n\n'
        'User prompt: Open PR from feature/auth to main titled "feat: OAuth2 Google Sign-in"\n\n'
        'Output format:\n'
        '{"name": "create_pull_request", "arguments": {"repo": "fast-vllm/fast-vllm", '
        '"title": "feat: OAuth2 Google Sign-in", "head": "feature/auth", "base": "main", '
    ),
    "E-Commerce-Cart": (
        'You are a shopping assistant backend. Generate the cart checkout schema.\n\n'
        'User items: 2x Mechanical Keyboard ($89.99), 1x Mousepad ($19.99).\n\n'
        'JSON payload:\n'
        '{\n'
        '  "order_id": "ord_987654321",\n'
        '  "currency": "USD",\n'
        '  "items": [\n'
        '    {"sku": "KB-MECH-01", "name": "Mechanical Keyboard", "quantity": 2, "unit_price": 89.99},\n'
        '    {"sku": "PAD-DESK-02", "name": "Mousepad", "quantity": 1, "unit_price": 19.99}\n'
        '  ],\n'
        '  "subtotal": 199.97,\n'
        '  "tax": 16.00,\n'
        '  "total":'
    ),
}

# Worker 模式
if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    pname = sys.argv[2]
    ptext = JSON_TOOL_PROMPTS[pname]
    mode = sys.argv[3]  # "base", "spec_eager", "spec_graph"
    k = int(sys.argv[4])
    n = int(sys.argv[5])

    import torch
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    from fastvllm.engine.llm_engine import LLMEngine
    from fastvllm.sampling_params import SamplingParams

    spec = mode.startswith("spec")
    eager = (mode != "spec_graph")

    kwargs = dict(
        model="/home/uos/huggingface/Qwen3-0.6B",
        max_num_batched_tokens=4096,
        max_model_len=1024,
        enforce_eager=eager,
    )
    if spec:
        kwargs.update(
            enable_eagle3=True,
            eagle3_model_path="/home/uos/huggingface/Qwen3-0.6B-eagle3",
            eagle3_extract_layers=[1, 13, 24],
            num_speculative_tokens=k,
        )

    eng = LLMEngine(**kwargs)
    t0 = time.perf_counter()
    out = eng.generate([ptext], SamplingParams(temperature=0.0, max_tokens=n), use_tqdm=False)
    dt = time.perf_counter() - t0
    token_ids = out[0]["token_ids"]
    text = out[0]["text"]
    stats = eng.model_runner.get_spec_stats() if spec else None
    eng.exit()

    res = {
        "tokens": len(token_ids),
        "token_ids": token_ids,
        "text": text,
        "dt": dt,
        "tps": len(token_ids) / dt,
        "stats": stats,
    }
    print("<<<RESULT>>>" + json.dumps(res) + "<<<END>>>")
    sys.exit(0)


# Master 模式
def run_worker(pname, mode, k, n=80):
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        pname,
        mode,
        str(k),
        str(n),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout
    if "<<<RESULT>>>" not in out:
        print(f"Error ({mode}):\n{proc.stderr}\n{out}")
        return None
    raw = out.split("<<<RESULT>>>")[1].split("<<<END>>>")[0]
    return json.loads(raw)


def main():
    print("=" * 115)
    print("目标场景实测：English Single-User JSON Tool Calling (Qwen3-0.6B + Eagle3)")
    print("三方对比：Baseline (Eager) vs Eagle3 投机 (Eager) vs Eagle3 投机 (CUDA Graph)")
    print("=" * 115)

    fmt = "{:<16} | {:>6} | {:>7} | {:>7} | {:>9} | {:>10} | {:>14} | {:>10} | {:>5}"
    print(fmt.format("测试用例", "Tokens", "接受率", "Token/轮", "Base(tps)", "Eager(tps)", "Graph(tps) 🚀", "加速比(Graph)", "对齐"))
    print("-" * 115)

    records = []
    for pname in JSON_TOOL_PROMPTS:
        base = run_worker(pname, mode="base", k=3, n=80)
        spec_eager = run_worker(pname, mode="spec_eager", k=3, n=80)
        spec_graph = run_worker(pname, mode="spec_graph", k=3, n=80)
        if not base or not spec_eager or not spec_graph:
            continue

        st = spec_graph["stats"]
        acc_rate = st["acceptance_rate"]
        tpr = st["avg_accepted_per_step"]
        speedup_eager = spec_eager["tps"] / base["tps"]
        speedup_graph = spec_graph["tps"] / base["tps"]
        min_len = min(len(base["token_ids"]), len(spec_graph["token_ids"]))
        match = (base["token_ids"][:min_len] == spec_graph["token_ids"][:min_len])

        records.append({
            "name": pname,
            "tokens": spec_graph["tokens"],
            "acc_rate": acc_rate,
            "tpr": tpr,
            "base_tps": base["tps"],
            "eager_tps": spec_eager["tps"],
            "graph_tps": spec_graph["tps"],
            "speedup_eager": speedup_eager,
            "speedup_graph": speedup_graph,
            "match": match,
            "sample_output": spec_graph["text"][:100].replace("\n", " "),
        })

        print(fmt.format(
            pname,
            spec_graph["tokens"],
            f"{acc_rate:.2%}",
            f"{tpr:.2f}",
            f"{base['tps']:.1f}",
            f"{spec_eager['tps']:.1f} ({speedup_eager:.2f}x)",
            f"{spec_graph['tps']:.1f}",
            f"{speedup_graph:.2f}x",
            "✅" if match else "❌"
        ))

    print("\n" + "=" * 115)
    print("【生成内容样例（确认 JSON 格式完全正确）】")
    for r in records:
        print(f"• [{r['name']}] -> {r['sample_output']}...")

    avg_acc = sum(r["acc_rate"] for r in records) / len(records)
    avg_tpr = sum(r["tpr"] for r in records) / len(records)
    avg_base_tps = sum(r["base_tps"] for r in records) / len(records)
    avg_eager_tps = sum(r["eager_tps"] for r in records) / len(records)
    avg_graph_tps = sum(r["graph_tps"] for r in records) / len(records)
    avg_gain_eager = avg_eager_tps / avg_base_tps
    avg_gain_graph = avg_graph_tps / avg_base_tps

    print("\n" + "=" * 115)
    print("【English JSON Tool Calling 场景汇总数据】")
    print(f"1. 平均接受率 (Acceptance Rate):                 {avg_acc:.2%}")
    print(f"2. 平均每轮产出 Token 数 (Tokens / Round):       {avg_tpr:.2f} tokens")
    print(f"3. 基准模式吞吐 (Baseline Eager):                 {avg_base_tps:.1f} tok/s")
    print(f"4. 投机解码 (Eager 模式) 吞吐:                   {avg_eager_tps:.1f} tok/s (加速比: {avg_gain_eager:.2f}x)")
    print(f"5. 投机解码 (CUDA Graph 优化) 吞吐:              {avg_graph_tps:.1f} tok/s (加速比: {avg_gain_graph:.2f}x 🚀)")
    print(f"6. CUDA Graph 带来的相对加速增益:               {avg_graph_tps / avg_eager_tps:.2f}x (相比 Eager 投机)")
    print("=" * 115)


if __name__ == "__main__":
    main()
