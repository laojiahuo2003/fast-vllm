"""
端到端验证：Eagle3 Speculative Decoding + CUDA Graph (enforce_eager=False)
验证文本生成、接受率统计和端到端加速
"""
import time
import torch
from fastvllm.engine.llm_engine import LLMEngine
from fastvllm.sampling_params import SamplingParams

MODEL = "/home/uos/huggingface/Qwen3-0.6B"
EAGLE3 = "/home/uos/huggingface/Qwen3-0.6B-eagle3"

prompt = (
    'You are a helpful AI assistant with tool calling capabilities.\n\n'
    'User: Can you check the current weather in San Francisco and Tokyo?\n\n'
    'Assistant: Certainly! I will invoke the weather API for both locations.\n\n'
    'Tool call: {"tool_calls": [{"id": "call_001", "type": "function", "function": '
    '{"name": "get_weather", "arguments": {"city": "San Francisco", "units": "celsius"}}}, '
    '{"id": "call_002", "type": "function", "function": {"name": "get_weather", "arguments": '
)

print("=" * 70)
print("1. 启动 Baseline 引擎 (Eager 模式基准)...")
print("=" * 70)
eng_base = LLMEngine(
    model=MODEL,
    max_num_batched_tokens=4096,
    max_model_len=1024,
    gpu_memory_utilization=0.4,
    enforce_eager=True,
)
t0 = time.perf_counter()
out_base = eng_base.generate([prompt], SamplingParams(temperature=0.0, max_tokens=60), use_tqdm=False)
dt_base = time.perf_counter() - t0
base_ids = out_base[0]["token_ids"]
base_text = out_base[0]["text"]
eng_base.exit()
import gc
gc.collect()
torch.cuda.empty_cache()
print(f"Base 耗时: {dt_base:.3f}s, 生成 tokens: {len(base_ids)}, 吞吐: {len(base_ids)/dt_base:.1f} tok/s")
print(f"Base 输出:\n{base_text}\n")

print("=" * 70)
print("2. 启动 Eagle3 投机解码引擎 + CUDA Graph (enforce_eager=False)...")
print("=" * 70)
eng_spec = LLMEngine(
    model=MODEL,
    enable_eagle3=True,
    eagle3_model_path=EAGLE3,
    num_speculative_tokens=3,
    max_num_batched_tokens=4096,
    max_model_len=1024,
    gpu_memory_utilization=0.4,
    enforce_eager=False,  # 启用 CUDA Graph!
)
t1 = time.perf_counter()
out_spec = eng_spec.generate([prompt], SamplingParams(temperature=0.0, max_tokens=60), use_tqdm=False)
dt_spec = time.perf_counter() - t1
spec_ids = out_spec[0]["token_ids"]
spec_text = out_spec[0]["text"]
stats = eng_spec.model_runner.get_spec_stats()
eng_spec.exit()

print(f"Spec+CUDA Graph 耗时: {dt_spec:.3f}s, 生成 tokens: {len(spec_ids)}, 吞吐: {len(spec_ids)/dt_spec:.1f} tok/s")
print(f"Spec 输出:\n{spec_text}\n")

print("=" * 70)
print("3. 结果严格比对与加速统计:")
print("=" * 70)
min_len = min(len(base_ids), len(spec_ids))
match_count = sum(1 for b, s in zip(base_ids[:min_len], spec_ids[:min_len]) if b == s)
print(f"• Token Match: {match_count}/{min_len} ({100*match_count/min_len:.1f}%)")
print(f"• 投机接受率:   {stats['acceptance_rate']*100:.2f}%")
print(f"• 每轮产出:     {stats['avg_accepted_per_step']:.2f} tokens")
print(f"• 端到端加速比: {dt_base / dt_spec:.2f}x")
print("=" * 70)
