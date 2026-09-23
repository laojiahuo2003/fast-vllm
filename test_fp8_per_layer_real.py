#!/usr/bin/env python3
"""
使用真实前向传播测试 per-layer K scale 的精度改善
"""

import json
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from fastvllm.models.qwen3 import Qwen3ForCausalLM
from fastvllm.utils.context import set_context, reset_context
from fastvllm.utils.loader import load_model

FP8_E4M3_MAX = 448.0

PROMPTS = [
    "什么是机器学习？它与深度学习有什么区别？",
    "请解释一下注意力机制的工作原理。",
    "如何理解大语言模型的上下文窗口？",
    "Transformer 架构的核心创新是什么？",
]

def quantize_k(k, scale):
    """K 量化 + 反量化"""
    k_fp8 = (k.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    k_dequant = (k_fp8.float() * scale).to(k.dtype)
    return k_dequant

def compute_metrics(k_orig, k_quant):
    """计算量化误差指标"""
    abs_err = (k_orig - k_quant).abs()
    rel_err = abs_err / (k_orig.abs() + 1e-12)

    return {
        "max_abs": abs_err.max().item(),
        "mean_abs": abs_err.mean().item(),
        "mean_rel": (rel_err.mean().item() * 100),  # 百分比
    }

def main():
    model_path = "/home/uos/huggingface/Qwen3-0.6B/"

    print("=" * 70)
    print("Per-layer K Scale 真实精度测试")
    print("=" * 70)

    # 加载 calibrated scales
    with open("fp8_k_scales.json", "r") as f:
        data = json.load(f)
    k_scales = data["k_scale"]
    num_layers = data["num_hidden_layers"]

    print(f"Loaded {num_layers} layer scales")
    print(f"Scale range: [{min(k_scales):.5f}, {max(k_scales):.5f}]")
    print()

    # 初始化模型
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", "tcp://localhost:29500", world_size=1, rank=0)

    hf_config = AutoConfig.from_pretrained(model_path)
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")

    print("Loading model...")
    model = Qwen3ForCausalLM(hf_config)
    load_model(model, model_path)
    model.eval()

    # 找到所有 attention 模块
    attn_modules = []
    for module in model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            attn_modules.append(module)

    print(f"Found {len(attn_modules)} attention layers")

    # 准备输入
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    all_ids, cu = [], [0]
    for p in PROMPTS:
        ids = tokenizer.encode(p, add_special_tokens=True)[:512]
        all_ids.extend(ids)
        cu.append(cu[-1] + len(ids))

    input_ids = torch.tensor(all_ids, dtype=torch.int64, device='cuda')
    positions = torch.arange(len(all_ids), dtype=torch.int64, device='cuda')
    cu_q = torch.tensor(cu, dtype=torch.int32, device='cuda')
    max_len = max(b - a for a, b in zip(cu[:-1], cu[1:]))

    print(f"Test data: {len(all_ids)} tokens, max seqlen: {max_len}")
    print()

    # 收集真实 K 值并测试两种量化方案
    global_results = []
    per_layer_results = []
    k_values = []

    def hook_fn(module, inp, out, lid):
        k = inp[1].detach().clone()  # [N, num_kv_heads, head_dim]
        k_values.append(k)

    # Hook 收集 K 值
    handles = []
    for layer_id, module in enumerate(attn_modules):
        h = module.register_forward_hook(lambda m, i, o, lid=layer_id: hook_fn(m, i, o, lid))
        handles.append(h)

    print("Running forward pass to collect K values...")
    set_context(True, cu_q, cu_q, max_len, max_len,
                torch.zeros(0, dtype=torch.int32, device='cuda'), None, None)
    with torch.inference_mode():
        model.compute_logits(model(input_ids, positions))
    reset_context()

    for h in handles:
        h.remove()

    print(f"Collected K from {len(k_values)} layers")
    print()

    # 对每层的 K 测试两种量化方案
    print("Testing quantization precision...")
    for layer_id, k in enumerate(k_values):
        # 全局 scale=1.0
        k_global = quantize_k(k, 1.0)
        global_res = compute_metrics(k, k_global)
        global_results.append(global_res)

        # Per-layer scale
        k_layer = quantize_k(k, k_scales[layer_id])
        layer_res = compute_metrics(k, k_layer)
        per_layer_results.append(layer_res)

    # 汇总统计
    global_avg_rel = sum(r["mean_rel"] for r in global_results) / len(global_results)
    per_layer_avg_rel = sum(r["mean_rel"] for r in per_layer_results) / len(per_layer_results)

    print("=" * 70)
    print("结果对比:")
    print("=" * 70)
    print(f"全局 scale=1.0 平均相对误差:        {global_avg_rel:.2f}%")
    print(f"Per-layer scale 平均相对误差:       {per_layer_avg_rel:.2f}%")
    print(f"改善幅度:                           {(global_avg_rel - per_layer_avg_rel) / global_avg_rel * 100:.1f}%")
    print()

    # 逐层对比
    print("=" * 70)
    print("逐层对比 (前 10 层):")
    print("=" * 70)
    print(f"{'Layer':<8}{'K Scale':<12}{'全局误差':<14}{'优化后误差':<14}{'改善':<10}")
    print("-" * 70)

    for i in range(min(10, num_layers)):
        improve = global_results[i]["mean_rel"] - per_layer_results[i]["mean_rel"]
        print(f"{i:<8}{k_scales[i]:<12.5f}{global_results[i]['mean_rel']:<14.2f}%"
              f"{per_layer_results[i]['mean_rel']:<14.2f}%{improve:<10.2f}%")

    print()

    # 找出改善最明显的层
    improvements = [(i, global_results[i]["mean_rel"] - per_layer_results[i]["mean_rel"], k_scales[i])
                   for i in range(num_layers)]
    improvements.sort(key=lambda x: x[1], reverse=True)

    print("=" * 70)
    print("改善最明显的 5 层:")
    print("=" * 70)
    print(f"{'Layer':<8}{'K Scale':<12}{'全局误差':<14}{'优化后误差':<14}{'改善':<10}")
    print("-" * 70)

    for i in range(min(5, len(improvements))):
        layer_id, improve, scale = improvements[i]
        print(f"{layer_id:<8}{scale:<12.5f}{global_results[layer_id]['mean_rel']:<14.2f}%"
              f"{per_layer_results[layer_id]['mean_rel']:<14.2f}%{improve:<10.2f}%")

    print("=" * 70)

    torch.set_default_device("cpu")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
