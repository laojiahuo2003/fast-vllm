#!/usr/bin/env python3
"""
诊断：global scale=1.0 vs per-layer scale，各自损失了哪部分精度

对每层真实 K 统计：
  - amax / p99.9 / std（数值分布）
  - global scale=1.0 下被 clamp 裁剪的元素数（amax > 448 才会发生）
  - per-layer scale 下跌进 e4m3 subnormal 区的元素数（|k| < scale * 2^-6）
  - 两种方案的 SQNR / 平均相对误差 / 余弦

用法: ./.venv/bin/python test_fp8_scale_diag.py
"""

import json

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from fastvllm.models.qwen3 import Qwen3ForCausalLM
from fastvllm.utils.context import set_context, reset_context
from fastvllm.utils.loader import load_model

FP8_E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2 ** -6  # 0.015625，低于此进入 subnormal

PROMPTS = [
    "什么是机器学习？它与深度学习有什么区别？请详细解释。",
    "请解释一下注意力机制的工作原理，包括 Q、K、V 的含义。",
    "如何理解大语言模型的上下文窗口？它受哪些因素限制？",
    "Write a detailed explanation of how flash attention works and why it is faster.",
    "Explain the difference between pre-training, fine-tuning, and RLHF in modern LLMs.",
]


def quant_dequant(k, scale):
    kq = (k.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).float() * scale
    return kq


def metrics(k_orig, k_q):
    diff = k_orig.float() - k_q
    sqnr = 10 * torch.log10((k_orig.float() ** 2).sum() / (diff ** 2).sum()).item()
    rel = (diff.abs() / (k_orig.float().abs() + 1e-8)).mean().item() * 100
    cos = F.cosine_similarity(k_orig.float().flatten(1), k_q.flatten(1), dim=1).mean().item()
    return sqnr, rel, cos


def main():
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", "tcp://localhost:29500", world_size=1, rank=0)
    hf_config = AutoConfig.from_pretrained("/home/uos/huggingface/Qwen3-0.6B/")
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")

    model = Qwen3ForCausalLM(hf_config)
    load_model(model, "/home/uos/huggingface/Qwen3-0.6B/")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("/home/uos/huggingface/Qwen3-0.6B/")

    with open("fp8_k_scales.json") as f:
        k_scales = json.load(f)["k_scale"]

    attn_modules = [m for m in model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
    assert len(attn_modules) == hf_config.num_hidden_layers

    # 收集每层真实 K（prefill，中英文混合 prompt）
    all_ids, cu = [], [0]
    for p in PROMPTS:
        ids = tokenizer.encode(p, add_special_tokens=True)
        all_ids.extend(ids)
        cu.append(cu[-1] + len(ids))
    input_ids = torch.tensor(all_ids, dtype=torch.int64, device="cuda")
    positions = torch.arange(len(all_ids), dtype=torch.int64, device="cuda")
    cu_q = torch.tensor(cu, dtype=torch.int32, device="cuda")
    max_len = max(b - a for a, b in zip(cu[:-1], cu[1:]))

    k_values = []
    handles = []
    for lid, m in enumerate(attn_modules):
        handles.append(m.register_forward_hook(
            lambda mod, inp, out, lid=lid: k_values.append((lid, inp[1].detach().float().clone()))))
    set_context(True, cu_q, cu_q, max_len, max_len,
                torch.zeros(0, dtype=torch.int32, device="cuda"), None, None)
    with torch.inference_mode():
        model.compute_logits(model(input_ids, positions))
    reset_context()
    for h in handles:
        h.remove()

    print("=" * 100)
    print("per-layer K 数值分布与两种 scale 的损失来源")
    print("=" * 100)
    print(f"{'layer':>5} {'amax':>9} {'p99.9':>8} {'std':>7} | "
          f"{'global: 裁剪':>12} {'SQNR':>7} {'relErr':>7} | "
          f"{'per-layer: subnormal':>19} {'SQNR':>7} {'relErr':>7}")
    print("-" * 100)

    agg = {"g_sqnr": [], "g_rel": [], "p_sqnr": [], "p_rel": [],
           "g_clip": 0, "p_sub": 0, "total": 0}
    for lid, k in sorted(k_values, key=lambda x: x[0]):
        s_layer = k_scales[lid]
        n = k.numel()

        k_g = quant_dequant(k, 1.0)
        k_p = quant_dequant(k, s_layer)

        # global scale=1.0: 只有 |k| > 448 才会被 clamp
        n_clip = (k.abs() > FP8_E4M3_MAX).sum().item()
        # per-layer: |k| < scale * 2^-6 会掉进 subnormal（相对精度下降）
        n_sub = (k.abs() < s_layer * E4M3_MIN_NORMAL).sum().item()

        g_sqnr, g_rel, _ = metrics(k, k_g)
        p_sqnr, p_rel, _ = metrics(k, k_p)

        agg["g_sqnr"].append(g_sqnr); agg["p_sqnr"].append(p_sqnr)
        agg["g_rel"].append(g_rel); agg["p_rel"].append(p_rel)
        agg["g_clip"] += n_clip; agg["p_sub"] += n_sub; agg["total"] += n

        if lid < 12 or n_clip > 0 or lid % 6 == 0:
            print(f"{lid:>5} {k.abs().max():>9.3f} {k.abs().quantile(0.999):>8.3f} {k.std():>7.4f} | "
                  f"{n_clip:>7}/{n:<5} {g_sqnr:>7.2f} {g_rel:>6.2f}% | "
                  f"{n_sub:>12}/{n:<6} {p_sqnr:>7.2f} {p_rel:>6.2f}%")

    print("-" * 100)
    t = agg["total"]
    print(f"全模型统计 ({len(k_values)} 层, {t} 个 K 元素):")
    print(f"  global scale=1.0 : 平均 SQNR {sum(agg['g_sqnr']) / len(agg['g_sqnr']):.2f} dB, "
          f"平均 relErr {sum(agg['g_rel']) / len(agg['g_rel']):.2f}%, "
          f"被裁剪 {agg['g_clip']}/{t} ({agg['g_clip'] / t * 100:.4f}%)")
    print(f"  per-layer scale  : 平均 SQNR {sum(agg['p_sqnr']) / len(agg['p_sqnr']):.2f} dB, "
          f"平均 relErr {sum(agg['p_rel']) / len(agg['p_rel']):.2f}%, "
          f"掉 subnormal {agg['p_sub']}/{t} ({agg['p_sub'] / t * 100:.4f}%)")
    n_sat = sum(1 for s in k_scales if s * FP8_E4M3_MAX < 448)  # amax < 448 的层
    print(f"\n  amax > 448（global 会裁剪）的层: {sum(1 for s in k_scales if s > 1.0)}/{len(k_scales)}")
    print(f"  amax < 448 的层: {sum(1 for s in k_scales if s <= 1.0)}/{len(k_scales)}"
          f"  ← 这些层用 per-layer scale 只会把小值推进 subnormal，没有收益")
    print("=" * 100)

    torch.set_default_device("cpu")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
