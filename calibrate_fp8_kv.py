#!/usr/bin/env python3
"""
FP8 KV Cache —— K 静态 scale 的按层 profiling 校准脚本

用真实文本跑 BF16 前向，hook 每层 Attention 的输入 k（RoPE 之后），
统计每层 K 的 amax，据此标定每层的最优 scale：
    k_scale[layer] = amax[layer] / 448.0

产出 JSON：
    {"num_hidden_layers": <N>, "k_scale": [s0, s1, ...]}

用法：
    ./.venv/bin/python calibrate_fp8_kv.py --model ~/huggingface/Qwen3-0.6B/ --out fp8_k_scales.json
"""

import argparse
import json
import os
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from fastvllm.models.qwen3 import Qwen3ForCausalLM
from fastvllm.utils.context import set_context, reset_context
from fastvllm.utils.loader import load_model

# E4M3 最大可表示值
FP8_E4M3_MAX = 448.0

# Profiling 用真实文本（覆盖不同领域，让 K 量级分布有代表性）
PROMPTS = [
    "什么是机器学习？它与深度学习有什么区别？",
    "请解释一下注意力机制的工作原理。",
    "如何理解大语言模型的上下文窗口？",
    "Transformer 架构的核心创新是什么？",
    "请简述自然语言处理的主要任务。",
    "什么是迁移学习？它在 NLP 中如何应用？",
    "解释一下什么是梯度下降算法。",
    "神经网络中的激活函数有哪些？各有什么特点？",
]


def _resolve_prompt_ids(tokenizer, max_len=512):
    """把一批 prompt 编码成 varlen prefill：返回拼接后的 input_ids 与 cu_seqlens。"""
    all_ids, cu = [], [0]
    for p in PROMPTS:
        ids = tokenizer.encode(p, add_special_tokens=True)
        if not ids:
            ids = [tokenizer.unk_token_id]
        ids = ids[:max_len]
        all_ids.extend(ids)
        cu.append(cu[-1] + len(ids))
    return torch.tensor(all_ids, dtype=torch.int64), cu


def _run_forward(model, input_ids, positions, cu_q, max_len):
    """执行一次前向传播"""
    set_context(True, cu_q, cu_q, max_len, max_len,
                torch.zeros(0, dtype=torch.int32, device='cuda'), None, None)
    with torch.inference_mode():
        model.compute_logits(model(input_ids, positions))
    torch.cuda.synchronize()
    reset_context()


def main():
    parser = argparse.ArgumentParser(description="Calibrate per-layer K scales for FP8 KV Cache")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
                       help="Path to model directory")
    parser.add_argument("--out", default="fp8_k_scales.json",
                       help="Output JSON file path")
    args = parser.parse_args()

    print("=" * 60)
    print("FP8 K Scale Calibration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.out}")
    print()

    # 初始化模型（单卡，BF16）
    torch.cuda.set_device(0)

    # 初始化分布式环境（单卡模式）
    dist.init_process_group("nccl", "tcp://localhost:29500", world_size=1, rank=0)

    hf_config = AutoConfig.from_pretrained(args.model)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")

    print("[1/4] Loading model...")
    model = Qwen3ForCausalLM(hf_config)
    load_model(model, args.model)
    model.eval()
    torch.cuda.synchronize()

    # 定位所有 Attention 子模块
    print("[2/4] Finding attention modules...")
    attn_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            attn_modules.append(module)

    num_layers = len(attn_modules)
    assert num_layers == hf_config.num_hidden_layers, \
        f"Found {num_layers} attention modules, expected {hf_config.num_hidden_layers}"
    print(f"   Found {num_layers} attention layers")

    # 准备输入
    print("[3/4] Preparing prompts...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids, cu = _resolve_prompt_ids(tokenizer)
    input_ids = input_ids.cuda()
    n = input_ids.size(0)
    positions = torch.arange(n, dtype=torch.int64, device='cuda')
    cu_q = torch.tensor(cu, dtype=torch.int32, device='cuda')
    max_len = max(b - a for a, b in zip(cu[:-1], cu[1:]))
    print(f"   Total tokens: {n}, Max seqlen: {max_len}")

    # Pass 1：统计每层 K 的 amax
    print("[4/4] Profiling K scales...")
    layer_amax = [0.0] * num_layers
    handles = []

    for layer_id, module in enumerate(attn_modules):
        def _amax_hook(module, inp, out, lid=layer_id):
            k = inp[1]  # [N, num_kv_heads, head_dim]
            layer_amax[lid] = max(layer_amax[lid], k.abs().amax().item())
        handles.append(module.register_forward_hook(_amax_hook))

    _run_forward(model, input_ids, positions, cu_q, max_len)

    for h in handles:
        h.remove()

    # 计算 scales
    scales = [max(1e-12, amax) / FP8_E4M3_MAX for amax in layer_amax]

    # 验证量化误差
    print("\n   Verifying quantization error...")
    max_rel_err = [0.0] * num_layers
    handles2 = []

    for layer_id, module in enumerate(attn_modules):
        def _err_hook(module, inp, out, lid=layer_id):
            k = inp[1].float()
            s = scales[lid]
            kq = (k / s).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).float() * s
            denom = layer_amax[lid]
            max_rel_err[lid] = max(max_rel_err[lid], (k - kq).abs().max().item() / denom)
        handles2.append(module.register_forward_hook(_err_hook))

    _run_forward(model, input_ids, positions, cu_q, max_len)

    for h in handles2:
        h.remove()

    # 保存结果
    result = {
        "num_hidden_layers": num_layers,
        "k_scale": scales
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    # 输出统计
    print("\n" + "=" * 60)
    print("Calibration Results")
    print("=" * 60)
    print(f"Layers           : {num_layers}")
    print(f"K scale range    : [{min(scales):.5f}, {max(scales):.5f}]")
    print(f"K scale mean     : {sum(scales)/len(scales):.5f}")
    print(f"Max rel error    : {max(max_rel_err)*100:.2f}%")
    print(f"Worst layer      : {max_rel_err.index(max(max_rel_err))}")
    print(f"\nSaved to: {args.out}")
    print("=" * 60)

    # 对比全局 scale=1.0 的误差
    global_scale_err = [max(0, (amax - FP8_E4M3_MAX) / FP8_E4M3_MAX) for amax in layer_amax]
    if any(e > 0 for e in global_scale_err):
        print(f"\n⚠️  Global scale=1.0 would saturate {sum(1 for e in global_scale_err if e > 0)} layers")
        print(f"   Max saturation: {max(global_scale_err)*100:.1f}%")

    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
