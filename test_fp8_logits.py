#!/usr/bin/env python3
"""
端到端 logits 对比：BF16 KV Cache vs FP8 KV Cache（统计版）

对多个 prompt、多档上下文长度跑真实 prefill + greedy decode，逐步对比 logits：
  - 相对平均绝对误差 mean|Δlogit| / mean|logit|
  - 余弦相似度（逐步，报 mean / min）
  - KL 散度 KL(bf16‖fp8)（报 mean / p95）
  - top-1 一致率、生成 token 一致率
  - 首次分叉步数（greedy 下 argmax 第一次不同的 decode step）

长上下文由多条不同 prompt 循环拼接而成，避免重复文本退化成简单场景。

用法:
    ./.venv/bin/python test_fp8_logits.py                      # 默认全量测试
    ./.venv/bin/python test_fp8_logits.py --lengths 256 1024   # 指定上下文长度
    ./.venv/bin/python test_fp8_logits.py --decode-steps 16
"""

import argparse
import json

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from fastvllm.models.qwen3 import Qwen3ForCausalLM
from fastvllm.utils.context import set_context, reset_context
from fastvllm.utils.loader import load_model

PROMPTS = [
    "什么是机器学习？它与深度学习有什么区别？请详细解释。",
    "请解释一下注意力机制的工作原理，包括 Q、K、V 的含义。",
    "如何理解大语言模型的上下文窗口？它受哪些因素限制？",
    "请用通俗的语言解释 Transformer 架构的核心创新。",
    "Write a detailed explanation of how flash attention works and why it is faster.",
    "Explain the difference between pre-training, fine-tuning, and RLHF in modern LLMs.",
    "请描述牛顿三大定律，并各举一个生活中的例子。",
    "What are the main causes of climate change, and what can individuals do about it?",
    "请介绍中国古代四大发明，并说明它们对世界历史的影响。",
    "Describe the process of photosynthesis step by step for a high school student.",
]


def build_kv_caches(hf_config, num_blocks, block_size):
    """分配 BF16 与 FP8 两套 KV Cache，以及 FP8 的 per-token V scale"""
    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

    shape = (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
    kv_bf16 = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    kv_fp8 = torch.zeros(shape, dtype=torch.float8_e4m3fn, device="cuda")
    v_scale = torch.ones(num_layers, num_blocks * block_size, dtype=torch.float32, device="cuda")
    return kv_bf16, kv_fp8, v_scale


def bind_caches(model, kv_cache, v_scale_cache, k_scales, is_fp8):
    """把 KV Cache 绑定到每个 Attention 模块"""
    layer_id = 0
    for module in model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = kv_cache[0, layer_id]
            module.v_cache = kv_cache[1, layer_id]
            if is_fp8:
                module.k_scale = k_scales[layer_id]
                module.v_scale_cache = v_scale_cache[layer_id]
                module.is_fp8_kv_cache = True
            else:
                module.is_fp8_kv_cache = False
            layer_id += 1
    assert layer_id == kv_cache.shape[1], f"绑定了 {layer_id} 层，cache 有 {kv_cache.shape[1]} 层"


def run_sequence(model, ids, num_decode_steps):
    """跑 prefill + greedy decode，返回每步 decode 的 logits 与生成的 token"""
    P = len(ids)
    num_blocks_needed = (P + num_decode_steps + 255) // 256 + 1

    input_ids = torch.tensor(ids, dtype=torch.int64, device="cuda")
    positions = torch.arange(P, dtype=torch.int64, device="cuda")
    set_context(
        True,
        cu_seqlens_q=torch.tensor([0, P], dtype=torch.int32, device="cuda"),
        cu_seqlens_k=torch.tensor([0, P], dtype=torch.int32, device="cuda"),
        max_seqlen_q=P,
        max_seqlen_k=P,
        slot_mapping=torch.arange(P, dtype=torch.int32, device="cuda"),
    )
    with torch.inference_mode():
        logits = model.compute_logits(model(input_ids, positions))
    reset_context()

    next_token = int(logits[-1].argmax())
    logits_steps, tokens = [], [next_token]

    block_table = list(range(num_blocks_needed)) + [-1] * (64 - num_blocks_needed)
    block_tables = torch.tensor([block_table], dtype=torch.int32, device="cuda")
    cur_len = P
    for _ in range(num_decode_steps):
        set_context(
            False,
            slot_mapping=torch.tensor([cur_len], dtype=torch.int32, device="cuda"),
            context_lens=torch.tensor([cur_len + 1], dtype=torch.int32, device="cuda"),
            block_tables=block_tables,
        )
        with torch.inference_mode():
            logits = model.compute_logits(
                model(torch.tensor([next_token], dtype=torch.int64, device="cuda"),
                      torch.tensor([cur_len], dtype=torch.int64, device="cuda"))
            )
        reset_context()
        next_token = int(logits[-1].argmax())
        logits_steps.append(logits[-1].float().cpu())
        tokens.append(next_token)
        cur_len += 1

    return torch.stack(logits_steps), tokens


def compare(a, b):
    """a: BF16 logits [T, V], b: FP8 logits [T, V]"""
    diff = (a - b).abs()
    cos = F.cosine_similarity(a, b, dim=1)
    kl = F.kl_div(F.log_softmax(b, dim=-1), F.log_softmax(a, dim=-1),
                  log_target=True, reduction="none").sum(dim=-1)
    top1_match = (a.argmax(-1) == b.argmax(-1))
    # 首次分叉步数：argmax 第一次不同的 step 下标（0-based），未分叉则记 -1
    diverge = (~top1_match).nonzero()
    first_diverge = int(diverge[0, 0]) if diverge.numel() else -1
    # BF16 分布的不确定性：熵越高、top-1 概率越低，量化噪声越容易翻转 argmax
    logp = F.log_softmax(a, dim=-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    top1_prob = logp.exp().max(dim=-1).values
    return {
        "rel_mean_abs": (diff.mean() / a.abs().mean()).item(),
        "abs_mean": diff.mean().item(),
        "max_abs": diff.max().item(),
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
        "kl_mean": kl.mean().item(),
        "kl_p95": kl.quantile(0.95).item(),
        "top1": top1_match.float().mean().item(),
        "first_diverge": first_diverge,
        "entropy": entropy.mean().item(),
        "top1_prob": top1_prob.mean().item(),
        "ambig_frac": (top1_prob < 0.5).float().mean().item(),
    }


def mean_std(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    return t.mean().item(), (t.std(unbiased=True).item() if len(vals) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/uos/huggingface/Qwen3-0.6B/")
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 256, 1024, 2048])
    parser.add_argument("--decode-steps", type=int, default=16)
    args = parser.parse_args()

    print("=" * 78)
    print("BF16 vs FP8 KV Cache: logits 端到端对比（统计版）")
    print("=" * 78)
    print(f"模型: {args.model}")
    print(f"prompt 数: {len(PROMPTS)}（中英混合）  上下文长度: {args.lengths}  decode 步数: {args.decode_steps}")
    print(f"每个数据点样本数: {len(PROMPTS)} prompt × {args.decode_steps} step = {len(PROMPTS) * args.decode_steps}\n")

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", "tcp://localhost:29500", world_size=1, rank=0)
    hf_config = AutoConfig.from_pretrained(args.model)
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")

    model = Qwen3ForCausalLM(hf_config)
    load_model(model, args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"模型加载完成: {hf_config.num_hidden_layers} 层, vocab={hf_config.vocab_size}")

    k_scales = [1.0] * hf_config.num_hidden_layers
    with open("fp8_k_scales.json") as f:
        k_scales = json.load(f)["k_scale"]
    print(f"per-layer K scale: [{min(k_scales):.5f}, {max(k_scales):.5f}]")

    max_len = max(args.lengths)
    num_blocks = (max_len + args.decode_steps + 511) // 256 + 2
    kv_bf16, kv_fp8, v_scale = build_kv_caches(hf_config, num_blocks, 256)
    print(f"KV Cache: {num_blocks} blocks × 256 tokens\n")

    tokenized = [tokenizer.encode(p, add_special_tokens=True) for p in PROMPTS]

    def build_context(target_len, start_idx):
        """多条不同 prompt 循环拼接，截断到目标长度"""
        ids, i = [], start_idx
        while len(ids) < target_len:
            ids += tokenized[i % len(tokenized)]
            i += 1
        return ids[:target_len]

    results = {}  # results[mode][length] = [metric dict, ...]
    for length in args.lengths:
        for mode, use_per_layer in [("global", False), ("per-layer", True)]:
            scales = k_scales if use_per_layer else [1.0] * hf_config.num_hidden_layers
            per_prompt = []
            for idx in range(len(PROMPTS)):
                ids = build_context(length, idx)

                bind_caches(model, kv_bf16, None, scales, is_fp8=False)
                logits_bf16, tokens_bf16 = run_sequence(model, ids, args.decode_steps)

                kv_fp8.view(torch.uint8).zero_()
                v_scale.fill_(1.0)
                bind_caches(model, kv_fp8, v_scale, scales, is_fp8=True)
                logits_fp8, tokens_fp8 = run_sequence(model, ids, args.decode_steps)

                m = compare(logits_bf16, logits_fp8)
                m["tok_match"] = sum(a == b for a, b in zip(tokens_bf16, tokens_fp8)) / len(tokens_bf16)
                per_prompt.append(m)

            results.setdefault(mode, {})[length] = per_prompt
            r = per_prompt
            print(f"[len={length:>5}] {mode:<10} "
                  f"rel|Δ|={mean_std([x['rel_mean_abs'] for x in r])[0] * 100:6.3f}%  "
                  f"cos={mean_std([x['cos_mean'] for x in r])[0]:.5f}  "
                  f"KL={mean_std([x['kl_mean'] for x in r])[0]:.2e}  "
                  f"top1={mean_std([x['top1'] for x in r])[0] * 100:5.1f}%  "
                  f"tok={mean_std([x['tok_match'] for x in r])[0] * 100:5.1f}%")

    # ---- 汇总表 ----
    print("\n" + "=" * 92)
    print("汇总（mean ± std，跨 prompt；每格样本 = 10 prompt × 16 step = 160）")
    print("=" * 92)
    header = (f"{'ctx':>5} {'scale':>10} {'mean|Δ|':>8} {'rel|Δ|':>10} {'cos':>15} "
              f"{'KL mean':>10} {'top-1':>7} {'tok':>7} {'首分叉':>7} {'熵':>7} {'top1概率':>8} {'模糊步':>7}")
    print(header)
    print("-" * 92)
    for length in args.lengths:
        for mode in ["global", "per-layer"]:
            r = results[mode][length]
            absm = mean_std([x["abs_mean"] for x in r])[0]
            rel, rel_s = mean_std([x["rel_mean_abs"] for x in r])
            cos, cos_s = mean_std([x["cos_mean"] for x in r])
            kl = mean_std([x["kl_mean"] for x in r])[0]
            t1 = mean_std([x["top1"] for x in r])[0] * 100
            tk = mean_std([x["tok_match"] for x in r])[0] * 100
            fd = [x["first_diverge"] for x in r]
            n_diverge = sum(1 for x in fd if x >= 0)
            ent = mean_std([x["entropy"] for x in r])[0]
            p1 = mean_std([x["top1_prob"] for x in r])[0]
            amb = mean_std([x["ambig_frac"] for x in r])[0] * 100
            print(f"{length:>5} {mode:>10} {absm:>8.4f} {rel * 100:>6.2f}±{rel_s * 100:<4.2f}% "
                  f"{cos:>8.5f}±{cos_s:<6.5f} {kl:>10.2e} {t1:>6.1f}% {tk:>6.1f}% "
                  f"{n_diverge:>4}/{len(fd)} {ent:>7.3f} {p1:>8.3f} {amb:>6.1f}%")
    print("-" * 92)
    print("mean|Δ| = mean|Δlogit|；rel|Δ| = mean|Δlogit|/mean|logit|；模糊步 = BF16 top-1 概率 < 0.5 的 step 占比")
    print("熵/模糊步越高 → 分布越平坦 → 量化噪声越容易翻转 argmax（短上下文分叉的主因）")
    print("=" * 92)

    torch.set_default_device("cpu")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
