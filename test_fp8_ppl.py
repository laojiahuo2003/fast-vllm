#!/usr/bin/env python3
"""
FP8 KV Cache 对长上下文质量的影响：teacher-forced 困惑度（自然文本）

用本地缓存的 HotpotQA 维基百科文本作为自然语料（非合成、非重复），
按引擎的 chunked prefill 方式分块前向——第 2 块起 KV 从 cache 读取，
因此完整走过「量化写入 → 反量化读取」路径，测得的是端到端困惑度差异。

    PPL = exp(mean_t NLL(t | t_0..t-1))

依赖：需要 pyarrow 读取本地缓存的 HotpotQA 数据集
      （uv pip install pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple）

用法:
    ./.venv/bin/python test_fp8_ppl.py                    # 默认 1024/2048
    ./.venv/bin/python test_fp8_ppl.py --lengths 512 2048 4096
    ./.venv/bin/python test_fp8_ppl.py --num-texts 8
"""

import argparse
import glob
import json

import pyarrow.ipc as ipc
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from fastvllm.models.qwen3 import Qwen3ForCausalLM
from fastvllm.utils.context import set_context, reset_context
from fastvllm.utils.loader import load_model

BLOCK_SIZE = 256


def load_natural_texts(num_chars=400000):
    """从本地 HotpotQA 缓存读维基百科段落，拼成自然英文语料"""
    path = glob.glob("/home/uos/.cache/huggingface/datasets/hotpotqa___hotpot_qa/"
                     "distractor/0.0.0/*/hotpot_qa-validation.arrow")[0]
    with open(path, "rb") as f:
        table = ipc.open_stream(f).read_all()

    texts = []
    for row in table.to_pylist():
        ctx = row["context"]
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            t = f"{title}. " + " ".join(sents)
            if len(t) > 200:
                texts.append(t)
        if sum(len(x) for x in texts) > num_chars:
            break
    return texts


def build_kv_caches(hf_config, num_blocks):
    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    shape = (2, num_layers, num_blocks, BLOCK_SIZE, num_kv_heads, head_dim)
    kv_bf16 = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    kv_fp8 = torch.zeros(shape, dtype=torch.float8_e4m3fn, device="cuda")
    v_scale = torch.ones(num_layers, num_blocks * BLOCK_SIZE, dtype=torch.float32, device="cuda")
    return kv_bf16, kv_fp8, v_scale


def bind_caches(model, kv_cache, v_scale_cache, k_scales, is_fp8):
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
    assert layer_id == kv_cache.shape[1]


def full_logits(model, ids, positions):
    """返回全部位置的 logits [N, V]

    注意：ParallelLMHead 在 prefill 时只取每个序列最后一个 token（采样只需要末位），
     teacher-forcing 需要全部位置，所以绕过它直接做 lm_head 的线性映射。
    """
    with torch.inference_mode():
        hidden = model.model(
            torch.tensor(ids, dtype=torch.int64, device="cuda"),
            torch.tensor(positions, dtype=torch.int64, device="cuda"))
        return F.linear(hidden.float(), model.lm_head.weight.float())


def chunked_prefill(model, ids):
    """按 BLOCK_SIZE 分块 prefill；第 2 块起 KV 从 cache 读（触发量化路径）"""
    n = len(ids)
    logits_all = []
    for start in range(0, n, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, n)
        q_len = end - start
        if start == 0:
            block_tables = None
        else:
            nb = (end + BLOCK_SIZE - 1) // BLOCK_SIZE
            bt = list(range(nb)) + [-1] * (64 - nb)
            block_tables = torch.tensor([bt], dtype=torch.int32, device="cuda")
        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32, device="cuda"),
            cu_seqlens_k=torch.tensor([0, end], dtype=torch.int32, device="cuda"),
            max_seqlen_q=q_len,
            max_seqlen_k=end,
            slot_mapping=torch.arange(start, end, dtype=torch.int32, device="cuda"),
            block_tables=block_tables,
        )
        logits_all.append(full_logits(model, ids[start:end], range(start, end)))
        reset_context()
    return torch.cat(logits_all)


def ppl_of(logits, ids):
    """teacher-forced 困惑度：logits[:-1] 预测 ids[1:]"""
    nll = F.cross_entropy(logits[:-1].float(), torch.tensor(ids[1:], device=logits.device),
                          reduction="none")
    return nll.mean().item(), torch.exp(nll.mean()).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/uos/huggingface/Qwen3-0.6B/")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--num-texts", type=int, default=8)
    args = parser.parse_args()

    print("=" * 84)
    print("FP8 KV Cache 长上下文质量：teacher-forced 困惑度（HotpotQA 维基百科自然文本）")
    print("=" * 84)
    print(f"模型: {args.model}  语料长度: {args.lengths}  样本数: {args.num_texts} 段\n")

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", "tcp://localhost:29500", world_size=1, rank=0)
    hf_config = AutoConfig.from_pretrained(args.model)
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")

    model = Qwen3ForCausalLM(hf_config)
    load_model(model, args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"模型加载完成: {hf_config.num_hidden_layers} 层")

    with open("fp8_k_scales.json") as f:
        k_scales = json.load(f)["k_scale"]
    print(f"per-layer K scale: [{min(k_scales):.5f}, {max(k_scales):.5f}]")

    max_len = max(args.lengths)
    num_blocks = max_len // BLOCK_SIZE + 2
    kv_bf16, kv_fp8, v_scale = build_kv_caches(hf_config, num_blocks)
    print(f"KV Cache: {num_blocks} blocks × {BLOCK_SIZE} tokens\n")

    texts = load_natural_texts()
    print(f"语料: {len(texts)} 段维基百科文本\n")

    # 切出不重叠的定长 token 窗口
    windows = {}
    for length in args.lengths:
        ws, i = [], 0
        while len(ws) < args.num_texts and i < len(texts):
            ids = tokenizer.encode(" ".join(texts[i:i + 20]), add_special_tokens=False)
            if len(ids) < length:
                i += 20
                continue
            ws.append(ids[:length])
            i += 20
        windows[length] = ws
        print(f"  ctx={length}: {len(ws)} 段")

    results = {}
    for length in args.lengths:
        for mode, use_pl in [("global", False), ("per-layer", True)]:
            scales = k_scales if use_pl else [1.0] * hf_config.num_hidden_layers
            ppls_bf16, ppls_fp8, dnnll = [], [], []
            for ids in windows[length]:
                bind_caches(model, kv_bf16, None, scales, is_fp8=False)
                logits_bf16 = chunked_prefill(model, ids)
                nll_bf16 = F.cross_entropy(logits_bf16[:-1],
                                           torch.tensor(ids[1:], device="cuda"), reduction="none")

                kv_fp8.view(torch.uint8).zero_()
                v_scale.fill_(1.0)
                bind_caches(model, kv_fp8, v_scale, scales, is_fp8=True)
                logits_fp8 = chunked_prefill(model, ids)
                nll_fp8 = F.cross_entropy(logits_fp8[:-1],
                                          torch.tensor(ids[1:], device="cuda"), reduction="none")

                ppls_bf16.append(torch.exp(nll_bf16.mean()).item())
                ppls_fp8.append(torch.exp(nll_fp8.mean()).item())
                dnnll.append((nll_fp8 - nll_bf16).abs().mean().item())

            results.setdefault(mode, {})[length] = {
                "ppl_bf16": sum(ppls_bf16) / len(ppls_bf16),
                "ppl_fp8": sum(ppls_fp8) / len(ppls_fp8),
                "dnll": sum(dnnll) / len(dnnll),
            }
            r = results[mode][length]
            print(f"\n[ctx={length:>5}] {mode:<10} PPL(BF16)={r['ppl_bf16']:.4f}  "
                  f"PPL(FP8)={r['ppl_fp8']:.4f}  相对劣化 {(r['ppl_fp8'] / r['ppl_bf16'] - 1) * 100:+.4f}%  "
                  f"mean|ΔNLL|={r['dnll']:.2e}")

    print("\n" + "=" * 84)
    print("汇总")
    print("=" * 84)
    print(f"{'ctx':>6} {'scale':>10} {'PPL(BF16)':>12} {'PPL(FP8)':>12} {'相对劣化':>12} {'mean|ΔNLL|':>12}")
    print("-" * 84)
    for length in args.lengths:
        for mode in ["global", "per-layer"]:
            r = results[mode][length]
            print(f"{length:>6} {mode:>10} {r['ppl_bf16']:>12.4f} {r['ppl_fp8']:>12.4f} "
                  f"{(r['ppl_fp8'] / r['ppl_bf16'] - 1) * 100:>11.4f}% {r['dnll']:>12.2e}")
    print("-" * 84)
    print("相对劣化 = PPL(FP8)/PPL(BF16) - 1；mean|ΔNLL| 为逐 token NLL 绝对差均值")
    print("=" * 84)

    torch.set_default_device("cpu")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
