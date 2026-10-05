"""
验证 target_extend 能否成功被 CUDA Graph 捕获并正确重放
对比 Eager 模式与 CUDA Graph 模式的数值一致性与耗时
"""
import time
import torch
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from fastvllm.engine.llm_engine import LLMEngine
from fastvllm.utils.context import set_context, reset_context, get_context
from fastvllm.layers.attention import store_kvcache
from flash_attn import flash_attn_with_kvcache

MODEL = "/home/uos/huggingface/Qwen3-0.6B"
EAGLE3 = "/home/uos/huggingface/Qwen3-0.6B-eagle3"
K = 3
M = K + 1  # 4 tokens per verify step

print("=" * 80)
print("测试：Eagle3 Verify (target_extend) CUDA Graph 捕获与回放")
print(f"K = {K}, Verify Tokens = {M}")
print("=" * 80)

eng = LLMEngine(
    model=MODEL,
    enable_eagle3=True,
    eagle3_model_path=EAGLE3,
    num_speculative_tokens=K,
    max_num_batched_tokens=4096,
    max_model_len=1024,
    enforce_eager=True,
)
mr = eng.model_runner
model = mr.model
target_model = model.target_model
dev = target_model.lm_head.weight.device
hf_config = eng.config.hf_config
bs = eng.config.kvcache_block_size
max_blocks = eng.config.num_kvcache_blocks

print("1. 准备静态缓冲区...")
static_ids = torch.zeros(M, dtype=torch.int64, device=dev)
static_pos = torch.zeros(M, dtype=torch.int64, device=dev)
static_slot = torch.zeros(M, dtype=torch.int32, device=dev)
static_context_lens = torch.zeros(1, dtype=torch.int32, device=dev)
static_block_tables = torch.zeros(1, (1024 + bs - 1) // bs, dtype=torch.int32, device=dev)
static_hidden = torch.zeros(M, hf_config.hidden_size, dtype=hf_config.dtype, device=dev)
static_logits = torch.zeros(M, hf_config.vocab_size, dtype=hf_config.dtype, device=dev)

static_feats = {
    lid: torch.zeros(M, hf_config.hidden_size, dtype=hf_config.dtype, device=dev)
    for lid in model.eagle3_config.extract_layers
}

print("2. 设定测试数据...")
prefix_len = 20
total_len = prefix_len + M
test_ids = torch.tensor([123, 456, 789, 1011], dtype=torch.long, device=dev)
test_pos = torch.arange(prefix_len, total_len, dtype=torch.long, device=dev)
test_slots = torch.arange(prefix_len, total_len, dtype=torch.int32, device=dev)
test_block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=dev)

with torch.inference_mode():
    # --- 3. 运行 Eager 模式下的 target_extend 作为 Ground Truth ---
    print("\n3. 运行 Eager 模式 target_extend (基准 Ground Truth)...")
    eager_feats, eager_hidden = model.target_extend(
        test_ids, test_pos, prefix_len=prefix_len,
        slot_mapping=test_slots, block_tables=test_block_table
    )
    eager_logits = F.linear(eager_hidden, target_model.lm_head.weight)
    eager_preds = eager_logits.argmax(dim=-1).tolist()
    print(f"   Eager Predictions: {eager_preds}")

# --- 4. 捕获 CUDA Graph ---
print("\n4. 开始 CUDA Graph 预热与捕获...")
# 设置静态 context
set_context(
    is_prefill=False,
    slot_mapping=static_slot,
    context_lens=static_context_lens,
    block_tables=static_block_tables,
    is_spec_verify=True,
)

# Warmup
static_ids.copy_(test_ids)
static_pos.copy_(test_pos)
static_slot.copy_(test_slots)
static_context_lens[0] = total_len
static_block_tables[:, :test_block_table.shape[1]].copy_(test_block_table)

with torch.inference_mode():
    h = target_model(static_ids, static_pos)
    static_hidden.copy_(h)
    static_logits.copy_(F.linear(static_hidden, target_model.lm_head.weight))
    for lid in model.eagle3_config.extract_layers:
        static_feats[lid].copy_(model._cached_features[lid])
    torch.cuda.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        h = target_model(static_ids, static_pos)
        static_hidden.copy_(h)
        static_logits.copy_(F.linear(static_hidden, target_model.lm_head.weight))
        for lid in model.eagle3_config.extract_layers:
            static_feats[lid].copy_(model._cached_features[lid])
    torch.cuda.synchronize()
    print("   ✅ CUDA Graph 捕获成功！")

with torch.inference_mode():
    # --- 5. 验证 CUDA Graph 回放正确性 ---
    print("\n5. 验证 CUDA Graph 回放正确性...")
    # 重放测试 1：相同输入
    graph.replay()
    torch.cuda.synchronize()
    graph_preds = static_logits.argmax(dim=-1).tolist()
    print(f"   Graph Predictions (相同输入): {graph_preds}")
    assert graph_preds == eager_preds, f"Mismatch! Eager: {eager_preds}, Graph: {graph_preds}"
    for lid in model.eagle3_config.extract_layers:
        feat_diff = (static_feats[lid] - eager_feats[lid]).abs().max().item()
        print(f"   Layer {lid} Aux Feature Max Diff: {feat_diff:.6f}")
        assert feat_diff < 1e-3, f"Feature diff too large: {feat_diff}"
    print("   ✅ 预测与中间层特征完全一致！")

    # 重放测试 2：全新输入
    new_prefix_len = 30
    new_total_len = new_prefix_len + M
    new_ids = torch.tensor([555, 666, 777, 888], dtype=torch.long, device=dev)
    new_pos = torch.arange(new_prefix_len, new_total_len, dtype=torch.long, device=dev)
    new_slots = torch.arange(new_prefix_len, new_total_len, dtype=torch.int32, device=dev)

    # Eager ground truth for new input
    eager_feats_new, eager_hidden_new = model.target_extend(
        new_ids, new_pos, prefix_len=new_prefix_len,
        slot_mapping=new_slots, block_tables=test_block_table
    )
    eager_preds_new = F.linear(eager_hidden_new, target_model.lm_head.weight).argmax(dim=-1).tolist()

    # Graph replay with new input
    static_ids.copy_(new_ids)
    static_pos.copy_(new_pos)
    static_slot.copy_(new_slots)
    static_context_lens[0] = new_total_len
    graph.replay()
    torch.cuda.synchronize()
    graph_preds_new = static_logits.argmax(dim=-1).tolist()
    print(f"   Eager New Predictions: {eager_preds_new}")
    print(f"   Graph New Predictions: {graph_preds_new}")
    assert graph_preds_new == eager_preds_new, f"Mismatch on new input! Eager: {eager_preds_new}, Graph: {graph_preds_new}"
    print("   ✅ 全新输入回放预测 100% 精确吻合！")

    # --- 6. 速度基准测试 (100 次迭代) ---
    print("\n6. 速度对比测试 (100 轮 Verify 前向)...")
    N_ITERS = 100

    # Eager 耗时
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _f, h = model.target_extend(test_ids, test_pos, prefix_len=prefix_len, slot_mapping=test_slots, block_tables=test_block_table)
        _l = F.linear(h, target_model.lm_head.weight)
    torch.cuda.synchronize()
    dt_eager = (time.perf_counter() - t0) * 1000 / N_ITERS

    # CUDA Graph 耗时
    t1 = time.perf_counter()
    for _ in range(N_ITERS):
        static_ids.copy_(test_ids)
        static_pos.copy_(test_pos)
        static_slot.copy_(test_slots)
        static_context_lens[0] = total_len
        graph.replay()
    torch.cuda.synchronize()
    dt_graph = (time.perf_counter() - t1) * 1000 / N_ITERS

    print(f"   • Eager 模式单步 Verify:      {dt_eager:.3f} ms")
    print(f"   • CUDA Graph 单步 Verify:     {dt_graph:.3f} ms")
    print(f"   • Verify 单步加速比:          {dt_eager / dt_graph:.2f}x ({100*(1 - dt_graph/dt_eager):.1f}% 延迟消除)")

reset_context()
eng.exit()
print("\n✅ 所有测试完美通过！")
