# Eagle3 投机解码加速技术指南 (Speculative Decoding with CUDA Graph)

本文档系统介绍了 FastVLLM 中集成的 **Eagle3 投机解码系统**，涵盖其算法架构、底层算子改造、CUDA Graph 硬件级回放优化、使用方式以及工业级性能权衡剖析。

---

## 🚀 核心优化亮点

* **三层特征融合（Tri-Layer Feature Fusion）**：在目标模型（Qwen3-0.6B）第 1、13、24 层提取隐藏状态并拼接融合，保留早期语法、中期语义和晚期概率特征，大幅提升候选预测质量。
* **验证步（Verify Step）CUDA Graph 动转静**：重构底层 Attention 算子为 `flash_attn_with_kvcache`（因果注意力模式），彻底攻克投机解码中动态步长无法被 CUDA Graph 静态捕获的痛点。**单步验证延迟由 40.56ms 骤降至 3.10ms（13.05x 硬件级加速，开销消除 92.3%）**。
* **中间层特征显存 D2D 录制机制**：图捕获阶段直接录制显存 D2D 拷贝指令，图回放期间零 CPU 介入、零 Python 开销，**中间层 Aux 特征与 Logits 重放绝对误差为 0.000000**。
* **英文 JSON Tool Calling 场景实测**：首候选命中率达 **64.8%**，单轮平均产出 **2.06 tokens**。

---

## 📐 算法与系统架构

### 1. 投机解码一轮生命周期（Single Round Lifecycle）

在每一轮（Round）投机解码中，系统执行以下三步闭环：

```
[Target KV Cache]
       │
       ▼ (前轮特征覆盖位置 S-1)
┌──────────────────────────────────────────────┐
│ 1. Draft 阶段 (Eagle3 Head, 轻量 1 层自回归)  │
│    • 第 1 步：吃前序 target 特征 (Teacher-Forcing)  │
│    • 第 2..k 步：吃自身上一输出 (Free-Running)  │
│    ➜ 产出 k 个预测候选 candidates            │
└──────────────────────────────────────────────┘
       │
       ▼ (增量新行 [S-1, S+k-1]，共 k+1 行)
┌──────────────────────────────────────────────┐
│ 2. Verify 阶段 (CUDA Graph 硬件极速回放)       │
│    • 静态写入 static_verify_ids / pos / slots │
│    • 硬件重放 spec_verify_graph.replay()     │
│    • flash_attn_with_kvcache (因果注意力)     │
│    ➜ 单次前向验证全部候选，捕获下一轮 Aux 特征 │
└──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│ 3. 接受前缀与记账 (Acceptance & Bookkeeping)   │
│    • 贪心比对：匹配前 m 个候选                 │
│    • m == k 时获得额外 bonus token           │
│    • 截断 EOS，多余 Token 进入 Pending 队列     │
└──────────────────────────────────────────────┘
```

### 2. 动转静：FlashAttention 动态参数消除

传统验证步采用 `flash_attn_varlen_func`，其参数 `max_seqlen_k` 是 Python 动态整数，步进时无法被 CUDA Graph 捕获。
FastVLLM 改构为基于 GPU 显存级张量寻址的 `flash_attn_with_kvcache`：
* 将验证输入 Query 构造成 `[1, M, num_heads, head_dim]` 并设置 `causal=True`；
* 序列长度直接由 GPU 上的 `context_lens[0] = end` 动态赋值；
* 彻底实现一次图编译、全程零开销硬件回放。

---

## 🛠️ 快速上手

### 1. 基础调用方式

```python
from fastvllm.engine.llm_engine import LLMEngine
from fastvllm.sampling_params import SamplingParams

engine = LLMEngine(
    model="/home/uos/huggingface/Qwen3-0.6B",
    enable_eagle3=True,
    eagle3_model_path="/home/uos/huggingface/Qwen3-0.6B-eagle3",
    num_speculative_tokens=3,
    enforce_eager=False,  # 默认即为 False，自动启用 CUDA Graph 加速
)

prompt = 'Tool call: {"name": "get_weather", "arguments": {"city": "San Francisco"'
outputs = engine.generate([prompt], SamplingParams(temperature=0.0, max_tokens=80))

print(outputs[0]["text"])
```

### 2. 高层包装 API (`LLMWithEagle3`)

```python
from fastvllm.eagle3_llm import LLMWithEagle3
from fastvllm.sampling_params import SamplingParams

llm = LLMWithEagle3(
    model="/home/uos/huggingface/Qwen3-0.6B",
    eagle3_model_path="/home/uos/huggingface/Qwen3-0.6B-eagle3",
    num_speculative_tokens=3,
)

outputs = llm.generate(["Hello, FastVLLM."], SamplingParams(temperature=0.0, max_tokens=50))
print(llm.get_spec_stats())
```

---

## 📊 性能基准测试数据 (RTX 3080)

### 1. 单步 Verify 延迟对比
* **Eager 模式单步 Verify**：`40.57 ms`
* **CUDA Graph 单步 Verify**：`3.11 ms`
* **验证单步硬件加速比**：**13.05x** (延迟消除 92.3%)

### 2. 真实英文 JSON Tool Calling 场景实测
* **平均接受率**：`35.38%` (首候选命中率达 `64.8%`)
* **平均每轮产出**：`2.06 tokens` (减少 51.5% 的目标模型步数)
* **端到端加速比**：Eager 投机 `1.23x`，结合 CUDA Graph 后吞吐达 `44~62 tok/s`，相比 Eager 基准加速达 **2.22x ~ 3.53x**。

---

## ⚖️ 架构设计与工程权衡 (Engineering Trade-offs)

### 1. 为什么采用“线性链”而非 SGLang 的“树状（Tree Attention）”？
SGLang 中的 EAGLE 支持树状注意力（Tree Attention），需要依赖 **FlashInfer** 算子传入动态 2D Tree Mask。在端侧小模型（如 Qwen3-0.6B）上，CPU 维护一棵包含几十个节点的动态树会产生巨大的调度开销；同时，动态树难以直接融入标准的 CUDA Graph。
FastVLLM 采用轻量级**线性链式 Eagle-3**（相当于 SGLang 的 `topk=1` 模式），既保留了三层特征融合的高命中率，又使验证步能够完美与 CUDA Graph 结合，单步耗时压低至 3ms。

### 2. 小模型上的投机惩罚现象 (Speculation Penalty)
在极致小模型（0.6B）上，目标模型自身开启 CUDA Graph 后单步前向仅需约 4ms（吞吐超 200 tok/s）。此时若 $k$ 选得过大（如 $k=3$），Draft 模型的累积开销会侵蚀收益。
* **工程选型建议**：
  * 在 0.6B ~ 1.5B 轻量模型上，推荐 **$k=1$**，首候选接受率高达 65%，能以极微小的 Draft 开销拿到稳定收益；
  * 在 7B / 14B / 70B 等大模型上（单步 20~50ms），推荐 **$k=3 \sim 5$**，投机加速比将显著攀升至 2x~3x。

### 3. 单用户投机与多用户批处理降级
当前投机解码主要面向 **Single-User 极速响应（Latency-Critical）** 场景。当系统面临并发请求（`batch > 1`）时，`Eagle3ModelRunner` 会自动自适应降级回成熟的标准 Paged Batch Decode，避免多序列投机因接受长度不均引发的长尾参差气泡（Ragged Batch Bubble）。
