# Fast-vLLM Speculative Decoding 现状总结

## 🔍 发现的问题

1. **Eagle3 不是独立模型**
   - 它是 SGLang 专用的 speculative draft head
   - 架构特殊：只有 1 层 + auxiliary connections
   - 需要 SGLang 的特殊推理代码支持
   - **无法直接用 HuggingFace API 加载**

2. **Eagle3 的设计**
   ```
   Target Model (Qwen3-0.6B, 28层)
         ↓ 提取 hidden states (layer 1/13/24)
   Eagle3 Draft Head (1层 Llama)
         ↓ 映射到 32k draft vocab
   候选 tokens
   ```

## ✅ 可行方案对比

### 方案 A：SGLang + Eagle3（最快验证）

**优点**：
- ✅ **0 代码**，10分钟启动
- ✅ **1.7-1.8x 加速**（官方实测 955 vs 553 tok/s）
- ✅ 专门为 Qwen3-0.6B 优化
- ✅ 成熟方案，SGLang 官方支持

**缺点**：
- ❌ 需要安装 SGLang（~2GB）
- ❌ 不是 fast-vllm

**适合**：
- 你想**立即**看到加速效果
- 验证 speculative decoding 在你的场景是否有用
- 之后再决定要不要移植到 fast-vllm

**启动命令**：
```bash
# 1. 安装 SGLang
pip install "sglang[all]"

# 2. 启动服务
python -m sglang.launch_server \
  --model-path ~/huggingface/Qwen3-0.6B \
  --speculative-algo EAGLE3 \
  --speculative-draft-model-path ~/huggingface/Qwen3-0.6B-eagle3 \
  --speculative-num-steps 3 \
  --port 30000

# 3. 测试（OpenAI API）
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0,
    "max_tokens": 100
  }'
```

---

### 方案 B：fast-vllm + 通用小模型（自己实现）

**Draft 模型选择**：
- **Qwen2.5-0.5B**（推荐）：和 Qwen3 架构相似
- MiniCPM-1B：中文更好，但显存多一倍

**优点**：
- ✅ 完全掌控代码
- ✅ 学习 speculative decoding 原理
- ✅ 可以继续优化（adaptive K, tree-based, etc.）

**缺点**：
- ❌ 需要写 ~300 行代码
- ❌ 加速比较低（1.3-1.5x，不如 Eagle3 的 1.7-1.8x）
- ❌ 需要调试 KV cache 管理

**适合**：
- 你想深入理解 speculative decoding
- 需要自定义优化
- 愿意花 2-3 天时间实现

**实现清单**：
```
□ fastvllm/config.py          (+10 行) - 添加配置
□ fastvllm/engine/model_runner.py (+150 行) - draft + verify 逻辑
□ fastvllm/engine/scheduler.py    (+20 行) - 处理多 token 输出
□ 测试和调试                    (2-3 天)
```

---

## 🎯 我的建议

### 立即行动（今晚）：
```bash
# 方案 A：先用 SGLang 验证
cd /home/uos/code
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[all]"

# 启动服务
python -m sglang.launch_server \
  --model-path ~/huggingface/Qwen3-0.6B \
  --speculative-algo EAGLE3 \
  --speculative-draft-model-path ~/huggingface/Qwen3-0.6B-eagle3 \
  --speculative-num-steps 3 \
  --port 30000
```

**测试 10 分钟**，看加速效果：
- 如果 < 1.3x：speculative 在你场景收益不大
- 如果 > 1.5x：值得投入时间实现

### 之后决策：

**如果加速明显**：
1. 短期（1周内）：继续用 SGLang
2. 中期（有空时）：移植到 fast-vllm
   - 先实现基础版（固定 K=4）
   - 再加自适应、tree-based 等优化

**如果加速不明显**：
- 放弃 speculative decoding
- 尝试其他优化：FP8 quantization, continuous batching, etc.

---

## 📊 性能预期（RTX 3080 10GB）

| 配置 | 吞吐量 | 延迟 | 显存 |
|------|--------|------|------|
| fast-vllm baseline | ~400 tok/s | 2.5ms/tok | 2.5GB |
| + Eagle3 (SGLang) | ~700 tok/s | 1.4ms/tok | 3.3GB |
| + Qwen2.5-0.5B (自实现) | ~550 tok/s | 1.8ms/tok | 3.1GB |

**结论**：Eagle3 效果最好，但只能用 SGLang。

---

## 📝 项目文件清单

我已经给你准备了：

1. ✅ `fastvllm/engine/speculative_runner.py` - 通用实现框架
2. ✅ `test_speculative_simple.py` - 测试脚本（但 Eagle3 不兼容）
3. ✅ `SPECULATIVE_DECODING_GUIDE.md` - 完整集成指南
4. ✅ `USE_SGLANG_EAGLE3.md` - SGLang 快速上手
5. ✅ `GENERIC_SPECULATIVE_PLAN.md` - 通用方案设计

---

## 🚀 下一步行动

### 选项 1（推荐）：立即验证

```bash
# 安装 SGLang
cd /home/uos/code
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/

# 启动
python -m sglang.launch_server \
  --model-path ~/huggingface/Qwen3-0.6B \
  --speculative-algo EAGLE3 \
  --speculative-draft-model-path ~/huggingface/Qwen3-0.6B-eagle3 \
  --speculative-num-steps 3
```

### 选项 2：直接实现通用方案

```bash
# 下载 Qwen2.5-0.5B
huggingface-cli download Qwen/Qwen2.5-0.5B \
  --local-dir ~/huggingface/Qwen2.5-0.5B/

# 我帮你完成 fast-vllm 集成（需要 2-3 小时写完整代码）
```

### 选项 3：先观望

- 继续用现有的 fast-vllm
- 等我把代码写完再测试

---

## 💡 最终建议

**今天晚上花 30 分钟试一下 SGLang + Eagle3**：

如果效果好 → 短期用 SGLang，长期移植
如果效果一般 → 省下时间做其他优化

你想现在试哪个方案？我可以：
- A) 帮你跑 SGLang（给你完整命令）
- B) 写完整的 fast-vllm 实现代码
- C) 先暂停，你自己试试再说
