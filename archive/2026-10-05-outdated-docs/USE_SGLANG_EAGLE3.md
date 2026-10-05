# 使用 SGLang + Eagle3 加速 Qwen3-0.6B

## 快速开始

### 1. 安装 SGLang

```bash
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/
```

### 2. 启动服务（带 Eagle3 加速）

```bash
python -m sglang.launch_server \
  --model-path ~/huggingface/Qwen3-0.6B \
  --speculative-algo EAGLE3 \
  --speculative-draft-model-path ~/huggingface/Qwen3-0.6B-eagle3 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --port 30000
```

### 3. 测试

```python
import openai

client = openai.Client(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0,
    max_tokens=100,
)
print(response.choices[0].message.content)
```

## 性能预期

根据 Eagle3 README：
- Baseline (无 speculation): **553 tok/s**
- Eagle3 v3: **955 tok/s** (**+73% 加速**)

在你的 RTX 3080 上预期：
- Baseline: ~400-500 tok/s
- Eagle3: ~700-850 tok/s (**1.7-1.8x 加速**)

## 对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **SGLang + Eagle3** | ✅ 即开即用<br>✅ 1.7x+ 加速<br>✅ 官方支持 | ❌ 需要安装 SGLang<br>❌ 不是 fast-vllm |
| fast-vllm + 自己实现 | ✅ 控制权<br>✅ 学习经验 | ❌ 工作量大<br>❌ Eagle3 架构特殊 |

## 建议

**先用 SGLang 验证效果**，如果加速明显，再考虑移植到 fast-vllm。
