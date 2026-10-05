"""
Eagle3 使用示例

演示如何用 Eagle3 加速 Qwen3-0.6B 推理
"""
import os
from fastvllm.eagle3_llm import LLMWithEagle3
from fastvllm.sampling_params import SamplingParams
from transformers import AutoTokenizer


def main():
    # 模型路径
    target_model = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    eagle3_model = os.path.expanduser("~/huggingface/Qwen3-0.6B-eagle3/")

    print("="*80)
    print("Fast-vLLM with Eagle3 Speculative Decoding")
    print("="*80)

    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model)

    # 创建 LLM（带 Eagle3 加速）
    print("\n初始化模型...")
    llm = LLMWithEagle3(
        model=target_model,
        enforce_eager=True,
        tensor_parallel_size=1,
        # Eagle3 配置
        eagle3_model_path=eagle3_model,
        num_speculative_tokens=3,  # 每次推测 3 个 token（实测这个 head 的最优点，见 k 扫描）
    )

    # 测试 prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
    ]

    # 格式化为 chat 格式
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in test_prompts
    ]

    # 推理参数
    sampling_params = SamplingParams(
        temperature=0.01,  # 接近贪心（fast-vllm 不允许 0）
        max_tokens=100
    )

    # 生成
    print("\n开始推理...\n")
    outputs = llm.generate(prompts, sampling_params)

    # 打印结果
    print("\n" + "="*80)
    print("生成结果")
    print("="*80)
    for i, (prompt, output) in enumerate(zip(test_prompts, outputs)):
        print(f"\n[{i+1}] Prompt: {prompt}")
        print(f"Output: {output['text'][:200]}...")
        print(f"Tokens: {len(output['token_ids'])}")

    # 打印 Eagle3 统计
    print("\n" + "="*80)
    print("Eagle3 Speculative Decoding 统计")
    print("="*80)
    stats = llm.get_spec_stats()
    print(f"总共生成候选 tokens: {stats['total_drafted']}")
    print(f"被接受的 tokens: {stats['total_accepted']}")
    print(f"接受率: {stats['acceptance_rate']:.2%}")
    print(f"平均每步接受: {stats['avg_accepted_per_step']:.2f} tokens")
    print(f"理论加速比: {stats['avg_speedup']:.2f}x")

    # 加速说明
    print("\n" + "="*80)
    print("性能解读")
    print("="*80)
    print(f"• 接受率 {stats['acceptance_rate']:.0%} 表示 Eagle3 生成的候选中，")
    print(f"  有 {stats['acceptance_rate']:.0%} 被 target model 接受")
    print(f"• 平均每步接受 {stats['avg_accepted_per_step']:.2f} tokens，")
    print(f"  相比传统方式（每步 1 token）快约 {stats['avg_speedup']:.2f}x")
    print(f"• 实际加速比还受 draft model overhead 影响")


if __name__ == "__main__":
    main()
