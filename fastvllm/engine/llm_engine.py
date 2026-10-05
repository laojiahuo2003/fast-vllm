import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from fastvllm.config import Config
from fastvllm.sampling_params import SamplingParams
from fastvllm.engine.sequence import Sequence
from fastvllm.engine.scheduler import Scheduler
from fastvllm.engine.model_runner import ModelRunner
from fastvllm.engine.eagle3_model_runner import Eagle3ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.config = config      # 留着，add_request 等处要读
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        # 选择 ModelRunner 类型
        if config.enable_eagle3:
            print(f"✅ Eagle3 Speculative Decoding 已启用")
            print(f"  - Eagle3 模型: {config.eagle3_model_path}")
            print(f"  - 候选 token 数: {config.num_speculative_tokens}")
            runner_class = Eagle3ModelRunner
        else:
            runner_class = ModelRunner

        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=runner_class, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = runner_class(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # Eagle3 一次要写 k+1 个新 token 的 KV，得自己管 block 追加
        if config.enable_eagle3:
            self.model_runner.block_manager = self.scheduler.block_manager
        atexit.register(self.exit)

    def exit(self):
        # model_runner 可能已经被外部置 None（提前释放显存的情况）
        if getattr(self, 'model_runner', None) is not None:
            self.model_runner.call("exit")
            del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 投机路径目前只支持贪心：_spec_round 直接拿 logits.argmax() 当结果，
        # 完全绕过 sampler。 temperature != 0 时不会报错，而是**静默给出贪心输出**，
        # 那比直接报错难查得多。要支持随机采样得在 verify 段实现拒绝采样
        # （按概率比接受候选），不是加一行判断的事。
        if self.config.enable_eagle3 and sampling_params.temperature != 0:
            raise NotImplementedError(
                "Eagle3 投机解码目前只支持贪心采样（temperature=0）。"
                f"收到 temperature={sampling_params.temperature}。"
                "投机路径不经过 Sampler，直接取 argmax；支持随机采样需要"
                "在 verify 段实现按概率比的拒绝采样。")
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs = self.scheduler.schedule()  # 可混合 decode+prefill
        prefill_tokens = sum(s.num_scheduled_tokens for s in seqs if s.is_prefill)
        decode_tokens = sum(1 for s in seqs if not s.is_prefill)
        token_ids = self.model_runner.call("run", seqs)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        for seq_id, _ in outputs:
            if hasattr(self.model_runner, "cleanup_seq"):
                self.model_runner.cleanup_seq(seq_id)
            elif hasattr(self.model_runner, "call"):
                self.model_runner.call("cleanup_seq", seq_id)
        return outputs, prefill_tokens, decode_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, prefill_tokens, decode_tokens = self.step()
            if prefill_tokens:
                prefill_throughput = prefill_tokens / (perf_counter() - t)
            if decode_tokens:
                decode_throughput = decode_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        # 打印 Eagle3 统计信息
        if hasattr(self.model_runner, 'get_spec_stats'):
            self._print_eagle3_stats()

        return outputs

    def _print_eagle3_stats(self):
        """打印 Eagle3 Speculative Decoding 统计信息"""
        stats = self.model_runner.get_spec_stats()
        if stats['total_steps'] == 0:
            return

        print("\n" + "="*80)
        print("Eagle3 Speculative Decoding 统计")
        print("="*80)
        print(f"总共生成候选 tokens: {stats['total_drafted']}")
        print(f"被接受的 tokens: {stats['total_accepted']}")
        print(f"接受率: {stats['acceptance_rate']:.2%}")
        print(f"平均每步接受: {stats['avg_accepted_per_step']:.2f} tokens")

        if stats['acceptance_rate'] > 0:
            speedup = stats['avg_accepted_per_step']
            print(f"理论加速比: {speedup:.2f}x")

        print("\n" + "="*80)
        print("性能解读")
        print("="*80)
        if stats['acceptance_rate'] > 0.8:
            print("✅ Eagle3 工作良好！")
            print(f"• 接受率 {stats['acceptance_rate']:.0%} 表示 Eagle3 生成的候选中，")
            print(f"  有 {stats['acceptance_rate']:.0%} 被 target model 接受")
            print(f"• 平均每步接受 {stats['avg_accepted_per_step']:.2f} tokens，")
            print(f"  相比传统方式（每步 1 token）快约 {speedup:.2f}x")
        elif stats['acceptance_rate'] > 0.5:
            print("⚠️  Eagle3 工作一般")
            print(f"• 接受率 {stats['acceptance_rate']:.0%} 有提升空间")
            print("• 可能需要：")
            print("  - 调整 num_speculative_tokens")
            print("  - 检查 Eagle3 模型是否匹配")
        else:
            print("❌ Eagle3 效果不佳")
            print(f"• 接受率 {stats['acceptance_rate']:.0%} 太低")
            print("• 建议：")
            print("  - 确认 Eagle3 模型是否正确加载")
            print("  - 减小 num_speculative_tokens")
        print()
