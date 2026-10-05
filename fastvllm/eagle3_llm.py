"""
带 Eagle3 支持的 LLM Engine

使用方法：
    from fastvllm.eagle3_llm import LLMWithEagle3

    llm = LLMWithEagle3(
        model="~/huggingface/Qwen3-0.6B/",
        eagle3_model_path="~/huggingface/Qwen3-0.6B-eagle3/",
        num_speculative_tokens=3,
    )

    outputs = llm.generate(prompts, sampling_params)

    # 查看加速统计
    print(llm.get_spec_stats())
"""
import atexit
from dataclasses import fields, dataclass
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from fastvllm.config import Config
from fastvllm.sampling_params import SamplingParams
from fastvllm.engine.sequence import Sequence
from fastvllm.engine.scheduler import Scheduler
from fastvllm.engine.eagle3_model_runner import Eagle3ModelRunner


@dataclass
class Eagle3Config:
    """Eagle3 专用配置"""
    # Eagle3 模型路径
    eagle3_model_path: str = None

    # Speculative decoding 参数
    # 默认 3，与 config.py 的 num_speculative_tokens 保持一致：实测这个 draft
    # head 的链上条件一致率约 45%/71%/40%/0%，第 4 个候选已经基本死掉，
    # k=3 与 k=4/k=5 每轮产出的 token 数一样（1.90），但 k=3 少跑 1 行 draft
    # + 1 行 verify，墙钟更快。checkpoint 模型卡上 "steps 3 / topk 1 / draft 4
    # 是上限，更深反而掉" 也是同一个结论。
    num_speculative_tokens: int = 3

    # 提取特征的层（Qwen3-0.6B 有 28 层）
    eagle3_extract_layers: list = None

    def __post_init__(self):
        if self.eagle3_extract_layers is None:
            # 默认提取 1/13/24 层（早期、中期、后期）
            self.eagle3_extract_layers = [1, 13, 24]


class ConfigWithEagle3:
    """包装 Config，添加 Eagle3 属性（因为 Config 用了 slots=True 不能动态添加）"""
    def __init__(self, base_config: Config, eagle3_config: Eagle3Config):
        self._base = base_config
        self._eagle3 = eagle3_config

    def __getattr__(self, name):
        # 先查 eagle3 配置
        if hasattr(self._eagle3, name):
            return getattr(self._eagle3, name)
        # 再查 base 配置
        return getattr(self._base, name)


class LLMWithEagle3:
    """
    支持 Eagle3 Speculative Decoding 的 LLM Engine

    完全兼容原始 fast-vllm API，只是内部使用 Eagle3 加速
    """

    def __init__(self, model, **kwargs):
        # 分离 Eagle3 配置和原始配置
        eagle3_kwargs = {}
        base_kwargs = {}

        eagle3_fields = {f.name for f in fields(Eagle3Config)}
        config_fields = {field.name for field in fields(Config)}

        for k, v in kwargs.items():
            if k in eagle3_fields:
                eagle3_kwargs[k] = v
            elif k in config_fields:
                base_kwargs[k] = v

        # 创建配置
        eagle3_config = Eagle3Config(**eagle3_kwargs)
        base_config = Config(model, **base_kwargs)

        # 包装成统一的 config（绕过 slots=True 限制）
        config = ConfigWithEagle3(base_config, eagle3_config)

        Sequence.block_size = config.kvcache_block_size

        # 启动多进程（如果需要）
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=Eagle3ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 创建 Eagle3 ModelRunner
        self.model_runner = Eagle3ModelRunner(config, 0, self.events)

        # 其余和原始 LLMEngine 一样
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # 投机解码需要自己管 block 追加，必须注入 block_manager
        self.model_runner.block_manager = self.scheduler.block_manager
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs = self.scheduler.schedule()
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
    ) -> list[dict]:
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
        return outputs

    def get_spec_stats(self) -> dict:
        """
        获取 Eagle3 speculative decoding 统计信息

        Returns:
            {
                'total_drafted': 总共生成的候选 token 数,
                'total_accepted': 被接受的 token 数,
                'acceptance_rate': 接受率,
                'avg_speedup': 平均加速比,
            }
        """
        stats = self.model_runner.get_spec_stats()

        # 计算理论加速比
        if stats['total_steps'] > 0:
            # 平均每步接受的 token 数
            avg_accepted = stats['avg_accepted_per_step']
            # 理论加速比 = 接受的 tokens / 实际 forward 次数
            # 每步 speculative: 1 次 draft + 1 次 verify = 2 次
            # 传统方式: N 个 token 需要 N 次 forward
            # 加速比 = avg_accepted / 2（简化估算）
            stats['avg_speedup'] = avg_accepted / 2.0 if avg_accepted > 0 else 1.0
        else:
            stats['avg_speedup'] = 1.0

        return stats


# 为了兼容原有代码，提供别名
LLM = LLMWithEagle3
