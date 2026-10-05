from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        # 支持贪心采样：temperature=0 会在采样时自动转为 argmax
        assert self.temperature >= 0, "temperature must be non-negative"
