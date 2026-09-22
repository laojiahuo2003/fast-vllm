import torch
from torch import nn

from fastvllm.kernels.fused_ops import silu_and_mul


class SiluAndMul(nn.Module):
    # MergedColumnParallelLinear [gate0 gate1 ... gate7 | up0 up1 ... up7]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # kernel 用 row * stride + cols 寻址，只认行主序连续布局；非连续 view 喂进来
        # 不会报错，只会静默算错。.contiguous() 对已连续张量是 no-op（不发 kernel 不分配）。
        #
        # 实测（ncu, N=8192 I=3072）：inductor 融成的单 kernel 读写已精确等于理论值
        #   （读 96MiB / 写 47MiB），手写 kernel 143.39MiB vs 它 143.35MiB，耗时 221.73 vs
        #   220.42us —— 访存完全打平，还慢 0.6%。所以这一项没有带宽收益，接进来是为了
        #   统一走 kernels/ 这条路径、少一次 dynamo guard，不是为提速。
        return silu_and_mul(x.contiguous())
