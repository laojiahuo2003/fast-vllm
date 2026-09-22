import torch
from torch import nn

from fastvllm.kernels.fused_ops import add_rms_norm


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps# 小常数，防止除0
        self.weight = nn.Parameter(torch.ones(hidden_size))# 缩放因子

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()# 半精度做高次方求和（如平方）时极易发生数值溢出（Overflow）或下溢，用 FP32 保证数值稳定性
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    # 把残差相加和RMSNorm融合在一起
    # 不再套 @torch.compile：body 只剩一次 Triton launch，再套 dynamo 只是每次白付
    # guard 开销（~5-10us CPU）。实测（ncu, N=8192 H=1024）：
    #   inductor 版 79.05MiB / 123.52us —— 它生成了 3 个 store 但只返回 2 个 tensor，
    #                       那个 fp16(归一化后) 的中间量写出去后再也没人读，白写 16MiB
    #   手写 kernel  62.89MiB /  97.44us —— 流量回到理论值，快 1.26x
    # 两边都跑在 ~630GB/s，耗时比 == 流量比，说明纯 DRAM bound，省多少字节就快多少。
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # kernel 用 row * H + cols 寻址，只认行主序连续布局；喂进非连续 view 不会报错，
        # 只会静默算错。这里防御性 .contiguous()：对已连续张量是 no-op（直接返回自身，
        # 不发 kernel 不分配），只有真遇到非连续时才付一次拷贝。
        # 用 .contiguous() 而不是 assert：这是 serving 路径，assert 命中就是崩进程。
        return add_rms_norm(
            x.contiguous(),
            residual.contiguous(),
            self.weight,
            self.eps,
        )
    # 根据是否有残差，选择不同的前向传播路径
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
