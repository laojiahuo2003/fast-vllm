import torch
from torch import nn


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
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype# 显存里都是FP16，相较于32位，内存占用一半
        x = x.float().add_(residual.float())# 把残差也转换为FP32，因为后面要做pow，怕数值溢出或下溢
        residual = x.to(orig_dtype)
        # 和RMSNorm一样
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual
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
