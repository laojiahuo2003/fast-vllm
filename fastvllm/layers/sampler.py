import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 贪心采样：temperature=0 时直接 argmax
        greedy_mask = temperatures == 0

        if greedy_mask.all():
            # 全部贪心
            return logits.argmax(dim=-1)
        elif greedy_mask.any():
            # 部分贪心，部分采样
            sample_tokens = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

            # 贪心部分
            sample_tokens[greedy_mask] = logits[greedy_mask].argmax(dim=-1)

            # 采样部分
            sampling_mask = ~greedy_mask
            logits_sampling = logits[sampling_mask].float().div_(temperatures[sampling_mask].unsqueeze(dim=1))
            probs = torch.softmax(logits_sampling, dim=-1)
            sample_tokens[sampling_mask] = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

            return sample_tokens
        else:
            # 全部采样
            logits = logits.float().div_(temperatures.unsqueeze(dim=1))
            probs = torch.softmax(logits, dim=-1)
            sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
            return sample_tokens
