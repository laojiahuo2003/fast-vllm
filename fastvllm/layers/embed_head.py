import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from fastvllm.utils.context import get_context

# 词表并行层，tokenid -> embedding vector
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()# 当前卡的rank
        self.tp_size = dist.get_world_size()# 有几张卡
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings# 总词表大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size# 每张卡负责的词表大小
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank# 当前卡负责的词表起始索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition# 当前卡负责的词表结束索引
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))# 当前卡负责的词表权重
        self.weight.weight_loader = self.weight_loader# 加载词表权重的函数

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 步骤 1：看当前输入 token id 属不属于本张卡管辖
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)# 当前卡负责的词表索引
            # 步骤 2：把全局 token id 转为当前卡的局部索引（不在本卡的 token 会被 mask 变成 0，防止越界 crash）
            x = mask * (x - self.vocab_start_idx)# 只保留当前卡负责的词表索引
            # 步骤 3：根据当前卡的词表索引，从当前卡的词表权重中查出对应的向量
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 步骤 4：不在本卡的 token，查出来的向量强制清零（变成 0 向量）
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)# 合并所有卡的向量
        return y

# LM Head 的任务是把 Transformer 最后一层的隐状态（维度 hidden_dim）映射为词表大小的打分分布（logits，维度 num_embeddings）。
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1 # 取出每个序列的最后一个 token 的索引
            x = x[last_indices].contiguous()# 取出每个序列的最后一个 token 的向量，然后显存中开辟一块连续内存，把筛选出来的这些向量按顺序紧挨着拷贝进去。
        logits = F.linear(x, self.weight)# 权重共享，从hidden dim映射到vocab dim
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 把logits发到目标卡上面
            dist.gather(logits, all_logits, 0)
            # 把列表里的按最后一维度合并，最后一维变成词表大小
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
