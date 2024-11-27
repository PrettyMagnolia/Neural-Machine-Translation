import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import clones, attention


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 定义词嵌入层
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 将输入的词索引映射为词向量，并放大至 d_model 的尺度
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        # 生成一个位置下标 tensor 矩阵
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # 计算各位置的位置编码，存放在 pe 中
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        # 添加维度，变为 [1 * max_len * d_model] (便于与句子所有词的 embedding 相加)
        pe = pe.unsqueeze(0)
        # 注册为常量，避免训练中更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将 batch 中句子所有词的 embedding 与已构建好的 pe 相加
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # d_k 为一个 head 的 attention 表示维度
        self.d_k = d_model // h  # 每个头的维度
        self.h = h  # 注意力头的数量

        # 定义 4 个全连接函数：Q, K, V 的变换矩阵 (3) + 结果变换矩阵 (1)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query 的第一个维度为batch size
        nbatches = query.size(0)

        # 将 W，K，V 分别乘以WQ，WK，WV矩阵 -> [batch * seq_len * embedding_dim]
        # 将结果拆成 h 块(分 h 个头) -> [batch * seq_len * h * embedding_dim / h]
        # 交换第2,3维(便于计算) -> [batch * h * seq_len * embedding_dim / h]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 合并多头注意力的结果(换回2,3维)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # 对结果进行变换(最后一个全连接函数)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 经过第一层线性变换，映射到隐藏层空间
        hidden = F.relu(self.w_1(x))
        # 经过第二层线性变换，映射回原空间
        return self.w_2(self.dropout(hidden))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 缩放参数 α(初始化为1)，平移参数 β(初始化为0)
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps  # 平滑项

    def forward(self, x):
        # 按最后一个维度计算均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # sublayer 表示的是某种子层操作（比如 Multi-Head Attention 或 Feed Forward）
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))
