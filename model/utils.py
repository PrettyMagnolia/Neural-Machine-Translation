import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    '''
    克隆多个模块实例，保证各模块互相独立。
    参数：
        module：需要克隆的神经网络模块
        N：克隆模块的数量
    返回：
        nn.ModuleList：包含 N 个独立克隆模块的列表
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    '''
    计算注意力权重并返回加权和。
    参数：
        query：查询向量 Q
        key：键向量 K
        value：值向量 V
        mask：掩码，用于屏蔽无效位置
        dropout：dropout 层，用于正则化
    返回：
        加权和结果，以及注意力权重矩阵
    '''
    d_k = query.size(-1)  # 查询向量的维度

    # QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 将掩码位置填充为一个极小值，防止影响 softmax
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)  # 按行计算注意力权重
    if dropout is not None:
        p_attn = dropout(p_attn)  # 正则化防止过拟合

    # 返回加权和与注意力权重
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    '''
    生成一个子序列掩码矩阵，屏蔽未来时间步的信息。
    参数：
        size：序列长度
    返回：
        下三角掩码矩阵，未来时间步位置为 False
    '''
    attn_shape = (1, size, size)
    # 右上角为 1，主对角线及以下为 0
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 转为布尔类型，下三角为 True，其他为 False
    return torch.from_numpy(subsequent_mask) == 0
