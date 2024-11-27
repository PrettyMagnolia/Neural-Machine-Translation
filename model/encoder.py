import torch.nn as nn
from .utils import clones
from .modules import LayerNorm, SublayerConnection


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 使用两个 SublayerConnection 将自注意力和前馈网络连接起来
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size  # d_model

    def forward(self, x, mask):
        """
        mask: 用于屏蔽 <pad> 位置
        """
        # 多头注意力层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 前馈层
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N=6):
        super(Encoder, self).__init__()
        # 克隆 N 个编码器层
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # 遍历每个编码器层，将输入依次传递
        for layer in self.layers:
            x = layer(x, mask)
        
        # 对最终的编码器输出进行层归一化   
        return self.norm(x)
