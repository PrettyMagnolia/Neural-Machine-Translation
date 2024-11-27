import torch.nn as nn
from .utils import clones
from .modules import LayerNorm, SublayerConnection


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn  # 自注意力
        self.src_attn = src_attn   # 编码-解码注意力
        self.feed_forward = feed_forward

        # 使用三个 SublayerConnection 连接各子模块
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        src_mask: 编码器输入的 mask，用于屏蔽 <pad> 位置
        tgt_mask: 解码器输入的 mask，用于屏蔽未来位置信息，以及 <pad> 位置
        """
        # 编码器输出作为上下文表示 memory
        m = memory

        # 自注意力层：Q、K、V 都是解码器当前的隐层表示 x
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 编码-解码交叉注意力层：Q 为解码器隐层表示，K 和 V 为编码器输出 memory
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 前馈层
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N=6):
        super(Decoder, self).__init__()
        # 克隆 N 个解码器层
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 依次通过每一层解码器
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # 对最终结果进行层归一化
        return self.norm(x)
