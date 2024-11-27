import copy
import torch.nn as nn
from config import DEVICE
from .modules import Embeddings, MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward
from .decoder import Decoder, DecoderLayer
from .encoder import Encoder, EncoderLayer
from .generator import Generator


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder      # 编码器模块
        self.decoder = decoder      # 解码器模块
        self.src_embed = src_embed  # 源序列嵌入模块
        self.tgt_embed = tgt_embed  # 目标序列嵌入模块
        self.generator = generator  # 输出生成器

    def encode(self, src, src_mask):
        # 编码器生成源序列
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # 解码器生成目标序列
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 完整的编码-解码流程
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    # 实例化Transformer模型对象
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)

    # 参数初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # Xavier 均匀分布初始化
    return model.to(DEVICE)
