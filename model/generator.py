import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # 将解码后的结果通过一个全连接层映射到词汇表大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 对映射后的结果进行 log_softmax 操作
        return F.log_softmax(self.proj(x), dim=-1)
