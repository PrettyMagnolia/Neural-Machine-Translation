import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from config import DEVICE, PAD_IDX
from model.utils import subsequent_mask


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, trg=None, pad=PAD_IDX):
        src = src.to(DEVICE)
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            trg = trg.to(DEVICE)
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask


class NMTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):

        src_tokens = [x['src'] for x in batch]
        tgt_tokens = [x['tgt'] for x in batch]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=PAD_IDX)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=PAD_IDX)

        return Batch(batch_input, batch_target)
