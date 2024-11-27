import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """
    标签平滑处理
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # 使用 KL 散度计算损失，指定 reduction='sum' 以便累加损失。
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        # 创建真实分布的副本，并初始化所有类别的概率为平滑项
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))

        # 为目标类别分配置信度概率
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # 将填充符号的概率设为 0
        true_dist[:, self.padding_idx] = 0

        # 查找目标类别中为填充符号的位置
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            # 对填充符号对应的行，完全清零
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist.detach())


class SimpleLossCompute:
    """
    简单的计算损失和进行参数反向传播更新训练的函数
    """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()
