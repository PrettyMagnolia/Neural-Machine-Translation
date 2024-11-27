import torch


class NoamOpt:
    """
    实现学习率调度的优化器封装类
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        # 实现学习率调度公式
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    # 返回标准配置的 NoamOpt 优化器
    return NoamOpt(
        model.src_embed[0].d_model,  # 模型嵌入层的维度
        2,  # 学习率缩放因子
        4000,  # 学习率预热步数
        torch.optim.Adam(
            model.parameters(),
            lr=0,  # 初始学习率
            betas=(0.9, 0.98),  # Adam 优化器的动量参数
            eps=1e-9  # Adam 优化器的数值稳定性参数
        )
    )
