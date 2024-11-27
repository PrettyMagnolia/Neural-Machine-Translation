import os
import logging
import pickle

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from model.loss import SimpleLossCompute, LabelSmoothing
from model.transformer import make_model
from model.optimizer import get_std_opt
from dataset import NMTDataset

from torch.utils.tensorboard import SummaryWriter


def run_epoch(dataloader, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(dataloader):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    return total_loss / total_tokens


def train(train_data, dev_data, model, model_par, criterion, optimizer):
    """训练并保存模型"""
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_dev_loss = float('inf')
    early_stop = config.EARLY_STOP
    for epoch in range(config.EPOCH_NUM):
        # 模型训练
        model.train()
        train_loss = run_epoch(train_data, model_par, SimpleLossCompute(
            model.generator, criterion, optimizer))
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        # Tensorboard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        # 模型验证
        model.eval()
        dev_loss = run_epoch(dev_data, model_par,
                             SimpleLossCompute(model.generator, criterion, None))
        logging.info('Epoch: {}, Dev loss: {}'.format(
            epoch, dev_loss))
        writer.add_scalar('Loss/valid', dev_loss, epoch)

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_dev_loss = dev_loss
            early_stop = config.EARLY_STOP
            logging.info("-------- Save Best Model! --------")
        else:
            early_stop -= 1
            logging.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            logging.info("-------- Early Stop! --------")
            break


def main():
    # 配置日志记录
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s: %(message)s', handlers=[
        logging.FileHandler(config.LOG_PATH, mode='w'),
        logging.StreamHandler()
    ])

    # 加载数据
    with open(config.DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    train_dataset = NMTDataset(data['train'])
    dev_dataset = NMTDataset(data['valid'])

    logging.info("-------- Dataset Build! --------")

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
                                collate_fn=dev_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")

    # 初始化模型
    model = make_model(config.SRC_VOCAB_SIZE, config.TGT_VOCAB_SIZE, config.N_LAYERS,
                       config.D_MODEL, config.D_FF, config.N_HEADS, config.DROPOUT)
    model_par = torch.nn.DataParallel(model)

    # 选择损失函数
    if config.USE_SMOOTHING:
        criterion = LabelSmoothing(
            size=config.TGT_VOCAB_SIZE, padding_idx=config.PAD_IDX, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    # 选择优化器
    if config.USE_NOAMOPT:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

    # 开始训练
    train(train_dataloader, dev_dataloader,
          model, model_par, criterion, optimizer)


if __name__ == '__main__':
    main()
