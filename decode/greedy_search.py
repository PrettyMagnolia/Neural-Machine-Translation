import torch
from model.utils import subsequent_mask


def greedy_search(model, src, src_mask, max_len, start_symbol, end_symbol):
    batch_size, src_seq_len = src.size()
    results = [[] for _ in range(batch_size)]  # 存储每个样本的解码结果
    stop_flag = [False for _ in range(batch_size)]  # 标记每个样本是否解码结束
    count = 0
    memory = model.encode(src, src_mask)

    # 初始化解码输入，填充起始符
    tgt = torch.Tensor(batch_size, 1).fill_(start_symbol).type_as(src.data)

    for _ in range(max_len):
        tgt_mask = subsequent_mask(tgt.size(1)).expand(
            batch_size, -1, -1).type_as(src.data)
        out = model.decode(memory, src_mask, tgt, tgt_mask)

        prob = model.generator(out[:, -1, :])
        pred = torch.argmax(prob, dim=-1)
        # 将预测结果拼接到输出序列中，作为下一时刻的输入
        tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)

        # 更新结果并检查解码是否结束
        pred = pred.cpu().numpy()
        for i in range(batch_size):
            if stop_flag[i] is False:
                if pred[i] == end_symbol:
                    count += 1
                    stop_flag[i] = True
                else:
                    results[i].append(pred[i].item())
            if count == batch_size:
                break

    return results
