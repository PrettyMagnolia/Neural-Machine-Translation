import pickle
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu

import config
from dataset import NMTDataset
from model.transformer import make_model
from decode.greedy_search import greedy_search
from decode.beam_search import beam_search


def decode_text(text, vocab, split=''):
    # 首先根据vocab建立反向映射
    id_to_token = {id: token for token, id in vocab.items()}

    # 然后根据text中的id找到对应的token
    decoded_text = [id_to_token.get(id) for id in text if id > 3]

    return split.join(decoded_text)


def test(vocab, dataloader, model):
    tgt_sents = []
    pred_sents = []
    src_sents = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            src_batch = batch.src
            tgt_batch = [[token.item() for token in seq] for seq in batch.trg]
            src_mask = (src_batch != 0).unsqueeze(-2)

            if config.BEAM_SIZE > 1:
                decoded_batch, _ = beam_search(model, src_batch, src_mask, config.MAX_OUTPUT_LEN,
                                               config.PAD_IDX, config.BOS_IDX, config.EOS_IDX,
                                               config.BEAM_SIZE, config.DEVICE)
            else:
                decoded_batch = greedy_search(model, src_batch, src_mask, max_len=config.MAX_OUTPUT_LEN,
                                              start_symbol=config.BOS_IDX, end_symbol=config.EOS_IDX)

            src_batch = [[token.item() for token in seq] for seq in src_batch]
            src_sents.extend([decode_text(seq, vocab['src'], ' ')
                             for seq in src_batch])
            tgt_sents.extend([decode_text(seq, vocab['tgt'])
                             for seq in tgt_batch])
            pred_sents.extend([decode_text(seq, vocab['tgt'])
                              for seq in decoded_batch])

    assert len(tgt_sents) == len(pred_sents)

    bleu_scores = []
    with open(config.OUTPUT_PATH, 'w') as out_file:
        for src, pred, tgt in zip(src_sents, pred_sents, tgt_sents):
            bleu = sentence_bleu([list(tgt)], list(pred))
            bleu_scores.append(bleu)
            out_file.write(f"{src}|||{tgt}|||{pred}|||{bleu}\n")

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    return float(avg_bleu)


def main():
    # 加载词汇表
    with open(config.VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    # 加载测试数据
    with open(config.DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # 创建测试数据集和数据加载器
    test_dataset = NMTDataset(data['test'])
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
                                 collate_fn=test_dataset.collate_fn)

    # 初始化模型
    model = make_model(config.SRC_VOCAB_SIZE, config.TGT_VOCAB_SIZE, config.N_LAYERS,
                       config.D_MODEL, config.D_FF, config.N_HEADS, config.DROPOUT)

    # 加载预训练模型权重
    model.load_state_dict(torch.load(config.LOAD_MODEL_PATH))
    model = model.to(config.DEVICE).eval()

    # 测试模型并计算BLEU分数
    bleu = test(vocab, test_dataloader, model)
    print("Bleu Score: ", bleu)


if __name__ == '__main__':
    main()
