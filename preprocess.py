import re
import jieba
import nltk
import pickle
import unicodedata
import numpy as np
from tqdm import tqdm
from collections import Counter
import config


def tokenize_en(sentence):

    def normalize_string(s):
        s = ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
        s = re.sub(r"[^a-zA-Z0-9,.!?]", r" ", s.lower().strip())
        return s

    sentence = normalize_string(sentence)
    tokens = nltk.word_tokenize(sentence)
    return [token for token in tokens if token.strip()]


def tokenize_zh(sentence):

    def normalize_string(s):
        s = re.sub(r"（.*?）", "", s)
        s = re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z，。！？]", r" ", s.strip())
        return s

    sentence = normalize_string(sentence)
    tokens = jieba.lcut(sentence)
    return [token for token in tokens if token.strip()]


def load_data(file_path, lang):
    """Load and tokenize sentences from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    word_freq = Counter()
    samples = []

    for line in tqdm(lines, desc=f"Processing {file_path} data"):
        sentence = line.strip()
        tokens = tokenize_en(
            sentence) if lang == 'en' else tokenize_zh(sentence)
        word_freq.update(tokens)
        samples.append(tokens)

    return samples, word_freq


def build_vocab(file_path, lang, vocab_len=50000):
    _, word_freq = load_data(file_path, lang)
    word_freq = word_freq.most_common(vocab_len - 4)
    vocab = {word: idx + 4 for idx, (word, _) in enumerate(word_freq)}
    vocab['<pad>'] = 0
    vocab['<bos>'] = 1
    vocab['<eos>'] = 2
    vocab['<unk>'] = 3

    print('Vocab size:', len(vocab))
    return vocab


def build_data(file_path, lang, vocab, max_len):
    samples, _ = load_data(file_path, lang)
    data = []
    for sample in samples:
        if len(sample) > max_len - 2:
            sample = sample[:max_len]
        sample = ['<bos>'] + sample + ['<eos>']
        tokens = [vocab.get(token, vocab['<unk>']) for token in sample]
        data.append(tokens)
    return data


def build_dataset(src_file, tgt_file, src_vocab, tgt_vocab):

    src_data = build_data(src_file, 'en', src_vocab, config.MAX_INPUT_LEN)
    tgt_data = build_data(tgt_file, 'zh', tgt_vocab, config.MAX_OUTPUT_LEN)

    assert len(src_data) == len(tgt_data)

    samples = []
    for src_sample, tgt_sample in zip(src_data, tgt_data):
        sample = {
            'src': src_sample,
            'tgt': tgt_sample
        }
        samples.append(sample)

    return samples


if __name__ == '__main__':
    # 构建词表
    en_vocab = build_vocab('./data/train.en', 'en', config.SRC_VOCAB_SIZE)
    zh_vocab = build_vocab('./data/train.zh', 'zh', config.TGT_VOCAB_SIZE)

    vocab = {
        'src': en_vocab,
        'tgt': zh_vocab
    }
    with open('./data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # Build dataset
    train_data = build_dataset(
        './data/train.en', './data/train.zh', en_vocab, zh_vocab)
    valid_data = build_dataset(
        './data/valid.en', './data/valid.zh', en_vocab, zh_vocab)
    test_data = build_dataset(
        './data/test.en', './data/test.zh', en_vocab, zh_vocab)

    data = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }
    with open('./data/data.pkl', 'wb') as f:
        pickle.dump(data, f)
