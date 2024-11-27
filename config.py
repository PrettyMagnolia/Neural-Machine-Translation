import torch
import datetime
import os

# Model hyperparameters
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
D_K = 64
D_V = 64
D_FF = 2048
DROPOUT = 0.1

# Vocabulary and token indices
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
SRC_VOCAB_SIZE = 40000
TGT_VOCAB_SIZE = 50000

# Training parameters
BATCH_SIZE = 256
EPOCH_NUM = 50
EARLY_STOP = 5
LR = 3e-4

# Sequence lengths
MAX_INPUT_LEN = 64
MAX_OUTPUT_LEN = 64

# Decoding and optimization
BEAM_SIZE = 3  # beam size for bleu
USE_SMOOTHING = True  # Label Smoothing
USE_NOAMOPT = True  # NoamOpt

# Paths and directories
DATA_DIR = './data'
TRAIN_DATA_PATH = './data/train'
DEV_DATA_PATH = './data/valid'
TEST_DATA_PATH = './data/test'
DATA_PATH = './data/data.pkl'
VOCAB_PATH = './data/vocab.pkl'

# Logging and model saving
DATE_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_DIR = f'./runs/{DATE_TIME}'
LOG_PATH = os.path.join(LOG_DIR, 'train.log')
MODEL_PATH = os.path.join(LOG_DIR, 'model.pth')
OUTPUT_PATH = os.path.join('./runs', f'output_{"beam" if BEAM_SIZE > 1 else "greedy"}.txt')
LOAD_MODEL_PATH = r'/home/yifei/code/MY_NMT/runs/base/model.pth'

# Device configuration
GPU_ID = '0'
DEVICE = torch.device(f"cuda:{GPU_ID}") if GPU_ID != '' else torch.device('cpu')
