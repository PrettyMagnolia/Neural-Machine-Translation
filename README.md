# Neural Machine Translation(NMT)

## Data Preparation

Perpare you data in the `data` directory. Each line in the source and target files should be corresponding to each other.

Modify the `SRC_VOCAB_SIZE` and `TGT_VOCAB_SIZE` in `config.py` to the number of unique words in the source and target languages, respectively. 

Run `preprocess.py` to generate the data file `data.pkl` and vocabulary file `vocab.pkl`.


## Train

Modify the hyperparameters and path arguments in `config.py`. Run `train.py` to train the model.

If you want to see the training progress, run `tensorboard --logdir=runs/{LOG_DIR}`.

## Test

Modify the `MODEL_PATH` in `config.py` to the path of the trained model. Run `test.py` to test the model on the test set.

If `BEAM_SIZE` is greater than 1, the model will use beam search to generate translations. Otherwise, it will use greedy search.

Use `nltk.translate.bleu_score` to calculate the BLEU score of the model.