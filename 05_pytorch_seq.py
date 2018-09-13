import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchtext import data, datasets
import torchtext

import tqdm
import random

from TwitterPipeline import TwitterPipeline

# some constants
SEED = 762
#IN_FILE = 'germeval2018.try.txt'
IN_FILE = 'germeval2018.training.txt'
IN_FILE_TEST = 'germeval2018.test.txt'
BATCH_SIZE = 16

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# define Fields
f_text = data.Field(sequential=True, use_vocab=True)
f_pos_tag = data.Field(sequential=True, use_vocab=False,
                       pad_token=1, unk_token=0)
f_lemma = data.Field(sequential=True, use_vocab=True)
f_label = data.LabelField(tensor_type=torch.FloatTensor)
fields = [('text', f_text), ('pos', f_pos_tag),
          ('lemma', f_lemma), ('label', f_label)]

pipe = TwitterPipeline()

full_examples = pipe.process_data(
    IN_FILE, fields)[0]
full_ds = data.Dataset(full_examples, fields)

# do the splitting with torchtext
trn_ds, val_ds = full_ds.split(
    split_ratio=[0.8, 0.2], stratified=True, random_state=random.seed(SEED))
test_examples = pipe.process_data(
    IN_FILE_TEST, fields)[0]
tst_ds = data.Dataset(test_examples, fields)
print(f'train len {len(trn_ds.examples)}')
print(f'val len {len(val_ds.examples)}')
print(f'test len {len(tst_ds.examples)}')

# vec = torchtext.vocab.Vectors('embed_tweets_de_100D_fasttext',
#                              cache='/Users/michel/Downloads/')

# build vocab
# validation + test data should by no means influence the model, so build the vocab just on trn
#f_text.build_vocab(trn_ds, vectors=vec)
f_text.build_vocab(trn_ds, max_size=20000)
f_lemma.build_vocab(trn_ds)
f_label.build_vocab(trn_ds)

print(f'text vocab size {len(f_text.vocab)}')
print(f'lemma vocab size {len(f_lemma.vocab)}')
print(f'label vocab size {len(f_label.vocab)}')

trn_iter, val_iter, tst_iter = data.BucketIterator.splits((trn_ds, val_ds, tst_ds),
                                                          batch_size=BATCH_SIZE,
                                                          device=-1,
                                                          sort_key=lambda t: len(
                                                              t.text),
                                                          sort_within_batch=False,
                                                          repeat=False)


class SimpleRNN(nn.Module):
    def __init__(self, vocab_dim, emb_dim=100, hidden_dim=200):
        super().__init__()

        self.embedding = nn.Embedding(vocab_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)  # 1 is output dim

    def forward(self, x):
        # x type is Tensor[sentence len, batch size]. Internally pytorch does not use 1-hot

        embedded = self.embedding(x)
        # embedded type is Tensor[sentence len, batch size, emb dim]

        output, hidden_state = self.rnn(embedded)
        # output type is Tensor[sentence len, batch size, hidden dim]
        # hidden_state type is Tensor[1, batch size, hidden dim]

        return self.fc(hidden_state.squeeze(0))


def binary_accuracy(preds, y):
    """
    return accuracy per batch as ratio of correct/all
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds))
    # convert into float for division
    pred_is_correct = (rounded_preds == y).float()
    acc = pred_is_correct.sum()/len(pred_is_correct)
    return acc


def train(model, iterator, optimizer, criterion, metric):
    epoch_loss = 0
    epoch_meter = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        y_hat = model(batch.text).squeeze(1)
        loss = criterion(y_hat, batch.label)
        meter = metric(y_hat, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_meter += meter.item()

    return epoch_loss / len(iterator), epoch_meter / len(iterator)


def evaluate(model, iterator, criterion, metric):
    epoch_loss = 0
    epoch_meter = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:
            y_hat = model(batch.text).squeeze(1)
            loss = criterion(y_hat, batch.label)
            meter = metric(y_hat, batch.label)

            epoch_loss += loss.item()
            epoch_meter += meter.item()

    return epoch_loss / len(iterator), epoch_meter / len(iterator)


EMB_SIZE = 100
HID_SIZE = 200
NUM_EPOCH = 5

# RNN variant SETUP
model = SimpleRNN(len(f_text.vocab), EMB_SIZE, HID_SIZE)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# model.embedding.weight.data.copy_(f_text.vocab.vectors)

# TRAINING
for epoch in range(NUM_EPOCH):
    train_loss, train_acc = train(
        model, trn_iter, optimizer, criterion, binary_accuracy)
    valid_loss, valid_acc = evaluate(
        model, val_iter, criterion, binary_accuracy)

    print(f'EPOCH: {epoch:02} - TRN_LOSS: {train_loss:.3f} - TRN_ACC: {train_acc*100:.2f}% - VAL_LOSS: {valid_loss:.3f} - VAL_ACC: {valid_acc*100:.2f}%')

test_loss, test_acc = evaluate(model, tst_iter, criterion, binary_accuracy)
print(f'TEST_LOSS: {test_loss:.3f}, TEST_ACC: {test_acc*100:.2f}%')
