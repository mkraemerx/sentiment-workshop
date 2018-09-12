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
# IN_FILE = 'germeval2018.try.txt'
IN_FILE = 'germeval2018.training.txt'
IN_FILE_TEST = 'germeval2018.test.txt'
BATCH_SIZE = 32

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
test_examples = pipe.process_data(
    IN_FILE_TEST, fields)[0]
full_ds = data.Dataset(full_examples, fields)
tst_ds = data.Dataset(test_examples, fields)

# do the splitting for trn/val with torchtext
trn_ds, val_ds = full_ds.split(
    split_ratio=[0.95, 0.05], stratified=True, random_state=random.seed(SEED))

print(f'train len {len(trn_ds.examples)}')
print(f'val len {len(val_ds.examples)}')
print(f'test len {len(tst_ds.examples)}')

vec = torchtext.vocab.Vectors('embed_tweets_de_100D_fasttext',
                              cache='/Users/michel/Downloads/')
# validation + test data should by no means influence the model, so build the vocab just on trn
f_text.build_vocab(trn_ds, vectors=vec)
# ALT: f_text.build_vocab(trn_ds, max_size=20000)
f_lemma.build_vocab(trn_ds, vectors=vec)
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


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_dim, emb_dim=100, hidden_dim=200, num_rnn_layers=1, bidirectional=False, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_rnn_layers,
                           bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, 1)  # 1 is output dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x type is Tensor[sentence len, batch size]. Internally pytorch does not use 1-hot

        embedded = self.dropout(self.embedding(x))
        # embedded type is Tensor[sentence len, batch size, emb dim]

        output, (hidden_state, cell) = self.rnn(embedded)
        # output type is Tensor[sentence len, batch size, hidden dim]
        # hidden_state type is Tensor[1, batch size, hidden dim]

        hidden_state_comp = self.dropout(
            torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1))
        # TODO out = self.lin(out.view(-1, out.size(2))

        # hidden_state_comp type is Tensor[1, batch size, hidden dim * 2] (forward/backward from bidirectional)

        return self.fc(hidden_state_comp.squeeze(0))


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds))
    # rounded_preds type is Tensor[32] either 0.0 or 1.0 each

    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def detail_metric(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds)).long()
    # rounded_preds type is Tensor[32] either 0 or 1 each

    y = y.long()

    tp = (rounded_preds * y).sum()
    tn = (torch.max(rounded_preds, y) ==
          torch.zeros(rounded_preds.shape[0]).long()).sum()
    fp = rounded_preds.sum() - tp
    fn = rounded_preds.shape[0] - tn - rounded_preds.sum()

    return tp, tn, fp, fn


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
    epoch_meter_tp = 0
    epoch_meter_tn = 0
    epoch_meter_fp = 0
    epoch_meter_fn = 0

    with torch.no_grad():
        model.eval()

        for batch in iterator:
            y_hat = model(batch.text).squeeze(1)
            loss = criterion(y_hat, batch.label)
            meter = metric(y_hat, batch.label)

            epoch_loss += loss.item()
            epoch_meter_tp += meter[0].item()
            epoch_meter_tn += meter[1].item()
            epoch_meter_fp += meter[2].item()
            epoch_meter_fn += meter[3].item()

    if epoch_meter_tp == 0:
    recall = float(epoch_meter_tp) / (epoch_meter_tp + epoch_meter_fn)
    precision = float(epoch_meter_tp) / (epoch_meter_tp + epoch_meter_fp)
    f_score = 2 * precision * recall / (precision + recall)
    return epoch_loss / len(iterator), (epoch_meter_tp + epoch_meter_tn) / (epoch_meter_tp + epoch_meter_tn + epoch_meter_fp + epoch_meter_fn), recall, precision, f_score


EMB_SIZE = 100
HID_SIZE = 100
NUM_RNN_LAYERS = 3
DROPOUT = 0.7
BIDIRECTIONAL = True
NUM_EPOCH = 7

# RNN variant SETUP
model = SimpleLSTM(len(f_text.vocab), EMB_SIZE, HID_SIZE,
                   NUM_RNN_LAYERS, bidirectional=BIDIRECTIONAL, dropout=DROPOUT)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model.embedding.weight.data.copy_(f_text.vocab.vectors)

# TRAINING
for epoch in range(NUM_EPOCH):
    train_loss, train_acc = train(
        model, trn_iter, optimizer, criterion, binary_accuracy)
    valid_loss, valid_acc, recall, precision, f_score = evaluate(
        model, val_iter, criterion, detail_metric)

    print(f'EPOCH: {epoch:02}\nTRN_LOS: {train_loss:.3f} - TRN_ACC: {train_acc*100:.2f}% - VAL_LOS: {valid_loss:.3f} - VAL_ACC: {valid_acc*100:.2f}% - REC: {recall:.4f} - PRE: {precision:.4f} - F1: {f_score:.4f}')

test_loss, test_acc, recall, precision, f_score = evaluate(
    model, tst_iter, criterion, detail_metric)
print(f'TEST_LOS: {test_loss:.3f}, TEST_ACC: {test_acc*100:.2f}% - TEST_REC: {recall:.4f} - TEST_PRE: {precision:.4f} - TEST_F1: {f_score:.4f}')
