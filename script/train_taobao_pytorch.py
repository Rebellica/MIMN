import argparse
import numpy as np
import torch
import torch.nn as nn
from data_iterator import DataIterator
from pytorch_model import PyTorchDNN
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embed_dim', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--maxlen', type=int, default=200)
args = parser.parse_args()

EMBED_DIM = args.embed_dim
BATCH_SIZE = args.batch_size
MAXLEN = args.maxlen

def prepare(batch):
    src, tgt = batch
    item = torch.tensor(src[1], dtype=torch.long)
    cate = torch.tensor(src[2], dtype=torch.long)
    hist_item = torch.tensor(tgt[1], dtype=torch.long)
    hist_cate = torch.tensor(tgt[2], dtype=torch.long)
    mask = torch.tensor(tgt[5], dtype=torch.float32)
    label = torch.tensor(np.argmax(tgt[0], axis=1), dtype=torch.long)
    return item, cate, hist_item, hist_cate, mask, label


def train():
    train_file = './data/taobao_data/taobao_train.txt'
    feature_file = './data/taobao_data/taobao_feature.pkl'
    train_data = DataIterator(train_file, BATCH_SIZE, MAXLEN)
    vocab_size = pkl.load(open(feature_file, 'rb'))
    model = PyTorchDNN(vocab_size, EMBED_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for batch in train_data:
        item, cate, hist_i, hist_c, mask, label = prepare(batch)
        optimizer.zero_grad()
        out = model(item, cate, hist_i, hist_c, mask)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        print('loss: %.4f' % loss.item())

if __name__ == '__main__':
    if args.p == 'train':
        train()
