import torch
import torch.nn as nn
from mimn import MIMNCell
from rnn import dynamic_rnn

class PyTorchDNN(nn.Module):
    """DNN model implemented in PyTorch following the TensorFlow version."""

    def __init__(self, vocab_size, embed_dim):
        super(PyTorchDNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 4, 200)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(200, 80)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(80, 2)

    def forward(self, item, cate, hist_item, hist_cate, mask):
        item_emb = torch.cat(
            [self.embedding(item), self.embedding(cate)], dim=1
        )
        hist_item_emb = self.embedding(hist_item)
        hist_cate_emb = self.embedding(hist_cate)
        hist_emb = torch.cat([hist_item_emb, hist_cate_emb], dim=2)
        hist_sum = (hist_emb * mask.unsqueeze(2)).sum(1)
        x = torch.cat([item_emb, hist_sum], dim=1)
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
        logits = self.fc3(x)
        return logits


class PyTorchMIMN(nn.Module):
    """Minimal MIMN-based model implemented in PyTorch."""

    def __init__(self, vocab_size, embed_dim, memory_size=4,
                 mem_induction=0, util_reg=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cell = MIMNCell(
            controller_units=embed_dim,
            memory_size=memory_size,
            memory_vector_dim=embed_dim * 2,
            read_head_num=1,
            write_head_num=1,
            output_dim=embed_dim,
            mem_induction=mem_induction,
            util_reg=util_reg,
        )
        self.fc1 = nn.Linear(embed_dim * 5, 200)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(200, 80)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(80, 2)

    def forward(self, item, cate, hist_item, hist_cate, mask):
        item_emb = torch.cat(
            [self.embedding(item), self.embedding(cate)], dim=1
        )
        hist_item_emb = self.embedding(hist_item)
        hist_cate_emb = self.embedding(hist_cate)
        hist_emb = torch.cat([hist_item_emb, hist_cate_emb], dim=2)
        seq_len = mask.sum(1).long()
        outputs, _ = dynamic_rnn(
            self.cell, hist_emb, sequence_length=seq_len
        )
        last_out = outputs[range(outputs.size(0)), seq_len - 1, :]
        hist_sum = (hist_emb * mask.unsqueeze(2)).sum(1)
        x = torch.cat([item_emb, hist_sum, last_out], dim=1)
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
        logits = self.fc3(x)
        return logits
