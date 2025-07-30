import torch
import torch.nn as nn
import torch.nn.functional as F


def din_attention(query, facts, attention_size, mask=None, stag='null', mode='SUM', softmax_stag=1,
                  time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        facts = torch.cat(facts, 2)
        query = torch.cat([query, query], dim=1)
    if time_major:
        facts = facts.transpose(0, 1)
    queries = query.unsqueeze(1).expand_as(facts)
    din_all = torch.cat([queries, facts, queries - facts, queries * facts], dim=-1)
    fc1 = F.sigmoid(nn.Linear(din_all.size(-1), 80)(din_all))
    fc2 = F.sigmoid(nn.Linear(80, 40)(fc1))
    scores = nn.Linear(40, 1)(fc2).squeeze(-1)
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float('-inf'))
    if softmax_stag:
        scores = F.softmax(scores, dim=-1)
    if mode == 'SUM':
        output = torch.bmm(scores.unsqueeze(1), facts)
    else:
        output = facts * scores.unsqueeze(-1)
    if return_alphas:
        return output, scores
    return output


class VecAttGRUCell(nn.GRUCell):
    def forward(self, inputs, state, att_score=None):
        out = super().forward(inputs, state)
        if att_score is None:
            att_score = torch.ones_like(state[:, :1])
        new_h = att_score * out + (1 - att_score) * state
        return new_h, new_h


def prelu(x, init=0.1):
    alpha = nn.Parameter(torch.full((x.size(-1),), init, dtype=x.dtype, device=x.device))
    return F.prelu(x, alpha)


def calc_auc(raw_arr):
    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for _, label in arr:
        if label == 1.:
            pos += 1
        else:
            neg += 1
    fp, tp = 0., 0.
    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for score, label in arr:
        if label == 1.:
            tp += 1
        else:
            fp += 1
        x = fp / neg if neg > 0 else 0
        y = tp / pos if pos > 0 else 0
        auc += (x - prev_x) * (y + prev_y) / 2.
        prev_x = x
        prev_y = y
    return auc


def calc_gauc(raw_arr, nick_index):
    last_index = 0
    gauc = 0.
    pv_sum = 0
    for idx in range(len(nick_index)):
        if nick_index[idx] != nick_index[last_index]:
            input_arr = raw_arr[last_index:idx]
            auc_val = calc_auc(input_arr)
            gauc += max(auc_val, 0.0) * len(input_arr)
            pv_sum += len(input_arr)
            last_index = idx
    return gauc / max(pv_sum, 1)


def attention(query, facts, attention_size, mask, stag='null', mode='LIST', softmax_stag=1,
              time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        facts = torch.cat(facts, 2)
    if time_major:
        facts = facts.transpose(0, 1)
    mask = mask.bool()
    w1 = torch.randn(facts.size(-1), attention_size, device=facts.device)
    w2 = torch.randn(query.size(-1), attention_size, device=query.device)
    b = torch.randn(attention_size, device=facts.device)
    v = torch.randn(attention_size, device=facts.device)
    tmp1 = torch.tensordot(facts, w1, dims=1)
    tmp2 = torch.tensordot(query, w2, dims=1).unsqueeze(1)
    tmp = torch.tanh(tmp1 + tmp2 + b)
    v_dot_tmp = torch.tensordot(tmp, v, dims=1)
    v_dot_tmp = v_dot_tmp.masked_fill(~mask, float('-inf'))
    alphas = F.softmax(v_dot_tmp, dim=-1)
    if mode == 'SUM':
        output = torch.bmm(alphas.unsqueeze(1), facts)
    else:
        output = facts * alphas.unsqueeze(-1)
    if return_alphas:
        return output, alphas
    return output


def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1,
                      time_major=False, return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        facts = torch.cat(facts, 2)
    if facts.dim() == 2:
        facts = facts.unsqueeze(1)
    if time_major:
        facts = facts.transpose(0, 1)
    facts_size = facts.size(-1)
    query = nn.Linear(query.size(-1), facts_size)(query)
    query = prelu(query)
    queries = query.unsqueeze(1).expand_as(facts)
    din_all = torch.cat([queries, facts, queries - facts, queries * facts], dim=-1)
    d_layer_1_all = F.sigmoid(nn.Linear(din_all.size(-1), 80)(din_all))
    d_layer_2_all = F.sigmoid(nn.Linear(80, 40)(d_layer_1_all))
    d_layer_3_all = nn.Linear(40, 1)(d_layer_2_all).squeeze(-1)
    if mask is not None:
        mask = mask.bool()
        d_layer_3_all = d_layer_3_all.masked_fill(~mask, float('-inf'))
    if softmax_stag:
        scores = F.softmax(d_layer_3_all, dim=-1)
    else:
        scores = d_layer_3_all
    if mode == 'SUM':
        output = torch.bmm(scores.unsqueeze(1), facts)
    else:
        output = facts * scores.unsqueeze(-1)
    if return_alphas:
        return output, scores
    return output


def self_attention(facts, attention_size, mask, stag='null'):
    if facts.dim() == 2:
        facts = facts.unsqueeze(1)
    outputs = []
    for i in range(facts.size(1)):
        tmp = din_fcn_attention(facts[:, i, :], facts[:, :i+1, :], attention_size, mask[:, :i+1], stag=stag,
                                softmax_stag=1, mode='LIST')
        tmp = tmp.sum(1)
        outputs.append(tmp)
    outputs = torch.stack(outputs, dim=1)
    return outputs


def self_all_attention(facts, attention_size, mask, stag='null'):
    if facts.dim() == 2:
        facts = facts.unsqueeze(1)
    outputs = []
    for i in range(facts.size(1)):
        tmp = din_fcn_attention(facts[:, i, :], facts, attention_size, mask, stag=stag, softmax_stag=1, mode='LIST')
        tmp = tmp.sum(1)
        outputs.append(tmp)
    outputs = torch.stack(outputs, dim=1)
    return outputs


def din_fcn_shine(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1,
                  time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        facts = torch.cat(facts, 2)
    if time_major:
        facts = facts.transpose(0, 1)
    mask = mask.bool()
    facts_size = facts.size(-1)
    query = nn.Linear(query.size(-1), facts_size)(query)
    query = prelu(query)
    queries = query.unsqueeze(1).expand_as(facts)
    din_all = torch.cat([queries, facts, queries - facts, queries * facts], dim=-1)
    d_layer_1_all = F.sigmoid(nn.Linear(din_all.size(-1), facts_size)(din_all))
    d_layer_2_all = F.sigmoid(nn.Linear(facts_size, facts_size)(d_layer_1_all))
    d_layer_2_all = d_layer_2_all.reshape_as(facts)
    output = d_layer_2_all
    return output
