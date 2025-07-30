import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def expand(x, dim, N):
    return torch.cat([x.unsqueeze(dim) for _ in range(N)], dim=dim)


def learned_init(units):
    lin = nn.Linear(1, units, bias=False)
    return lin(torch.ones(1, 1)).squeeze(0)


class RUMCell(nn.Module):
    """Simplified RUM cell implemented with PyTorch."""

    def __init__(self, controller_units, memory_size, memory_vector_dim, read_head_num, write_head_num,
                 output_dim=None, clip_value=20, batch_size=128):
        super().__init__()
        self.controller_units = controller_units
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.controller = nn.GRUCell(memory_vector_dim * read_head_num + memory_vector_dim*2, controller_units)
        self.o2p = nn.Linear(controller_units, (memory_vector_dim + 1) * (read_head_num + write_head_num))
        self.o2o = nn.Linear(controller_units + memory_vector_dim * read_head_num, output_dim or memory_vector_dim)

    def forward(self, x, prev_state):
        prev_read_vector_list = prev_state["read_vector_list"]
        controller_input = torch.cat([x] + prev_read_vector_list, dim=1)
        controller_state = self.controller(controller_input, prev_state["controller_state"])
        parameters = torch.tanh(self.o2p(controller_state))
        head_params = torch.split(parameters, self.memory_vector_dim + 1, dim=1)
        prev_M = prev_state["M"]
        read_vector_list = []
        for i in range(self.read_head_num):
            k = torch.tanh(head_params[i][:, :self.memory_vector_dim])
            beta = F.softplus(head_params[i][:, self.memory_vector_dim]) + 1
            w = self.addressing(k, beta, prev_M)
            read_vector = torch.sum(w.unsqueeze(2) * prev_M, dim=1)
            read_vector_list.append(read_vector)
        M = prev_M
        output = self.o2o(torch.cat([controller_state] + read_vector_list, dim=1))
        output = torch.clamp(output, -self.clip_value, self.clip_value)
        next_state = {
            "controller_state": controller_state,
            "read_vector_list": read_vector_list,
            "M": M
        }
        return output, next_state

    def addressing(self, k, beta, M):
        key = k.unsqueeze(2)
        inner = torch.bmm(M, key).squeeze(2)
        k_norm = k.norm(dim=1, keepdim=True)
        m_norm = M.norm(dim=2)
        K = inner / (m_norm * k_norm + 1e-8)
        K = torch.exp(beta.unsqueeze(1) * K)
        w_c = K / (K.sum(dim=1, keepdim=True) + 1e-8)
        return w_c

    def zero_state(self, batch_size, dtype=torch.float32):
        read_vector_list = [expand(torch.tanh(learned_init(self.memory_vector_dim)), 0, batch_size)
                            for _ in range(self.read_head_num)]
        controller_state = torch.zeros(batch_size, self.controller_units, dtype=dtype)
        M = expand(torch.zeros(self.memory_size, self.memory_vector_dim, dtype=dtype), 0, batch_size)
        state = {
            "controller_state": controller_state,
            "read_vector_list": read_vector_list,
            "M": M
        }
        return state
