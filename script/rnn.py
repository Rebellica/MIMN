import torch
import torch.nn as nn


def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None):
    """A very small dynamic RNN wrapper using PyTorch GRU/LSTM cells."""
    batch_size, seq_len, _ = inputs.size()
    if initial_state is None:
        if hasattr(cell, 'zero_state'):
            state = cell.zero_state(batch_size, dtype=inputs.dtype)
        elif isinstance(cell, nn.LSTMCell):
            h = torch.zeros(batch_size, cell.hidden_size, dtype=inputs.dtype, device=inputs.device)
            c = torch.zeros(batch_size, cell.hidden_size, dtype=inputs.dtype, device=inputs.device)
            state = (h, c)
        else:
            state = torch.zeros(batch_size, cell.hidden_size, dtype=inputs.dtype, device=inputs.device)
    else:
        state = initial_state

    outputs = []
    for t in range(seq_len):
        inp = inputs[:, t, :]
        if hasattr(cell, 'zero_state'):
            out, state, _ = cell(inp, state)
        else:
            state = cell(inp, state)
            out = state[0] if isinstance(cell, nn.LSTMCell) else state
        outputs.append(out)
    outputs = torch.stack(outputs, dim=1)
    if sequence_length is not None:
        mask = torch.arange(seq_len, device=inputs.device).expand(batch_size, seq_len) < sequence_length.unsqueeze(1)
        outputs = outputs * mask.unsqueeze(-1).to(outputs.dtype)
    if isinstance(cell, nn.LSTMCell):
        final_state = state
    else:
        final_state = state
    return outputs, final_state
