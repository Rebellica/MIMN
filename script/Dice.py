import torch
import torch.nn as nn
import torch.nn.functional as F


class Dice(nn.Module):
    """Data Adaptive Activation Function."""

    def __init__(self, input_dim, epsilon=1e-9):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(input_dim))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=0, keepdim=True)
        x_normed = (x - mean) / torch.sqrt(var + self.epsilon)
        p = torch.sigmoid(x_normed)
        return self.alpha * (1.0 - p) * x + p * x


def dice(x):
    """Convenience function that creates a Dice layer on the fly."""
    layer = Dice(x.size(-1))
    return layer(x)


def parametric_relu(x, init=0.0):
    """Parametric ReLU using PyTorch."""
    alpha = nn.Parameter(torch.full((x.size(-1),), init, dtype=x.dtype, device=x.device))
    return F.prelu(x, alpha)
