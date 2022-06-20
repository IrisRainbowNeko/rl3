import math
import torch
from torch import nn

class EpsScheduler:
    def __init__(self, eps_start, eps_end, eps_decay):
        self.idx=0
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_delta=eps_start-eps_end
        self.eps_decay=eps_decay

    def step(self):
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.idx / self.eps_decay)
        self.idx+=1
        return eps

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

def to_onehot(data, n=None):
    if n is None:
        result = torch.zeros_like(data)
        result[data.argmax()]=1.
    else:
        result = torch.zeros((n,))
        result[data] = 1.
    return result