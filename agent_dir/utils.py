import math
import torch

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


def to_onehot(data):
    result = torch.zeros_like(data)
    result[data.argmax()]=1.
    return result