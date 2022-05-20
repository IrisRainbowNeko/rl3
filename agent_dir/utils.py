import math


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