import torch.optim
import math


class Optim():
    def __init__(self, model):
        self.params = list(filter(lambda p: p.requires_grad, model.parameters()))
        self.optimizer = torch.optim.Adam(self.params, lr=0.001, weight_decay=1e-4)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clip_grad_norm(self, max_norm):
        """" Clips gradient norm"""
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        else:
            return math.sqrt(sum(p.grad.data.norm() ** 2 for p in self.params if p.grad is not None))
