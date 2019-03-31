import torch
from torch import nn


class Test_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2,3)

    def forward(self, x):
        y = self.model(x)
        return y

