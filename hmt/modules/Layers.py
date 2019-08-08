from torch import nn
import torch


class LayerNorm(nn.Module):
    """
    Layer normalization module
    """

    def __init__(self, depth, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.depth = depth
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(depth), requires_grad=True)
        self.bias = nn.Parameter(torch.ones(depth), requires_grad=True)

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        norm_x = (x-mean) / (std+self.eps)
        return norm_x*self.scale + self.bias

