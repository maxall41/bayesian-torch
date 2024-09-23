'''
wrapper for pooling functions
'''

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class MaxPool1d(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, size, inplace=False):
        super(MaxPool1d, self).__init__()
        self.inplace = inplace
        self.pool = nn.MaxPool1d(size)

    def forward(self, input):
        kl = 0
        return self.pool(input[0]), kl

class MaxPool2d(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, size, inplace=False):
        super(MaxPool2d, self).__init__()
        self.inplace = inplace
        self.pool = nn.MaxPool2d(size)

    def forward(self, input):
        kl = 0
        return self.pool(input[0]), kl


class MaxPool3d(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, size, inplace=False):
        super(MaxPool3d, self).__init__()
        self.inplace = inplace
        self.pool = nn.MaxPool3d(size)

    def forward(self, input):
        kl = 0
        return self.pool(input[0]), kl