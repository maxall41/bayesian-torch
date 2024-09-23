'''
wrapper for ReLU
'''

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class ReLU(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        kl = 0
        return F.relu(input[0], inplace=self.inplace), kl

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class Mish(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        kl = 0
        return F.mish(input[0], inplace=self.inplace), kl

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class SiLU(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        kl = 0
        return F.silu(input[0], inplace=self.inplace), kl

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class LeakyReLU(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(LeakyReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        kl = 0
        return F.leaky_relu(input[0], inplace=self.inplace), kl

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class Sigmoid(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        kl = 0
        return F.sigmoid(input[0], inplace=self.inplace), kl

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class Tanh(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        kl = 0
        return F.tanh(input[0], inplace=self.inplace), kl

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
