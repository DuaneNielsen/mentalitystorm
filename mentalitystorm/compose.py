import torch
import torch.nn as nn
from collections import namedtuple

SpaceInvadersCategories = ('SpaceInvadersCategories', 'shots, invaders, barriers, player')


class CategoricalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.categories = nn.ModuleDict()

    """
    Accepts a dictionary of of image channels    
    """
    def forward(self, x_dict):
        y_dict = {}
        for x_category in x_dict:
            y_dict[x_category] = self.categories[x_category](x_dict[x_category])
        return y_dict


class PassThroughWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, args):
        y = self.module(args[0])
        return (y, *args[1:])


class FlattenInputWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.view(x.shape[0], -1))


class ReshapeOutputWrapper(nn.Module):
    def __init__(self, module, shape):
        super().__init__()
        self.shape = shape
        self.module = module

    def forward(self, x):
        y = self.module(x)
        return y.view(x.shape[0], *self.shape)
