import torch
import torch.nn as nn
import numpy as np


class NaiveIdentity(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        '''We implement an identity transformation to support our code
        '''
        super().__init__()
    
    def forward(self, x, **kwargs):
        '''
        Input: x
        Output: x
        '''
        return x
