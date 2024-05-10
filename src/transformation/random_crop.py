import torch
import torch.nn as nn
import numpy as np

from kornia.augmentation import RandomCrop


class ImageRandomCropper(nn.Module):
    def __init__(
        self,
        size,
        n_view=1,
        padding=None,
        cropping_mode="slice",
        p=1.0,
        **kwargs
    ):
        '''We implement an easy random cropping operation
        '''
        super().__init__()
        
        self.transformation = RandomCrop(
            size=size,
            padding=padding,
            p=p,
            cropping_mode=cropping_mode
        )
        self.n_view = n_view
    
    def forward(self, x):
        '''
        Input: (1, C, H, W)
        Output: (n_view, C, h, w)
        '''
        if x.shape[0] == 1:
            x = x.repeat(self.n_view, 1, 1, 1)
        
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.transformation(x)
        return x
