import torch
import torch.nn as nn
import numpy as np

from kornia.augmentation import RandomCrop


class ImageToSpec(nn.Module):
    def __init__(
        self,
        inverse=False,
        flip=False,
        rgb2gray='mean',
        **kwargs
    ):
        '''We implement an simple image-to-spectrogram transformation
        '''
        super().__init__()

        self.inverse = inverse
        self.flip = flip
        self.rgb2gray = rgb2gray

        if self.rgb2gray == 'mean':
            self.coefficients = torch.ones(3).float() / 3.0
        elif self.rgb2gray in ['NTSC', 'ntsc']:
            self.coefficients = torch.tensor([0.299, 0.587, 0.114])
        elif self.rgb2gray == 'luminance':
            self.coefficients = torch.tensor([0.2126, 0.7152, 0.0722])
        elif self.rgb2gray == 'r_channel':
            self.coefficients = torch.tensor([1.0, 0.0, 0.0])
        elif self.rgb2gray == 'g_channel':
            self.coefficients = torch.tensor([0.0, 1.0, 0.0])
        elif self.rgb2gray == 'b_channel':
            self.coefficients = torch.tensor([0.0, 0.0, 1.0])
    
    def forward(self, x):
        '''
        Input: (1, C, H, W)
        Output: (1, 1, H, W)
        '''

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        coefficients = self.coefficients.view(1, -1, 1, 1).to(dtype=x.dtype, device=x.device)
        x = torch.sum(x * coefficients, dim=1, keepdim=True)
        
        if self.inverse:
            x = 1.0 - x
        
        if self.flip:
            x = torch.flip(x, [2])
        
        return x
