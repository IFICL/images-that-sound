import torch
import torch.nn as nn
import numpy as np

from kornia.augmentation import RandomCrop


class BlockRearranger(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        '''We implement an easy block rearrange operation
        '''
        super().__init__()
    
    def forward(self, latents, inverse=False):
        '''
        Input: (1, C, H, W)
        Output: (n_view, C, h, w)
        '''
        B, C, H, W = latents.shape
        if not inverse: # convert square to rectangle
            transformed_latents = torch.cat([latents[:, :, :H//2, :], latents[:, :, H//2:, :]], dim=-1)
        else: # convert rectangle to square
            transformed_latents = torch.cat([latents[:, :, :, :W//2 ], latents[:, :, :, W//2:]], dim=-2)
        return transformed_latents
