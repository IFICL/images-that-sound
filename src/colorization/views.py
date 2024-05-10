import torch


class ColorLView():
    def __init__(self):
        pass

    def view(self, im):
        return im

    def inverse_view(self, noise):
        # Get L color by averaging color channels
        noise[:3] = 2 * torch.stack([noise[:3].mean(0)] * 3)

        return noise


class ColorABView():
    def __init__(self):
        pass

    def view(self, im):
        return im

    def inverse_view(self, noise):
        # Get AB color by taking residual
        noise[:3] = 2 * (noise[:3] - torch.stack([noise[:3].mean(0)] * 3))

        return noise

