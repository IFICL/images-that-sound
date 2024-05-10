import torch
import torch.nn as nn
import numpy as np


class LearnableImage(nn.Module):
    def __init__(
        self,
        height      :int,
        width       :int,
        num_channels:int
    ):
        '''This is an abstract class, and is meant to be subclassed before use upon calling forward(), retuns a tensor of shape (num_channels, height, width)
        '''
        super().__init__()
        
        self.height = height      
        self.width = width       
        self.num_channels = num_channels
    
    def as_numpy_image(self):
        image = self.forward()
        image = image.cpu().numpy()
        image = image.transpose(1, 2, 0) 
        return image

    
class LearnableImageFourier(LearnableImage):
    def __init__(
        self,
        height      :int=256, # Height of the learnable images
        width       :int=256, # Width of the learnable images
        num_channels:int=3  , # Number of channels in the images
        hidden_dim  :int=256, # Number of dimensions per hidden layer of the MLP
        num_features:int=128, # Number of fourier features per coordinate
        scale       :int=10 , # Magnitude of the initial feature noise
        renormalize :bool=False
    ):        
        super().__init__(height, width, num_channels)
        
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.scale = scale
        self.renormalize = renormalize
        
        # The following objects do NOT have parameters, and are not changed while optimizing this class
        self.uv_grid = nn.Parameter(get_uv_grid(height, width, batch_size=1), requires_grad=False)
        self.feature_extractor = GaussianFourierFeatureTransform(2, num_features, scale)
        self.features = nn.Parameter(self.feature_extractor(self.uv_grid), requires_grad=False) # pre-compute this if we're regressing on images
        
        H = hidden_dim # Number of hidden features. These 1x1 convolutions act as a per-pixel MLP
        C = num_channels  # Shorter variable names let us align the code better
        M = 2 * num_features
        self.model = nn.Sequential(
            nn.Conv2d(M, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, C, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self):
        features = self.features

        output = self.model(features).squeeze(0)
        
        assert output.shape==(self.num_channels, self.height, self.width)
        
        if self.renormalize:
            output = output * 2 - 1.0 # renormalize to [-1, 1]

        return output
    

class LearnableImageParam(LearnableImage):
    def __init__(
        self,
        height      :int=256, # Height of the learnable images
        width       :int=256, # Width of the learnable images
        num_channels:int=3  , # Number of channels in the images\
        **kwargs
    ):        
        super().__init__(height, width, num_channels)
        
        self.model = nn.Parameter(torch.randn(num_channels, height, width))


    def forward(self):
        x = self.model
        return x


##################################
######## HELPER FUNCTIONS ########
##################################

class GaussianFourierFeatureTransform(nn.Module):
    """
    Original authors: https://github.com/ndahlquist/pytorch-fourier-feature-networks
    
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
        https://arxiv.org/abs/2006.10739
        https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
        returns a tensor of size [batches, num_features*2, width, height].
    """
    def __init__(self, num_channels, num_features=256, scale=10):
        #It generates fourier components of Arandom frequencies, not all of them.
        #The frequencies are determined by a random normal distribution, multiplied by "scale"
        #So, when "scale" is higher, the fourier features will have higher frequencies    
        #In learnable_image_tutorial.ipynb, this translates to higher fidelity images.
        #In other words, 'scale' loosely refers to the X,Y scale of the images
        #With a high scale, you can learn detailed images with simple MLP's
        #If it's too high though, it won't really learn anything but high frequency noise
        
        super().__init__()

        self.num_channels = num_channels
        self.num_features = num_features
        
        #freqs are n-dimensional spatial frequencies, where n=num_channels
        self.freqs = nn.Parameter(torch.randn(num_channels, num_features) * scale, requires_grad=False)

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batch_size, num_channels, height, width = x.shape

        assert num_channels == self.num_channels,\
            "Expected input to have {} channels (got {} channels)".format(self.num_channels, num_channels)

        # Make shape compatible for matmul with freqs.
        # From [B, C, H, W] to [(B*H*W), C].
        x = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, num_channels)

        # [(B*H*W), C] x [C, F] = [(B*H*W), F]
        x = x @ self.freqs

        # From [(B*H*W), F] to [B, H, W, F]
        x = x.view(batch_size, height, width, self.num_features)
        # From [B, H, W, F] to [B, F, H, W 
        x = x.permute(0, 3, 1, 2)

        x = 2 * torch.pi * x
        
        output = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        
        assert output.shape==(batch_size, 2*self.num_features, height, width)
        
        return output  


def get_uv_grid(height:int, width:int, batch_size:int=1)->torch.Tensor:
    #Returns a torch cpu tensor of shape (batch_size,2,height,width)
    #Note: batch_size can probably be removed from this function after refactoring this file. It's always 1 in all usages.
    #The second dimension is (x,y) coordinates, which go from [0 to 1) from edge to edge
    #(In other words, it will include x=y=0, but instead of x=y=1 the other corner will be x=y=.999)
    #(this is so it doesn't wrap around the texture 360 degrees)

    # import pdb; pdb.set_trace()
    assert height>0 and width>0 and batch_size>0,'All dimensions must be positive integers'
    
    y_coords = np.linspace(0, 1, height, endpoint=False)
    x_coords = np.linspace(0, 1, width , endpoint=False)
    
    uv_grid = np.stack(np.meshgrid(y_coords, x_coords), -1)
    uv_grid = torch.tensor(uv_grid).unsqueeze(0).permute(0, 3, 2, 1).float().contiguous()
    uv_grid = uv_grid.repeat(batch_size,1,1,1)
    
    assert tuple(uv_grid.shape)==(batch_size,2,height,width)
    
    return uv_grid

