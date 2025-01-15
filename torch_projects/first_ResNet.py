import numpy as np
import einops
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int

class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.max(t.tensor(0.), x)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """Linear transform"""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        limit = 1/np.sqrt(in_features)
        # t.rand outputs random entries in [0,1] -> rescale to [-1,1]
        weight = limit * (2 * t.rand(out_features, in_features) - 1)
        self.weight = nn.Parameter(weight)

        if bias:
            bias = limit * (2 * t.rand(out_features,) - 1)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        x = einops.einsum(x, self.weight, '... in_feats, out_feats in_feats -> ... out_feats')

        # cannot pass 'if self.bias' as bool value of tensor is ambiguous
        if self.bias is not None:
            x += self.bias
        return x

    def extra_repr(self) -> str:
        # use self.bias is not None -> bias is either a tensor or None
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        return t.flatten(input, self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return f'start_dim={self.start_dim}, end_dim={self.end_dim}'

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define the computational blocks of the MLP (also subclass of nn.Module)
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=28**2, out_features=100)
        self.relu = ReLU()
        self.linear2 = Linear(in_features=100, out_features=10)

    def forward(self, x: t.Tensor) -> t.Tensor:

        # PyTorch automatically calls the forward method of subclasses of nn.Module
        return self.linear2(self.relu(self.linear1(self.flatten(x))))
        
class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0
    ):
        super().__init__()
        # number of colour channels
        self.in_channels = in_channels
        # number of convolutional layers
        self.out_channels = out_channels
        # size of convolutional kernel which passes over each channel (matrix)
        self.kernel_size = kernel_size
        # step length of kernel
        self.stride = stride
        self.padding = padding

        # define a square kernel
        kernel_height = kernel_width = kernel_size
        scaling_factor = 1 / np.sqrt(in_channels * kernel_size**2)

        # dimensions of weight tensor are
        weight = scaling_factor * (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1)
        self.weight = nn.Parameter(weight)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
        return ', '.join([f'{key}={getattr(self, key)}' for key in keys])

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.nn.functional.max_pool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        keys = ['kernel_size', 'stride', 'padding']
        return ', '.join([f'{key}={getattr(self, key)}' for key in keys])
    
class Sequential(nn.Module):
    _modules: dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules) # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules) # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: t.Tensor) -> t.Tensor:
        for mod in self._modules.values():
            x = mod(x)
        return x
    
class BatchNorm2d(nn.Module):
    # type hints
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""]

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # learnable parameters
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        # testing parameters
        self.register_buffer('running_mean', t.zeros(num_features))
        self.register_buffer('running_var', t.ones(num_features))
        self.register_buffer('num_batches_tracked', t.tensor(0))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''

        if self.training:
            # want to calculate statistics for each channel - sum over dims: batch, height, width
            mean = t.mean(x, dim=(0,2,3), keepdim=True)
            var = t.var(x, dim=(0,2,3), unbiased=False, keepdim=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        # in testing state we call on the buffer variables instead
        else:
            # resize to correct shape
            mean = einops.rearrange(self.running_mean, 'channels -> 1 channels 1 1')
            var = einops.rearrange(self.running_var, 'channels -> 1 channels 1 1')

        weight = einops.rearrange(self.weight, 'channels -> 1 channels 1 1')
        bias = einops.rearrange(self.bias, 'channels -> 1 channels 1 1')

        return ((x - mean) / t.sqrt(var + self.eps)) * weight + bias


    def extra_repr(self) -> str:
        return ', '.join([f'{key}={getattr(self, key)}' for key in ['num_features', 'eps', 'momentum']])

class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return t.mean(x, dim=(2, 3))
    
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()
        
        self.left = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_feats)
        )

        if first_stride > 1:
            self.right = Sequential(
                Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride),
                BatchNorm2d(out_feats)
            )
        else:
            assert in_feats == out_feats
            self.right = nn.Identity()

        self.relu = ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, in_feats, height, width)
        Return: shape (batch, out_feats, height / stride, width / stride)
        '''
        x_left = self.left(x)
        x_right = self.right(x)

        return self.relu(x_left + x_right)
    
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        
        blocks = [ResidualBlock(in_feats, out_feats, first_stride)] + [
            ResidualBlock(out_feats, out_feats) for n in range(n_blocks - 1)
        ]
        self.blocks = Sequential(*blocks)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, in_feats, height, width)
        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.blocks(x)
    
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ): 
        
        super().__init__()
        in_feats_0 = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        self.input_layers = Sequential(
            Conv2d(in_channels=3, out_channels=in_feats_0, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(in_feats_0),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # list of all input feature counts for each block
        all_in_feats = [in_feats_0] + out_features_per_group[:-1]
        
        self.residual_layers = Sequential(
            *(
                BlockGroup(*args) 
                for args in zip(
                    n_blocks_per_group,
                    all_in_feats,
                    out_features_per_group,
                    first_strides_per_group
                )
            )
        )

        self.output_layers = Sequential(
            AveragePool(),
            Linear(out_features_per_group[-1], n_classes)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        return self.output_layers(self.residual_layers(self.input_layers(x)))

my_resnet = ResNet34()
