import numpy as np
import einops
from typing import Union, Optional, Tuple, List, Dict
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from dataclasses import dataclass
from tqdm import tqdm

import matplotlib.pyplot as plt

device = t.device("cuda" if t.cuda.is_available() else "cpu")

class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.max(t.tensor(0.), x)
    
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''Simple linear transformation'''
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
        '''x: shape (*, in_features); Return: shape (*, out_features)'''
        x = einops.einsum(x, self.weight, '... in, out in -> ... out')

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
        '''Flatten out dimensions from start_dim to end_dim, inclusive of both'''
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
    
# mean = 0.1307, sd = 0.3081 are precoumpted for the MNIST dataset
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    # allows for taking subset of training/testing data
    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset

class SimpleMLPTrainingArgs():
    """Implicity sets an __init__ function with arguments as below"""
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    subset: int = 10

def train(args: SimpleMLPTrainingArgs):
    """Trains model using parameters from 'args' object"""

    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist(subset=args.subset)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)

    # optimizer calculates amount by which to alter training parameters
    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_lst = []
    accuracy_lst = []

    for epoch in tqdm(range(args.epochs)):
        for imgs, labels in mnist_trainloader:
            # move training images & labels to CPU
            imgs = imgs.to(device)
            labels = labels.to(device)

            # model evaluates entered img returning logit
            # logit is model output before softmax
            logits = model(imgs)

            # loss calculation & backprop, accumulating gradients in optimization of model parameters
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            # zero the gradients for next optimization cycle
            optimizer.zero_grad()
            loss_lst.append(loss.item())

        # validation step
        num_correct = 0
        for imgs, labels in mnist_testloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            with t.inference_mode():
                logits = model(imgs)

            # find index of largest entry over the columns (0-9 digit)
            predictions = t.argmax(logits, dim=1)
            num_correct += (predictions == labels).sum().item()

        accuracy = num_correct / len(mnist_testset)
        accuracy_lst.append(accuracy)

    # generating figures
    plt.figure(figsize=(10,6))
    plt.plot(loss_lst)
    plt.ylim(0, max(loss_lst) + 0.1)
    plt.xlabel('Num batches seen')
    plt.ylabel('Cross entropy loss')
    plt.title('Simple MLP training on MNIST')
    plt.grid(True)
    plt.savefig('Simple_MLP_training.jpg', format='jpeg', dpi=300)
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(accuracy_lst)
    plt.ylim(0, max(accuracy_lst) + 0.1)
    plt.xlabel('Num epochs')
    plt.ylabel('Accuracy = correct_predictions / len(test_set)')
    plt.title('Simple MLP accuracy on MNIST')
    plt.grid(True)
    plt.savefig('Simple_MLP_accuracy.jpg', format='jpeg', dpi=300)
    plt.show()

args = SimpleMLPTrainingArgs()
train(args)
