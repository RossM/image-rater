import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from torch import Tensor
from abc import ABC, abstractmethod

class LogisticRegression(nn.Module, ABC):
    @abstractmethod
    def get_score(self, x: Tensor):
        pass
    
    def forward(self, x: Tensor):
        scores = self.get_score(x)
        return torch.sigmoid(scores[:,0] - scores[:,1])
        
    def aux_loss(self):
        sum = 0
        for p in self.parameters():
            sum += p.pow(2).sum()
        return sum

class MultifactorLogisticRegression(LogisticRegression):
    def __init__(self, dim: int, factors: int, activation = nn.ReLU()):
        super().__init__()
        
        self.dim = dim
        self.factors = factors
        self.input_conv = nn.Linear(dim, factors)
        self.activation = activation
        self.output_conv = nn.Linear(factors, 1, bias=False)
        
    def get_score(self, x: Tensor):
        x = self.input_conv(x)
        x = self.activation(x)
        x = self.output_conv(x)
        return x
