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

class LinearLogisticRegression(LogisticRegression):
    def __init__(self, dim: int):
        super().__init__()
        
        self.dim = dim
        self.c = nn.Parameter(torch.zeros((dim)))
        
    def get_score(self, x: Tensor):
        return x @ self.c.t()
        
class MultifactorLogisticRegression(LogisticRegression):
    def __init__(self, dim: int, factors: int, activation = nn.Sigmoid()):
        super().__init__()
        
        self.dim = dim
        self.factors = factors
        self.input_conv = nn.Linear(dim, factors)
        self.activation = activation
        self.c = nn.Parameter(torch.zeros((factors)))
        
    def get_score(self, x: Tensor):
        return self.activation(self.input_conv(x)) @ self.c.t()
