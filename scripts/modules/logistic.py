import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from torch import Tensor

class LogisticRegression(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        
        self.dim = dim
        self.c = nn.Parameter(torch.zeros((dim)))
    
    def forward(self, x: Tensor):
        return torch.sigmoid(x @ self.c.t())