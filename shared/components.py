from torch import nn
from torch import Tensor
import torch
from typing import List, Tuple, Union
from jaxtyping import Float

class Bilinear(nn.Linear):
    """A bilinear layer with optional gate and noise"""
    def __init__(self, d_in: int, d_out: int, bias=False, gate=None) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.gate = {"relu":nn.ReLU(), "silu":nn.SiLU(), "gelu":nn.GELU(), None: nn.Identity()}[gate]
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        left, right = super().forward(x).chunk(2, dim=-1)
        return self.gate(left) * right
    
    @property
    def w_l(self):
        return self.weight.chunk(2, dim=0)[0]
    
    @property
    def w_r(self):
        return self.weight.chunk(2, dim=0)[1]


class Linear(nn.Linear):
    """A linear layer with optional gate and noise"""
    def __init__(self, d_in: int, d_out: int, bias=False, gate=None) -> None:
        super().__init__(d_in, d_out, bias=bias)
        self.gate = {"relu": nn.ReLU(), "silu": nn.SiLU(), "gelu": nn.GELU(), None: nn.Identity()}[gate]
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return self.gate(super().forward(x))


class MLP(nn.Module):
    """A general MLP implementation supporting bilinear, gated and ReLU activations"""
    def __init__(self, d_model: int, d_hidden: int, bias=False, bilinear=True, gate=None) -> None:
        super().__init__()
        self.w = (Bilinear if bilinear else Linear)(d_model, d_hidden, bias=bias, gate=gate)
        self.p = nn.Linear(d_hidden, d_model, bias=bias)
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.p(self.w(x))
    

class RMSNorm(nn.Module):
    """PyTorch doesn't yet have RMSNorm implemented, this is the canonical implementation"""
    def __init__(self):
        super().__init__()
        self.eps = 1e-8
    
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)


class Norm(nn.Module):
    """A multi-function normalization layer with noise and bias options"""
    def __init__(self, norm):
        super().__init__()
        self.norm = RMSNorm() if norm else nn.Identity()
        
    def forward(self, x):
        return self.norm(x)