from torch import nn
from torch import Tensor
import torch
from jaxtyping import Float


class Noise(nn.Module):
    """Adding normed Gaussian noise to the activations"""
    def __init__(self, scale: float | None = None) -> None:
        super().__init__()
        self.scale = scale
    
    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        """Forwards the input with and adds Gaussian noise if in training mode"""
        if self.training and self.scale is not None:
            x = x + torch.randn_like(x) * self.scale * x.std()
        return x


class Bilinear(nn.Linear):
    """A bilinear layer with optional gate and noise"""
    def __init__(self, d_in: int, d_out: int, bias=False, gate=False, noise=None) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.noise = Noise(scale=noise) if noise else nn.Identity()
        self.gate = nn.ReLU() if gate else nn.Identity()
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Performs an element-wise bilinear transformation. Note that the gate is optional and off by default"""
        left, right = super().forward(self.noise(x)).chunk(2, dim=-1)
        return self.gate(left) * right


class Linear(nn.Linear):
    """A linear layer with optional gate and noise"""
    def __init__(self, d_in: int, d_out: int, bias: bool = False, gate: bool = False, noise: bool = None) -> None:
        super().__init__(d_in, d_out, bias=bias)
        self.noise = Noise(scale=noise) if noise else nn.Identity()
        self.gate = nn.ReLU() if gate else nn.Identity()
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Conditionally adds noise and a gate atop the linear transformation"""
        return self.gate(super().forward(self.noise(x)))


class MLP(nn.Module):
    """A general MLP implementation supporting bilinear, gated and ReLU activations"""
    def __init__(self, d_model: int, d_hidden: int, bias: bool = False, bilinear: bool = True, gate: bool = False) -> None:
        super().__init__()
        self.w = (Bilinear if bilinear else Linear)(d_model, d_hidden, bias=bias, gate=gate)
        self.o = nn.Linear(d_hidden, d_model, bias=bias) # should rename this to p
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """The default MLP transformation"""
        return self.o(self.w(x))
    

class RMSNorm(nn.Module):
    """PyTorch doesn't yet have RMSNorm implemented, this is the canonical implementation"""
    def __init__(self, dims: int, bias: bool=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.bias = nn.Parameter(torch.zeros(dims), requires_grad=bias)
        self.eps = 1e-8
    
    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        """Divide by the length of the vector and multiply by the weight and add the bias"""
        x = x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return x * self.weight + self.bias


class Norm(nn.Module):
    """A multi-function normalization layer with noise and bias options"""
    def __init__(self, d_model, normalization, noise, bias=False):
        super().__init__()
        
        self.norm = RMSNorm(d_model, bias) if normalization else nn.Identity()
        self.noise = Noise(noise)
        
    def forward(self, x):
        return self.noise(self.norm(x))