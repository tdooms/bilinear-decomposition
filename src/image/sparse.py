import torch
from torch import nn

from transformers import PretrainedConfig, PreTrainedModel
from einops import einsum


class Config(PretrainedConfig):
    def __init__(
        self,
        rank: int = 64,
        d_input: int = 784,
        d_output: int = 10,
        seed: int = 42,
        **kwargs,
    ):
        self.rank = rank
        self.d_input = d_input
        self.d_output = d_output
        self.seed = seed
        super().__init__(**kwargs)


class Model(PreTrainedModel):
    config_class = Config

    def __init__(self, config):
        super().__init__(config)
        torch.manual_seed(config.seed)

        self.left = nn.Parameter(torch.randn(config.d_input, config.rank) / config.d_input**0.5)
        self.right = nn.Parameter(torch.randn(config.d_input, config.rank) / config.d_input**0.5)
        self.down = nn.Parameter(torch.randn(config.d_output, config.rank) / config.d_output**0.5)

    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(Config(*args, **kwargs))

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return ((x @ self.left) * (x @ self.right)) @ self.down.T

    def tensor(self):
        """Symmetrized CPD tensor: B[c,i,j] = 0.5 * Σ_r (L[i,r]R[j,r] + L[j,r]R[i,r]) D[c,r]."""
        t = einsum(self.down, self.left, self.right, "c r, i r, j r -> c i j")
        return 0.5 * (t + t.mT)

    def similarity(self, original):
        """Cosine similarity between this CPD tensor and the original model's tensor."""
        wl, wr = original.w_lr[0].unbind()
        wl, wr = wl @ original.w_e, wr @ original.w_e
        target = einsum(original.w_u, wl, wr, "c o, o i, o j -> c i j")
        target = 0.5 * (target + target.mT)
        pred = self.tensor()
        return einsum(target, pred, "c i j, c i j ->") / (target.norm() * pred.norm())

    def decompose(self):
        """Return normalized (l+r, l-r, d, sigma) sorted by sigma."""
        sigma = self.left.norm(dim=0) * self.right.norm(dim=0) * self.down.norm(dim=0)
        idx = sigma.argsort(descending=True)
        plus = (self.left.data + self.right.data)[:, idx]
        minus = (self.left.data - self.right.data)[:, idx]
        down = self.down.data[:, idx]
        plus = plus / plus.norm(dim=0, keepdim=True).clamp(min=1e-8)
        minus = minus / minus.norm(dim=0, keepdim=True).clamp(min=1e-8)
        down = down / down.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return plus, minus, down, sigma[idx]
