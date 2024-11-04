import torch
from torch import nn
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
import itertools


@dataclass
class Config:
    """A configuration class for toy models."""
    n_inputs: int = 4
    n_hidden: int = 4
    n_outputs: int = 6
    
    n_epochs: int = 1_000
    batch_size: int = 512
    lr: float = 0.01
    seed: Optional[int] = 0
    device: str = "cpu"
    
    sparsity: float = 0.7
    bias: bool = False
    operation: dict = field(default_factory=dict)


class Model(nn.Module):
    """The base model for studying computation. This provides most default behavior for some toy computation tasks."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pairs = list(itertools.combinations(range(self.cfg.n_inputs), 2))
        
        # The bilinear (left/right) and head can be seen as an MLP.
        self.left = nn.Linear(self.cfg.n_inputs, self.cfg.n_hidden, bias=self.cfg.bias)
        self.right = nn.Linear(self.cfg.n_inputs, self.cfg.n_hidden, bias=self.cfg.bias)
        self.head = nn.Linear(self.cfg.n_hidden, self.cfg.n_outputs, bias=self.cfg.bias)

    def from_config(**kwargs):
        return Model(Config(**kwargs))
    
    def labels(self):
        return [f"{i} ∧ {j}" for i, j in self.pairs]
        
    def compute(self, x):
        operation = self.cfg.operation
        accum = torch.zeros(x.size(0), self.cfg.n_outputs, device=self.cfg.device)
        
        pairs = torch.tensor(self.pairs, device=self.cfg.device)
        left, right = pairs[:, 0], pairs[:, 1]
        
        accum += (x[..., left] ^ x[..., right]) * operation.get("xor", 0)
        accum += (~(x[..., left] ^ x[..., right])) * operation.get("xnor", 0)
        accum += (x[..., left] & x[..., right]) * operation.get("and", 0)
        accum += (~(x[..., left] & x[..., right])) * operation.get("nand", 0)
        accum += (x[..., left] | x[..., right]) * operation.get("or", 0)
        accum += (~(x[..., left] | x[..., right])) * operation.get("nor", 0)
        
        return accum
    
    def criterion(self, y_hat, x):
        y = self.compute(x)
        return ((y - y_hat) ** 2).mean()
    
    @property
    def w_l(self):
        return self.left.weight
    
    @property
    def w_r(self):
        return self.right.weight
    
    @property
    def w_p(self):
        return self.head.weight

    def forward(self, x):
        return self.head(self.left(x) * self.right(x))
        
    def generate_batch(self):
        return torch.rand(self.cfg.batch_size, self.cfg.n_inputs, device=self.cfg.device) > self.cfg.sparsity

    def fit(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.n_epochs)
        
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
        
        losses = []

        torch.set_grad_enabled(True)
        for _ in tqdm(range(self.cfg.n_epochs)):
            features = self.generate_batch()
            y_hat = self(features.float())
            loss = self.criterion(y_hat, features)
            losses += [loss.item()]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        torch.set_grad_enabled(False)
        return losses