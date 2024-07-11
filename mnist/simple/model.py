import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from transformers import PretrainedConfig, PreTrainedModel
from jaxtyping import Float
from tqdm import tqdm
from pandas import DataFrame
from einops import *

from shared import Linear, Bilinear


class Config(PretrainedConfig):
    def __init__(
        self,
        lr: float = 1e-3,           # learning rate
        wd: float = 0.5,            # weight decay
        input_noise: float = 0.0,   # input noise
        latent_noise: float = 0.0,  # latent noise
        epochs: int = 100,          # number of epochs
        batch_size: int = 2048,     # batch size
        d_hidden: int = 512,        # hidden dimension
        n_layer: int = 3,           # number of layers
        d_input: int = 784,         # input dimension
        d_output: int = 10,         # output dimension
        bias: bool = False,         # include model bias
        device: str = "cuda",  
        seed: int = 42,
        **kwargs
    ):
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.input_noise = input_noise
        self.latent_noise = latent_noise
        
        self.d_hidden = d_hidden
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_output = d_output
        self.bias = bias
        
        self.device = device
        self.seed = seed
        
        super().__init__(**kwargs)


class Model(PreTrainedModel):
    """A default model for the MNIST dataset with some helper functions for training and evaluation."""
    def __init__(self, config) -> None:
        super().__init__(config)
        torch.manual_seed(config.seed)
        
        # Define the model architecture
        self.embed = Linear(config.d_input, config.d_hidden, bias=False, noise=config.input_noise)
        
        self.blocks = nn.ModuleList([
            Bilinear(config.d_hidden, config.d_hidden, bias=config.bias, noise=config.latent_noise) 
            for _ in range(config.n_layer)
        ])
        
        self.head = Linear(config.d_hidden, config.d_output, bias=False, noise=config.latent_noise)
        
        # Define the loss and accuracy functions
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    
    def forward(self, x: Float[Tensor, "... inputs"]) -> Float[Tensor, "... outputs"]:
        x = self.embed(x)
        
        for layer in self.blocks:
            x = layer(x)
        
        return self.head(x)
    
    @property
    def w_e(self):
        return self.embed.weight.data
    
    @property
    def w_u(self):
        return self.head.weight.data
    
    @property
    def w_b(self):
        return torch.stack([rearrange(layer.weight.data, "(s o) h -> s o h", s=2) for layer in self.blocks])
    
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(Config(*args, **kwargs))

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        new = cls(Config(*args, **kwargs))
        new.load_state_dict(torch.load(path))
        return new
    
    def step(self, x, y):
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        return loss, accuracy
    
    def fit(self, train, test):
        """A default training function for the model. It records some metrics into a dataframe for later analysis."""
        torch.manual_seed(self.config.seed)
        
        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        
        # MNIST is small enough not to bother with a validation dataloader
        loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        test_x, test_y = test.x, test.y
        
        pbar = tqdm(range(self.config.epochs))
        history = []
        
        for _ in pbar:
            epoch = []
            for x, y in loader:
                loss, acc = self.train().step(x, y)
                epoch += [(loss.item(), acc.item())]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            val_loss, val_acc = self.eval().step(test_x, test_y)

            metrics = {
                "train/loss": sum(loss for loss, _ in epoch) / len(epoch),
                "train/acc": sum(acc for _, acc in epoch) / len(epoch),
                "val/loss": val_loss.item(),
                "val/acc": val_acc.item()
            }
            
            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in metrics.items()))
        
        return DataFrame.from_records(history, columns=['train/loss', 'train/acc', 'val/loss', 'val/acc'])
