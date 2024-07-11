from torch.utils.data import Dataset
from torchvision import datasets
from einops import rearrange, einsum
import torch
from torch import Tensor

class MNIST(Dataset):
    """A dataset helper class which puts MNIST on the GPU. This *drastically* speeds up training."""
    def __init__(self, train=True, download=False, device="cuda"):
        dataset = datasets.MNIST(root='./data', train=train, download=download)
        x = dataset.data.float().to(device) / 255.0
        
        self.x = rearrange(x, "batch width height -> batch (width height)")
        self.y = dataset.targets.to(device)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)


def eigen(model, out: int | Tensor, *args) -> None:
    """
    This is a helper function to compute the eigenvector decomposition in a chained manner.
    Using this interface, you can access the eigenvectors of the model at any depth.
    The syntax is ``eigen(model, 0, -1)`` to get the positive eigenvecs of the first positive eigenvec for output 0.
    Currently this doesn't support arbitrary output vectors but that's not hard to add.
    """
    assert len(args) == model.w_b.size(0) - 1, "Index must have same size as model depth"

    if isinstance(out, int):
        out = model.w_u[out]
    elif isinstance(out, Tensor):
        out = einsum(model.w_u, out, "out hid, out -> hid")
    else:
        raise ValueError("Invalid output type, should be int or Tensor")
    
    for i in range(0, 1+len(args)):
        if i > 0: out = vecs[:, args[i-1]]
        
        l, r = model.w_b[-(i+1)].unbind()
        q = einsum(out, l, r, "out, out in1, out in2 -> in1 in2")
        q = 0.5 * (q + q.mT)
        
        vals, vecs = torch.linalg.eigh(q)
    
    vecs = einsum(vecs, model.w_e, "emb batch, emb inp -> batch inp")
    return vals, vecs