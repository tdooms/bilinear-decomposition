import torch
from einops import *

from sae import SAE
from tqdm import tqdm


class Tracer:
    """A class to encapsulate the analysis and utils for SAE interactions."""
    def __init__(self, model, layer, out: dict = dict(), inp: dict = dict(), use_encoder=True, device="cuda"):
        self.model = model
        self.layer = layer
        
        repo = f"{model.config.repo}-scope"
        
        inp = dict(name="mlp-in", expansion=4, k=30) | inp
        out = dict(name="mlp-out", expansion=4, k=30) | out
        
        self.inp = SAE.from_pretrained(repo, point=(inp["name"], layer), expansion=inp["expansion"], k=inp["k"]).to(device)
        self.out = SAE.from_pretrained(repo, point=(out["name"], layer), expansion=out["expansion"], k=inp["k"]).to(device)
        
        self.out_latents = self.out.w_enc.weight if use_encoder else self.out.w_dec.weight.T
        self.inp_latents = self.inp.w_dec.weight
    
    def q(self, idx: int | slice, project=False):
        """Compute the Q tensor for a given output index and optionally project onto the input latents."""
        # Computing these (honestly trivial) einsums can be *very* expensive.
        # Installing `opt_einsum` and setting a proper strategy speeds it up (500x on my machine).
        model, layer = self.model, self.layer
        res = torch.einsum("mi,mj,om,...o->...ij", model.w_l[layer], model.w_r[layer], model.w_p[layer], self.out_latents[idx])

        if project:
            res = torch.einsum("il,jk,...ij->...lk", self.inp_latents, self.inp_latents, res)
            return 0.5 * (res + res.mT)
        return res
    
    def compute(self, fn, batch_size=32, project=False, *args, **kwargs):
        """Execute certain functions batch-wise on the interactions of the B tensor, useful for getting summary statistics."""
        accum = []
        for start in tqdm(range(0, self.out_latents.size(0), batch_size)):
            matrices = self.q(slice(start, start + batch_size), project=project)
            accum.append(fn(matrices, *args, **kwargs))
            del matrices
            
        return torch.cat(accum)