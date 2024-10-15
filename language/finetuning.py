from einops import einsum
import torch
from torch.utils.data import DataLoader
import gc
import numpy as np
from language.utils import Sight
from transformers import TrainerCallback
from torch import nn

def _regression_metrics(a, b, x):
    b_pred = einsum(a, x, "... b i, ... i o -> ... b o")
    residuals = b - b_pred

    ss_total = (b - b.mean(-1, keepdim=True)).pow(2).sum()
    ss_residual = residuals.pow(2).sum()
    r_squared = 1 - (ss_residual / ss_total)
    
    return dict(r_squared=r_squared.item())

def exp_decay(x, k=1.05):
    return min(x * np.e ** (k - k * x), 1.0)

def logistic_decay(x, d=4):
    return 1 / (1 + np.exp(-2*d*x + d))

def smootherstep_decay(x, s=0.5):
    return (6*x**5 - 15*x**4 + 10*x**3)**s

def no_decay(x):
    return 1e-5

class GateReplacer:
    def extract(self, sight, batch):
        with sight.trace(batch["input_ids"], scan=False, validate=False):
            g_inp = [layer.mlp.w.gate.input.save() for layer in sight.transformer.h]
            g_out = [layer.mlp.w.gate.output.save() for layer in sight.transformer.h]
        return torch.stack(g_inp), torch.stack(g_out)
        
    def overwrite(self, model, x):
        for layer, x1 in zip(model.transformer.h, x):
            layer.mlp.w.gate = Interpolator(layer.mlp.w.gate, x1)

class NormReplacer:
    def extract(self, sight, batch):
        with sight.trace(batch["input_ids"], scan=False, validate=False):
            n1_inp = [layer.n1.input.save() for layer in sight.transformer.h]
            n1_out = [layer.n1.output.save() for layer in sight.transformer.h]
            
            n2_inp = [layer.n2.input.save() for layer in sight.transformer.h]
            n2_out = [layer.n2.output.save() for layer in sight.transformer.h]
            
            nf_inp = sight.transformer.n_f.input.save()
            nf_out = sight.transformer.n_f.output.save()
    
        return torch.stack([*n1_inp, *n2_inp, nf_inp]), torch.stack([*n1_out, *n2_out, nf_out])
    
    def overwrite(self, model, x):
        for layer, x1, x2 in zip(model.transformer.h, x[0:6], x[6:12]):
            layer.n1 = Interpolator(layer.n1, x1)
            layer.n2 = Interpolator(layer.n2, x2)

        model.transformer.n_f = Interpolator(model.transformer.n_f, x[-1])


def replace_components(model, dataset, which="norm", n_batches=1, compute_metrics=True):
    replacer = dict(norm=NormReplacer, gate=GateReplacer)[which]()
    sight = Sight(model)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    a, b = [], []
    
    for _, batch in zip(range(n_batches), loader):
        inp, out = replacer.extract(sight, batch)
        a.append(inp)
        b.append(out)
    
    a = torch.cat(a, dim=1).flatten(1, 2)
    b = torch.cat(b, dim=1).flatten(1, 2)
    x = torch.linalg.lstsq(a, b).solution
    
    if compute_metrics:
        print(_regression_metrics(a, b, x))
        
    del a, b
    gc.collect()
    torch.cuda.empty_cache()

    replacer.overwrite(model, x)
    return model

class Interpolator(nn.Module):
    def __init__(self, original, approximation) -> None:
        super().__init__()
        self.original = original
        
        self.linear = nn.Linear(*approximation.shape, bias=False)
        self.alpha = 0.0
        
        if approximation is not None:
            self.linear.weight = nn.Parameter(approximation.detach())
    
    def forward(self, x):
        return (1 - self.alpha) * self.original(x) + self.alpha * self.linear(x)

class Annealer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.0
        self.c = nn.Parameter(torch.ones(2048))
    
    def forward(self, x):
        beta = 1.0 - self.alpha
        return self.c * x * torch.sigmoid(beta * x)
    

def _set_alpha(module, alpha):
    if isinstance(module, Interpolator):
        module.alpha = alpha
    elif isinstance(module, Annealer):
        module.alpha = alpha
        
        
def replace_with_annealer(model, *args, **kwargs):
    for layer in model.transformer.h:
        layer.mlp.w.gate = Annealer(*args, **kwargs)
    return model

  
class AlphaDecay(TrainerCallback):
    def __init__(self, model, decay=exp_decay):
        self.model = model
        self.decay = decay
 
    def on_step_begin(self, args, state, control, **kwargs):
        fraction = state.global_step / state.max_steps
        self.model.apply(lambda x: _set_alpha(x, self.decay(fraction)))
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        fraction = state.global_step / state.max_steps
        logs['alpha'] = self.decay(fraction)
        