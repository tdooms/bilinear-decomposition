import torch
from einops import *
from torch import nn
from tqdm import tqdm
from collections import namedtuple
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from shared.components import Bilinear
from huggingface_hub import HfApi
from safetensors.torch import save_model, load_model
from huggingface_hub import hf_hub_download
import json
import os
import shutil
from language.utils import Sight
from sae.samplers import ShuffleSampler

Point = namedtuple('Point', ['name', 'layer'])


class ConstrainedAdam(torch.optim.Adam):
    def __init__(self, params, constrained_params, lr, dim=-2):
        super().__init__(params, lr=lr)
        self.dim = dim
        self.constrained_params = list(constrained_params)
        
    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=self.dim, keepdim=True)
                p.grad -= (p.grad * normed_p).sum(dim=self.dim, keepdim=True) * normed_p
        
        super().step(closure=closure)
        
        with torch.no_grad():
            for p in self.constrained_params:
                p /= p.norm(dim=self.dim, keepdim=True)


class SAEConfig:
    def __init__(
        self,
        point: Point | None = None,     # Hook point of the SAE on the model
        target: Point | None = None,    # Point at which to reconstruct in the model
        lr: float = 1e-4,               # Learning rate
        in_batch: int = 32,             # batch size for the transformer
        out_batch: int = 4096,          # batch size for the SAE
        n_batches: int = 2**8,          # Number of batches to buffer
        n_buffers: int | None = None,   # Number of buffers to train for, defaults to the full dataset
        d_model: int | None = None,     # Model dimension at the hook point
        n_ctx: int = 256,               # Context length
        expansion: int = 4,             # SAE expansion factor
        k: int = 30,                    # Top-k sparsity, no other sparsity is supported
        val_steps: int = 100,           # Validation interval
        dead_thresh: int = 2,           # Steps before a neuron is considered dead
        normalize: float | None = None, # A normalization value for all inputs
        init_scale: float = 1.0,        # Encoder initialization scale
        bilinear_encoder: bool = False, # Whether to use a bilinear encoder
        token_lookup: bool = False,     # Whether to use a token lookup table
        decoder_decay: float = 0.0,     # Decoder weight decay factor
        normalize_decoder: bool = True, # Whether to normalize the decoder weights
        encoder_bias: bool = False,     # Whether to use a bias in the encoder
        tag: str | None = None,         # Tag for the model
        **kwargs
    ):
        assert point is not None, "A hook point must be provided"
        assert d_model is not None, "Model dimension must be provided"
        
        # SAE related parameters
        self.point = Point(*point) if isinstance(point, list) or isinstance(point, tuple) else point
        self.target = Point(*target) if isinstance(target, list) or isinstance(target, tuple) else target
        self.expansion = expansion
        self.lr = lr
        
        # Data related parameters
        self.in_batch = in_batch
        self.out_batch = out_batch
        self.n_buffers = n_buffers
        self.n_batches = n_batches
        
        # Model related parameters
        self.d_model = d_model
        self.n_ctx = n_ctx
        
        # Sparsity related parameters
        self.k = k
        
        # Metric related parameters
        self.val_steps = val_steps
        self.dead_thresh = dead_thresh
        
        # Setup related parameters
        self.normalize = normalize
        
        # Encoder related parameters
        self.init_scale = init_scale
        self.bilinear_encoder = bilinear_encoder
        self.encoder_bias = encoder_bias
        
        # Decoder related parameters
        self.token_lookup = token_lookup
        self.decoder_decay = decoder_decay
        self.normalize_decoder = normalize_decoder

        # Miscellaneous parameters
        self.tag = tag
        self.kwargs = kwargs
    
    @property
    def d_features(self):
        return self.expansion * self.d_model
    
    @property
    def n_tokens(self):
        return self.n_ctx * self.in_batch * self.n_batches * self.n_buffers
    
    @property
    def name(self):
        ret = f"{self.point.layer}-{self.point.name}-x{self.expansion}-k{self.k}".replace("_", "-")
        return f"{ret}-{self.tag}" if self.tag else ret

class SAE(nn.Module):
    """And end-to-end top-k sparse autoencoder"""
    
    def __init__(self, config) -> None:
        super().__init__()
        
        self.config = config
        self.point = config.point
        self.target = config.target if config.target is not None else config.point
        self.d_model = config.d_model
        self.d_features = config.d_features
        self.n_ctx = config.n_ctx
        
        self.inactive = torch.zeros(self.d_features)
        
        self.w_dec = nn.Linear(self.d_features, self.d_model, bias=False)
        self.w_dec.weight.data /= torch.norm(self.w_dec.weight.data, dim=-2, keepdim=True)
        
        if config.bilinear_encoder:
            self.w_enc = Bilinear(self.d_model, self.d_features, bias=config.encoder_bias)
        else:
            self.w_enc = nn.Linear(self.d_model, self.d_features, bias=config.encoder_bias)
            self.w_enc.weight.data = self.w_dec.weight.data.T.contiguous().clone()
        
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))
        self.register_parameter('b_dec', self.b_dec)

    def preprocess(self, x):
        """Performs any operation before everything else"""
        if self.config.normalize is not None:
            return x / self.config.normalize
        return x
    
    def postprocess(self, x):
        """Performs any operation after everything else"""
        if self.config.normalize is not None:
            return x * self.config.normalize
        return x
        
    def decode(self, x):
        """Standard decoder operation"""
        return self.w_dec(x) + self.b_dec
    
    def encode(self, x):
        """Top-k encoder operation"""
        x_hid = self.w_enc(x - self.b_dec)
        indices = x_hid.topk(self.config.k, dim=-1).indices

        mask = torch.zeros_like(x_hid)
        mask.scatter_(-1, indices, 1)
        
        return x_hid * mask
    
    def forward(self, x):
        """Chained encoder-decoder operation, returning the hidden state as well"""
        x_hid = self.encode(x)
        x_hat = self.decode(x_hid)
        return x_hat, x_hid
    
    @staticmethod
    def from_pretrained(repo_id, point, expansion, k):
        config = SAEConfig(point=point, expansion=expansion, k=k, d_model=0)

        config_path = hf_hub_download(repo_id=repo_id, filename=f"{config.name}/config.json")
        model_path = hf_hub_download(repo_id=repo_id, filename=f"{config.name}/model.safetensors")

        sae = SAE.from_config(**json.load(open(config_path)))
        load_model(sae, model_path)
        return sae
    
    @staticmethod
    def from_config(*args, **kwargs):
        return SAE(SAEConfig(*args, **kwargs))
    
    def metrics(self, x, x_hid, x_hat):
        """Computes all interesting metrics for the model"""
        self.inactive[rearrange(x_hid, "... d -> (...) d").sum(0) > 0] = 0
        mse = (x - x_hat).pow(2).mean()
        
        metrics = dict()
        metrics["dead_fraction"] = (self.inactive > self.config.dead_thresh).float().mean()
        metrics["mean_inactive"] = self.inactive.mean()
        
        metrics["mse"] = mse
        metrics["nmse"] = (mse / x.pow(2).mean())
        
        metrics["l1"] = x_hid.sum(-1).mean()
        
        self.inactive += 1
        return metrics
    
    def push_to_hub(self, repo_id, tmp_dir="tmp"):
        os.makedirs(tmp_dir, exist_ok=True)
        json.dump(vars(self.config), open(f'{tmp_dir}/config.json', 'w'), indent=2)
        save_model(self, f'{tmp_dir}/model.safetensors')

        HfApi().upload_folder(folder_path=tmp_dir, path_in_repo=self.config.name, repo_id=repo_id)
        shutil.rmtree(tmp_dir)
            
    def _reconstruct_step(self, batch):
        """Sample and reconstruct a batch, returning the local loss"""
        
        x_hat, x_hid = self(batch)
        metrics = self.metrics(batch, x_hid, x_hat)
        return metrics["mse"], metrics
    
    def _e2e_step(self, sight, batch):
        """Sample and patch in the reconstruction, retuning the global loss"""
        
        with torch.no_grad(), sight.trace(batch, validate=False, scan=False):
            x = sight[self.point].save()
            clean = sight.output.loss.save()
        
        x_hat, x_hid = self(x)
        
        with torch.no_grad(), sight.trace(batch, validate=False, scan=False):
            sight[self.target][:] = x_hat
            loss = sight.output.loss.save()
        
        with torch.no_grad(), sight.trace(batch, validate=False, scan=False):
            sight[self.target][:] = 0
            corrupt = sight.output.loss.save()
            
        metrics = self.metrics(x, x_hid, x_hat)
        return clean, corrupt, loss, metrics
    
    def fit(self, model, train, validate, project: str | None = None):
        """A general fit function with a default training loop"""
        if project: wandb.init(project=project, config=self.config)
        
        sight = Sight(model)
        sampler = ShuffleSampler(sight, train, **vars(self.config))
        
        parameters = [
            dict(params=list(self.w_enc.parameters()) + [self.b_dec], weight_decay=0.0),
            dict(params=self.w_dec.parameters(), weight_decay=self.config.decoder_decay)
        ]
        
        total = self.config.n_buffers
        
        lr = self.config.lr
        optimizer = ConstrainedAdam(parameters, [self.w_dec.weight], lr=lr) if self.config.normalize_decoder else Adam(parameters, lr=lr)
        scheduler = CosineAnnealingLR(optimizer, total, 2e-5)
        
        pbar = tqdm(zip(range(total), sampler), total=total, smoothing=0.001)
        added = float("nan")

        for idx, batch in pbar:
            loss, metrics = self._reconstruct_step(batch["activations"])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if validate is not None and idx % self.config.val_steps == 0:
                clean, corrupt, patched, _ = self._e2e_step(sight, validate)
                added = (patched.item() - clean.item()) / clean.item()
                metrics['val/added'] = added
                metrics['val/patched'] = patched.item()
                metrics['val/recovered'] = 1 - ((patched.item() - clean.item()) / (corrupt.item() - clean.item()))
            pbar.set_description(f"L1: {metrics['l1']:.4f}, NMSE: {metrics['nmse']:.4f}, added: {added:.4f}")
            
            if project: wandb.log(metrics)
        if project: wandb.finish()