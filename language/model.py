from typing import Optional

import torch
import pandas as pd
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from einops import *
from transformers.modeling_outputs import CausalLMOutput
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer, DataCollatorForLanguageModeling
from torch import Tensor
from jaxtyping import Float
from datasets import load_dataset
import wandb
from transformers import TrainingArguments, Trainer

from language.utils import UBE, Vocab
from shared import MLP, Norm


class Config(PretrainedConfig):
    def __init__(
        self,
        n_vocab: int = 4096,
        n_head: int = 4,
        n_layer: int= 4,
        n_ctx: int = 256,
        d_model: int = 4 * 64,
        d_hidden: int = 4 * 4 * 64,
        bilinear: bool = True,
        gate: bool = False,
        normalization: bool = True,
        modifier: str | None = None,
        noise: float | None = None,
        **kwargs
    ):
        self.n_vocab = n_vocab
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_hidden = d_hidden
        
        self.bilinear = bilinear
        self.gate = gate
        self.normalization = normalization
        self.noise = noise
        
        self.modifier = modifier
        
        super().__init__(**kwargs)
    
    @property
    def d_head(self):
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        return self.d_model // self.n_head


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Rotary(torch.nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q, k, device):
        seq_len = q.size(-2)
        
        # Using isinstance does not work, this is necessary for NNSight compatibility
        if (seq_len != self.seq_len_cached) or type(self.cos_cached) != torch.Tensor:
            self.seq_len_cached = seq_len
            
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        
        return apply_rotary_pos_emb(q, k, self.cos_cached, self.sin_cached)
    

class Attention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        
        self.rotary = Rotary(config.d_model // config.n_head)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)
        self.mask = torch.tril(torch.ones(config.n_ctx, config.n_ctx))[None, None, :, :]
    
    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        n_head = self.config.n_head
        
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'batch seq (n n_head d_head) -> n batch n_head seq d_head', n=3, n_head=n_head).unbind(dim=0)
        q, k = self.rotary(q, k, q.device)
        
        if self.training:
            z = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            scores = einsum(q, k, "batch n_head seq_q d_head, batch n_head seq_k d_head -> batch n_head seq_q seq_k")
            scores = scores / (torch.tensor(q.size(-1), device=x.device).sqrt())
            scores = scores.masked_fill(self.mask[:,:,:x.size(1),:x.size(1)].to(scores.device) == 0, -torch.inf)
            
            pattern = self.softmax(scores)
            z = einsum(pattern, v, "batch n_head seq_q seq_k, batch n_head seq_k d_head -> batch n_head seq_q d_head")
            
        z = rearrange(z, 'batch n_head seq d_head -> batch seq (n_head d_head)')
        return self.o(z)
   

class Layer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.attn = Attention(config)
        self.mlp = MLP(config.d_model, config.d_hidden, bilinear=config.bilinear, gate=config.gate)
        
        self.n1 = Norm(config.d_model, config.normalization, config.noise)
        self.n2 = Norm(config.d_model, config.normalization, config.noise)
    
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x
        

class Transformer(PreTrainedModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        
        self.url = "tdooms/TinyStories"
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.url}-{config.n_vocab}", pad_token="[EOS]")
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.n_vocab, config.d_model),
            h = nn.ModuleList([Layer(config) for _ in range(config.n_layer)]),
            n_f = Norm(config.d_model, config.normalization, config.noise)
        ))
        
        self.lm_head = nn.Linear(config.d_model, config.n_vocab, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids=None, labels=None, **kwargs):
        x = self.transformer.wte(input_ids)
        
        for layer in self.transformer.h:
            x = layer(x)
    
        x = self.transformer.n_f(x)
        logits = self.lm_head(x)
        
        if labels is None:
            return CausalLMOutput(logits=logits)
        else:
            shifted_labels = labels[..., 1:].contiguous()
            shifted_logits = logits[..., :-1, :].contiguous()
            
            loss = self.criterion(shifted_logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
            return CausalLMOutput(loss=loss, logits=logits)
    
    def center_unembed(self):
        """Centers the unembedding layer of the model along the input dim.

        Returns:
            self: The model, also updated in place.
        """
        self.lm_head.weight = nn.Parameter(self.lm_head.weight - self.lm_head.weight.mean(dim=1, keepdim=True))
        return self
    
    def fold_norms(self):
        for layer in self.transformer.h:
            layer.attn.qkv.weight.data = layer.attn.qkv.weight.data * layer.n1.weight.data[None, :]
            layer.mlp.w.weight.data = layer.mlp.w.weight.data * layer.n2.weight.data[None, :]
            
            layer.n1.weight.data = torch.ones_like(layer.n1.weight.data)
            layer.n2.weight.data = torch.ones_like(layer.n2.weight.data)
        
        self.lm_head.weight.data = self.lm_head.weight.data * self.transformer.n_f.weight.data
        self.transformer.n_f.weight.data = torch.ones_like(self.transformer.n_f.weight.data)
        
        return self
    
    @classmethod
    def from_config(csl, *args, **kwargs):
        config = Config(*args, **kwargs)
        return Transformer(config)
        
    @classmethod
    def from_pretrained(cls, n_layer=1, d_model=512, modifier=None, device='cuda', **kwargs):
        url = "tdooms/TinyStories"
        name = f"{url}-{n_layer}-{d_model}"
        if modifier is not None: name += f"-{modifier}"
        
        config = Config.from_pretrained(name)
        return super(Transformer, Transformer).from_pretrained(name, config=config, device_map=device, **kwargs)
    
    def dataset(self, tokenized: bool = False, split: str | None = None):
        location = f"{self.url}-tokenized-{self.config.n_vocab}" if tokenized else self.url
        return load_dataset(location, split=split)
    
    def tokenize(self, dataset):
        return self.tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)
    
    @property
    def sight(self):
        return StoriesSight(self, tokenizer=self.tokenizer)
    
    @property
    def vocab(self):
        return Vocab(self.tokenizer)
    
    @property
    def w_qkv(self):
        qkv = torch.stack([self.transformer.h[i].attn.qkv.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(qkv, "n_layer (n_proj n_head d_head) d_model -> n_proj n_layer n_head d_head d_model", n_proj=3, n_head=self.config.n_head)
    
    @property
    def w_lr(self):
        lr = torch.stack([self.transformer.h[i].mlp.w.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(lr, "n_layer (n_proj d_hidden) d_model -> n_proj n_layer d_hidden d_model", n_proj=2)
    
    @property
    def b(self):
        w_l, w_r, w_p = self.w_l.detach(), self.w_r.detach(), self.w_p.detach()
        b = einsum(w_l, w_r, w_p, "... hid in1, ... hid in2, ... out hid -> ... out in1 in2")
        return 0.5 * (b + b.mT)
    
    @property
    def w_l(self):
        return self.w_lr[0]
    
    @property
    def w_r(self):
        return self.w_lr[1]
    
    @property
    def w_p(self):
        return torch.stack([self.transformer.h[i].mlp.o.weight for i in range(self.config.n_layer)], dim=0)
    
    @property
    def w_q(self):
        return self.w_qkv[0]
    
    @property
    def w_k(self):
        return self.w_qkv[1]
    
    @property
    def w_v(self):
        return self.w_qkv[2]
    
    @property
    def w_o(self):
        o = torch.stack([self.transformer.h[i].attn.o.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(o, "n_layer d_model (n_head d_head) -> n_layer n_head d_model d_head", n_head=self.config.n_head)
    
    @property
    def w_e(self):
        return self.transformer.wte.weight.T
    
    @property
    def w_u(self):
        return self.lm_head.weight
    
    @property 
    def qk(self):
        return self.w_q.mT @ self.w_k
    
    @property 
    def ov(self):
        return self.w_o @ self.w_v
    
    @property
    def ube(self):
        return UBE(self)
    
    @property
    def name(self):
        modifier = "" if self.config.modifier is None else f"-{self.config.modifier}"
        return f"{self.url}-{self.config.n_layer}-{self.config.d_model}{modifier}"
    
    @property
    def collator(self, **kwargs):
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
    
    def fit(self, log=True, lr=1e-3, wd=0.2, batch_size=128, epochs=1, eval_steps=10_000, **kwargs):
        dataset = self.dataset(tokenized=True)
        train, validation = dataset["train"], dataset["validation"]
        
        training_args = TrainingArguments(
            # use_cpu=True,
            output_dir="_checkpoints",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=wd,
            do_eval=True,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            report_to="wandb" if log else None,
            remove_unused_columns=False,
            **kwargs
        )

        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=train,
            eval_dataset=validation,
            tokenizer=self.tokenizer,
            data_collator=self.collator,
        )
        
        if log: wandb.init(project="stories", config=self.config)
        trainer.train()
        if log: wandb.finish()
        
        return trainer

    @torch.no_grad()
    def generate(self,prompt: str = "", max_length: int | None = None, temperature: float = 1.0, top_k: int | None = None):
        """The default naive generation method for the model.

        Args:
            prompt (str, optional): the prompt. Defaults to "".
            max_length (Optional[int], optional): the generation length, is always capped to the ctx length. Defaults to None.
            temperature (float, optional): a scale in the logits when sampling, makes outputs more volatile. Defaults to 1.0.
            top_k (Optional[int], optional): the number of top tokens to choose from. Defaults to None.

        Returns:
            str: a string with the generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        max_length = min(max_length or self.config.n_ctx, self.config.n_ctx - input_ids.size(-1) - 1)
        
        for _ in range(max_length):
            logits = self(input_ids).logits
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -torch.inf

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)

        out = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return out.replace(" ##", "")
    
    def summary(self) -> pd.DataFrame:
        """Summarizes the model's architecture and parameter count into a dataframe.

        Returns:
            pd.Dataframe: the summary dataframe
        """
        names = [
            "total", 
            "emb.tok", 
            "head", 
            "attn.qkv", 
            "attn.out", 
            "mlp.bilin", 
            "mlp.out"
        ]
        
        parameters = [
            sum(p.numel() for p in self.parameters()),
            self.transformer.wte.weight.numel(),
            self.lm_head.weight.numel(),
            self.transformer.h[0].attn.qkv.weight.numel(),
            self.transformer.h[0].attn.o.weight.numel(),
            self.transformer.h[0].mlp.w.weight.numel(),
            self.transformer.h[0].mlp.o.weight.numel()
        ]
        
        dims = [
            "",
            f"{self.config.d_model} x {self.config.n_vocab}",
            f"{self.config.n_vocab} x {self.config.d_model}",
            f"3 x {self.config.d_model} x {self.config.d_model}",
            f"{self.config.d_model} x {self.config.d_model}",
            f"2 x {self.config.d_hidden} x {self.config.d_model}",
            f"{self.config.d_model} x {self.config.d_hidden}"
        ]
        
        return pd.DataFrame(dict(name=names, parameters=parameters, dimensions=dims))
