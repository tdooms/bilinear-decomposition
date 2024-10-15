import torch
import pandas as pd
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from einops import *
from transformers.modeling_outputs import CausalLMOutput
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer, DataCollatorForLanguageModeling
from torch import Tensor
from jaxtyping import Float
import wandb
from transformers import TrainingArguments, Trainer
from shared.components import MLP, Norm


class Config(PretrainedConfig):
    def __init__(
        self,
        n_head: int = 4,
        n_layer: int= 4,
        n_ctx: int = 256,
        d_model: int = 4 * 64,
        d_hidden: int = 4 * 4 * 64,
        bilinear: bool = True,
        gate: str | None = None,
        bias: bool = False,
        normalization: bool = True,
        tokenizer: str = None,
        repo: str = None,
        **kwargs
    ):
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_hidden = d_hidden
        
        self.bilinear = bilinear
        self.gate = gate
        self.bias = bias
        self.normalization = normalization
        
        self.tokenizer = tokenizer
        self.repo = repo
        
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
    def __init__(self, dim: int, n_ctx: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q, k, device="cuda"):
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
        
        self.rotary = Rotary(config.d_model // config.n_head, config.n_ctx)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)
        self.mask = torch.tril(torch.ones(config.n_ctx, config.n_ctx))[None, None, :, :]
    
    def forward(self, x: Float[Tensor, "batch seq d_model"], attn_mask=None) -> Float[Tensor, "batch seq d_model"]:
        n_head = self.config.n_head
        
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'batch seq (n n_head d_head) -> n batch n_head seq d_head', n=3, n_head=n_head).unbind(dim=0)
        q, k = self.rotary(q, k)
        
        if self.training:
            z = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[:, None, None, :], dropout_p=0.0, is_causal=True)
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
        self.scale = 1.0 / ((2.0 * config.n_layer) ** 0.5)
        
        self.attn = Attention(config)
        self.mlp = MLP(config.d_model, config.d_hidden, bilinear=config.bilinear, gate=config.gate, bias=config.bias)
        
        self.n1 = Norm(config.normalization)
        self.n2 = Norm(config.normalization)
    
    def forward(self, x, attn_mask=None):
        x = x + self.scale * self.attn(self.n1(x), attn_mask)
        x = x + self.mlp(self.n2(x))
        return x
        

class Transformer(PreTrainedModel):
    def __init__(self, config: Config, tokenizer: AutoTokenizer):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(tokenizer.vocab_size, config.d_model),
            h = nn.ModuleList([Layer(config) for _ in range(config.n_layer)]),
            n_f = Norm(config.normalization)
        ))
        
        self.lm_head = nn.Linear(config.d_model, tokenizer.vocab_size, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
        x = self.transformer.wte(input_ids)
        
        for layer in self.transformer.h:
            x = layer(x, attention_mask.bool() if attention_mask is not None else None)
    
        x = self.transformer.n_f(x)
        logits = self.lm_head(x)
        
        if labels is None:
            return CausalLMOutput(logits=logits)
        else:
            shifted_labels = labels[..., 1:].contiguous()
            shifted_logits = logits[..., :-1, :].contiguous()
            
            loss = self.criterion(shifted_logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
            return CausalLMOutput(loss=loss, logits=logits)
    
    @staticmethod
    def get_tokenizer(name):
        name, pad = {
            "ts-4096": ("tdooms/ts-tokenizer-4096", "[EOS]"),
            "mistral": ("mistral-community/Mixtral-8x22B-v0.1", "</s>"),
            "gpt2": ("openai-community/gpt2", "<|endoftext|>")
        }[name]
        
        return AutoTokenizer.from_pretrained(name, pad_token=pad, padding_side="right")
        # return AutoTokenizer.from_pretrained(name, pad_token=pad)
    
    @classmethod
    def from_pretrained(cls, repo, device='cuda', **kwargs):
        config = Config.from_pretrained(repo, repo=repo)
        tokenizer = cls.get_tokenizer(config.tokenizer)
        
        return super(Transformer, Transformer).from_pretrained(repo, config=config, tokenizer=tokenizer, device_map=device, **kwargs)
    
    @classmethod
    def from_config(cls, *args, **kwargs):
        config = Config(*args, **kwargs)
        return Transformer(config, cls.get_tokenizer(config.tokenizer))
    
    def tokenize(self, dataset):
        return self.tokenizer(dataset["text"], truncation=True, padding=True, max_length=self.config.n_ctx)
    
    @property
    def collator(self, **kwargs):
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
    
    @property
    def w_qkv(self):
        qkv = torch.stack([self.transformer.h[i].attn.qkv.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(qkv, "n_layer (n_proj n_head d_head) d_model -> n_proj n_layer n_head d_head d_model", n_proj=3, n_head=self.config.n_head)
    
    @property
    def w_lr(self):
        lr = torch.stack([self.transformer.h[i].mlp.w.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(lr, "n_layer (n_proj d_hidden) d_model -> n_proj n_layer d_hidden d_model", n_proj=2)
    
    @property
    def w_l(self):
        return self.w_lr[0]
    
    @property
    def w_r(self):
        return self.w_lr[1]
    
    @property
    def w_p(self):
        return torch.stack([self.transformer.h[i].mlp.p.weight for i in range(self.config.n_layer)], dim=0)
    
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
    
    def fit(self, train, project, lr=1e-3, wd=0.1, batch_size=128, callbacks=None, **kwargs):
        training_args = TrainingArguments(
            output_dir="_checkpoints",
            learning_rate=lr,
            warmup_steps=50,
            logging_steps=10,
            adam_beta1=0.9,
            adam_beta2=0.95,
            optim="adamw_torch_fused",
            per_device_train_batch_size=batch_size,
            weight_decay=wd,
            do_eval=False,
            report_to="wandb" if project else "none",
            remove_unused_columns=True,
            **kwargs
        )

        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=train,
            tokenizer=self.tokenizer,
            data_collator=self.collator,
            callbacks=callbacks,
        )
        
        if project: wandb.init(project=project, config=self.config)
        trainer.train()
        if project: wandb.finish()
        
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
            self.transformer.h[0].mlp.p.weight.numel()
        ]
        
        dims = [
            "",
            f"{self.config.d_model} x {self.tokenizer.vocab_size}",
            f"{self.tokenizer.vocab_size} x {self.config.d_model}",
            f"3 x {self.config.d_model} x {self.config.d_model}",
            f"{self.config.d_model} x {self.config.d_model}",
            f"2 x {self.config.d_hidden} x {self.config.d_model}",
            f"{self.config.d_model} x {self.config.d_hidden}"
        ]
        
        return pd.DataFrame(dict(name=names, parameters=parameters, dimensions=dims))
