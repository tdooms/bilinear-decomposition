import torch
from sae.sae import SAE, Point
from sae.samplers import MultiSampler
from einops import *
from datasets import load_dataset, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, HfApi
from safetensors.torch import save_file
from safetensors import safe_open
from language import Sight
import os


class Visualizer:
    def __init__(self, model, sae):
        path = hf_hub_download(repo_id=f"{model.config.repo}-scope", filename="top-activations.safetensors")
        with safe_open(path, framework="pt", device="cpu") as f:
            self.top_activations = f.get_tensor(f"{sae.point.layer}-{sae.point.name}")
    
        self.sae = sae
        self.sight = Sight(model)
        self.dataset = load_dataset(f"tdooms/{model.config.dataset}-16k", split="train").with_format("torch")
    
    @staticmethod
    def create_dataset_subset(model):
        tokenized = model.dataset().map(model.tokenize, batched=True)
        input_ids = torch.stack([batch["input_ids"] for batch in tokenized.take(2**14)])
        
        dataset = Dataset.from_dict({"input_ids": input_ids})
        dataset.push_to_hub(f"tdooms/{model.config.dataset}-16k")
        
        return dataset
        
    @staticmethod
    def compute_max_activations(model, dataset=None, in_batch=16):
        dataset = Visualizer.create_dataset_subset(model) if dataset is None else dataset
        
        # TODO: this should be generalized by looking at all the SAEs in the repo
        points = [Point(name, layer) for layer in range(model.config.n_layer) for name in ["mlp-in", "mlp-out"]]
        saes = [SAE.from_pretrained(f"{model.config.repo}-scope", point=point, expansion=8, k=30).cuda() for point in points]
        
        # Hacky wacky
        points += [Point("resid-mid", 1)]
        saes += [SAE.from_pretrained(f"{model.config.repo}-scope", point=points[-1], expansion=4, k=30).cuda()]
        
        sampler = MultiSampler(Sight(model), points, dataset=dataset, d_model=model.config.d_model, n_ctx=512, in_batch=in_batch)

        total = 2**14 // in_batch
        tops = []

        for batch, _ in tqdm(zip(sampler, range(total)), total=total):
            top = [saes[i].encode(batch["activations"][:, i]).max(1).values for i, _ in enumerate(saes)]
            tops.append(top)

        transposed = list(zip(*tops))
        stacked = [torch.cat(t, dim=0) for t in transposed]
        idxs = [s.topk(dim=0, k=100).indices for s in stacked]
        tensors = {f"{p.layer}-{p.name}": idx for idx, p in zip(idxs, points)}

        save_file(tensors, "tmp.safetensors")
        HfApi().upload_file(path_or_fileobj="tmp.safetensors", path_in_repo="top-activations.safetensors", repo_id=f"{model.config.repo}-scope")
        os.remove("tmp.safetensors")
    
    @staticmethod
    def color_str(str, color, value):
        r, g, b = color
        pre = " " if str.startswith("▁") else ""
        return pre + str[len(pre):] if value == 0 else f"{pre}\033[48;2;{int(r)};{int(g)};{int(b)}m{str[len(pre):]}\033[0m"

    @staticmethod
    def color_line(line, colors, values, view):
        idx = values.argmax(dim=-1)
        start, end = max(0, idx + view.start), min(len(line), idx + view.stop)
        return "".join([Visualizer.color_str(line[i], colors[i], values[i]) for i in range(start, end)])

    def color_input_ids(self, input_ids, feature=0, view=range(-10, 20), dark=False):
        with torch.no_grad(), self.sight.trace(input_ids, validate=False, scan=False):
            features = self.sae.encode(self.sight[self.sae.point]).save()
        
        values = features[..., feature]
        tokens = [self.sight.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        
        maxes = values.max(dim=-1, keepdim=True).values
        denom = maxes.where(maxes > 0, torch.ones_like(maxes))
        normalized = (values / denom) * 0.6
        
        colors = plt.cm.magma(normalized.cpu())[..., :3] if dark else plt.cm.Blues(normalized.cpu())[..., :3]
        colors = (colors * 255).astype(int)
        
        for line, color, value in zip(tokens, colors, values):
            print(f"{value.max().item():<4.1f}:  {Visualizer.color_line(line, color, value, view)}")
    
    def show_logit_influence(self, feature):
        sims = einsum(self.sae.w_dec.weight[:, feature], self.sight.w_u, "d, b d -> b")
        pos, neg = sims.topk(k=5), sims.topk(k=5, largest=False)
        
        for idx, val in zip(pos.indices, pos.values):
            print(f"{self.sight.tokenizer.decode(idx)}: {val.item():.2f}", end=', ')
        
        for idx, val in zip(neg.indices, neg.values):
            print(f"{self.sight.tokenizer.decode(idx)}: {val.item():.2f}", end=', ')
        
    def __call__(self, *args, k=5, **kwargs):
        assert k <= 100, "Amount must be less than or equal to 100"
        
        # TODO: be somewhat smarter about this
        args = [x for arg in args for x in ([arg] if not isinstance(arg, (list, tuple)) else arg)]
        
        for feature in args:
            print(f"feature {feature}")
            input_ids = self.dataset["input_ids"][self.top_activations[:, feature]][:k]
            self.color_input_ids(input_ids, feature, **kwargs)
            print()

