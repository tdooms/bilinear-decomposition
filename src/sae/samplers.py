import torch
from einops import *
from torch.utils.data import DataLoader
from abc import abstractmethod
import gc


class BaseSampler:
    """This class is a dynamic dataset that samples activations from a model on the fly, buffering and shuffling them."""
    def __init__(self, dataset, d_model, n_ctx=256, n_batches=1, in_batch=32, extra=[], device="cuda", **kwargs):
        self.extra, self.device = extra, device
        self.n_ctx, self.d_model, self.n_batches, self.in_batch = n_ctx, d_model, n_batches, in_batch
        
        self.loader = DataLoader(dataset, batch_size=in_batch)
        self.iter = iter(self.loader)
        
        self.start, self.end = 0, 0
        self.buffer = None
    
    @abstractmethod
    def _extract(self, input_ids):
        raise NotImplementedError
    
    def _buffer(self):
        input_ids = torch.empty(self.n_batches, self.in_batch, self.n_ctx, dtype=torch.long, device=self.device)
        activations = torch.empty(self.n_batches, self.in_batch, *self.extra, self.n_ctx, self.d_model, device=self.device)
        
        for i, batch in zip(range(self.n_batches), self.iter):
            inputs = batch["input_ids"][..., :self.n_ctx]
            activations[i] = self._extract(inputs)
            input_ids[i] = inputs
        
        return dict(activations=activations, input_ids=input_ids)


class ShuffleSampler(BaseSampler):
    """A sampler based on NNsight, using the Sight helper"""
    def __init__(self, sight, train, d_model, point, out_batch = 2**12, **kwargs):
        super().__init__(train,  d_model, **kwargs)
        self.sight, self.point, self.out_batch = sight, point, out_batch
    
    def _buffer(self):
        buffer = super()._buffer()
        
        buffer = {key: rearrange(value, "n b ... -> (n b) ...") for key, value in buffer.items()}
        perm = torch.randperm(buffer["input_ids"].size(0))
        
        buffer["input_ids"] = buffer["input_ids"][perm]
        buffer["activations"] = rearrange(buffer["activations"][perm], "... d -> (...) d")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return buffer
    
    @torch.no_grad()
    def _extract(self, input_ids):
        with torch.no_grad(), self.sight.trace(input_ids, validate=False, scan=False):
            saved = self.sight[self.point].save()
            self.sight[self.point].stop()
        return saved
    
    def __iter__(self):
        n_steps = (self.n_batches * self.in_batch * self.n_ctx) // self.out_batch
        step = 0
        
        while True:
            self.buffer = self._buffer() if (self.buffer is None or step == 0) else self.buffer

            start, end = step * self.out_batch, (step+1) * self.out_batch
            step = (step + 1) % n_steps
            
            yield {key: value[start:end] for key, value in self.buffer.items()}


class MultiSampler(BaseSampler):
    """A sampler based on NNsight, using the Sight helper"""
    def __init__(self, sight, points, **kwargs):
        super().__init__(**kwargs, extra=[len(points)])
        self.sight, self.points = sight, points
    
    @torch.no_grad()
    def _extract(self, input_ids):
        with torch.no_grad(), self.sight.trace(input_ids, validate=False, scan=False):
            saved = [self.sight[point].save() for point in self.points]
        return torch.stack(saved, dim=1)
    
    def __iter__(self):
        while True:
            yield {key: rearrange(value, "n b ... -> (b n) ...") for key, value in self._buffer().items()}
            
