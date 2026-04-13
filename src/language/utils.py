import torch
import pandas as pd
from einops import *
from transformers import AutoTokenizer
from bidict import bidict
from typing import List, Tuple
from torch import Tensor
from jaxtyping import Int
from collections import Counter
from nnsight import LanguageModel


class Vocab:
    def __init__(self, tokenizer: AutoTokenizer):
        self.vocab = bidict(tokenizer.vocab)
        self.tokenizer = tokenizer
    
    def label_top_weights(self, tensor, axes, k=10, largest=True, val_name="value"):
        """Utility function to relate the top-k weights in a tensor to their correct tokens.

        Args:
            tensor (Tensor): a data tensor corresponding to the weights. Must have same dims as tokens.
            axes (List[str]): names of the axes in the tensor.
            k (int, optional): number of top weights to consider. Defaults to 10.
            largest (bool, optional): take the largest or the smallest. Defaults to True.
            val_name (str, optional): custom name for the value column (weights). Defaults to "value".

        Returns:
            DataFrame: dataframe which token names and their corresponding weights.
        """
        top = torch.topk(tensor.flatten(), k=k, largest=largest)
        dims = torch.unravel_index(top.indices, tensor.size())
        
        data = {k: self[v.cpu()] for k, v in zip(axes, dims)}
        data[val_name] = top.values.cpu()
        
        return pd.DataFrame(data)
        
    def make_labels(self, prompt: str):
        """Generate tokens for a certain prompt and use a unicode trick to make them unique.

        Args:
            prompt (str): a prompt to generate tokens for.

        Returns:
            List[str]: a list of tokens with empty unicode characters to make them unique. 
        """
        tokens = self.tokenizer.tokenize(prompt)
        counts = [Counter(tokens[:i])[token] for i, token in enumerate(tokens)]
        empty = '‎'
        return ["BOS"] + [f"{tok} {empty * cnt}" for tok, cnt in zip(tokens, counts)] + ["EOS"]
    
    @property
    def tokens(self):
        """Gets the full list of tokens in the vocabulary.

        Returns:
            List[str]: The list of tokens in the vocabulary. 
        """
        
        return [self.inv[i] for i in range(len(self))]
    
    def __getitem__(self, key: str | int | List[int] | Int[Tensor, "indices"] | Tuple[int] | Tuple[str]):
        """Gets the token/index associated with the given key.

        Args:
            key (Union[str, int, List[int], List[str], Tensor]): The key to look up. Can be a handful of types.

        Raises:
            TypeError: If the key is not of type str, int, list, or Tensor.

        Returns:
            token, index, or list of tokens: The token(s) associated with the given key.
        """
        
        if isinstance(key, str):
            return self.vocab[key]
        elif isinstance(key, int):
            return self.inv[key]
        elif (isinstance(key, list), isinstance(key, tuple)) and all(isinstance(i, int) for i in key):
            return [self.inv[i] for i in key]
        elif (isinstance(key, list), isinstance(key, tuple)) and all(isinstance(i, str) for i in key):
            return [self[i] for i in key]
        elif isinstance(key, torch.Tensor):
            return [self.inv[i.item()] for i in key]
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")
    
    def __len__(self):
        return len(self.vocab)
    
    @property
    def inv(self):
        return self.vocab.inv
    
    @property
    def inverse(self):
        return self.vocab.inverse


class Sight(LanguageModel):
    """A helper class to more cleanly interface with NNsight."""
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, tokenizer=model.tokenizer, *args, **kwargs)
    
    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
             args = args[0]
        
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], int):
            point, layer = args
        else:
            raise ValueError("Invalid arguments, should be a tuple or two arguments.")
        
        point = point.replace("-", "_")
        
        return dict(
            resid_pre=self._envoy.transformer.h[layer].input,
            resid_mid=self._envoy.transformer.h[layer].n2.input,
            resid_post=self._envoy.transformer.h[layer].output,
            mlp_in=self._envoy.transformer.h[layer].n2.output,
            mlp_out=self._envoy.transformer.h[layer].mlp.output,
            attn_out=self._envoy.transformer.h[layer].attn.output,
            pattern=self._envoy.transformer.h[layer].attn.softmax.output,
            scores=self._envoy.transformer.h[layer].attn.softmax.input[0][0],
        )[point]