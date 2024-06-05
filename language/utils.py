import torch
import pandas as pd
from einops import *
from transformers import AutoTokenizer
from bidict import bidict
from typing import List, Tuple
from torch import Tensor
from jaxtyping import Int
from collections import Counter

class UBE:
    def __init__(self, inner) -> None:
        self.inner = inner
    
    def diagonal(self, residual: bool = False, layer: int = 0):
        """The diagonal or direct token interactions of an MLP.

        Args:
            residual (bool, optional): Whether to include the residual. Defaults to False.
            layer (int, optional):Which MLP layer to consider. Defaults to 0.

        Returns:
            Tensor: A matrix containing the diagonal
        """
        inner = self.inner
        
        diag = einsum(inner.b[layer], inner.w_e, inner.w_u, "res emb emb, emb inp, out res -> out inp")
            
        if residual:
            diag += einsum(inner.w_e, inner.w_u, "res inp, out res -> out inp")
        
        return diag
    
    def interaction(self, token: int, residual: bool = False, layer:int = 0):
        """Compute the interaction matrix of an MLP for a certain output token.

        Args:
            idx (int): Vocab token index
            residual (bool, optional): Whether to include the residual. Defaults to False.
            layer (int, optional): Which MLP layer to consider. Defaults to 0.

        Returns:
            Tensor: A matrix containing the token interactions
        """
        inner = self.inner
        
        inter = einsum(
            inner.w_e, inner.w_e, inner.b[layer], inner.w_u[token],
            "emb1 inp1, emb2 inp2, out emb1 emb2, out -> inp1 inp2"
        )
        
        if residual:
            inter += einsum(inner.w_e, inner.w_e, inner.w_u[token], "res inp1, res inp2, res -> inp1 inp2")
        
        return inter
    
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