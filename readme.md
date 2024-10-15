# Bilinear MLPs enable weight-based mechanistic interpretability

This is the official code repository for the above paper [[link](https://arxiv.org/pdf/2410.08417)].

## Abstract

A mechanistic understanding of how MLPs do computation in deep neural networks
remains elusive. Current interpretability work can extract features from
hidden activations over an input dataset but generally cannot explain how MLP
weights construct features. One challenge is that element-wise nonlinearities
introduce higher-order interactions and make it difficult to trace computations
through the MLP layer. In this paper, we analyze bilinear MLPs, a type of
Gated Linear Unit (GLU) without any element-wise nonlinearity that nevertheless
achieves competitive performance. Bilinear MLPs can be fully expressed in
terms of linear operations using a third-order tensor, allowing flexible analysis of
the weights. Analyzing the spectra of bilinear MLP weights using eigendecomposition
reveals interpretable low-rank structure across toy tasks, image classification, and language modeling.
We use this understanding to craft adversarial
examples, uncover overfitting, and identify small language model circuits directly
from the weights alone. Our results demonstrate that bilinear layers serve as an
interpretable drop-in replacement for current activation functions and that weight-based
interpretability is viable for understanding deep-learning models.

## Progress

The current code only contains the infrastructure code to load models and analyze them without much documentation.
We will add more documentation and add several examples soon™.
