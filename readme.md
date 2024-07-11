# Weight-based Decomposition: A Case for Bilinear MLPs

This is the official code repository for the above paper [[link](https://arxiv.org/abs/2406.03947)].

## Abstract

Gated Linear Units (GLUs) have become a common building block in modern foundation models. Bilinear layers drop the non-linearity in the ``gate'' but still have comparable performance to other GLUs. An attractive quality of bilinear layers is that they can be fully expressed in terms of a third-order tensor and linear operations. Leveraging this, we develop a method to decompose the bilinear tensor into a set of sparsely interacting eigenvectors that show promising interpretability properties in preliminary experiments for shallow image classifiers (MNIST) and small language models (Tiny Stories). Since the decomposition is fully equivalent to the model's original computations, bilinear layers may be an interpretability-friendly architecture that helps connect features to the model weights. Application of our method may not be limited to pre-trained bilinear models since we find that language models such as TinyLlama-1.1B can be finetuned into bilinear variants.

## Code

Code is organized into two folders:

- **MNIST**: The training setup and analysis of small MNIST and FMNIST models. Refer to [here](mnist/simple/example.ipynb) for a simple demo.
- **Language**: The transformer model used for the TinyStories analysis.

## Research

This is a snapshot of our work for the paper. Please refer to [this repo](https://github.com/tdooms/bilinear-interp) for an up-to-date version of our research. Note that the code there will be messy!
