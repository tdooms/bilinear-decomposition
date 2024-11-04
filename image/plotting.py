from jaxtyping import Float
from torch import Tensor
from einops import *
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import plotly.express as px
import torch

def plot_explanation(model, sample: Float[Tensor, "w h"], eigenvalues=10):
    """Creates a plot showing the top eigenvector activations for a given input sample."""
    colors = px.colors.qualitative.Plotly
    
    logits = model(sample)[0].cpu()
    classes = logits.topk(3).indices.sort().values.cpu()
    
    # compute the activations of the eigenvectors for a given sample
    vals, vecs = model.decompose()
    vals, vecs = vals.cpu(), vecs.cpu()
    acts = einsum(sample.flatten().cpu(), vecs, "inp, cls comp inp -> cls comp").pow(2) * vals

    # compute the contributions of the top 3 classes
    contrib, idxs = acts[classes].sort(dim=-1)

    titles = [''] + [f"{c}" for c in classes] + ['input', ''] + [f"{c}" for c in classes] + ['logits']
    fig = make_subplots(rows=2, cols=5, subplot_titles=titles, vertical_spacing=0.1)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    
    # add line plot for eigenvalues
    for i in range(3):
        params = dict(showlegend=False, marker=dict(color=colors[i]))
        fig.add_scatter(y=contrib[i, -eigenvalues-2:].flip(0), mode="lines", **params, row=1, col=1)
        fig.add_scatter(y=contrib[i, -1:].flip(0), mode="markers", **params, row=1, col=1)
        
        fig.add_scatter(y=contrib[i, :eigenvalues+2], mode="lines", **params, row=2, col=1)
        fig.add_scatter(y=contrib[i, :1], mode="markers", **params, row=2, col=1)
    
    # add heatmaps for the top 3 classes
    for i in range(3):
        params = dict(showscale=False, colorscale="RdBu", zmid=0)
        fig.add_heatmap(z=vecs[classes[i]][idxs[i, -1]].view(28, 28).flip(0), **params, row=1, col=i+2)
        fig.add_heatmap(z=vecs[classes[i]][idxs[i, 0]].view(28, 28).flip(0), **params, row=2, col=i+2)
    
    # add tickmarks for the heatmaps
    for i in range(2):
        tickvals = [0] + list(contrib[:3, [-1, 0][i]])
        ticktext = [f'{val:.2f}' for val in tickvals]
        fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=i+1)
    
    bars, text = ["gray"] * 10, [""] * 10
    for i, c in zip(classes, colors):
        bars[i], text[i] = c, f"{i}"

    fig.add_bar(y=logits, marker_color=bars, text=text, showlegend=False, textposition='outside', textfont=dict(size=12), row=2, col=5)
    fig.update_yaxes(range=[logits.min(), logits.max() * 1.5], row=2, col=5)
    
    fig.add_heatmap(z=sample[0].flip(0).cpu(), colorscale="RdBu", zmid=0, showscale=False, row=1, col=5)
    fig.update_annotations(font_size=13)
    
    fig.update_xaxes(visible=True, tickvals=[eigenvalues], ticktext=[f'{eigenvalues}'], zeroline=False, col=1)
    fig.update_layout(width=800, height=320, margin=dict(l=0, r=0, b=0, t=20), template="plotly_white")

    return fig


def plot_eigenspectrum(model, digit, eigenvectors=3, eigenvalues=20, ignore_pos=[], ignore_neg=[]):
    """Plot the eigenspectrum for a given digit."""
    colors = px.colors.qualitative.Plotly
    fig = make_subplots(rows=2, cols=1 + eigenvectors)
    
    vals, vecs = model.decompose()
    vals, vecs = vals[digit].cpu(), vecs[digit].cpu()
    
    negative = torch.arange(eigenvectors)
    positive = -1 - negative

    fig.add_trace(go.Scatter(y=vals[-eigenvalues-2:].flip(0), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=negative.flip(0), y=vals[positive].flip(0), mode='markers', marker=dict(color=colors[0])), row=1, col=1)

    fig.add_trace(go.Scatter(y=vals[:eigenvalues+2], mode="lines", marker=dict(color=colors[1])), row=2, col=1)
    fig.add_trace(go.Scatter(x=negative, y=vals[negative], mode='markers', marker=dict(color=colors[1])), row=2, col=1)

    for i, idx in enumerate(positive):
        fig.add_trace(go.Heatmap(z=vecs[idx].view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=1, col=i+2)

    for i, idx in enumerate(negative):
        fig.add_trace(go.Heatmap(z=vecs[idx].view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=2, col=i+2)

    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.update_xaxes(visible=True, tickvals=[eigenvalues], ticktext=[f'{eigenvalues}'], zeroline=False, col=1)
    fig.update_yaxes(zeroline=True, rangemode="tozero", col=1)
    
    tickvals = [0] + [x.item() for i, x in enumerate(vals[positive]) if i not in ignore_pos]
    ticktext = [f'{val:.2f}' for val in tickvals]
    
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=1)

    tickvals = [0] + [x.item() for i, x in enumerate(vals[negative]) if i not in ignore_neg]
    ticktext = [f'{val:.2f}' for val in tickvals]
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=2)

    fig.update_coloraxes(showscale=False)
    fig.update_layout(autosize=False, width=170*(eigenvectors+1), height=300, margin=dict(l=0, r=0, b=0, t=0), template="plotly_white")
    fig.update_legends(visible=False)
    
    return fig