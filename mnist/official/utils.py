from numpy import False_
import torch
import itertools
import einops
import copy
from mnist.model import *

def get_pixel_label_mutual_info(train_loader, img_size=(28,28), num_classes = 10):
    # TODO: make more generic for non-image inputs
    class_means = torch.zeros((num_classes,img_size[0]*img_size[1]))
    class_counts = torch.zeros(num_classes)
    for images, labels in train_loader:
      images = images.reshape(-1, img_size[0]*img_size[1])
      for idx, label in enumerate(labels):
        class_means[label] += images[idx]
        class_counts[label] += 1
    class_means /= class_counts.unsqueeze(1)

    pixel_probs = class_means.mean(dim=0)
    class_probs = (class_counts/class_counts.sum()).unsqueeze(1)
    pixel_class_on_prob = class_means * class_probs
    pixel_class_off_prob = (1-class_means) * class_probs
    
    pixel_on = pixel_class_on_prob * torch.log(pixel_class_on_prob/(pixel_probs * class_probs))
    pixel_off =pixel_class_off_prob * torch.log(pixel_class_off_prob/((1-pixel_probs)* class_probs))
    
    pixel_on = pixel_on.nan_to_num(0)
    pixel_off = pixel_off.nan_to_num(0)
    mutual_info = (pixel_on + pixel_off).sum(dim=0)
    return mutual_info

def get_top_pixel_idxs(train_loader, num_pixels, bias_idx = None, **kwargs):
    mutual_info = get_pixel_label_mutual_info(train_loader)
    top_mi = mutual_info.topk(num_pixels)
    pixel_idxs = top_mi.indices.sort().values
    if bias_idx is not None:
        pixel_idxs = torch.cat([pixel_idxs, torch.tensor([bias_idx])], dim=0)
    return pixel_idxs

def get_B_tensor(W,V, idxs = None):
    device = W.device
    with torch.no_grad():
        if idxs is not None:
            idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(idxs,2))).to(device)
            B = (1/2) * W[:,idx_pairs[:,0]] * V[:,idx_pairs[:,1]] + \
                (1/2) * W[:,idx_pairs[:,1]] * V[:,idx_pairs[:,0]]
        else:
            B = einops.einsum(W,V, "out in1, out in2 -> out in1 in2").to(device)
            B = 0.5 * B + 0.5 * B.transpose(-2,-1)
            B = einops.rearrange(B, "out in1 in2 -> out (in1 in2)")
    return B

def compute_symmetric_svd(W, V, idxs = None, return_B = False):
    B = get_B_tensor(W,V,idxs=idxs)
    with torch.no_grad():
        svd = torch.svd(B)
    if return_B:
        return svd, B
    else:
        del B
        if torch.cuda.is_available: torch.cuda.empty_cache()
        return svd

def compute_svds_for_deep_model(model, svd_components, input_idxs = None,
    svd_type='symmetric', sing_val_type='with R', bias = False, device = None):
    
    if device is None:
        device = model.layers[0].linear1.weight.device
    svds = [None] * len(model.layers)
    for layer_idx, layer in enumerate(model.layers):
        if layer_idx == 0:
            idxs = input_idxs
            W = layer.linear1.weight.to(device).detach()
            V = layer.linear2.weight.to(device).detach()
        else:
            R = svds[layer_idx-1].U[:,:svd_components]
            if sing_val_type == 'with R':
                S = svds[layer_idx-1].S[:svd_components]
                R = R @ torch.diag(S)
            if bias:
                idxs = torch.arange(svd_components+1).to(device)
                ones = torch.ones(1).to(device)
                R = torch.block_diag(R, ones)
            else:
                idxs = torch.arange(svd_components).to(device) if input_idxs is not None else None
            W = layer.linear1.weight.to(device).detach() @ R
            V = layer.linear2.weight.to(device).detach() @ R

        if svd_type == 'symmetric':
            svd = compute_symmetric_svd(W, V, idxs=idxs)
            svds[layer_idx] = svd
    return svds

def reduce_svds(svds, svd_components):
    class ReducedSVD():
        def __init__(self, svd, svd_components):
            self.U = svd.U[:,:svd_components]
            self.S = svd.S[:svd_components]
            self.V = svd.V[:,:svd_components]
    return [ReducedSVD(svd, svd_components) for svd in svds]

def get_topK_tensors(svds, topK_list, input_idxs, svd_components, sing_val_type,
    bias = False):

    device = svds[0].V.device
    B_tensors = []
    R_tensors = []
    for layer_idx, svd in enumerate(svds):
        if layer_idx == 0:
            idxs = input_idxs.clone().to(device)
            Q_idxs = input_idxs.clone().to(device)
        else:
            Q_idxs = torch.arange(topK_list[layer_idx-1]).to(device)
            if bias:
                idxs = torch.arange(svd_components+1).to(device)
                Q_idxs = torch.cat([Q_idxs, torch.tensor([svd_components])]).to(device)
            else:
                idxs = torch.arange(svd_components).to(device)
        
        topK = topK_list[layer_idx]
        B = torch.zeros((topK, len(Q_idxs), len(Q_idxs))).to(device)

        idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(idxs,2))).to(device)
        mask0 = torch.isin(idx_pairs[:,0], Q_idxs)
        mask1 = torch.isin(idx_pairs[:,1], Q_idxs)
        mask = torch.logical_and(mask0, mask1)
        idx_pairs_reduced = idx_pairs[mask]
        if sing_val_type == 'with R':
            Q_reduced = svd.V[mask, :topK]
        elif sing_val_type == 'with Q':
            Q_reduced = svd.V[mask, :topK] @ torch.diag(svd.S[:topK])
    
        idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(range(len(Q_idxs)),2))).to(device)
        B[:, idx_pairs[:,0],idx_pairs[:,1]] = Q_reduced.T
        B[:, idx_pairs[:,1],idx_pairs[:,0]] = Q_reduced.T
        
        if sing_val_type == 'with R':
            R = svd.U[:,:topK] @ torch.diag(svd.S[:topK])
        elif sing_val_type == 'with Q':
            R = svd.U[:,:topK]

        B_tensors.append(B)
        R_tensors.append(R)
    return B_tensors, R_tensors

def get_topK_model(model, svds, topK_list, input_idxs, svd_components, sing_val_type = 'with R'):
    B_tensors, R_tensors = get_topK_tensors(svds, topK_list, input_idxs, svd_components, 
                                            sing_val_type, bias=model.cfg.bias)
    W_out = model.linear_out.weight @ R_tensors[-1]
    bias_out = model.linear_out.bias
    
    topk_model = BilinearModelTopK(B_tensors, W_out, bias_out, input_idxs,
                                    norm = model.cfg.rms_norm, bias=model.cfg.bias)
    return topk_model

def get_topK_baseline_model(model, input_idxs):
    topk_model = copy.deepcopy(model)
    device = model.layers[0].linear1.weight.device
    W1 = torch.zeros(*model.layers[0].linear1.weight.shape)
    W1[:,input_idxs] = model.layers[0].linear1.weight[:,input_idxs]
    W2 = torch.zeros(*model.layers[0].linear1.weight.shape)
    W2[:,input_idxs] = model.layers[0].linear2.weight[:,input_idxs]
    topk_model.layers[0].linear1.weight = torch.nn.Parameter(W1)
    topk_model.layers[0].linear2.weight = torch.nn.Parameter(W2)
    return topk_model.to(device)

def get_max_pos_neg_activations(Q):
    # Q : square matrix for quadratic filter
    cfg = MaxActivationConfig()

    model = MaxActivationModel(cfg, Q)
    model.train()
    x_pos = model.get_activation()
    act_pos = model.forward()

    model = MaxActivationModel(cfg, -Q)
    model.train()
    x_neg = model.get_activation()
    act_neg = -1 * model.forward()
    return x_pos, x_neg, act_pos, act_neg

