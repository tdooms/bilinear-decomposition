import torch

def compute_outliers(data, factor):
    q1 = torch.quantile(data, 0.25, dim=1, keepdim=True)
    q3 = torch.quantile(data, 0.75, dim=1, keepdim=True)
    
    iqr = q3 - q1
    
    lower, upper = q1 - (factor * iqr), q3 + (factor * iqr)
    mask = (data < lower) | (data > upper)
    
    indices = mask.nonzero().T
    values = data[mask]
    return torch.sparse_coo_tensor(indices, values, data.size())

def compute_self_and_cross_outliers(data, cross_factor, self_factor):
    idxs = torch.tril_indices(*data.shape[1:], offset=-1)
    
    cross = compute_outliers(data[:, idxs[0], idxs[1]], cross_factor).coalesce()
    
    cidxs = cross.indices()
    uidxs = torch.unravel_index(cidxs[1], data.shape[1:])
    
    cross = torch.sparse_coo_tensor(torch.stack([cidxs[0], *uidxs], dim=0), cross.values(), data.size())
    
    selv = compute_outliers(data.diagonal(dim1=-2, dim2=-1), self_factor).coalesce()
    sidxs = selv.indices()
    selv = torch.sparse_coo_tensor(torch.stack([sidxs[0], sidxs[1], sidxs[1]], dim=0), selv.values(), data.size())
    
    return 2*cross + selv
    
def compute_kurtosis(data):
    flat = data.flatten(start_dim=1)
    mean = torch.mean(flat, dim=0, keepdim=True)
    diffs = flat - mean
    var = torch.mean(torch.pow(diffs, 2.0), dim=0)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    return torch.mean(torch.pow(zscores, 4.0), dim=1) - 3.0

def compute_truncated_eigenvalues(data, k=2):
    vals = torch.linalg.eigvalsh(data)
    return vals.abs().topk(k=k).values.sum(-1)

def compute_effective_rank(data):
    vals = torch.linalg.eigvalsh(data)
    
    l2 = vals.pow(2).sum(-1).sqrt()
    l1 = vals.abs().sum(-1)
    
    return (l1/l2).pow(2)