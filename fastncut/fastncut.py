"""
This is the implementation of a fast algorithm for estimating the minimal normalized cut of a given feature map.
Andrej Lucny, 2025, lucny@fmph.uniba.sk
"""

from typing import Optional, Union, Tuple, List, Literal
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math

def unpr(
    u: Tensor, 
    v: Tensor
) -> Tensor:
    """
    Unnormalized projection of vector of features `u` onto vector of features `v`.
    This function processes a batches, each pair `u` `v` in the batch is processed as follows:

        unpr(u, v) = v @ (vᵀ @ u)

    Parameters
    ----------
    u : Tensor
        A tensor of shape (B, N, C)
    v : Tensor
        A tensor of shape (B, N, C)
        where:
        - B is the batch size
        - N is the length of the vector (the number of regions in the feature map)
        - C is the number of features

    Returns
    -------
    A tensor of shape (B, N, C) containing the unnormalized projection of `u` onto `v`.

    Notes
    -----
    We aim to compute projection of a vector u by Gram matrix (v @ vᵀ) without its materialization.
    This way, it is much more efficient, O(BNC) instead of O(BN²C) in space and time.

    Examples
    --------
    >>> B, N, C = 1, 784, 384
    >>> u = torch.randn(B, N, K)
    >>> v = torch.randn(B, N, M)
    >>> out = unpr(u, v)
    >>> out.shape
    torch.Size([1, 784, 384])
    """
    return torch.bmm(v, torch.bmm(torch.transpose(v, 1, 2), u))

def ncut(
    features: Tensor,
    num_iters: int = 2,
    mask: Optional[Tensor] = None, 
    init: Union[None, Tensor, Tuple[int, int], List[Tuple[int, int]], Literal["frame", "full", "random", "chessboard"]] = "frame",
    data_format: Literal["hwc", "chw", "bhwc", "bchw"] = "chw",
    patience: int = 1,
    return_all: bool = False,
    border_size: int = 2,
    eps: float = 1e-5
) -> Tensor:
    """
    Estimates normalized cut on feature maps.

    This function calculates the bipartition of the given feature map by the fastncut algorithm

    Parameters
    ----------
    features : torch.Tensor (float)
        Input feature tensor. The expected shape depends on the `data_format`
        argument:
        - "hwc": (Height, Width, Channels) 
        - "chw": (Channels, Height, Width) - Default
        - "bhwc": (Batch, Height, Width, Channels) 
        - "bchw": (Batch, Channels, Height, Width) 

    num_iters : int, optional
        Number of iterations to perform. 
        Default is 1. 
        The optimal value for intensity features is 2. 
        For hundreds of features 4 is recommended.
        When `num_iters` is zero, the algorithm iterates until the bipartition is stable for `patience` times.
        (This mechanism serves for the investigation only; it is not proper for production.)

    mask : torch.Tensor (bool), optional
        An optional tensor specifying a part of the given feature map to be processed by the fastncut algorithm
        - float tensor: (Height, Width)
        Note: Unfortunately, the shape (Batch, Height, Width) is not supported.
              More calls with the batch size one are preferred in this case.
              It is the same as we cannot have various resolutions in one batch.
    
    init : torch.Tensor or None, optional
        An optional initial tensor used to initialize the fastncut process or a kind of its generation.
        It indicates the resulting polarity of the bipartition mask.
        - float tensor: (Height, Width)
        - "frame": ones in interior and zeros on the border of the `border_size` (Default)
        - "full": all ones
        - "random": random values [-1,1]
        - "chessboard": half -1, half 1, like the chessboard
        - a tuple of int: (x,y) specifies a point/region, which guides the polarity of the bipartition (prompt)
        - a list of such tuples, if the prompt differs for each item in the batch 

    return_all : bool, optional
        Determines the output format:
        - False: return only the final bipartition
        - True: return all bipartitions, and further info

    Returns
    -------
    torch.Tensor (bool)
        The resulting bipartition provided by the fastncut algorithm: (Batch, Height, Width) 
        If `return_all` is true, returns dictionary containing the complete info including:
        - intermediate results from all iterations (Batch, Height, Width) x num_iters
        - value of the generated init

    Notes
    -----
    - This algorithm stems from Shi-Malik's solution for the estimation of the normalized cut.
      However, it does not require the materialization of the similarity matrix quadratic to the number of pixels/regions.
      As a result, it is linear instead of quadratic, working for a specific kind of similarity only (cosine similarity).

    """
    # features are expected to have shape (B,H,W,C) 
    if data_format == "hwc" or data_format == "chw":
        features = features.unsqueeze(0) # add B
    if data_format == "bchw" or data_format == "chw":
        features = features.permute(0,2,3,1) # (B,C,H,W) -> (B,H,W,C)
    B, H, W, C = features.shape   
    
    # flatten and normalize the image features
    if mask is None:
        feats = features.view(B, H*W, C)  # (B,H,W,C) -> (B,H*W,C)
    else:
        indices = torch.nonzero(mask, as_tuple=False) 
        feats = features[:,indices[:,0],indices[:,1],:] # -> (B,N,C)
   
    # Use power iteration to find eigenvector corresponding to the second smallest eigenvalue of 
    # diag(1/√d) @ (diag(d) - featsᵀ @ feats) @ diag(1/√d) for each sample in the batch
    
    # generate initial value of the vector we will project (H,W)
    prompt = None
    if init is None or init == "frame":
        # frame, b ~= 0
        init = - torch.ones(H, W, device=feats.device) 
        init[border_size:H-border_size, border_size:W-border_size] = 1.0
    elif init == "full":
        # full, b == infty
        init = torch.ones(H, W, device=feats.device)
    elif init == "random":
        # random, b ~= -1
        init = torch.rand(H, W, device=feats.device)
    elif init == "chessboard":
        # chessboard, b = -1
        init = ((torch.arange(H, device=feats.device).unsqueeze(1) + torch.arange(W, device=feats.device))%2).float()
    elif isinstance(init, tuple):
        prompt_x, prompt_y = init
        prompt = features[:,prompt_y,prompt_x,:] # (B,C)
    elif isinstance(init, list):
        prompt_x, prompt_y = map(list, zip(*init))
        advanced_indices = torch.arange(features.shape[0],device=features.device)
        prompt = features[advanced_indices,prompt_y,prompt_x,:] # (B,C)
    elif init.device != feats.device:
        init.to(feats.device)

    # initialize the fastncut algorithm
    d = torch.bmm(feats,feats.sum(dim=1).unsqueeze(2))  # (B,H*W or N,1) 
    # d == unpr(torch.ones(1, feats.shape[1], 1, dtype=feats.dtype, device=feats.device), feats)
    d = torch.clamp(d, min=eps) # for safety reasons
    z0norm = torch.nn.functional.normalize(torch.sqrt(d),dim=1) # (B,H*W or N,1)
    pr = feats * torch.rsqrt(d) # (B,H*W,1)

    # initialize the vector we will project
    if prompt is not None:
        eigenvector = torch.bmm(feats, prompt.unsqueeze(-1)) # (B,H*W or N,1)
        if return_all:
            if mask is None:
                init = eigenvector.view(B,H,W)
            else:
                init = torch.zeros(B, H, W, dtype=eigenvector.dtype, device=eigenvector.device)
                init[:,indices[:,0],indices[:,1]] = eigenvector.squeeze(2)
                init = init.view(B,H,W)
            if B == 1:
                init = init.squeeze(0)
    elif mask is None:
        eigenvector = init.view(1,H*W).repeat(B, 1).unsqueeze(-1) # (B,H*W,1)
    else:
        eigenvector = init[indices[:,0],indices[:,1]].repeat(B, 1).unsqueeze(-1) # (B,N,1)
    eigenvector = torch.nn.functional.normalize(eigenvector,dim=1)
    
    # repeat projection of the vector to let it converge to the eigenvector
    if num_iters == 0 or return_all:
        intermediates = []
        stable = 0
    
    i = 0
    while True:

        if num_iters == 0 or return_all or i == num_iters:
            if mask is None:
                bipartition = (eigenvector > 0).view(B,H,W)
            else:
                bipartition = torch.zeros(B, H, W, dtype=torch.bool, device=eigenvector.device)
                bipartition[:,indices[:,0],indices[:,1]] = (eigenvector.squeeze(2) > 0)
                bipartition = bipartition.view(B,H,W)

        # termination conditions
        if num_iters == 0:
            if i > 0:
                if torch.equal(intermediates[-1],bipartition):
                    stable +=1
                    if stable >= patience:
                        break
                else:
                    stable = 0
        elif i == num_iters: 
            break

        if num_iters == 0 or return_all:
            intermediates.append(bipartition)
        
        # project the vector
        projection = unpr(eigenvector,pr) - unpr(eigenvector,z0norm)
        # normalize it
        eigenvector = torch.nn.functional.normalize(projection,dim=1)
        
        i += 1

    if data_format == "hwc" or data_format == "chw":
        bipartition = bipartition.squeeze(0) # remove B
        if return_all:
            intermediates = [ intermediate.squeeze(0) for intermediate in intermediates ]
    
    if return_all:
        return {
            'bipartition': bipartition,
            'intermediates': intermediates,
            'mask': mask,
            'init': init,
            'num_iters': i,
            'prompt_x': prompt_x if prompt is not None else None,
            'prompt_y': prompt_y if prompt is not None else None,
        }
    else:
        return bipartition

class Ncut(nn.Module):
    """Building block of NN performing fastncut algorithm, i.e., agnostic bipartition (optionally masked)"""
    def __init__(self, 
        num_iters: int = 1,
        init: Union[None, Tensor, Literal["frame", "full", "random", "chessboard"]] = "frame",
        data_format: Literal["bhwc", "bchw"] = "bchw",
    ):
        super(Ncut, self).__init__()
        self.num_iters = num_iters
        self.init = init
        self.data_format = data_format

    def forward(self, 
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return ncut(features, self.num_iters, mask, init=self.init, data_format=self.data_format, return_all=False)

format2dim = {
    "hwc": 2, 
    "chw": 0,
    "bhwc": 3, 
    "bchw": 1, 
}

format2height = {
    "hwc": 0, 
    "chw": 1,
    "bhwc": 1, 
    "bchw": 2, 
}

format2width = {
    "hwc": 1, 
    "chw": 2,
    "bhwc": 2, 
    "bchw": 3, 
}

def toCosSin(
    features: Tensor, # (B,C,H,W), (B,H,W,C), (C,H,W) or (H,W,C) accordig to `data_format`
    scale: float = 1.0,
    data_format: Literal["hwc", "chw", "bhwc", "bchw"] = "chw",
    wrap_around: bool = False, # e.g. for length or angle 0-π/2 this is False, but for angle 0-2π this should be True
    eps: float = 1e-5,
) -> Tensor:
    """Convert features [0, 1/scale] → [0, π/2-ε] and return [cos, sin] channels."""
    dim = format2dim[data_format]
    coef = 2*torch.pi - eps if wrap_around else torch.pi/2 - eps
    x = coef * features * scale # scale 0–1 → 0–π/2 or 0-2π
    return torch.cat([
        torch.cos(x), 
        torch.sin(x),
    ], dim=dim) # (B,2*C,H,W), (B,H,W,2*C), (2*C,H,W) or (H,W,2*C) accordig to `data_format`

class ToCosSin(nn.Module):
    """Convert features [0, 1/scale] → [0, π/2-ε] and return [cos, sin] channels."""
    def __init__(self, 
        scale: float = 1.0,
        data_format: Literal["bhwc", "bchw"] = "bchw",
        wrap_around: bool = False,
        eps: float = 1e-5,
    ):
        super(ToCosSin, self).__init__()
        self.scale = scale
        self.data_format = data_format
        self.wrap_around = wrap_around
        self.eps = eps
    
    def __call__(self, 
        features: Tensor, 
    ) -> Tensor: 
        return toCosSin(features, self.scale, self.data_format, self.wrap_around, self.eps)

def extendWithPositionEncoding(
    features: Tensor, # (B,C,H,W), (B,H,W,C), (C,H,W) or (H,W,C) accordig to `data_format` 
    weight: float = 0.59, 
    data_format: Literal["hwc", "chw", "bhwc", "bchw"] = "chw",
    wrap_around_x: bool = False, # for "tire" should be True
    wrap_around_y: bool = False, 
    eps: float = 1e-5,
) -> Tensor:
    """Features extension with the positional encoding"""
    dim = format2dim[data_format]
    shape = features.shape
    H, W = shape[format2height[data_format]], shape[format2width[data_format]]
    coef_y = 2*torch.pi - eps if wrap_around_y else torch.pi/2 - eps
    coef_x = 2*torch.pi - eps if wrap_around_x else torch.pi/2 - eps
    y, x = torch.meshgrid(
        coef_y * torch.arange(H).to(features.device) / H,
        coef_x * torch.arange(W).to(features.device) / W,
        indexing="ij",
    )
    if dim in (1,3):
        B = shape[0]
        y = y.repeat(B,1,1)
        x = x.repeat(B,1,1)
    y = y.unsqueeze(dim)
    x = x.unsqueeze(dim)
    pe = torch.cat([y.cos(), y.sin(), x.cos(), x.sin()], dim=dim)  # e.g. (B, H, W, 4)
    return torch.cat([
        (1-weight) * features / shape[dim], 
        weight * pe / 4,
    ], dim=dim) # (B,C+1,H,W), (B,H,W,C+1), (C+1,H,W) or (H,W,C+1) accordig to `data_format` 

class ExtendWithPositionEncoding(nn.Module):
    """Features extension with the positional encoding"""
    def __init__(self,
        weight: float = 0.59,    
        data_format: Literal["bhwc", "bchw"] = "bchw",
        wrap_around_x: bool = False,
        wrap_around_y: bool = False,
        eps: float = 1e-5,
    ):
        super(ExtendWithPositionEncoding, self).__init__()
        self.weight = weight
        self.data_format = data_format
        self.wrap_around_x = wrap_around_x
        self.wrap_around_y = wrap_around_y
        self.eps = eps
    
    def __call__(self, 
        features: Tensor, 
    ) -> Tensor: 
        return extendWithPositionEncoding(features, self.weight, self.data_format, self.wrap_around_x, self.wrap_around_y, self.eps)

def extendWithFix(
    features: Tensor, # (B,C,H,W), (B,H,W,C), (C,H,W) or (H,W,C) accordig to `data_format`
    data_format: Literal["hwc", "chw", "bhwc", "bchw"] = "chw",
    eps: float = 1e-5,
) -> Tensor:
    """Extend the cosine similarity features by adding a feature that ensures the ncut algorithm is applicable."""
    dim = format2dim[data_format]
    value = features.norm(dim=dim).abs().max() + eps
    shape = list(features.shape)
    shape[dim] = 1
    shape = tuple(shape)
    return torch.cat([
        features, 
        torch.full(shape, value, device=features.device, dtype=features.dtype),
    ], dim=dim) # (B,C+1,H,W), (B,H,W,C+1), (C+1,H,W) or (H,W,C+1) accordig to `data_format`

class ExtendWithFix(nn.Module):
    """Extend the cosine similarity features by adding a feature that ensures the ncut algorithm is applicable."""
    def __init__(self, 
        data_format: Literal["bhwc", "bchw"] = "bchw",
        eps: float = 1e-5,
    ):
        super(ExtendWithFix, self).__init__()
        self.data_format = data_format
        self.eps = eps
    
    def __call__(self, 
        features: Tensor
    ) -> Tensor: 
        return extendWithFix(features, self.data_format, self.eps)

def correlateWithPrompt(
    features: Tensor, # (B,C,H,W), (B,H,W,C), (C,H,W) or (H,W,C) accordig to `data_format`
    prompt: List[Tuple[int,int]], # a list of tuples of integers (x,y), pointing to regions of the segmented object
    data_format: Literal["hwc", "chw", "bhwc", "bchw"] = "chw",
) -> Tensor:
    """Turn general features into the features describing similarity to a given list of regions (samples of segmented object)"""
    coords = torch.tensor(prompt, device=features.device)  # shape (C, 2)
    x, y = coords[:, 0], coords[:, 1]  # (C,)
    
    # features are expected to have shape (B,H,W,C) 
    if data_format == "hwc" or data_format == "chw":
        features = features.unsqueeze(0) # add B
    if data_format == "bchw" or data_format == "chw":
        features = features.permute(0,2,3,1) # (B,C,H,W) -> (B,H,W,C)
    
    # features'[b,h,w,c] == features[b,h,w] @ features[b,y[c],x[c]]
    B, H, W, C = features.shape
    features = torch.bmm(features.view(B, H*W, C), features[:, y, x, :].permute(0,2,1)).view(B, H, W, -1)
    
    # return features to the original shape
    if data_format == "bchw" or data_format == "chw":
        features = features.permute(0,3,1,2) # (B,H,W,C) -> (B,C,H,W)
    if data_format == "hwc" or data_format == "chw":
        features = features.squeeze(0) # remove B
    
    return features

class CorrelateWithPrompt(nn.Module):
    """Extend the cosine similarity features by adding a feature that ensures the ncut algorithm is applicable."""
    def __init__(self, 
        data_format: Literal["bhwc", "bchw"] = "bchw",
    ):
        super(CorrelateWithPrompt, self).__init__()
        self.data_format = data_format
    
    def __call__(self, 
        features: Tensor, 
        prompt: List[Tuple[int,int]], # a list of tuples of integers (x,y), pointing to regions of the segmented object    
    ) -> Tensor: 
        return correlateWithPrompt(features, prompt, self.data_format)
