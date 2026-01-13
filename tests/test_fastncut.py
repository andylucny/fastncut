import sys
sys.path.append('fastncut')

import math
import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur
import pytest

from fastncut import ncut, toCosSin, extendWithPositionEncoding, extendWithFix, correlateWithPrompt
from fastncut import Ncut, ToCosSin, ExtendWithPositionEncoding, ExtendWithFix, CorrelateWithPrompt

import numpy as np
import cv2

device = 'cuda'
eps = 1e-5

def leftright(size=(224,224)):
    x = torch.zeros(*size).float()
    x[:,size[1]//2:] = 1.0
    return x

def island(size=(224,224)):
    x = torch.zeros(*size).float()
    x[size[0]//4:size[0]-size[0]//4,size[1]//4:size[1]-size[1]//4] = 1.0
    return x

def gradient(size=(224,224)):
    x = torch.arange(size[0]).unsqueeze(0) + torch.arange(size[0]).unsqueeze(1)
    x = x.float() / x.max()
    return x

def noised(x, probability=0.1):
    mask = torch.rand_like(x, dtype=torch.float) < probability
    x = x.clone()
    x[mask] = 1 - x[mask]
    return x

def blur(x, kernel_size=9, sigma=2.0):
    return gaussian_blur(x.unsqueeze(0), kernel_size=kernel_size, sigma=sigma).squeeze(0)

def eq(x, y, tolerance=1.0):
    return (x == y).float().mean().item() >= tolerance

def topng(x, prompt_x=None, prompt_y=None):
    img = (x.detach().cpu().numpy()*255).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    if prompt_x is not None and prompt_y is not None:
        cv2.circle(img,(prompt_x, prompt_y),2,(0,0,255),cv2.FILLED)
    return img
    
def totif(x):
    return x.detach().float().cpu().numpy()
    
def one(input, init="frame", num_iters=2, any_polarity=True, save=None, pe=0, mask=None, pattern=None, tolerance=1.0, fix=False, hwc=False, colored=False, return_all=True):
    if len(input.shape) == 2:
        x = input.unsqueeze(0)
        if colored:
            x = torch.cat([x,x,x],dim=0)
    elif len(input.shape) == 3 and input.shape[0] > 1:
        x = input.unsqueeze(1)
        if colored:
            x = torch.cat([x,x,x],dim=1)
    else:
        x = input
    
    x = x.to(device)
    x = toCosSin(x, data_format="chw" if len(x.shape) == 3 else "bchw")
    if fix:
        x = extendWithFix(x, data_format="chw" if len(x.shape) == 3 else "bchw")
    if pe > 0:
        x = extendWithPositionEncoding(x, weight=pe, data_format="chw" if len(x.shape) == 3 else "bchw")
    
    data_format = "chw" if len(x.shape) == 3 else "bchw"
    if hwc:
        data_format = "hwc" if len(x.shape) == 3 else "bhwc"
        x = x.permute(1,2,0) if len(x.shape) == 3 else x.permute(0,2,3,1)
    
    result = ncut(x, num_iters=num_iters, data_format=data_format, init=init, mask=mask, return_all=return_all)
    output = result['bipartition'].cpu() if return_all else result.cpu()
    pattern = pattern if pattern is not None else (input > 0.5)
    
    if save and return_all:
        name = save + (init if isinstance(init,str) else 'prompt')
        if len(x.shape) == 3:
            cv2.imwrite('input-'+name+'.png',topng(input, result['prompt_x'], result['prompt_y']))
            cv2.imwrite('output-'+name+'.png',topng(output))
            cv2.imwrite('init-'+name+'.tif',totif(result['init']))
        else:
            prompt_x, prompt_y = result['prompt_x'], result['prompt_y']
            if prompt_x is None or isinstance(prompt_x,int):
                prompt_x = [prompt_x] * len(output)
            if prompt_y is None or isinstance(prompt_y,int):
                prompt_y = [prompt_y] * len(output)
            for i, (input_item, output_item, init_item, prompt_x_item, prompt_y_item) in enumerate(zip(input, output, result['init'], prompt_x, prompt_y)):
                cv2.imwrite('input-'+name+str(i)+'.png',topng(input_item, prompt_x_item, prompt_y_item))
                cv2.imwrite('output-'+name+str(i)+'.png',topng(output_item))
                cv2.imwrite('init-'+name+'.tif',totif(init_item))
        if mask is not None:
            cv2.imwrite('mask-'+name+'.png',topng(result['mask']))
    
    assert eq(pattern,output,tolerance) or (eq(pattern,~output,tolerance) and any_polarity)

def test_one_leftright():
    one(leftright(), "frame", 1, True, "LR")
    one(leftright(), "full", 1, True, "LR")
    one(leftright(), "random", 1, True, "LR")
    one(leftright(), "chessboard", 1, True, "LR")
    one(leftright(), (200,200), 1, False, "LR")

def test_one_island():
    for return_all in [False, True]:
        one(island(), "frame", 1, False, "I", return_all=return_all)
        one(island(), "full", 1, True, "I", return_all=return_all)
        one(island(), "random", 1, True, "I", return_all=return_all)
        one(island(), "chessboard", 1, True, "I", return_all=return_all)
        one(island(), (112,112), 1, False, "I", return_all=return_all)

def test_one_noised_island():
    pattern = island()
    one(noised(pattern), "frame", 1, False, "NI")

def test_one_noised_island_pe():
    pattern = island()
    one(blur(noised(pattern,0.005),5,1.0), "frame", 2, False, "NIPE", pe=0.9, pattern=pattern, tolerance=0.95)
    
def test_one_gradient():
    one(gradient(), "full", 2, True, "G")

def test_one_island_fix():
    one(island(), "frame", 1, False, "IF", fix=True)

def test_one_hwc():
    one(island(), "frame", 1, False, fix=True, hwc=True)

def test_batch():
    batch = torch.stack([ 
        island(),
        island(),
        island(),
    ])
    one(batch, "frame", 1, False)
    one(batch, "frame", 1, False, hwc=True)

def test_mask():
    input = island()
    mask = (leftright() > 0.5)
    pattern = (input > 0.5) & mask
    one(input, "frame", 1, False, "M", mask=mask, pattern=pattern)
    one(input, (140,90), 1, False, "MP", mask=mask, pattern=pattern)
    pattern = ~(input > 0.5) & mask
    one(input, (210,210), 1, False, "MN", mask=mask, pattern=pattern)
    
def test_mask_batch():
    input = island()
    mask = (leftright() > 0.5)
    pattern = (input > 0.5) & mask
    input = torch.stack([input,input,input])
    pattern = torch.stack([pattern,pattern,pattern])
    one(input, "frame", 1, False, "M", mask=mask, pattern=pattern)
    one(input, (140,90), 1, False, "MP", mask=mask, pattern=pattern)
    pattern[0] = ~pattern[0] & mask
    pattern[2] = ~pattern[2] & mask
    one(input, [(210,210),(140,90),(210,210)], 1, False, "MX", mask=mask, pattern=pattern)

def test_color():
    one(island(), "frame", 1, False, "IC", colored=True)
   
def test_model():
    batch = torch.stack([ 
        torch.stack([island(),island(),island()]),
        torch.stack([island(),island(),island()]),
    ]).to(device)
    model = nn.Sequential(
        ToCosSin(),
        ExtendWithFix(), # redundant
        ExtendWithPositionEncoding(), # optional
        Ncut(num_iters=1),
    ).to(device)
    with torch.no_grad():
        result = model(batch)
        pattern = (batch > 0.5).max(dim=1).values
    assert eq(result,pattern)

def test_model_hwc():
    batch = torch.stack([ 
        torch.stack([island(),island(),island()]),
        torch.stack([island(),island(),island()]),
    ]).to(device).permute(0,2,3,1)
    print('batch',batch.shape)
    model = nn.Sequential(
        ToCosSin(data_format='bhwc'),
        ExtendWithFix(data_format='bhwc'), # redundant
        ExtendWithPositionEncoding(data_format='bhwc'), # optional
        Ncut(num_iters=1, data_format='bhwc'),
    ).to(device)
    with torch.no_grad():
        result = model(batch)
        pattern = (batch > 0.5).max(dim=3).values
    assert eq(result,pattern)

def test_convergence():
    feats = torch.rand(6,224,244).to(device)
    result = ncut(feats, num_iters=0, return_all=True)
    with open('iterations.txt','wt') as f:
        f.write(str(result['num_iters']))
    assert(result['num_iters'] > 0)

def test_wrap_around():
    hue = torch.arange(18,device=device).unsqueeze(0).unsqueeze(-1).float()/18
    feats = toCosSin(hue, data_format='hwc', wrap_around=True)
    assert( feats[0,0] @ feats[0,-1] > feats[0,3] @ feats[0,-4] )
    feats = extendWithPositionEncoding(feats, data_format='hwc', wrap_around_x=True)
    pe = feats[...,2:]
    assert( pe[0,0] @ pe[0,-1] > pe[0,3] @ pe[0,-4] )

def test_correlate():
    prompt = [(0,0)]
    feats = torch.rand(6,224,224).to(device)
    feats /= torch.linalg.norm(feats, dim=0, keepdim=True) + 1e-8
    feats = correlateWithPrompt(feats, prompt)
    assert(len(feats.shape) == 3)
    assert(feats.shape[0] == 1)
    assert(feats.max() == feats[0,0,0])
    feats = feats.permute(1,2,0)
    feats = correlateWithPrompt(feats, prompt*4, data_format='hwc')
    assert(len(feats.shape) == 3)
    assert(feats.shape[2] == 4)
    assert(feats.max() == feats[0,0,0])
    