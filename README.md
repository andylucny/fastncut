# fastncut
A Fast Algorithm for Normalized Cut with Applications on Bipartitioning Feature Maps in Deep Learning

A fast agnostic algorithm for bipartitioning images or feature maps. 
Implemented in Pytorch.
Really cool! 
The ingenious idea of Shi & Malik was brought into practice.

## Installation

```bash
pip install fastncut
```

## Documentation

<a href="https://www.agentspace.org/fastncut/fastncut.html" target="_blank"> API </a> &nbsp; &nbsp;
<a href="https://pypi.org/project/fastncut/" target="_blank"> PyPI </a>


## Usage

```bash
import torch
import numpy as np
from fastncut import ncut, toCosSin, extendWithFix
device = 'cpu' # 'cuda'
```

### for intensity image (H,W), 0-255:

load the image:

```bash
import cv2
image = cv2.imread('man.png',cv2.IMREAD_GRAYSCALE)
```

or

```bash
from PIL import Image
image = np.array(Image.open('man.png'))
```

then process it into a feature map (2,H,W) and apply the bipartioning

```bash
blob = toCosSin(torch.tensor(image,device=device).float().unsqueeze(0)/255.0)
bipartition = ncut(blob,num_iters=2).cpu().numpy().astype(np.uint8)*255
```

and save the result:

```bash
cv2.imwrite('bipartition.png',bipartition) # or ~bipartition
```

### for feature map (C,H,W):

```bash
features = F.interpolate(features, size=(image_height, image_width), mode="bilinear", align_corners=False)
bipartition = ncut(features, num_iters=4)
```

or

```bash
features = F.interpolate(features, size=(image_height, image_width), mode="bilinear", align_corners=False)
bipartition = ncut(extendWithFix(features), num_iters=4)
```

set `num_iters=8` for small resolutions (without interpolation)

See notebooks (each notebook is provided in two versions: with and without outputs)

## Model zoo

- NCut can be included in a neural network. From a mere 500 examples, we trained this 
  <a href="http://www.agentspace.org/download/fgsegmentation.pth"> foreground segmentation model</a>. 
  (It is not competitive for the task; we have just tested that we can train through NCut.)

## Please, cite:

<a href="https://doi.org/10.2139/ssrn.6482332"> preprint </a>

Lúčny, A. (2026). A Fast Algorithm for Normalized Cut with Applications on Bipartitioning Feature Maps in Deep Learning. SSRN Electronic Journal. https://doi.org/10.2139/ssrn.6482332
