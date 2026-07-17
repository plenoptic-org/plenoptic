---
jupyter:
  jupytext:
    custom_cell_magics: kql
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: plenoptic (3.13.12)
    language: python
    name: python3
---

```python
import plenoptic as po
import torch
import numpy as np
# needed for the plotting/animating:
import matplotlib.pyplot as plt
plt.rcParams['animation.html'] = 'html5'
# use single-threaded ffmpeg for animation writer
plt.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams['animation.ffmpeg_args'] = ['-threads', '1']
from torchvision.models.feature_extraction import get_graph_node_names
import torchvision
import einops
import os
import os
```

```python
seed = 2
```

```python
if seed is not None:
    po.set_seed(seed)
```

```python
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

```python
cpu_or_gpu = 'cpu' # 'cpu' or 0
```

```python
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
tv_model = torchvision.models.resnet50(weights=weights).eval()
tv_transform = weights.transforms()
norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
```

```python
tv_transform
```

```python
norm
```

```python
train_nodes, eval_nodes = get_graph_node_names(tv_model)
```

```python
eval_nodes[-10:]
```

```python
model = po.models.DeepNetFeatures(tv_model, 'fc', norm)
```

```python
po.remove_grad(model)
model.to(cpu_or_gpu).to(torch.float64)
```

```python
imagenet_categories = np.asarray(weights.meta['categories'])
```

```python
img = po.data.macaque()
print(img.shape)
img = po.process.blur_downsample(img, 2)[...,:-60,:]
print(img.shape)
img = po.process.center_crop(img, tv_transform.crop_size[0])
img = img.to(cpu_or_gpu).to(torch.float64)
print(img.shape)
print(f"Min pixel value: {img.min().item()}, Max pixel value: {img.max().item()}")
img_label = 'selfie_monkey'
po.plot.imshow(img, as_rgb=True, title=img_label)
```

```python
def convert_logits_to_probs(logits):
    return torch.nn.functional.softmax(logits, dim=1).squeeze()

def get_category(image):
    img_cat = convert_logits_to_probs(tv_model(norm(image))).detach().cpu()
    category = imagenet_categories[img_cat.argmax()]
    return img_cat, category
```

```python
img_cat, category  = get_category(img)
```

```python
po.plot.stem_plot(img_cat)
```

```python
category
```

```python
img_cat[img_cat>.1]
```

```python
logit_distance = lambda x, y: torch.sqrt(torch.sum((model(x)-model(y))**2))
exponent = 10
penalty_factor = 1
l2_penalty_y = lambda x, y: torch.pow(convert_logits_to_probs(model(y)), exponent).sum()
l2_penalty_x = lambda x, y: torch.pow(convert_logits_to_probs(model(x)), exponent).sum()
metric = lambda x,y: logit_distance(x,y) + penalty_factor*(l2_penalty_y(x,y) - l2_penalty_x(x,y))
```

```python
mad = po.MADCompetition(img, metric, lambda x,y: po.metric.mse(x,y).mean(), "max", metric_tradeoff_lambda=1e10)
mad.setup(initial_noise=0.001,optimizer_kwargs={"lr": 0.01})
```

```python
mad.synthesize(1000)
```

```python
po.plot.synthesis_status(mad);
```

```python
imgs = [img, mad.initial_image, mad.mad_image]
mse = [po.metric.mse(img, i) for i in imgs]
titles = [get_category(i)[1] for i in imgs]
diffs = [(i+1)/2 for i in [img-img, mad.initial_image-img, mad.mad_image-img]]
titles.extend([f"MSE={m.mean().item():.2e}" for m in mse])
imgs.extend(diffs)
po.plot.imshow(imgs, as_rgb=True, title=titles, col_wrap=3, vrange='auto1');
```

```python
channelwise_diffs = [mad.initial_image-img, mad.mad_image-img]
po.plot.imshow(channelwise_diffs, col_wrap=3, vrange='auto0');
```

```python
fig, axes = plt.subplots(4, 2, figsize=(8, 20))
for i, img in enumerate([mad.image, mad.initial_image, mad.mad_image, mad.image-mad.mad_image]):
    img_cat, category = get_category(img)
    likely_cats = '\n- '.join(list(imagenet_categories[img_cat>.05]))
    most_likely_cat = imagenet_categories[img_cat.argmax()]
    if (img<0).any():
        img = (img+1)/2
    po.plot.imshow(img, ax=axes[i, 0], as_rgb=True, title=most_likely_cat)
    po.plot.stem_plot(img_cat, ax=axes[i,1], ylim=False)
    axes[i,1].set_title("Categories")
    axes[i,0].xaxis.set_visible(False)
    axes[i,0].yaxis.set_visible(False)
    axes[i,1].text(1, .5, f"Likely categories:\n- {likely_cats}", transform=axes[i,1].transAxes)
```

```python

```
