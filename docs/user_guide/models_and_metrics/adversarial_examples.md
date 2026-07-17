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

:::{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`adversarial_examples.ipynb`**!

Run it in your browser: **{binder}`adversarial_examples.ipynb`**!

:::

(adversarial_examples)=
# Using MAD to generate adversarial examples

:::{warning}
This notebook requires the optional dependency `torchvision`, which can be installed with `pip`.
:::

Adversarial examples are tiny perturbations to an image that makes Deep Neural Networks misclasify an image to a different class. In this notebook we demonstrate how we can use the {class}`~plenoptic.MADCompetition` class to synthesize adversarial examples.

```python
import matplotlib.pyplot as plt
import numpy as np
import torch

import plenoptic as po

# this notebook uses torchvision, which is an optional dependency. if this import fails,
# install torchvision in your plenoptic environment and restart the notebook kernel.
try:
    import torchvision
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "optional dependency torchvision not found!"
        " please install it in your plenoptic environment "
        "and restart the notebook kernel"
    )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

%load_ext autoreload
%autoreload 2

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72

# set seed for reproducibility
po.set_seed(2)
```

## Prepare model and image for synthesis

In this section, we walk through how to initialize a plenoptic-compatible model using the weights from {external+torchvision:ref}`TorchVision <models>`. Then, at the end of this section, we briefly show to do the same with models from {external+timm:doc}`timm <models>`.

To use one of these deep nets in `plenoptic`, we have to specify three things:
1. The deep net model.
2. The layer(s) to extract.
3. The image pre-processing to use.

### Initialize deep neural network and pre-trained weights

First, we download the model weights for ResNet50 trained on [ImageNet-1K](https://en.wikipedia.org/wiki/ImageNet#ImageNet-1K) and initialize the `torchvision` model.

```python
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
deepnet = torchvision.models.resnet50(weights=weights)
deepnet.eval()
transform = weights.transforms()
norm = torchvision.transforms.Normalize(transform.mean, transform.std)
```

```python
target_layer = "fc"
model = po.models.DeepNetFeatures(deepnet, target_layer, norm)
```

```python
img = po.data.macaque()
# here we downsample the original image by a factor of 4 and then lop off the bottom.
# that way, when we take the central 224 pixels in the following block, we end up with a
# decent image.
img = po.process.blur_downsample(img, 2)[..., :-59, :]
img = po.process.center_crop(img, transform.crop_size[0])
po.plot.imshow(img, as_rgb=True);
```

```python
img = img.to(DEVICE).to(torch.float64)
model.to(DEVICE).to(torch.float64)
deepnet.to(DEVICE).to(torch.float64)
po.remove_grad(model)
```

## Visualizing classification of the clean image

```python
imagenet_categories = np.asarray(weights.meta['categories'])
```

```python
def convert_logits_to_probs(logits):
    return torch.nn.functional.softmax(logits, dim=1).squeeze()

def get_category(image):
    img_cat = convert_logits_to_probs(deepnet(norm(image))).detach().cpu()
    category = imagenet_categories[img_cat.argmax()]
    return img_cat, category
```

```python
img_cat, category  = get_category(img)
```

```python
po.plot.stem_plot(img_cat)
```

## Define optimized and reference metric

```python
logit_distance = lambda x, y: torch.sqrt(torch.sum((model(x)-model(y))**2))
exponent = 10
penalty_factor = 1
l2_penalty_y = lambda x, y: torch.pow(convert_logits_to_probs(model(y)), exponent).sum()
l2_penalty_x = lambda x, y: torch.pow(convert_logits_to_probs(model(x)), exponent).sum()
metric = lambda x,y: logit_distance(x,y) + penalty_factor*(l2_penalty_y(x,y) - l2_penalty_x(x,y))
```

## Synthesize the adversarial image

```python
mad = po.MADCompetition(img, metric, lambda x,y: po.metric.mse(x,y).mean(), "max", metric_tradeoff_lambda=1e10)
mad.setup(initial_noise=0.001)
```

```python
mad.synthesize(50)
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
