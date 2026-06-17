---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: plenoptic
  language: python
  name: python3
---

:::{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`Feature_Extractor.ipynb`**!

Run it in your browser: **{binder}`Feature_Extractor.ipynb`**!

:::

# Using Deep Neural Networks with plenoptic

:::{warning}
This notebook requires the optional dependency `torchvision`, which can be installed with `pip`.
:::

plenoptic is compatible with any model written in pytorch, including deep neural networks from the model zoos {external+torchvision:ref}`TorchVision <models>` and {external+timm:doc}`timm <models>`. In this notebook, we'll show how to adapt a deep net from these two packages for use with plenoptic, recreating some ResNet50 metamers shown in {cite:alp}`Feather2023-model-metam`, figure 2e.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import torch

import plenoptic as po

# this notebook uses torchvision, which is an optional dependency.
# if this fails, install torchvision in your plenoptic environment
# and restart the notebook kernel.
try:
    import torchvision
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "optional dependency torchvision not found!"
        " please install it in your plenoptic environment "
        "and restart the notebook kernel"
    )


dtype = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

%load_ext autoreload

%autoreload 2

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72
```

When synthesizing images for deep nets, as in {cite:alp}`Feather2023-model-metam`, it is common to pick a specific intermediate layer whose representation we wish to use. `torchvision` contains a "feature extractor" to grab activity from intermediate layers, and plenoptic's {class}`plenoptic.models.FeatureExtractorModel` is a small wrapper to simplify this process.

First, let's specify the layer to target. You can view possible layer names with {external+torchvision:func}`torchvision.models.feature_extraction.get_graph_node_names`. (For more details on the node naming conventions used here, please see the {external+torchvision:ref}`About Node Names <about-node-names>` heading in the {external+torchvision:doc}`torchvision documentation <feature_extraction>`.)

The metamer synthesis procedure in this notebook works with any of `"layer2"`, `"layer3"`, or `"layer4"` (and possibly others, I just haven't tested them all). Let's start with `"layer3"`:

```{code-cell} ipython3
target_layer = "layer3"
```

Below, we show how to initialize a plenoptic-compatible model using the weights from either the {external+torchvision:ref}`TorchVision <models>` or {external+timm:doc}`timm <models>` model zoos; their behavior after this step is the same.


::::{tab-set}
:::{tab-item} torchvision

```{code-cell} ipython3
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
tv_model = torchvision.models.resnet50(weights=weights).eval()
# This model's transform consists of resizing, cropping, and normalizing.
# We recommend only including the normalizing in the model.
tv_transform = weights.transforms()
norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
model = po.models.FeatureExtractorModel(tv_model, target_layer, norm)
po.remove_grad(model)
# this model requires a 3d input, and expects it to have a certain input size.
img = po.process.center_crop(po.data.parrot(False), tv_transform.crop_size[0])
```

:::

:::{tab-item} timm

Note that to run this cell, you must install `timm` as well (`pip install timm`)!

```{code-block} python
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
timm_model = timm.create_model("timm/resnet50.tv_in1k", pretrained=True).eval()
# This model's transform consists of resizing, cropping, and normalizing.
# We recommend only including the normalizing in the model.
timm_transform = create_transform(
    **resolve_data_config(timm_model.pretrained_cfg, model=timm_model)
)
timm_crop = timm_transform.transforms[1]
timm_norm = timm_transform.transforms[-1]
model = po.models.FeatureExtractorModel(timm_model, target_layer, timm_norm)
po.remove_grad(model)
# this model requires a 3d input, and expects it to have a certain input size.
img = timm_crop(po.data.parrot(False))
```
:::
::::

The above cell is doing several important things:
- Download the model weights for ResNet50, trained on
  [ImageNet-1K](https://en.wikipedia.org/wiki/ImageNet#ImageNet-1K).
- Initialize the `torchvision` / `timm` version of ResNet50 using these weights.
- Grab the transform defining the preprocessing procedure. https://docs.pytorch.org/vision/stable/models.html#using-the-pre-trained-models
- Include norm in model, use crop outside model (in general, that's what we recommend, as in {cite:alp}`Feather2023-model-metam`)
- Note remove_grad and eval mode


next:
- metamer synthesis, show result

also of interest:
- can match multiple layers
- convert to dict/tensor
- plot
