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

```{code-cell} ipython3
:tags: [hide-input]

import pooch

# don't have pooch output messages about downloading or untarring
logger = pooch.get_logger()
logger.setLevel("WARNING")
```

:::{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`feather2023.ipynb`**!

Run it in your browser: **{binder}`feather2023.ipynb`**!

:::

(feather2023)=
# Reproducing Feather et al., 2023

:::{warning}
This notebook requires the optional dependency `torchvision`, which can be installed with `pip`.
:::

In this notebook, we will reproduce some of the model metamers presented in {cite:alp}`Feather2023-model-metam`: metamers for layers 2, 3, and 4 of the standard ImageNet-trained ResNet50, as shown in Figure 2e of that paper.

:::{admonition} Using Deep Nets with plenoptic
:class: warning

We strongly recommended reading [](deep_nets) before this notebook, to better understand the use of {class}`~plenoptic.models.DeepNetFeatures`.

:::

```{code-cell} ipython3
import matplotlib.pyplot as plt
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


dtype = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

%load_ext autoreload
%autoreload 2

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72

# set seed for reproducibility
po.set_seed(1)
```

- just do the torchvision version, with no explanation, separate models for each of the three layers (single image)
- create metamer for each layer and load, then show the separate figures for each (each in their own subheader)
- discuss synthesis success

```{code-cell} ipython3
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
deepnet = torchvision.models.resnet50(weights=weights)
deepnet.eval()
target_layer = "layer3"
```

(feather-synthesis-success)=
## Synthesis success?
