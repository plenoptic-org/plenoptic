---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

:::{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`penalty_visual_diversity.ipynb`**!

Run it in your browser: **{binder}`penalty_visual_diversity.ipynb`**!

:::

(penalty-visual-diversity)=
# Using penalties to incentive metamer diversity

[](how-to-penalty) showed the basics of working with custom penalty functions, but their flexibility allows for us to bias stimulus synthesis in a wide variety of different ways. In this notebook, we will show how to use the {class}`~plenoptic.process.SteerablePyramidFreq` to increase perceptual diversity among model metamers for the {class}`~plenoptic.models.LuminanceGainControl` model.

:::{admonition} Seaborn
:class: attention

This notebook uses an additional package, [seaborn](https://seaborn.pydata.org/), to create heatmaps. Install it in your environment in order to run the notebook successfully.

:::

```{code-cell} ipython3
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import plenoptic as po

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72

# On a cpu, we won't run this to completion (takes too long). If you would like to run
# it to completion on a local CPU-only device, increase the value from 100 below
MAX_ITER = 100 if DEVICE.type == "cpu" else 4000
N_IMGS = 2
```

## Metamer synthesis without custom model

```{code-cell} ipython3
:tags: [hide-input]

def pairwise_image_mse(imgs):
    """Get pair-wise MSE between images, along batch dimension."""
    n = imgs.shape[0]
    idx = itertools.combinations(range(n), 2)
    mse = torch.nan * torch.zeros((n, n))
    for i, j in idx:
        mse[j, i] = po.loss.mse(imgs[i], imgs[j])
    return po.to_numpy(mse)


def create_metamer_figure(met):
    n_cols = len(met.image) + 2
    fig = plt.figure(figsize=(5 * n_cols + 2, 16))
    gs = mpl.gridspec.GridSpec(3, n_cols, figure=fig)
    im_axes = [fig.add_subplot(gs[0, i]) for i in range(N_IMGS + 1)]
    im_axes += [fig.add_subplot(gs[1, i]) for i in range(N_IMGS + 1)]
    # met.image has more than one dimension, but they're all identical, so just use the
    # first. image[:1] is the same as image[0], but preserves the number of dimensions.
    imgs = torch.cat([met.image[:1], met.metamer])
    # concatenate the representation of those images
    reps = met.model(imgs)
    titles = ["Target image"] + [f"Model metamer[{i}]" for i in range(N_IMGS)]
    titles += ["Representation of target"] + [
        f"Representation of metamer[{i}]" for i in range(N_IMGS)
    ]
    for ax, im, t in zip(im_axes, torch.cat([imgs, reps]), titles):
        po.plot.imshow(im.unsqueeze(0), ax=ax, title=t, zoom=1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    imgs_mse = pairwise_image_mse(imgs)
    reps_mse = pairwise_image_mse(reps)
    labels = ["Target"] + [f"Metamer[{i}]" for i in range(N_IMGS)]
    ax = fig.add_subplot(gs[0, -1])
    sns.heatmap(imgs_mse, ax=ax, annot=True, xticklabels=labels, yticklabels=labels)
    ax.set_title("MSE of Images")
    ax = fig.add_subplot(gs[1, -1])
    sns.heatmap(reps_mse, ax=ax, annot=True, xticklabels=labels, yticklabels=labels)
    ax.set_title("MSE of Representations")
    po.plot.synthesis_loss(met, ax=fig.add_subplot(gs[2, 0:2]), plot_penalties=True)
    return fig
```

```{code-cell} ipython3
model = po.models.LuminanceGainControl(
    31, pretrained=True, pad_mode="reflect", cache_filt=True
).eval()
po.remove_grad(model)
model.to(DEVICE).to(torch.float64)
img = po.data.einstein().to(DEVICE).to(torch.float64)
met = po.Metamer(img.repeat(N_IMGS, 1, 1, 1), model)
met.synthesize(MAX_ITER, stop_criterion=1e-16)
```

```{code-cell} ipython3

```
