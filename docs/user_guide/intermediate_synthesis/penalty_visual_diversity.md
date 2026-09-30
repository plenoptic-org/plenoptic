---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
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

import einops
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyiqa
import seaborn as sns
import torch

import plenoptic as po

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72

# On a cpu, we won't run this to completion (takes too long). If you would like to run
# it to completion on a local CPU-only device, increase the value from 100 below
MAX_ITER = 100 if DEVICE.type == "cpu" else 2000
# hard-code this...
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


def drop_all_nans(matrix, label_x=None, label_y=None):
    # find the rows/cols that are all nans...
    col = np.isnan(matrix).sum(0) == matrix.shape[0]
    row = np.isnan(matrix).sum(1) == matrix.shape[1]
    # ... and drop them
    matrix = matrix[~row][:, ~col]
    if label_x is not None:
        label_x = np.asarray(label_x)[~col]
    if label_y is not None:
        label_y = np.asarray(label_y)[~row]
    return matrix, label_x, label_y


def create_metamer_figure(met):
    n_cols = len(met.image) + 2
    zoom = 256 // met.image.shape[-1]
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
        vr = (0, 1) if "Representation" not in t else (0.68, 0.71)
        po.plot.imshow(im.unsqueeze(0), ax=ax, title=t, zoom=zoom, vrange=vr)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    imgs_mse = pairwise_image_mse(imgs)
    reps_mse = pairwise_image_mse(reps)
    labels = ["Target"] + [f"Metamer[{i}]" for i in range(N_IMGS)]
    imgs_mse, labels_x, labels_y = drop_all_nans(imgs_mse, labels, labels)
    reps_mse = drop_all_nans(reps_mse)[0]
    ax = fig.add_subplot(gs[0, -1])
    sns.heatmap(
        imgs_mse,
        ax=ax,
        annot=True,
        xticklabels=labels_x,
        yticklabels=labels_y,
        vmin=0,
        vmax=0.25,
        cmap="Reds",
    )
    ax.set_title("MSE of Images")
    ax = fig.add_subplot(gs[1, -1])
    sns.heatmap(
        reps_mse,
        ax=ax,
        annot=True,
        xticklabels=labels_x,
        yticklabels=labels_y,
        vmin=0,
        vmax=2e-8,
        cmap="Reds",
    )
    ax.set_title("MSE of Representations")
    po.plot.synthesis_loss(met, ax=fig.add_subplot(gs[2, 0:2]), plot_penalties=True)
    return fig


def summarize_power(image, original_coeffs=None, mask=None):
    power = spyr_power(image)
    if original_coeffs is None:
        original_coeffs = torch.ones_like(power)
    if mask is None:
        mask = torch.ones_like(power)
    return (mask * (power / original_coeffs)).diff(dim=0)


def rearrange_power(power):
    place_holder = torch.ones(
        1, pyr.num_orientations - 1, dtype=power.dtype, device=power.device
    )
    hi = power[:, 0] * place_holder
    lo = power[:, -1] * place_holder
    power = torch.cat([hi, power, lo], -1)
    power = einops.rearrange(
        power, "b (s o) -> b s o", o=pyr.num_orientations
    ).squeeze()
    return po.to_numpy(power)


def power_difference(image, original_coeffs=None, mask=None, how_mask="annot"):
    power = summarize_power(image, original_coeffs)
    power = rearrange_power(power)
    heatmap_kwargs = {}
    vmin = min(power.min(), 0)
    vmax = max(power.max(), 0)
    if mask is not None:
        mask = rearrange_power(mask.sum(0, keepdim=True)).astype(bool)
        if how_mask == "annot":
            tmp = np.empty(mask.shape).astype(str)
            tmp[~mask] = "X"
            tmp[mask] = ""
            heatmap_kwargs.update({"annot": tmp, "fmt": ""})
        elif how_mask == "mask":
            heatmap_kwargs["mask"] = ~mask
            vmin = min(power[mask].min(), 0)
            vmax = max(power[mask].max(), 0)
        else:
            raise ValueError(f"{how_mask=}!")
    ax = sns.heatmap(
        power,
        square=True,
        center=0,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        yticklabels=pyr(img).keys(),
        **heatmap_kwargs,
    )
    ax.set(xlabel="Orientation", ylabel="Scale")
    return ax
```

```{code-cell} ipython3
model = po.models.LuminanceGainControl(
    31, pretrained=True, pad_mode="circular", cache_filt=True
).eval()
po.remove_grad(model)
model.to(DEVICE).to(torch.float64)
img = po.data.einstein().to(DEVICE).to(torch.float64)
img = po.process.blur_downsample(img)
img = img.repeat(N_IMGS, 1, 1, 1)
```

```{code-cell} ipython3
met = po.Metamer(img, model, po.loss.l2_norm)
met.synthesize(MAX_ITER, stop_criterion=1e-16)
```

```{code-cell} ipython3
create_metamer_figure(met);
```

I think don't show the MSE heatmaps or the loss plots

- Can find different looking metamers with different initializations
- What adding penalty does is adds an additional constraint and so shrink the "good solution set" to somewhere else
- This interacts with the initialization point! you could use two different initializations with any of these penalties
  - show curie initialization for one of the steerpyr inits, which doesn't do much, compared to the dists one

```{code-cell} ipython3
curie = po.process.blur_downsample(po.data.curie())
white_noise = torch.rand_like(curie)
init_img = torch.cat([white_noise, curie])
po.plot.imshow(init_img);
```

```{code-cell} ipython3
met = po.Metamer(img, model, po.loss.l2_norm)
met.setup(initial_image=init_img)
met.synthesize(MAX_ITER, stop_criterion=1e-16)
```

```{code-cell} ipython3
create_metamer_figure(met);
```

# Power mask

```{code-cell} ipython3
pyr = po.process.SteerablePyramidFreq(img.shape[-2:])
pyr.to(DEVICE).to(img.dtype)


def spyr_power(x):
    if not isinstance(x, dict):
        x = pyr(x)
    power_list = torch.cat(
        [v.abs().mean(dim=(-2, -1)).flatten(1, -1) for k, v in x.items()], 1
    )
    return power_list


def construct_mask(x, include):
    mask = pyr(x)
    if "low" in include:
        # zero out high frequencies, all orientations
        for scale in ["residual_highpass", 0, 1, 2]:
            mask[scale] = torch.zeros_like(mask[scale])
    elif "vertical" in include:
        # zero out all non-vertical orientations, all frequencies
        for scale in mask:
            if scale in ["residual_highpass", "residual_lowpass"]:
                mask[scale] = torch.zeros_like(mask[scale])
            else:
                mask[scale][:, :, 1:] = 0
    return spyr_power(mask).to(bool)
```

```{code-cell} ipython3
mask = construct_mask(img, "vertical")
if mask.sum() == 0 or (~mask).sum() == 0:
    raise ValueError()
original_coeffs = spyr_power(img)


def penalty(x):
    penalty = spyr_power(x)
    penalty = mask * (penalty / original_coeffs)
    penalty = torch.exp(-torch.mean(penalty.diff(dim=0).pow(2)))
    return po.regularize.penalize_range(x) + penalty
```

```{code-cell} ipython3
met_penalty = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=1e-1
)
```

```{code-cell} ipython3
met_penalty.synthesize(1000, stop_criterion=1e-16, store_progress=True)
```

```{code-cell} ipython3
power_difference(met_penalty.metamer, original_coeffs, mask, "annot");
```

```{code-cell} ipython3
create_metamer_figure(met_penalty);
```

```{code-cell} ipython3
met_penalty = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=1e-1
)
met_penalty.setup(initial_image=init_img)
```

```{code-cell} ipython3
met_penalty.synthesize(1000, stop_criterion=1e-16, store_progress=True)
```

```{code-cell} ipython3
power_difference(met_penalty.metamer, original_coeffs, mask, "annot");
```

```{code-cell} ipython3
create_metamer_figure(met_penalty);
```

```{code-cell} ipython3
mask = construct_mask(img, "low")
if mask.sum() == 0 or (~mask).sum() == 0:
    raise ValueError()
original_coeffs = spyr_power(img)


def penalty(x):
    penalty = spyr_power(x)
    penalty = mask * (penalty / original_coeffs)
    penalty = torch.exp(-torch.mean(penalty.diff(dim=0).pow(2)))
    return po.regularize.penalize_range(x) + penalty
```

```{code-cell} ipython3
met_penalty = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=5e-2
)
```

```{code-cell} ipython3
met_penalty.synthesize(3000, stop_criterion=1e-16)
```

```{code-cell} ipython3
power_difference(met_penalty.metamer, original_coeffs, mask, "annot");
```

```{code-cell} ipython3
create_metamer_figure(met_penalty);
```

```{code-cell} ipython3
met_penalty = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=5e-2
)
met_penalty.setup(initial_image=init_img)
```

```{code-cell} ipython3
met_penalty.synthesize(3000, stop_criterion=1e-16, store_progress=True)
```

```{code-cell} ipython3
power_difference(met_penalty.metamer, original_coeffs, mask, "annot");
```

```{code-cell} ipython3
create_metamer_figure(met_penalty);
```

# Distance

```{code-cell} ipython3
metric = pyiqa.create_metric("dists", device=img.device, as_loss=True)
po.remove_grad(metric)
metric.to(img.dtype)


def dists(x):
    return metric(x[:1], x[-1:])


def penalty(x):
    return po.regularize.penalize_range(x) + torch.exp(-dists(x))
```

```{code-cell} ipython3
met = po.Metamer(img, model, po.loss.l2_norm)
met.synthesize(500, stop_criterion=1e-16)

met_penalty = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=0.5
)
met_penalty.setup(initial_image=met.metamer, optimizer_kwargs={"lr": 0.03})
met_penalty.synthesize(500, stop_criterion=1e-16)
```

```{code-cell} ipython3
create_metamer_figure(met_penalty);
```

```{code-cell} ipython3
met = po.Metamer(img, model, po.loss.l2_norm)
met.setup(initial_image=init_img)
met.synthesize(500, stop_criterion=1e-16)

met_penalty = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=0.5
)
met_penalty.setup(initial_image=met.metamer, optimizer_kwargs={"lr": 0.03})
met_penalty.synthesize(500, stop_criterion=1e-16)
```

```{code-cell} ipython3
create_metamer_figure(met_penalty);
```
