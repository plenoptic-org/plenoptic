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

:::{admonition} pyiqa
:class: attention

This notebook uses an additional package, [pyiqa](https://iqa-pytorch.readthedocs.io/), for their pytorch implementation of the DISTS image quality metric ([original publication](https://www.cns.nyu.edu/pub/lcv/ding20a-preprint.pdf)). Install it in your environment in order to run the notebook successfully.

:::

```{code-cell} ipython3
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyiqa
import torch

import plenoptic as po

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72

# We will not run synthesis in this notebook, instead loading cached results
cached_metamers = po.data.fetch_data("visual_diversity_penalty.tar.gz")
```

I think don't show the MSE heatmaps or the loss plots

- Can find different looking metamers with different initializations
- What adding penalty does is adds an additional constraint and so shrink the "good solution set" to somewhere else
- This interacts with the initialization point! you could use two different initializations with any of these penalties
  - show curie initialization for one of the steerpyr inits, which doesn't do much, compared to the dists one

## Metamer synthesis without custom model

```{code-cell} ipython3
:tags: [hide-input]

def create_metamer_figure(met):
    n_cols = len(met.image) + 1
    zoom = 256 // met.image.shape[-1]
    fig = plt.figure(figsize=(5.2 * n_cols, 10.4))
    gs = mpl.gridspec.GridSpec(2, n_cols, figure=fig)
    im_axes = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
    im_axes += [fig.add_subplot(gs[1, i]) for i in range(n_cols)]
    # met.image has more than one dimension, but they're all identical, so just use the
    # first. image[:1] is the same as image[0], but preserves the number of dimensions.
    imgs = torch.cat([met.image[:1], met.metamer])
    # get the representation of those images
    reps = met.model(imgs)
    titles = ["Target image"] + [
        f"Model metamer[{i}]" for i in range(met.metamer.shape[0])
    ]
    titles += ["Representation of target"] + [
        f"Representation of metamer[{i}]" for i in range(met.metamer.shape[0])
    ]
    for ax, im, t in zip(im_axes, torch.cat([imgs, reps]), titles):
        # ensure all have the same vrange
        vr = (0, 1) if "Representation" not in t else (0.68, 0.71)
        po.plot.imshow(im.unsqueeze(0), ax=ax, title=t, zoom=zoom, vrange=vr)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    return fig
```

```{code-cell} ipython3
model = po.models.LuminanceGainControl(
    31, pretrained=True, pad_mode="circular", cache_filt=True
).eval()
po.remove_grad(model)
model.to(DEVICE).to(torch.float64)
img = po.data.einstein().to(DEVICE).to(torch.float64)
img = po.process.blur_downsample(img)
img = img.repeat(2, 1, 1, 1)
```

```{code-cell} ipython3
met = po.Metamer(img, model, po.loss.l2_norm)
met.load(cached_metamers / "vis_div_penalty_basic.pt", map_location="cpu")
create_metamer_figure(met);
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestVisualDiversityPenalty.test_basic -->
```{code-block} python
met.synthesize(1000, stop_criterion=1e-16)
```

Note that for exact reproducibility, we also need to specify the seed and to use deterministic algorithms (see [](reproduce-best-practices) for more details) before the initialization of the {class}`~plenoptic.Metamer` object:

```{code-block} python
po.set_seed(0)
torch.use_deterministic_algorithms(True)
```
:::

```{code-cell} ipython3
curie = po.process.blur_downsample(po.data.curie())
white_noise = torch.rand_like(curie)
init_img = torch.cat([white_noise, curie])
po.plot.imshow(init_img, zoom=2);
```

```{code-cell} ipython3
met = po.Metamer(img, model, po.loss.l2_norm)
met.load(cached_metamers / "vis_div_penalty_basic_curie.pt", map_location="cpu")
create_metamer_figure(met);
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestVisualDiversityPenalty.test_basic_curie -->
```{code-block} python
met.setup(initial_image=init_img)
met.synthesize(1000, stop_criterion=1e-16)
```

Note that for exact reproducibility, we also need to specify the seed and to use deterministic algorithms (see [](reproduce-best-practices) for more details) before the initialization of the {class}`~plenoptic.Metamer` object:

```{code-block} python
po.set_seed(0)
torch.use_deterministic_algorithms(True)
```
:::

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
original_coeffs = spyr_power(img)


def penalty(x):
    penalty = spyr_power(x)
    penalty = mask * (penalty / original_coeffs)
    penalty = torch.exp(-torch.mean(penalty.diff(dim=0).pow(2)))
    return po.regularize.penalize_range(x) + penalty
```

```{code-cell} ipython3
:tags: [remove-cell]

# Check that our mask is the same as the one in the tests, which is what generates the
# saved metamers
saved_mask = torch.load(
    po.data.fetch_data("spyr_mask.tar.gz") / "spyr_mask_vertical.pt", map_location="cpu"
)
assert (saved_mask == mask).all()
```

```{code-cell} ipython3
met = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=0.1
)
met.load(cached_metamers / "vis_div_penalty_vertical.pt", map_location="cpu")
create_metamer_figure(met);
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestVisualDiversityPenalty.test_vertical -->
```{code-block} python
met.synthesize(1000, stop_criterion=1e-16)
```

Note that for exact reproducibility, we also need to specify the seed and to use deterministic algorithms (see [](reproduce-best-practices) for more details) before the initialization of the {class}`~plenoptic.Metamer` object:

```{code-block} python
po.set_seed(0)
torch.use_deterministic_algorithms(True)
```
:::


```{code-cell} ipython3
met = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=0.1
)
met.load(cached_metamers / "vis_div_penalty_vertical_curie.pt", map_location="cpu")
create_metamer_figure(met);
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestVisualDiversityPenalty.test_vertical_curie -->
```{code-block} python
met.setup(initial_image=init_img)
met.synthesize(1000, stop_criterion=1e-16)
```

Note that for exact reproducibility, we also need to specify the seed and to use deterministic algorithms (see [](reproduce-best-practices) for more details) before the initialization of the {class}`~plenoptic.Metamer` object:

```{code-block} python
po.set_seed(0)
torch.use_deterministic_algorithms(True)
```
:::

```{code-cell} ipython3
mask = construct_mask(img, "low")
original_coeffs = spyr_power(img)


def penalty(x):
    penalty = spyr_power(x)
    penalty = mask * (penalty / original_coeffs)
    penalty = torch.exp(-torch.mean(penalty.diff(dim=0).pow(2)))
    return po.regularize.penalize_range(x) + penalty
```

```{code-cell} ipython3
:tags: [remove-cell]

# Check that our mask is the same as the one in the tests, which is what generates the
# saved metamers
saved_mask = torch.load(
    po.data.fetch_data("spyr_mask.tar.gz") / "spyr_mask_low.pt", map_location="cpu"
)
assert (saved_mask == mask).all()
```

```{code-cell} ipython3
met = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=0.07
)
met.load(cached_metamers / "vis_div_penalty_low.pt", map_location="cpu")
create_metamer_figure(met);
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestVisualDiversityPenalty.test_low -->
```{code-block} python
met.synthesize(3000, stop_criterion=1e-16)
```

Note that for exact reproducibility, we also need to specify the seed and to use deterministic algorithms (see [](reproduce-best-practices) for more details) before the initialization of the {class}`~plenoptic.Metamer` object:

```{code-block} python
po.set_seed(0)
torch.use_deterministic_algorithms(True)
```
:::


```{code-cell} ipython3
met = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=0.05
)
met.load(cached_metamers / "vis_div_penalty_low_curie.pt", map_location="cpu")
create_metamer_figure(met);
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestVisualDiversityPenalty.test_low_curie -->
```{code-block} python
met.setup(initial_image=init_img)
met.synthesize(3000, stop_criterion=1e-16)
```

Note that for exact reproducibility, we also need to specify the seed and to use deterministic algorithms (see [](reproduce-best-practices) for more details) before the initialization of the {class}`~plenoptic.Metamer` object:

```{code-block} python
po.set_seed(0)
torch.use_deterministic_algorithms(True)
```
:::


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
met_penalty = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=0.5
)
met_penalty.load(cached_metamers / "vis_div_penalty_dists.pt", map_location="cpu")
create_metamer_figure(met_penalty);
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestVisualDiversityPenalty.test_dists -->
```{code-block} python
met = po.Metamer(img, lg_model, po.loss.l2_norm)
met.synthesize(500, stop_criterion=1e-16)

met_penalty = po.Metamer(
    img,
    lg_model,
    po.loss.l2_norm,
    penalty_function=penalty,
    penalty_lambda=0.5,
)
met_penalty.setup(initial_image=met.metamer, optimizer_kwargs={"lr": 0.03})
met_penalty.synthesize(500, stop_criterion=1e-16)
```

Note that for exact reproducibility, we also need to specify the seed and to use deterministic algorithms (see [](reproduce-best-practices) for more details) before the initialization of the {class}`~plenoptic.Metamer` object:

```{code-block} python
po.set_seed(0)
torch.use_deterministic_algorithms(True)
```
:::


```{code-cell} ipython3
met_penalty = po.Metamer(
    img, model, po.loss.l2_norm, penalty_function=penalty, penalty_lambda=0.5
)
met_penalty.load(cached_metamers / "vis_div_penalty_dists_curie.pt", map_location="cpu")
create_metamer_figure(met_penalty);
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestVisualDiversityPenalty.test_dists_curie -->
```{code-block} python
met = po.Metamer(img, lg_model, po.loss.l2_norm)
met.setup(initial_image=init_img)
met.synthesize(500, stop_criterion=1e-16)

met_penalty = po.Metamer(
    img,
    lg_model,
    po.loss.l2_norm,
    penalty_function=penalty,
    penalty_lambda=0.5,
)
met_penalty.setup(initial_image=met.metamer, optimizer_kwargs={"lr": 0.03})
met_penalty.synthesize(500, stop_criterion=1e-16)
```

Note that for exact reproducibility, we also need to specify the seed and to use deterministic algorithms (see [](reproduce-best-practices) for more details) before the initialization of the {class}`~plenoptic.Metamer` object:

```{code-block} python
po.set_seed(0)
torch.use_deterministic_algorithms(True)
```
:::
