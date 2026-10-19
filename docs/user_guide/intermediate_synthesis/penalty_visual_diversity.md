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
# Using penalties to incentivize visual diversity

[](how-to-penalty) showed the basics of working with custom penalty functions, but their flexibility allows for us to bias stimulus synthesis in a wide variety of different ways. In this notebook, we will show how to use the {class}`~plenoptic.process.SteerablePyramidFreq` to increase perceptual diversity among model metamers for the {class}`~plenoptic.models.LuminanceGainControl` model.

:::{admonition} pyiqa
:class: attention

The last section of this notebook uses an additional package, [pyiqa](https://iqa-pytorch.readthedocs.io/), for their pytorch implementation of the DISTS image quality metric ([original publication](https://www.cns.nyu.edu/pub/lcv/ding20a-preprint.pdf)). Install it in your environment in order to run the full notebook successfully.

You'll still be able to run the first several sections without it.

:::

```{code-cell} ipython3
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

import plenoptic as po

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72

# We will not run synthesis in this notebook, instead loading cached results
cached_metamers = po.data.fetch_data("visual_diversity_penalty.tar.gz")
```

:::{admonition} This notebook retrieves cached synthesis results / Reproducibility
:class: warning dropdown

Instead of performing synthesis in this notebook, we have cached the result of most of these syntheses online and download them for display.

This notebook contains collapsible admonitions which contain the steps necessary to run synthesis yourself after each call to {func}`~plenoptic.Metamer.load`. If you have a GPU available, the only synthesis calls that take longer than a minute are those which use DISTS, in the final section (if you do not have a GPU, they will all take longer).

If you wish to exactly reproduce our results, you will also need to do the following:
- Call `torch.use_deterministic_algorithms(True)`. This only needs to be done once.
- Set the seed to 0: `po.set_seed(0)`. The seed only affects the initialization of the metamer attribute as a patch of noise. It should thus be done either before the call to {func}`torch.rand_like` in the creation of `init_img` and `init_img_curie`, as we show below.

See [](reproduce-best-practices) for more details.
:::

- for experiments, we often need to generate many metamers, and it is typical to generate multiple metamers with the same model+target for different repeats
- often, synthesis initialized with different samples of noise (Feather, Portilla, Wallis, Freeman), often Gaussian uniform / white noise (Gatys just says white noise but torch style transfer tutorial uses randn (also checked their code), Broderick uses rand)
- and so these repeats use independent samples of noise
- here, we use LG model and demonstrate that this procedure leads to two metamers that are perceptually very similar (as pointed out by Wallis, Broderick)
- one way of addressing this (as used by Broderick) is to use different initializations, which does succeed in this
- here, present another way of doing so: using a custom penalty function to incentive the two metamers to be as different as possible, for some measure of "different"
    - this can be combined with different initializations, as we show
- we present two examples:
    - encourage their spatial frequency content to be as different as possible in specific steerpyr bands
    - use an image quality metric to be make them as different as possible
- try other bands! try other metrics!
- in both cases, transforming distance to penalty by using exp(-x), as in original example
- when initializing with Curie, that metamer is generally darker, we often see a high-frequency ghost of her face, and we don't have the high frequencies present in the noise-initialized metamer
- vertical penalty organizes the high frequencies in one of the metamers into vertical bands, whereas the other one looks a lot like the un-penalized metamer
- low penalty makes one super dark while the other is much brighter, has these white splotches, and looks to have a 2 cycles / image sinusoid subtly overlaid
    - only one where Curie-init'ed metamer isn't darker, though that may(?) change with different white noise initializations?
- DISTS arranges noise into a glyph like pattern (may have to zoom into see), while other stays more or less untouched
    - DISTS requires two-stage synthesis, more complex. only show how the second step is initialized here, original is like the un-penalized synthesis, see dropdown for details

- What adding penalty does is adds an additional constraint and so shrink the "good solution set" to somewhere else
- initializatnoi point is "start in different space", penalty is "shrink acceptable solution set"

The following hidden cells defines a function we'll use to visualize our metamer synthesis results, expand it for more details.

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

## Metamer synthesis without custom penalty

Throughout this notebook, we'll use the {class}`~plenoptic.models.LuminanceGainControl` model, which convolves its input with a center-surround filter, applies gain control / divisive normalization, and then rectifies the output. Thus, the model is sensitive only to mid-range frequencies and is insensitive to absolute pixel values.

```{code-cell} ipython3
model = po.models.LuminanceGainControl(
    31, pretrained=True, pad_mode="circular", cache_filt=True
).eval()
po.remove_grad(model)
model.to(torch.float64)
```

Before we start writing custom penalties, let's see what the default behavior is. By default, {class}`~plenoptic.Metamer` uses {func}`plenoptic.regularize.penalize_range` as the penalty function (to keep the metamer image's pixel values between 0 and 1). This function does nothing to encourage multiple metamers to be different (or similar) to each other, so let's see how it behaves!

Below, we create our target image, `img`. We start with a picture of Einstein and downsample it by a factor of 2 (this makes the high-frequency patterns generated in much of the following examples more obvious). We then repeat it along the batch dimension. Because {class}`~plenoptic.models.LuminanceGainControl` (like the other models in plenoptic) operates on each batch dimension independently, this allows us to synthesize two independent metamer images simultaneously.

```{code-cell} ipython3
img = po.data.einstein().to(torch.float64)
img = po.process.blur_downsample(img)
img = img.repeat(2, 1, 1, 1)

# Test that our model independently operates on these images
test_imgs = torch.cat([po.data.einstein(), po.data.curie()])
test_imgs = test_imgs.to(torch.float64)
rep = model(test_imgs)
print(f"Input shape: {test_imgs.shape}")
print(f"Output shape: {rep.shape}")
assert (rep[0] == model(test_imgs[0])).all()
assert (rep[1] == model(test_imgs[1])).all()
```

By default, {class}`~plenoptic.Metamer` initializes synthesis with independent samples of white noise. For this notebook, we'll create this initialization image explicitly so we can visualize it, but note that this is the same as the default behavior.

```{code-cell} ipython3
po.set_seed(0)
init_img = torch.rand_like(img)
title = [
    "Target image[0]",
    "Target image[1]",
    "Initial metamer[0]",
    "Initial metamer[1]",
]
po.plot.imshow([img, init_img], col_wrap=2, zoom=2, title=title);
```

In the figure above, we can see our target and initial images. Over the course of synthesis, plenoptic will update the pixels of the bottom image in each column so that the model representation matches that of the top.

The two images in the top row are identical, but the two in the bottom are not! Humans aren't very sensitive to the differences between independent samples of noise at this resolution, so they look pretty similar (I have to focus pretty hard in specific locations in order to find the differences). But in fact, on a pixel-by-pixel basis, these two initial images are more different from each other than they are from the target, which we can see by computing the mean-squared error:

```{code-cell} ipython3
init_diff = po.loss.mse(init_img[0], init_img[1])
print(f"MSE between indepedendent noise samples: {init_diff}")
img_diff = po.loss.mse(init_img[0], img[0])
print(f"MSE between Einstein image and noise: {img_diff}")
```

Okay, with all that out of the way, let's initialize our {class}`~plenoptic.Metamer` object using our target image and model. We will also set the loss function to {func}`plenoptic.loss.l2_norm` (rather than the default, {func}`plenoptic.loss.mse`), which improves the quality of our resulting metamer. We then load in and visualize the results of this synthesis:

```{code-cell} ipython3
met = po.Metamer(img, model, po.loss.l2_norm)
met.load(cached_metamers / "vis_div_penalty_basic.pt", map_location="cpu")
create_metamer_figure(met);
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestVisualDiversityPenalty.test_basic -->
```{code-block} python
met.setup(initial_image=init_img)
met.synthesize(1000, stop_criterion=1e-16)
```
:::

In the above figure, we can see the target image in the first column, followed by the two independent metamer images. The second row shows the model representation of each image in the top row, and you can see that they're all identical -- that is, metamer synthesis has succeeded and we've found two new metamers. The metamers both look like the target image of Einstein, except with lots of high frequency noise added and, if you look carefully, the finer details (e.g., in his hair) removed. Additionally, the image looks a little brighter (that is, the average pixel values are a little higher).

However, those two metamers look really perceptually similar! Again, they're not actually that similar on a pixel-by-pixel basis, as we can clarify by computing the mean-squared error between them:

```{code-cell} ipython3
met_diff = po.loss.mse(met.metamer[0], met.metamer[1])
print(f"MSE between metamers: {met_diff}")
```

But they do look really similar. This leads to the question: do all metamers for this model look like this? Or is this a result of our optimization procedure and the choices we made?

## Changing the initialization point

One simple way to try to find more visually-diverse metamers is to change the image used to initialize the synthesis procedure, as done by {cite:alp}`Broderick2025-foveat-metam`. In plenoptic, we can do that by creating an appropriately-shaped tensor and passing it to {func}`plenoptic.Metamer.setup` (as seen in the dropdown below). Let's create an initial image tensor that contains one sample of white noise and a picture of Marie Curie:

```{code-cell} ipython3
po.set_seed(0)
curie = po.process.blur_downsample(po.data.curie())
white_noise = torch.rand_like(curie)
init_img_curie = torch.cat([white_noise, curie])
po.plot.imshow(init_img_curie, zoom=2);
```

By changing where metamer synthesis starts, we affect where it ends up. That's because metamer synthesis is a high-dimensional non-convex optimization problem, the solutions of which are often dependent upon their initialization. There is (likely) a huge set of potential metamers for any given model, and so we're hoping that by choosing two starting points which appear very different, that our final metamers will also appear more distinct. As above, we initialize the {class}`~plenoptic.Metamer` object with the same target image, model, and loss, and then load and visualize our cached results:

```{code-cell} ipython3
met = po.Metamer(img, model, po.loss.l2_norm)
met.load(cached_metamers / "vis_div_penalty_basic_curie.pt", map_location="cpu")
create_metamer_figure(met);
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestVisualDiversityPenalty.test_basic_curie -->
```{code-block} python
met.setup(initial_image=init_img_curie)
met.synthesize(1000, stop_criterion=1e-16)
```
:::

The figure above has the same layout as before. As before, we can see that all three images in the bottom row are identical, and thus the three images in the top row are all metamers. The first metamer, in the second column, looks the same as the result from the default metamer synthesis: the target with high-frequency noise. The second, however, looks fairly different: it looks like a slightly blurred version of the target image with a faint outline of Marie Curie's face living in the highest frequencies, and the overall image looks a little darker than the target. All of those features are a result of our initialization: the initial image was dark everywhere except for the center and metamer synthesis has preserved the highest frequencies found in the initial image. In the first metamer, those preserved high frequencies show up as the noise pattern, and in the second, as the outline of Curie's face.

While this setting of the initialization to find visually different metamers works, let's see what we can accomplish if we choose the {attr}`~plenoptic.Metamer.penalty_function` to find more visually distinct metamers. Doing so is akin to shrinking the space of possible metamer images, and can be done in tandem with setting the initial image.

## Using penalty_function to find visually distinct metamers

In this section, we will create custom penalty functions to encourage our two simultaneously-synthesized metamers to be more perceptually different from each other. The question is "what do we mean by 'perceptually different'?" Here, we'll show three examples from two different types of penalties. We will first show that we can use {class}`~plenoptic.process.SteerablePyramidFreq` to compute the average spectral energy in a range of orientation and spatial frequency bands and use these numbers to encourage the spatial frequency content to be different. We'll do this first for vertical orientations across all frequencies, and then for low frequencies across all orientations. Finally, we'll show that we can set up a penalty that maximizes an image quality metric, [DISTS](https://www.cns.nyu.edu/pub/lcv/ding20a-preprint.pdf), to encourage the two metamers to be as different as possible according to that metric. For each of these examples, we'll demonstrate the results with the default and Marie Curie initializations, as above, so we can see how the initialization interacts with the penalty function.

Note that we are not claiming that our choices here are "the best" in any way --- you are encouraged to try tweaking the spectral energy penalty, try out different image quality metrics, or develop your own completely novel penalties! If you find something interesting, please [let us know!](https://github.com/plenoptic-org/plenoptic/discussions/new?category=show-and-tell)

### Encourage different spectral energy

```{code-cell} ipython3
pyr = po.process.SteerablePyramidFreq(img.shape[-2:])
pyr.to(img.dtype)


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

#### Different vertical power

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
met.setup(initial_image=init_img)
met.synthesize(1000, stop_criterion=1e-16)
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
met.setup(initial_image=init_img_curie)
met.synthesize(1000, stop_criterion=1e-16)
```
:::

#### Different low frequency power

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
met.setup(initial_image=init_img)
met.synthesize(3000, stop_criterion=1e-16)
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
met.setup(initial_image=init_img_curie)
met.synthesize(3000, stop_criterion=1e-16)
```
:::


### Maximize an image quality metric

:::{admonition} pyiqa
:class: attention

`pyiqa` must be installed in your environment to run the rest of this notebook.

:::

```{code-cell} ipython3
import pyiqa

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
met.setup(initial_image=init_img_curie)
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
:::
