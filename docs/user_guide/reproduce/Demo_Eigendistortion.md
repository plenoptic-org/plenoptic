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

Download the executed notebook: **{nb-download}`Demo_Eigendistortion.ipynb`**!

Run it in your browser: **{binder}`Demo_Eigendistortion.ipynb`**!

:::

:::{attention}
The eigendistortion synthesis investigated in this notebook takes a long time to run, especially if you don't have a GPU available. Therefore, we have cached the result of these syntheses online and only download them for investigation in this notebook.
:::

(demo-eigendistortions)=
# Reproducing Berardino et al., 2017 (Eigendistortions)
Author: Lyndon Duong, Jan 2021

In this demo, we will be reproducing eigendistortions first presented in [Berardino et al 2017](https://arxiv.org/abs/1710.02266). We'll be using a Front End model of the human visual system (called "On-Off" in the paper), as well as an early layer of VGG16. The [Front End model](plenoptic.models.OnOff) is a simple convolutional neural network with a normalization nonlinearity, loosely based on biological retinal/geniculate circuitry.

![Front-end model](/_static/images/front_end_model.png)

This signal-flow diagram shows an input being decomposed into two channels, with each being luminance and contrast normalized, and ending with a ReLu.

## What do eigendistortions tell us?

Our perception is influenced by our internal representation (neural responses) of the external world. Eigendistortions are rank-ordered directions in image space, along which a model's responses are more sensitive. `Plenoptic`'s {class}`~plenoptic.Eigendistortion` provides an easy way to synthesize eigendistortions for any PyTorch model.

```{code-cell} ipython3
import einops
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

# we do not actually run synthesis in this notebook, so the cpu is fine.
DEVICE = torch.device("cpu")
```

## Input preprocessing
Let's load the parrot image used in the paper and display it:

```{code-cell} ipython3
# crop the image to be square:
image_tensor = po.data.parrot().to(DEVICE).to(torch.float64)
image_tensor = po.process.center_crop(image_tensor, min(image_tensor.shape[-2:]))

print("Torch image shape:", image_tensor.shape)

po.plot.imshow(image_tensor);
```

Since the Front-end OnOff model only has two channel outputs, we can easily visualize the feature maps.
We'll apply a circular mask to this model's inputs to avoid edge artifacts in the synthesis.

```{code-cell} ipython3
mdl_f = po.models.OnOff(
    kernel_size=(31, 31), pretrained=True, apply_mask=True, cache_filt=True
)
po.remove_grad(mdl_f)
mdl_f = mdl_f.to(DEVICE).to(image_tensor.dtype)
mdl_f.eval()

response_f = mdl_f(image_tensor)
po.plot.imshow(
    response_f,
    title=["on channel response", "off channel response"],
);
```

## Synthesizing eigendistortions

### Front-end model: eigendistortion synthesis
Now that we have our Front End model set up, we can synthesize eigendistortions! This is done easily just by calling {func}`~plenoptic.Eigendistortion.synthesize` after instantiating the {class}`~plenoptic.Eigendistortion` object. We'll synthesize the top and bottom `k`, representing the most- and least-noticeable eigendistortions for this model.

The paper synthesizes the top and bottom `k=1` eigendistortions, but we'll set `k>1` so the algorithm converges/stabilizes faster.

```{code-cell} ipython3
# synthesize the top and bottom k distortions
eigendist_f = po.Eigendistortion(image=image_tensor, model=mdl_f)
# this synthesis takes a long time to run, so we load in a cached version.
# see the following admonition for how to run this yourself
eigendist_f.load(
    po.data.fetch_data("berardino_onoff.pt"),
    tensor_equality_atol=1e-7,
    map_location=DEVICE,
)
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestDemoEigendistortion.test_berardino_onoff[eigendist_f:eig] -->
```{code-block} python
eigendist_f.synthesize(k=3, method="power", max_iter=2000)
```
:::


### Front-end model: eigendistortion display

Once synthesized, we can plot the distortion on the image using {func}`~plenoptic.plot.eigendistortion_imshow_all`. Feel free to adjust the constant `distortion_scale` that scales the amount of each distortion on the image.

```{code-cell} ipython3
po.plot.eigendistortion_imshow_all(
    eigendist_f,
    distortion_scale=3,
    suptitle="OnOff",
);
```

### VGG16: eigendistortion synthesis

Following the lead of Berardino et al. (2017), let's compare the Front End model's eigendistortion to those of an early layer of VGG16! VGG16 takes as input color images, so we'll need to repeat the grayscale parrot along the RGB color dimension. We'll also apply the ImageNet normalization to the parrot image before initializing the {class}`~plenoptic.Eigendistortion` object.

:::{admonition} FeatureExtractorModel
:class: note

For more information about the way we're using VGG16 with plenoptic and an alternative way of handling `norm`, see [](feature_extractor) and {class}`~plenoptic.models.FeatureExtractorModel`.

:::

```{code-cell} ipython3
weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
transform = weights.transforms()
norm = torchvision.transforms.Normalize(transform.mean, transform.std)
image_tensor3 = norm(image_tensor.repeat(1, 3, 1, 1))
vgg = torchvision.models.vgg16(weights=weights, progress=False)
vgg.eval().to(DEVICE).to(image_tensor3.dtype)
# "layer 3" according to Berardino et al (2017)
mdl_v = po.models.FeatureExtractorModel(vgg, "features.11")
po.remove_grad(mdl_v)

eigendist_v = po.Eigendistortion(image_tensor3, mdl_v)
# this synthesis takes a long time to run, so we load in a cached version.
# see the following admonition for how to run this yourself
eigendist_v.load(
    po.data.fetch_data("berardino_vgg16.pt"),
    tensor_equality_atol=1e-7,
    map_location=DEVICE,
)
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestDemoEigendistortion.test_berardino_vgg16[eigendist_v:eig] -->
```{code-block} python
eigendist_v.synthesize(k=2, method="power", max_iter=5000)
```
:::

### VGG16: eigendistortion display

We can now display the most- and least-noticeable eigendistortions as before, then compare their quality to those of the Front-end model.

```{code-cell} ipython3
def unnormalize(x):
    std = torch.as_tensor(transform.std, device=x.device, dtype=x.dtype)
    std = einops.rearrange(std, "c -> 1 c 1 1")
    mean = torch.as_tensor(transform.mean, device=x.device, dtype=x.dtype)
    mean = einops.rearrange(mean, "c -> 1 c 1 1")
    return x * std + mean


po.plot.eigendistortion_imshow_all(
    eigendist_v,
    distortion_scale=[15, 100],
    suptitle="VGG16",
    process_image=unnormalize,
);
```

## Final thoughts

To rigorously test which of these model's representations are more human-like, we'll have to conduct a perceptual experiment. For now, we'll just leave it to you to eyeball and decide which distortions are more or less noticeable!
