---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: plenoptic_venv
  language: python
  name: plenoptic_venv
---

:::{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`2_simple_eigendistortion.ipynb`**! See the button at the top right to download as markdown or pdf.

:::

# Minimal eigendistortion synthesis example

See {external+plenoptic:doc}`plenoptic User Guide <user_guide/index>` for more details.

```{code-cell} ipython3
import plenoptic as po
import torch
# needed for the plotting/animating:
import matplotlib.pyplot as plt
plt.rcParams['animation.html'] = 'html5'
# use single-threaded ffmpeg for animation writer
plt.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams['animation.ffmpeg_args'] = ['-threads', '1']
# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

The following code block:
- initializes an image and a model
- runs eigendistortion synthesis until convergence

```{code-cell} ipython3
img = po.data.einstein().to(DEVICE)
model = po.models.LuminanceGainControl(
    kernel_size=(31, 31), pad_mode="circular",
    pretrained=True, cache_filt=True
)
model.to(DEVICE)
po.remove_grad(model)
model.eval()
eig = po.Eigendistortion(img, model)
eig.synthesize(max_iter=1000)
```

Unlike metamers (and MAD Competition), eigendistortion doesn't use an iterative optimization procedure in order to synthesize its images. It uses the "power iteration method" to find the eigenvector and eigenvalue of the model's Fisher Information Matrix (see [original paper](https://www.cns.nyu.edu/pub/lcv/berardino17c-final.pdf) for details).

This means that, to ensure we have actually found the eigendistortions, we need to let the synthesis run until the synthesized images have stopped changing, i.e., until we hit the stop criterion in the synthesis above. Assuming that happens, then synthesis has succeeded.

We can then visualize our outputs:

```{code-cell} ipython3
po.plot.imshow(eig.eigendistortions, title=["Max", "Min"]);
```

And we can add them to the reference image, to see what they look like (scaled up for visibility):

```{code-cell} ipython3
po.plot.imshow(eig.image + 5*eig.eigendistortions, title=["Max", "Min"]);
```

## Different target image

Try using a different target image than the one of Einstein above and running eigendistortion synthesis until completion:

:::{admonition} Loading other images
:class: hint

Try one of the other {external+plenoptic:ref}`included images <images-api>` or use {external+plenoptic:func}`plenoptic.load_images` to load one from disk.

:::

```{code-cell} ipython3
:tags: [skip-execution]

img = # WRITE SOMETHING NEW HERE
img = img.to(DEVICE)
model = po.models.LuminanceGainControl(
    kernel_size=(31, 31), pad_mode="circular",
    pretrained=True, cache_filt=True
)
model.to(DEVICE)
po.remove_grad(model)
model.eval()
eig = po.Eigendistortion(img, model)
eig.synthesize(max_iter=1000)
po.plot.imshow([eig.eigendistortions, eig.image+5*eig.eigendistortions]);
```

## Other models

Try any of the above with a different model! Try one of the other {external+plenoptic:ref}`LGN-inspired models <models-api>`.

If you want a more complex model, see the [](4_torchvision) notebooks.
