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

import warnings

warnings.filterwarnings(
    "ignore",
    message="The default behavior of tarfile extraction",
    category=RuntimeWarning,
)
```

:::{admonition} Under development
:class: warning
This currently contains examples of the earlier MAD synthesis, but we have yet to reproduce it using `plenoptic`.
:::

:::{admonition} Download
:class: important

Download this notebook: **{nb-download}`Original_MAD.ipynb`**!

:::

# Reproducing Wang and Simoncelli, 2008 (MAD Competition)

Goal here is to reproduce original MAD Competition results, as generated using the [matlab code](https://github.com/LabForComputationalVision/MAD_Competition) originally provided by Zhou Wang and then modified by the authors. MAD Competition is a synthesis method for efficiently computing two models, by generating sets of images that minimize/maximize one model's loss while holding the other's constant. For more details, see the [](mad-nb) and [](mad-concept) notebooks.

```{code-cell} ipython3
import plenoptic as po

%matplotlib inline

%load_ext autoreload
%autoreload 2

# Download some data we'll need for this notebook
import contextlib
import os

# the contextlib.redirect_stderr here is so that we don't print out the progressbar.
# If you would like to see it, remove this line.
with contextlib.redirect_stderr(open(os.devnull, "w")):
    po.data.fetch_data("MAD_results.tar.gz")
    po.data.fetch_data("ssim_images.tar.gz")
```

## SSIM

Before we discuss MAD Competition, let's look a little at SSIM, since that's the metric used in the original paper, and which we'll be using here. Important to remember that SSIM is a similarity metric, so higher is better, and thus a value of 1 means the images are identical and it's bounded between 0 and 1.

We have tests to show that this matches the output of the MATLAB code, won't show here.

```{code-cell} ipython3
img1 = po.data.einstein()
img2 = po.data.curie()
noisy = po.tools.add_noise(img1, [2, 4, 8])
```

We can see that increasing the noise level decreases the SSIM value, but not linearly

```{code-cell} ipython3
po.metric.ssim(img1, noisy)
```

And that our noise level does match the MSE

```{code-cell} ipython3
po.metric.mse(img1, noisy)
```

## MAD Competition

The following figure shows the results of MAD Competition synthesis using the original MATLAB code. It shows the original image in the top left. We then added some Gaussian noise (with a specified standard error) to get the image right below it. The four images to the right of that are the MAD-synthesized images. The first two have the same mean-squared error (MSE) as the first image (and each other), but the best and worst SSIM value (SSIM is a similarity metric, so higher is better), while the second two have the same SSIM as the first image, but the best and worst MSE. By comparing these images, we can get a sense for what MSE and SSIM consider important for image quality.

```{code-cell} ipython3
# We need to download some additional data for this portion of the notebook.
fig, results = po.tools.external.plot_MAD_results("samp6", [128], vrange="row1", zoom=3)
```

There's lots of info here, on the outputs of the MATLAB synthesis. We will later add stuff to investigate this using `plenoptic`.

```{code-cell} ipython3
results
```
