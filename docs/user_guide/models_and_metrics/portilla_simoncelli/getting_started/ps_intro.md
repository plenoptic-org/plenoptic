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

:::{admonition} Download
:class: important

Download this notebook: **{nb-download}`ps_intro.ipynb`**!

:::

(ps-intro)=
# What is a visual texture?

This notebook describes "(visual) textures", as used throughout this series of notebooks about the Portilla-Simoncelli texture model and the original paper, {cite:alp}`Portilla2000-param-textur`.

```{code-cell} ipython3
import matplotlib.pyplot as plt

import plenoptic as po

%load_ext autoreload
%autoreload 2

# We need to download some additional images for this notebook.
IMG_PATH = po.data.fetch_data("portilla_simoncelli_images.tar.gz")

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72
```

The simplest definition of a texture is a repeating visual pattern. Textures encompass a wide variety of images, including natural patterns such as bark or fur, artificial ones such as brick, and computer-generated ones such as the Julesz patterns ([Julesz 1978](https://link.springer.com/article/10.1007/BF00336998), [Yellot 1993](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-10-5-777)). Below we load some examples.

```{code-cell} ipython3
# Load and display a set of visual textures
def display_images(im_files, title=None):
    images = po.tools.load_images(im_files)
    fig = po.plot.imshow(images, col_wrap=4, title=None)
    if title is not None:
        fig.suptitle(title, y=1.05)


natural = [
    "3a",
    "6a",
    "8a",
    "14b",
    "15c",
    "15d",
    "15e",
    "15f",
    "16c",
    "16b",
    "16a",
]
artificial = ["4a", "4b", "14a", "16e", "14e", "14c", "5a"]
hand_drawn = ["5b", "13a", "13b", "13c", "13d"]

im_files = [IMG_PATH / f"fig{num}.jpg" for num in natural]
display_images(im_files, "Natural textures")
```

```{code-cell} ipython3
im_files = [IMG_PATH / f"fig{num}.jpg" for num in artificial]
display_images(im_files, "Artificial textures")
```

```{code-cell} ipython3
im_files = [IMG_PATH / f"fig{num}.jpg" for num in hand_drawn]
display_images(im_files, "Hand-drawn / computer-generated textures")
```

Why do vision scientists care about visual textures? Quoting from the beginning of {cite:alp}`Portilla2000-param-textur`:

> Vision is the process of extracting information from the images that enter the eye. The set of all possible images is vast, and yet only a small fraction of these are likely to be encountered in a natural setting... Nevertheless, it has proven difficult to characterize this set of "natural" images, using either deterministic or statistical models. The class of images that we commonly call "visual texture" seems most amenable to statistical modeling. Loosely speaking, texture images are spatially homogeneous and consist of repeated elements, often subject to some randomization in their location, size, color, orientation, etc. Julesz pioneered the statistical characterization of textures by hypothesizing that the Nth-order joint empirical densities of image pixels (for some unspecified N), could be used to partition textures into classes that are preattentively indistinguishable to a human observer (Julesz, 1962).

The phrase "indistinguishable to a human observer" in that last sentence should grab your attention. As discussed in the [Metamer tutorial](metamer-nb), this is the same idea as perceptual metamers!

In the Portilla-Simoncelli paper, the authors developed a model to measure the statistical properties of visual textures and then used metamer synthesis was used in conjunction with this model to demonstrate the necessity of different properties of the visual texture. Throughout the rest of the notebooks in this section, we will use some of these example textures to demonstrate aspects of the Portilla-Simoncelli model.

## Further reading

Now that you have a grasp on what textures are, check out the [](ps-basic-synth) notebook to see how to use the Portilla-Simoncelli model in plenoptic.
