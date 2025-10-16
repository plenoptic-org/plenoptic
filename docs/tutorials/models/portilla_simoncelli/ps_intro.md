---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
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

# We need to download some additional images for this notebook. In order to do so,
# we use an optional dependency, pooch. If the following raises an ImportError or
# ModuleNotFoundError
# then install pooch in your plenoptic environment and restart your kernel.
from plenoptic.data.fetch import fetch_data

IMG_PATH = fetch_data("portilla_simoncelli_images.tar.gz")

# so that relative sizes of axes created by po.imshow and others look right
plt.rcParams["figure.dpi"] = 72
```

The simplest definition is a repeating visual pattern. Textures encompass a wide variety of images, including natural patterns such as bark or fur, artificial ones such as brick, and computer-generated ones such as the Julesz patterns ([Julesz 1978](https://link.springer.com/article/10.1007/BF00336998), [Yellot 1993](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-10-5-777)). Below we load some examples.

The Portilla-Simoncelli model was developed to measure the statistical properties of visual textures. Metamer synthesis was used (and can be used) in conjunction with the Portilla-Simoncelli texture model to demonstrate the necessity of different properties of the visual texture. Throughout the notebooks in this section, we will use some of these example textures to demonstrate aspects of the Portilla-Simoncelli model.

```{code-cell} ipython3
# Load and display a set of visual textures
def display_images(im_files, title=None):
    images = po.tools.load_images(im_files)
    fig = po.imshow(images, col_wrap=4, title=None)
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
display_images(im_files, "Articial textures")
```

```{code-cell} ipython3
im_files = [IMG_PATH / f"fig{num}.jpg" for num in hand_drawn]
display_images(im_files, "Hand-drawn / computer-generated textures")
```

## Further reading

Now that you have a grasp on what textures are, check out the [](ps-basic-synth) notebook to see how to use the Portilla-Simoncelli model in plenoptic.
