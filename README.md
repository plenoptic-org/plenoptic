# plenoptic

[![PyPI Version](https://img.shields.io/pypi/v/plenoptic.svg)](https://pypi.org/project/plenoptic/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plenoptic/badges/version.svg)](https://anaconda.org/conda-forge/plenoptic)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/plenoptic-org/plenoptic/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.10|3.11|3.12-blue.svg)
[![Build Status](https://github.com/plenoptic-org/plenoptic/workflows/build/badge.svg)](https://github.com/plenoptic-org/plenoptic/actions?query=workflow%3Abuild)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10151131.svg)](https://doi.org/10.5281/zenodo.10151131)
[![codecov](https://codecov.io/gh/plenoptic-org/plenoptic/branch/main/graph/badge.svg?token=EDtl5kqXKA)](https://codecov.io/gh/plenoptic-org/plenoptic)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/plenoptic-org/plenoptic/1.2.0?filepath=examples)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Plenoptic_Logo_CMYK_Full_DarkMode_Wide.svg">
  <source media="(prefers-color-scheme: light)" srcset="Plenoptic_Logo_CMYK_Full_Wide.svg">
  <img alt="plenoptic logo" src="Plenoptic_Logo_CMYK_Full_Wide.svg">
</picture>


`plenoptic` is a python library for model-based synthesis of perceptual stimuli.
For `plenoptic`, models are those of visual[^1] information processing: they
accept an image[^2] as input, perform some computations, and return some output,
which can be mapped to neuronal firing rate, fMRI BOLD response, behavior on
some task, image category, etc. The intended audience is researchers in
neuroscience, psychology, and machine learning. The generated stimuli enable
interpretation of model properties through examination of features that are
enhanced, suppressed, or discarded. More importantly, they can facilitate the
scientific process, through use in further perceptual or neural experiments
aimed at validating or falsifying model predictions.

See our [documentation site](https://docs.plenoptic.org) for more details,
including how to get started!

### Installation

The best way to install `plenoptic` is via `pip`:

``` bash
$ pip install plenoptic
```

or `conda`:

``` bash
$ conda install plenoptic -c conda-forge
```

Our dependencies include [pytorch](https://pytorch.org/) and
[pyrtools](https://pyrtools.readthedocs.io/en/latest/). Installation should take
care of them (along with our other dependencies) automatically, but if you have
an installation problem (especially on a non-Linux operating system), it is
likely that the problem lies with one of those packages. [Open an
issue](https://github.com/plenoptic-org/plenoptic/issues) and we'll
try to help you figure out the problem!

See the [installation
page](https://docs.plenoptic.org/docs/branch/main/install.html) for more details,
including how to set up a virtual environment and jupyter.

### ffmpeg and videos

Several methods in this package generate videos. There are several backends
possible for saving the animations to file, see [matplotlib
documentation](https://matplotlib.org/stable/api/animation_api.html#writer-classes)
for more details. In order convert them to HTML5 for viewing (and thus, to view
in a jupyter notebook), you'll need [ffmpeg](https://ffmpeg.org/download.html)
installed and on your path as well. Depending on your system, this might already
be installed, but if not, the easiest way is probably through [conda]
(https://anaconda.org/conda-forge/ffmpeg): `conda install -c conda-forge
ffmpeg`.

To change the backend, run `matplotlib.rcParams['animation.writer'] = writer`
before calling any of the animate functions. If you try to set that `rcParam`
with a random string, `matplotlib` will tell you the available choices.

## Contents

### Synthesis methods

![](docs/images/example_synth.svg)

-   Metamers: given a model and a reference image, stochastically generate a new image whose model representation is identical to that of the reference image (a "metamer", as originally defined in the literature on Trichromacy). This method investigates what image features the model disregards entirely.
-   Eigendistortions: given a model and a reference image, compute the image perturbation that produces the smallest and largest changes in the model response space. This method investigates the image features the model considers the least and most important.
-   Maximal differentiation (MAD) competition: given two metrics that measure distance between images and a reference image, generate pairs of images that optimally differentiate the models. Specifically, synthesize a pair of images that the first model says are equi-distant from the reference while the second model says they are maximally/minimally distant from the reference. Then synthesize a second pair with the roles of the two models reversed. This method allows for efficient comparison of two metrics, highlighting the aspects in which their sensitivities differ.


### Models, Metrics, and Model Components

-   Portilla-Simoncelli texture model, which measures the statistical properties of visual textures, here defined as "repeating visual patterns."
-   Steerable pyramid, a multi-scale oriented image decomposition. The basis are oriented (steerable) filters, localized in space and frequency. Among other uses, the steerable pyramid serves as a good representation from which to build a primary visual cortex model. See the [pyrtools documentation](https://pyrtools.readthedocs.io/en/latest/index.html) for more details on image pyramids in general and the steerable pyramid in particular.
-   Structural Similarity Index (SSIM), is a perceptual similarity metric, returning a number between -1 (totally different) and 1 (identical) reflecting how similar two images are. This is based on the images' luminance, contrast, and structure, which are computed convolutionally across the images.
-   Multiscale Structrual Similarity Index (MS-SSIM), is a perceptual similarity metric similar to SSIM, except it operates at multiple scales (i.e., spatial frequencies).
-   Normalized Laplacian distance, is a perceptual distance metric based on transformations associated with the early visual system: local luminance subtraction and local contrast gain control, at six scales.


## Getting help

We communicate via several channels on Github:

-   [Discussions](https://github.com/plenoptic-org/plenoptic/discussions)
    is the place to ask usage questions, discuss issues too broad for a
    single issue, or show off what you've made with plenoptic.
-   If you've come across a bug, open an
    [issue](https://github.com/plenoptic-org/plenoptic/issues).
-   If you have an idea for an extension or enhancement, please post in the
    [ideas
    section](https://github.com/plenoptic-org/plenoptic/discussions/categories/ideas)
    of discussions first. We'll discuss it there and, if we decide to pursue it,
    open an issue to track progress.
-   See the [contributing guide](CONTRIBUTING.md) for how to get involved.

In all cases, please follow our [code of conduct](CODE_OF_CONDUCT.md).

## Citing us

If you use `plenoptic` in a published academic article or presentation, please
cite both the code by the DOI as well the JOV paper. If you are not using the
code, but just discussing the project, please cite the paper. You can click on
`Cite this repository` on the right side of the GitHub page to get a copyable
citation for the code, or use the following:

- Code: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10151131.svg)](https://doi.org/10.5281/zenodo.10151131)
- Paper:
  ``` bibtex
  @article{duong2023plenoptic,
    title={Plenoptic: A platform for synthesizing model-optimized visual stimuli},
    author={Duong, Lyndon and Bonnen, Kathryn and Broderick, William and Fiquet, Pierre-{\'E}tienne and Parthasarathy, Nikhil and Yerxa, Thomas and Zhao, Xinyuan and Simoncelli, Eero},
    journal={Journal of Vision},
    volume={23},
    number={9},
    pages={5822--5822},
    year={2023},
    publisher={The Association for Research in Vision and Ophthalmology}
  }
  ```

See the [citation
guide](https://docs.plenoptic.org/docs/branch/main/citation.html) for more
details, including citations for the different synthesis methods and
computational moels included in plenoptic.

## Support

This package is supported by the [Simons Foundation Flatiron Institute's Center
for Computational
Neuroscience](https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/).

![](docs/images/CCN-logo-wText.png)

[^1]: These methods also work with auditory models, such as in [Feather et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/ac27b77292582bc293a51055bfc994ee-Abstract.html), though we haven't yet implemented examples. If you're interested, please post in [Discussions](https://github.com/plenoptic-org/plenoptic/discussions)!

[^2]: Here and throughout the documentation, we use "image" to describe the input. The models and metrics that are included in plenoptic are intended to work on images, represented as 4d tensors. However, the synthesis methods should also work on videos (5d tensors), audio (3d tensors) and more! If you have a problem using a tensor with different dimensionality, please [open an issue](https://github.com/plenoptic-org/plenoptic/issues/new?template=bug_report.md)!
