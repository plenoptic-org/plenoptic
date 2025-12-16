(index-doc)=
# plenoptic

[![PyPI Version](https://img.shields.io/pypi/v/plenoptic.svg)](https://pypi.org/project/plenoptic/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plenoptic/badges/version.svg)](https://anaconda.org/conda-forge/plenoptic)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/plenoptic-org/plenoptic/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.10|3.11|3.12-blue.svg)
[![Build Status](https://github.com/plenoptic-org/plenoptic/workflows/build/badge.svg)](https://github.com/plenoptic-org/plenoptic/actions?query=workflow%3Abuild)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10151130.svg)](https://doi.org/10.5281/zenodo.10151130)
[![codecov](https://codecov.io/gh/plenoptic-org/plenoptic/branch/main/graph/badge.svg?token=EDtl5kqXKA)](https://codecov.io/gh/plenoptic-org/plenoptic)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/plenoptic-org/plenoptic/1.1.0?filepath=examples)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)

:::{figure} /_static/images/Plenoptic_Logo_CMYK_Full_Wide.svg
:alt: plenoptic logo
:figwidth: 100%
:align: center
:::

`plenoptic` is a python library for model-based synthesis of perceptual stimuli. For `plenoptic`, models are those of visual [^footnote-1] information processing: they accept an image[^footnote-2] as input, perform some computations, and return some output, which can be mapped to neuronal firing rate, fMRI BOLD response, behavior on some task, image category, etc. The intended audience is researchers in neuroscience, psychology, and machine learning. The generated stimuli enable interpretation of model properties through examination of features that are enhanced, suppressed, or discarded. More importantly, they can facilitate the scientific process, through use in further perceptual or neural experiments aimed at validating or falsifying model predictions.

## Getting started

- If you are unfamiliar with stimulus synthesis, see the [](conceptual-intro) for an in-depth introduction.
- Otherwise, see the [](quickstart-nb) tutorial.

### Installation

The best way to install `plenoptic` is via `pip`:

```{code-block} console
$ pip install plenoptic
```

or `conda`:

```{code-block} console
$ conda install plenoptic -c conda-forge
```

See the [](install-doc) page for more details, including how to set up an isolated virtual environment (recommended).

(package-contents)=

## Contents

:::{figure} /_static/images/example_synth.svg
:alt: The three synthesis methods included in plenoptic
:figwidth: 100%
:::

### Synthesis methods

- [Metamers](metamer-nb): given a model and a reference image, stochastically generate a new image whose model representation is identical to that of the reference image (a "metamer", as originally defined in the literature on Trichromacy). This method makes explicit those features that the model retains/discards.

  - Example papers: {cite:alp}`Portilla2000-param-textur`, {cite:alp}`Freeman2011-metam-ventr-stream`, {cite:alp}`Deza2019-towar-metam`, {cite:alp}`Feather2019-metam`, {cite:alp}`Wallis2019-image-conten`, {cite:alp}`Ziemba2021-oppos-effec`

- [Eigendistortions](eigendistortion-nb): given a model and a reference image, compute the image perturbations that produce the smallest/largest change in the model response space. These are the image changes to which the model is least/most sensitive, respectively.

  - Example papers: {cite:alp}`Berardino2017-eigen`

- [Maximal differentiation (MAD) competition](mad-nb): given a reference image and two models that measure distance
  between images, generate pairs of images that optimally differentiate the models. Specifically, synthesize a pair of images that are equi-distant from the reference image according to model-1, but maximally/minimally distant according to model-2. Synthesize a second pair with the roles of the two models reversed. This method allows for efficient comparison of two metrics, highlighting the aspects in which their sensitivities most differ.

  - Example papers: {cite:alp}`Wang2008-maxim-differ`

### Models, Metrics, and Model Components

- Steerable pyramid, {cite:alp}`Simoncelli1992-shift-multi, Simoncelli1995-steer-pyram`, is a multi-scale oriented image decomposition. Images are decomposed with a family of oriented filters, localized in space and frequency, similar to the "Gabor functions" commonly used to model receptive fields in primary visual cortex. The critical difference is that the pyramid organizes these filters so as to efficiently cover the 4D space of (x,y) positions, orientations, and scales, enabling efficient interpolation and interpretation ([further info](https://www.cns.nyu.edu/~eero/STEERPYR/)). See the [pyrtools documentation](https://pyrtools.readthedocs.io/en/latest/index.html) for more details on python tools for image pyramids in general and the steerable pyramid in particular.
- Portilla-Simoncelli texture model, {cite:alp}`Portilla2000-param-textur`, which computes a set of image statistics that capture the appearance of visual textures ([further info](https://www.cns.nyu.edu/~lcv/texture/)).
- Structural Similarity Index (SSIM), {cite:alp}`Wang2004-image-qualit-asses`, is a perceptual similarity metric, that takes two images and returns a value between -1 (totally different) and 1 (identical) reflecting their similarity ([further info](https://www.cns.nyu.edu/~lcv/ssim)).
- Multiscale Structural Similarity Index (MS-SSIM), {cite:alp}`Wang2003-multis`, is an extension of SSIM that operates jointly over multiple scales.
- Normalized Laplacian distance, {cite:alp}`Laparra2016-percep-image,Laparra2017-percep-optim`, is a perceptual distance metric based on transformations associated with the early visual system: local luminance subtraction and local contrast gain control, at six scales ([further info](https://www.cns.nyu.edu/~lcv/NLPyr/)).


## Getting help

We communicate via several channels on Github:

- To report a bug, open an [issue](https://github.com/plenoptic-org/plenoptic/issues).
- To send suggestions for extensions or enhancements, please post in the [ideas section](https://github.com/plenoptic-org/plenoptic/discussions/categories/ideas) of discussions first. We'll discuss it there and, if we decide to pursue it, open an issue to track progress.
- To ask usage questions, discuss broad issues, or show off what you've made with plenoptic, go to [Discussions](https://github.com/plenoptic-org/plenoptic/discussions).
- To contribute to the project, see the [contributing guide](https://github.com/plenoptic-org/plenoptic/blob/main/CONTRIBUTING.md).

In all cases, we request that you respect our [code of conduct](https://github.com/plenoptic-org/plenoptic/blob/main/CODE_OF_CONDUCT.md).

## Citing us

If you use `plenoptic` in a published academic article or presentation, please cite us! See the [](citation-doc) for more details.

:::{toctree}
:caption: Basic concepts
:glob: true
:titlesonly: true

install
jupyter
conceptual_intro
models
tutorials/*
citation
:::

:::{toctree}
:caption: Synthesis method introductions
:glob: true
:titlesonly: true

tutorials/intro/*
:::

:::{toctree}
:caption: Models and metrics
:glob: true
:titlesonly: true

tutorials/models/*
tutorials/models/portilla_simoncelli/ps_index
:::

:::{toctree}
:caption: Synthesis method examples
:glob: true
:titlesonly: true

tutorials/applications/*
:::

:::{toctree}
:caption: Advanced usage
:glob: true
:maxdepth: 1

synthesis
tips
reproducibility
API Documentation <api>
tutorials/advanced/*
:::

[^footnote-1]: These methods also work with auditory models, such as in [Feather et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/ac27b77292582bc293a51055bfc994ee-Abstract.html) though we haven't yet implemented examples. If you're interested, please post in [Discussions](<https://github.com/plenoptic-org/plenoptic/discussions)>)!

[^footnote-2]: Here and throughout the documentation, we use "image" to describe the input. The models and metrics that are included in plenoptic are intended to work on images, represented as 4d tensors. However, the synthesis methods should also work on videos (5d tensors), audio (3d tensors) and more! If you have a problem using a tensor with different dimensionality, please [open an issue](https://github.com/plenoptic-org/plenoptic/issues/new?template=bug_report.md)!

This package is supported by the [Center for Computational Neuroscience](https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/), in the Flatiron Institute of the Simons Foundation.

:::{image} /_static/images/CCN-logo-wText.png
:align: center
:alt: Flatiron Institute Center for Computational Neuroscience logo
:::
