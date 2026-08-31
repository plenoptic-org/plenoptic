(getting-started)=
# Getting started

Plenoptic can be installed using either pip or conda:

::::{tab-set}

:::{tab-item} pip
```{code-block} console
$ pip install plenoptic
```
:::

:::{tab-item} conda
```{code-block} console
$ conda install plenoptic -c conda-forge
```
:::

::::

If you are unfamiliar with stimulus synthesis, we recommend watching the [Video introduction](https://archive.org/details/vss2025-symposium-plenoptic) (and its [associated slides](https://presentations.plenoptic.org/2025-05-16_vss-symposium/slides.html)) and reading the [](conceptual_intro).

To see a minimum example of writing your own model to use with plenoptic, see [](quickstart).

New users are encouraged to work through the [](introductory_tutorial), which uses two of plenoptic's synthesis methods and steps through the kind of scientific reasoning that they facilitate. That tutorial starts with a simple {class}`~plenoptic.models.Gaussian` model and gradually adds complexity, demonstrating how synthesis methods allow us to reason about the sensitivities and invariances of computational visual models.

The exercises provide opportunities to practice using plenoptic. They all start with a brief demonstration and then prompt the user to change different aspects of the synthesis procedure. They are roughly ordered by difficulty, so users are encouraged to start with the first and proceed onwards.

::::{card}
:::{toctree}
:maxdepth: 2

quickstart
introductory_tutorial
:::
::::

::::{card}

(exercises-index)=
:::{toctree}
:caption: Exercises
:maxdepth: 1
:glob:

exercises/*
:::
::::

::::{card}
:::{toctree}
:caption: Installation

install
jupyter
:::
::::

::::{card}
:::{toctree}
:caption: Background

conceptual_intro
Video introduction <https://archive.org/details/vss2025-symposium-plenoptic>
Introduction slides <https://presentations.plenoptic.org/2025-05-16_vss-symposium/slides.html>
:::
::::
