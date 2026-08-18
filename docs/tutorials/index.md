(tutorials-exercises)=
# Tutorial + Exercises

The pages linked here are notebooks users can work through in order to practice using plenoptic. New users are encouraged to start with [](introductory_tutorial), which uses two of plenoptic's synthesis methods and steps through the kind of scientific reasoning that they facilitate. That starts with a simple {class}`~plenoptic.models.Gaussian` model and gradually adds complexity, demonstrating how synthesis methods allow us to reason about the sensitivities and invariances of computational visual models.

The exercises linked here provide opportunities to practice using plenoptic. They all start with a brief demonstration and then prompt the user to change different aspects of the synthesis procedure. They are roughly ordered by difficulty, so users are encouraged to start with the first and proceed onwards.

:::{admonition} Before getting started
:class: attention

You are encouraged to work through these notebooks locally, so that you gain practice writing the code yoruself, familiarizing yourself with plenoptic's syntax. View [](getting-started) for details on how to set up your environment.

If you are new to the idea of stimulus synthesis, you will probably also find it useful to peruse the Background materials linked in that section. In particular, you may find the 17 minute [introductory presentation video](https://archive.org/details/vss2025-symposium-plenoptic) informative.

:::

::::{card}
:::{toctree}

introductory_tutorial

:::
::::

::::{card}
:::{toctree}
:caption: Exercises
:maxdepth: 1
:glob:

exercises/*

:::
::::
