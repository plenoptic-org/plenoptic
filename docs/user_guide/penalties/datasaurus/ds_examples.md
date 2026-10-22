(ds-examples)=
# Example notebooks

Now that we've seen that plenoptic can create these metameric datasets, you are encouraged to peruse the following notebooks for details.

:::{admonition} Synthesis efficiency
:class: attention

These synthesis procedures are all pretty quick, less than a minute on a CPU. This is because our dataset has shape `(2, 142)`, which is a good deal smaller than the `(1, 1, 256, 256)` seen in much of the other tutorials. Additionally, the computations in the `DatasaurusModel` are all relatively quick.

We thus haven't paid much attention to efficiency in the definitions of the penalty functions found in the following notebooks. This means there are some inefficient operations in the penalties themselves, such as converting numpy arrays or lists to tensors and if statements. Because the input is small and the model is quick, this doesn't slow us down much, but if you wanted to use similar penalties on much larger inputs or with slower models, it would be beneficial to ensure the penalty functions are more efficient.

:::

The following notebooks are roughly ordered by complexity. The first one, [plenoptic-logo](ds-plenoptic-logo), does not use a penalty function beyond {func}`~plenoptic.regularize.penalize_range` and thus allows us to more easily visualize the effect of the optimizers and statistical redundancies in the model. The last one, [](ds_star.md), uses a penalty that is more difficult to match while also matching the target statistics and thus uses a two-stage synthesis method. The remaining notebooks only differ in which penalty they use and are broken into groups based on the type of computations their penalty functions perform.

Each of these notebooks is intentionally brief: most of the code is hidden, and we only explain the penalty. See [](datasaurus-index) for an overview of the datasaurus dozen dataset.

## Optimizers and Redundancies

::::{grid} auto
:gutter: 1

:::{grid-item-card}
:link: ds-plenoptic-logo
:link-type: ref

```{eval-rst}
.. plot:: scripts/datasaurus.py plenoptic_logo
  :class: datasaurus-card
  :include-source: false
```
:::
::::

## Circular Penalties

::::{grid} auto
:gutter: 1

:::{grid-item-card}
:link: ds_circle
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py circle
  :class: datasaurus-card
  :include-source: false
```
:::

:::{grid-item-card}
:link: ds_bullseye
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py bullseye
  :class: datasaurus-card
  :include-source: false
```
:::

:::{grid-item-card}
:link: ds_away
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py away
  :class: datasaurus-card
  :include-source: false
```
:::

:::{grid-item-card}
:link: ds_dots
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py dots
  :class: datasaurus-card
  :include-source: false
```
:::
::::

## Line-based penalties

::::{grid} auto
:gutter: 1

:::{grid-item-card}
:link: ds_hlines
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py hlines
  :class: datasaurus-card
  :include-source: false
```
:::

:::{grid-item-card}
:link: ds_vlines
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py vlines
  :class: datasaurus-card
  :include-source: false
```
:::

:::{grid-item-card}
:link: ds_xshape
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py xshape
  :class: datasaurus-card
  :include-source: false
```
:::

:::{grid-item-card}
:link: ds_slantup
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py slantup
  :class: datasaurus-card
  :include-source: false
```
:::

:::{grid-item-card}
:link: ds_slantdown
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py slantdown
  :class: datasaurus-card
  :include-source: false
```
:::
::::

## Cluster Distance Penalties

::::{grid} auto
:gutter: 1

:::{grid-item-card}
:link: ds_polygons
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py polygons
  :class: datasaurus-card
  :include-source: false
```
:::

:::{grid-item-card}
:link: ds_oval
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py oval
  :class: datasaurus-card
  :include-source: false
```
:::
::::

## Line + Cluster Distance Penalties

::::{grid} auto
:gutter: 1

:::{grid-item-card}
:link: ds_hwidelines
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py hwidelines
  :class: datasaurus-card
  :include-source: false
```
:::

:::{grid-item-card}
:link: ds_vwidelines
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py vwidelines
  :class: datasaurus-card
  :include-source: false
```
:::
::::

## Two-stage synthesis

::::{grid} auto
:gutter: 1

:::{grid-item-card}
:link: ds_star
:link-type: doc

```{eval-rst}
.. plot:: scripts/datasaurus.py star
  :class: datasaurus-card
  :include-source: false
```
:::
::::


:::{toctree}
:maxdepth: 1
:hidden:

ds_circle.md
ds_bullseye.md
ds_away.md
ds_dots.md

ds_hlines.md
ds_vlines.md
ds_xshape.md
ds_slantup.md
ds_slantdown.md

ds_polygons.md
ds_oval.md

ds_hwidelines.md
ds_vwidelines.md

ds_star.md

:::
