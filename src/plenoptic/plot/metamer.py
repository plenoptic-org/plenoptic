"""Plots for understanding Metamer objects."""  # numpydoc ignore=EX01

from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from torch import Tensor

from .._synthesize import Metamer
from . import display

__all__ = [
    "metamer_representation_error",
]


def __dir__() -> list[str]:
    return __all__


def _representation_error(
    metamer: Metamer,
    iteration: int | None = None,
    iteration_selection: Literal["floor", "ceiling", "round"] = "round",
    **kwargs: Any,
) -> Tensor:
    r"""
    Get the representation error.

    This is ``metamer.model(metamer) - target_representation)``. If
    ``iteration`` is not ``None``, we use
    ``metamer.model(saved_metamer[iteration])`` instead.

    Parameters
    ----------
    metamer
        Metamer object whose representation error we want to compute.
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``metamer.store_progress>1`` (that is, the metamer was not cached on every
        iteration), then we show the cached metamer from the nearest iteration.
    iteration_selection

        How to select the relevant iteration from :attr:`saved_metamer`
        when the request iteration wasn't stored.

        When synthesis was run with ``store_progress=n`` (where ``n>1``),
        metamers are only saved every ``n`` iterations. If you request an
        iteration where a metamer wasn't saved, this determines which available
        iteration is used instead:

        * ``"floor"``: use the closest saved iteration **before** the
          requested one.

        * ``"ceiling"``: use the closest saved iteration **after** the
          requested one.

        * ``"round"``: use the closest saved iteration.

    **kwargs
        Passed to ``metamer.model.forward``.

    Returns
    -------
    representation_error
        The representation error at the specified iteration, for displaying.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.

    Warns
    -----
    UserWarning
        If the iteration for the used metamer is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``metamer.store_progress=2``).
    """  # numpydoc ignore=EX01
    if iteration is not None:
        progress = metamer.get_progress(iteration)
        image = progress["saved_metamer"].to(metamer.target_representation.device)
        metamer_rep = metamer.model(image, **kwargs)
    else:
        metamer_rep = metamer.model(metamer.metamer, **kwargs)
    return metamer_rep - metamer.target_representation


def metamer_representation_error(
    metamer: Metamer,
    batch_idx: int = 0,
    iteration: int | None = None,
    ylim: tuple[float, float] | None | Literal[False] = None,
    ax: mpl.axes.Axes | None = None,
    as_rgb: bool = False,
    **kwargs: Any,
) -> list[mpl.axes.Axes]:
    r"""
    Plot representation error showing how close we are to convergence.

    We plot ``_representation_error(metamer, iteration)``. For more details, see
    :func:`plenoptic.plot.plot_representation`.

    Parameters
    ----------
    metamer
        Metamer object whose synthesized metamer we want to display.
    batch_idx
        Which index to take from the batch dimension.
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``metamer.store_progress>1`` (that is, the metamer was not cached on every
        iteration), then we show the cached metamer from the nearest iteration.
    ylim
        If ``ylim`` is ``None``, we sets the axes' y-limits to be ``(-y_max,
        y_max)``, where ``y_max=np.abs(data).max()``. If it's ``False``, we do
        nothing. If a tuple, we use that range.
    ax
        Pre-existing axes for plot. If ``None``, we call :func:`matplotlib.pyplot.gca`.
    as_rgb
        The representation can be image-like with multiple channels, and we
        have no way to determine whether it should be represented as an RGB
        image or not, so the user must set this flag to tell us. It will be
        ignored if the response doesn't look image-like or if the model has its
        own ``plot_representation_error()`` method. Else, it will be passed to
        :func:`~plenoptic.plot.imshow`, see that methods docstring for details.
    **kwargs
        Passed to ``metamer.model.forward``.

    Returns
    -------
    axes :
        List of created axes.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.

    Warns
    -----
    UserWarning
        If the iteration for the metamer used to compute the error is not the same as
        the argument ``iteration`` (because e.g., you set ``iteration=3`` but
        ``metamer.store_progress=2``).

    See Also
    --------
    :func:`~plenoptic.plot.plot_representation`
        Function used by this one to plot representation.
    :func:`~plenoptic.plot.synthesis_status`
        Create a figure combining this with other axis-level plots to summarize
        synthesis status at a given iteration.
    :func:`~plenoptic.plot.synthesis_animshow`
        Create a video animating this and other axis-level plots changing over
        the course of synthesis.

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import torch
      >>> img = po.data.einstein()
      >>> model = po.models.Gaussian(30).eval()
      >>> po.remove_grad(model)
      >>> met = po.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> po.plot.metamer_representation_error(met)
      [<Axes: title=...Representation error...>]

    Plot on an existing axis:

    .. plot::
      :context: close-figs

      >>> import matplotlib.pyplot
      >>> fig, axes = plt.subplots(1, 2, figsize=(8, 4))
      >>> po.plot.metamer_representation_error(met, ax=axes[1])
      [<Axes: title=...Representation error...>]

    The function uses :func:`~plenoptic.plot.plot_representation`,
    which switches between :func:`~plenoptic.plot.imshow` and
    :func:`~plenoptic.plot.stem_plot` based on the shape of the
    model's output:

    .. plot::
      :context: close-figs

      >>> # Flatten the last two dimensions of the output, so it looks like a vector.
      >>> class TestModel(po.models.Gaussian):
      ...     def __init__(self, *args, **kwargs):
      ...         super().__init__(*args, **kwargs)
      ...
      ...     def forward(self, x):
      ...         return super().forward(x).flatten(-2)
      >>> model = TestModel(30).eval()
      >>> po.remove_grad(model)
      >>> met = po.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.synthesize(5)
      >>> po.plot.metamer_representation_error(met)
      [<Axes: title=...Representation error...>]

    If model has its own ``plot_representation`` method, this function will use it,
    potentially creating multiple axes (see
    :func:`~plenoptic.models.PortillaSimoncelli.plot_representation`
    ):

    .. plot::
      :context: close-figs

      >>> img = po.data.reptile_skin()
      >>> model = po.models.PortillaSimoncelli(img.shape[-2:])
      >>> met = po.MetamerCTF(img, model, po.loss.l2_norm)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamerCTF_ps.pt"))
      >>> po.plot.metamer_representation_error(met)
      [<Axes: ...>, ..., <Axes: ...>]

    If plotting on an existing axis, this function will sub-divide that axis as
    needed:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 2, figsize=(8, 4))
      >>> po.plot.synthesis_imshow(met, ax=axes[0])
      <Axes: title=...Metamer[0] [iteration=150]...>
      >>> po.plot.metamer_representation_error(met, ax=axes[1])
      [<Axes: ...>, ..., <Axes: ...>]
    """
    representation_error = _representation_error(
        metamer=metamer, iteration=iteration, **kwargs
    )
    if ax is None:
        ax = plt.gca()
    return display.plot_representation(
        metamer.model,
        representation_error,
        ax,
        title="Representation error",
        ylim=ylim,
        batch_idx=batch_idx,
        as_rgb=as_rgb,
    )
