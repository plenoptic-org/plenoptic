"""Plots for understanding Metamer objects."""  # numpydoc ignore=EX01

import re
import warnings
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from ..synthesize import Metamer
from ..tools import data, display


def plot_loss(
    metamer: Metamer,
    iteration: int | None = None,
    ax: mpl.axes.Axes | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Plot synthesis loss with log-scaled y axis.

    Plots ``metamer.losses`` over all iterations. Also plots a red dot at
    ``iteration``, to highlight the loss there. If ``iteration=None``, then the
    dot will be at the final iteration.

    Parameters
    ----------
    metamer
        Metamer object whose loss we want to plot.
    iteration
        Which iteration to display. If ``None``,  we show
        the most recent one. Negative values are also allowed.
    ax
        Pre-existing axes for plot. If ``None``, we call
        :func:`matplotlib.pyplot.gca()`.
    **kwargs
        Passed to :func:`matplotlib.pyplot.semilogy`.

    Returns
    -------
    ax
        The matplotlib axes containing the plot.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.

    See Also
    --------
    plot_synthesis_status
        Create a figure combining this with other axis-level plots to summarize
        synthesis status at a given iteration.
    animate
        Create a video animating this and other axis-level plots changing over
        the course of synthesis.

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import torch
      >>> img = po.data.einstein()
      >>> model = po.simul.Gaussian(30).eval()
      >>> po.tools.remove_grad(model)
      >>> met = po.synth.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> po.synth.metamer.plot_loss(met)
      <Axes: ... ylabel='Loss'>

    Specify an iteration:

    .. plot::
      :context: close-figs

      >>> po.synth.metamer.plot_loss(met, iteration=10)
      <Axes: ... ylabel='Loss'>

    Plot on an axis in an existing figure:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 2)
      >>> po.synth.metamer.plot_loss(met, ax=axes[1])
      <Axes: ... ylabel='Loss'>
    """
    # this warning is not relevant for this plotting function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="loss iteration and iteration for")
        progress = metamer.get_progress(iteration)

    if ax is None:
        ax = plt.gca()
    ax.semilogy(metamer.losses, **kwargs)
    ax.scatter(progress["iteration"], progress["losses"], c="r")
    ax.set(xlabel="Synthesis iteration", ylabel="Loss")
    return ax


def display_metamer(
    metamer: Metamer,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    zoom: float | None = None,
    iteration: int | None = None,
    ax: mpl.axes.Axes | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Display metamer.

    We use :func:`~plenoptic.tools.display.imshow` to display the metamer and attempt to
    automatically find the most reasonable zoom value. You can override this
    value using the zoom arg, but remember that :func:`~plenoptic.tools.display.imshow`
    is opinionated about the size of the resulting image and will throw an
    exception if the axis created is not big enough for the selected zoom.

    Parameters
    ----------
    metamer
        Metamer object whose synthesized metamer we want to display.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we assume
        image is RGB(A) and show all channels.
    zoom
        How much to zoom in / enlarge the metamer, the ratio of display pixels
        to image pixels. If ``None``, we attempt to find the best
        value ourselves.
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``metamer.store_progress>1`` (that is, the metamer was not cached on every
        iteration), then we show the cached metamer from the nearest iteration.
    ax
        Pre-existing axes for plot. If ``None``, we call :func:`matplotlib.pyplot.gca`.
    **kwargs
        Passed to :func:`~plenoptic.tools.display.imshow`.

    Returns
    -------
    ax
        The matplotlib axes containing the plot.

    Raises
    ------
    ValueError
        If ``batch_idx`` is not an int.
    IndexError
        If ``iteration`` takes an illegal value.
    IndexError
        If ``iteration`` is not ``None`` and
        :meth:`~plenoptic.synthesize.metamer.Metamer.synthesize` was called with
        ``store_progress=False``.

    Warns
    -----
    UserWarning
        If the iteration for the displayed metamer is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``metamer.store_progress=2``).

    See Also
    --------
    plot_synthesis_status
        Create a figure combining this with other axis-level plots to summarize
        synthesis status at a given iteration.
    animate
        Create a video animating this and other axis-level plots changing over
        the course of synthesis.

    Examples
    --------
    If a matplotlib figure exists, this function will use it (using
    :func:`matplotlib.pyplot.gca`):

    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import matplotlib.pyplot as plt
      >>> import torch
      >>> plt.figure()
      <Figure size ...>
      >>> img = po.data.einstein()
      >>> model = po.simul.Gaussian(30).eval()
      >>> po.tools.remove_grad(model)
      >>> met = po.synth.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> po.synth.metamer.display_metamer(met)
      <Axes: title=...Metamer [iteration=107]...>

    If no matplotlib figure exists, this function will create a new one:

    .. plot::
      :context: close-figs

      >>> # close all open figures to ensure none exist
      >>> plt.close("all")
      >>> po.synth.metamer.display_metamer(met)
      <Axes: title=...Metamer [iteration=107]...>

    Display metamer from a specified iteration (requires setting ``store_progress``
    when :meth:`~plenoptic.synthesize.metamer.Metamer.synthesize` was called):

    .. plot::
      :context: close-figs

      >>> po.synth.metamer.display_metamer(met, iteration=10)
      <Axes: title=...Metamer [iteration=10]...>

    Explicitly define the axis to use:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 2, figsize=(8, 4))
      >>> po.synth.metamer.display_metamer(met, ax=axes[1])
      <Axes: title=...Metamer [iteration=107]...>

    When plotting on an existing axis, if ``zoom=None``, this function will determine
    the best zoom level for the axis size.

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 1, figsize=(8, 8))
      >>> po.synth.metamer.display_metamer(met, ax=axes)
      <Axes: title=...Metamer [iteration=107]...dims: [256, 256] * 2.0'}>
    """
    progress = metamer.get_progress(iteration)
    try:
        image = progress["saved_metamer"]
        iter = progress["store_progress_iteration"]
    except KeyError:
        if iteration is not None:
            raise IndexError(
                "When metamer.store_progress=False, iteration must be None!"
            )
        image = metamer.metamer
        # losses will always have one extra value, the current loss.
        iter = len(metamer.losses) - 1

    if not isinstance(batch_idx, int):
        raise ValueError("batch_idx must be an integer!")
    # we're only plotting one image here, so if the user wants multiple
    # channels, they must be RGB
    as_rgb = bool(channel_idx is None and image.shape[1] > 1)
    if ax is None:
        ax = plt.gca()
    display.imshow(
        image,
        ax=ax,
        title=f"Metamer [iteration={iter}]",
        zoom=zoom,
        batch_idx=batch_idx,
        channel_idx=channel_idx,
        as_rgb=as_rgb,
        **kwargs,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax


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


def plot_representation_error(
    metamer: Metamer,
    batch_idx: int = 0,
    iteration: int | None = None,
    ylim: tuple[float, float] | None | Literal[False] = None,
    ax: mpl.axes.Axes | None = None,
    as_rgb: bool = False,
    **kwargs: Any,
) -> list[mpl.axes.Axes]:
    r"""
    Plot distance ratio showing how close we are to convergence.

    We plot ``_representation_error(metamer, iteration)``. For more details, see
    :func:`plenoptic.tools.display.plot_representation`.

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
        :func:`~plenoptic.tools.display.imshow`, see that methods docstring for details.
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
    plot_synthesis_status
        Create a figure combining this with other axis-level plots to summarize
        synthesis status at a given iteration.
    animate
        Create a video animating this and other axis-level plots changing over
        the course of synthesis.

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import torch
      >>> img = po.data.einstein()
      >>> model = po.simul.Gaussian(30).eval()
      >>> po.tools.remove_grad(model)
      >>> met = po.synth.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> po.synth.metamer.plot_representation_error(met)
      [<Axes: title=...Representation error...>]

    Plot on an existing axis:

    .. plot::
      :context: close-figs

      >>> import matplotlib.pyplot
      >>> fig, axes = plt.subplots(1, 2, figsize=(8, 4))
      >>> po.synth.metamer.plot_representation_error(met, ax=axes[1])
      [<Axes: title=...Representation error...>]

    The function uses :func:`~plenoptic.tools.display.plot_representation`,
    which switches between :func:`~plenoptic.tools.display.imshow` and
    :func:`~plenoptic.tools.display.clean_stem_plot` based on the shape of the
    model's output:

    .. plot::
      :context: close-figs

      >>> # Flatten the last two dimensions of the output, so it looks like a vector.
      >>> class TestModel(po.simul.Gaussian):
      ...     def __init__(self, *args, **kwargs):
      ...         super().__init__(*args, **kwargs)
      ...
      ...     def forward(self, x):
      ...         return super().forward(x).flatten(-2)
      >>> model = TestModel(30).eval()
      >>> po.tools.remove_grad(model)
      >>> met = po.synth.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.synthesize(5)
      >>> po.synth.metamer.plot_representation_error(met)
      [<Axes: title=...Representation error...>]

    If model has its own ``plot_representation`` method, this function will use it,
    potentially creating multiple axes (see
    :func:`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli.plot_representation`
    ):

    .. plot::
      :context: close-figs

      >>> img = po.data.reptile_skin()
      >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
      >>> met = po.synth.MetamerCTF(img, model, po.tools.optim.l2_norm)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamerCTF_ps.pt"))
      >>> po.synth.metamer.plot_representation_error(met)
      [<Axes: ...>, ..., <Axes: ...>]

    If plotting on an existing axis, this function will sub-divide that axis as
    needed:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 2, figsize=(8, 4))
      >>> po.synth.metamer.display_metamer(met, ax=axes[0])
      <Axes: title=...Metamer [iteration=150]...>
      >>> po.synth.metamer.plot_representation_error(met, ax=axes[1])
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


def plot_pixel_values(
    metamer: Metamer,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    iteration: int | None = None,
    ylim: tuple[float, float] | Literal[False] = False,
    ax: mpl.axes.Axes | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    r"""
    Plot histogram of pixel values of target image and its metamer.

    As a way to check the distributions of pixel intensities and see
    if there's any values outside the allowed range

    Parameters
    ----------
    metamer
        Metamer object with the images whose pixel values we want to compare.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) images).
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``metamer.store_progress>1`` (that is, the metamer was not cached on every
        iteration), then we show the cached metamer from the nearest iteration.
    ylim
        If tuple, the ylimit to set for this axis. If ``False``, we leave
        it untouched.
    ax
        Pre-existing axes for plot. If ``None``, we call
        :func:`matplotlib.pyplot.gca()`.
    **kwargs
        Passed to :func:`matplotlib.pyplot.hist`.

    Returns
    -------
    ax
        Created axes.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.

    Warns
    -----
    UserWarning
        If the iteration used for ``saved_metamer`` is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``metamer.store_progress=2``).

    See Also
    --------
    plot_synthesis_status
        Create a figure combining this with other axis-level plots to summarize
        synthesis status at a given iteration.
    animate
        Create a video animating this and other axis-level plots changing over
        the course of synthesis.

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import torch
      >>> img = po.data.einstein()
      >>> model = po.simul.Gaussian(30).eval()
      >>> po.tools.remove_grad(model)
      >>> met = po.synth.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> po.synth.metamer.plot_pixel_values(met)
      <Axes: ... 'Histogram of pixel values'...>

    Plot pixel values from a specified iteration (requires setting ``store_progress``
    when :meth:`~plenoptic.synthesize.metamer.Metamer.synthesize` was called):

    .. plot::
      :context: close-figs

      >>> po.synth.metamer.plot_pixel_values(met, iteration=10)
      <Axes: ... 'Histogram of pixel values'...>

    Plot on an existing axis:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 2, figsize=(8, 4))
      >>> po.synth.metamer.plot_pixel_values(met, ax=axes[1])
      <Axes: ... 'Histogram of pixel values'...>
    """

    def _freedman_diaconis_bins(a: np.ndarray) -> int:
        """
        Calculate number of hist bins using Freedman-Diaconis rule.

        Copied from seaborn.

        Parameters
        ----------
        a
            The array to histogram.

        Returns
        -------
        n_bins
            Number of bins to use for histogram.
        """  # numpydoc ignore=EX01
        # From https://stats.stackexchange.com/questions/798/
        a = np.asarray(a)
        iqr = np.diff(np.percentile(a, [0.25, 0.75]))[0]
        if len(a) < 2:
            return 1
        h = 2 * iqr / (len(a) ** (1 / 3))
        # fall back to sqrt(a) bins if iqr is 0
        if h == 0:
            return int(np.sqrt(a.size))
        else:
            return int(np.ceil((a.max() - a.min()) / h))

    kwargs.setdefault("alpha", 0.4)
    progress = metamer.get_progress(iteration)
    try:
        met = progress["saved_metamer"]
        iter = progress["store_progress_iteration"]
    except KeyError:
        if iteration is not None:
            raise IndexError(
                "When metamer.store_progress=False, iteration must be None!"
            )
        met = metamer.metamer
        # losses will always have one extra value, the current loss.
        iter = len(metamer.losses) - 1
    image = metamer.image[batch_idx]
    if channel_idx is not None:
        image = image[channel_idx]
        met = met[channel_idx]
    image = data.to_numpy(image).flatten()
    met = data.to_numpy(met).flatten()

    if ax is None:
        ax = plt.gca()
    ax.hist(
        met,
        bins=min(_freedman_diaconis_bins(image), 50),
        label=f"Metamer [iteration={iter}]",
        **kwargs,
    )
    ax.hist(
        image,
        bins=min(_freedman_diaconis_bins(image), 50),
        label="Target image",
        **kwargs,
    )
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title("Histogram of pixel values")
    return ax


def _check_included_plots(to_check: list[str] | dict[str, float], to_check_name: str):
    """
    Check whether the user wanted us to create plots that we can't.

    Helper function for :func:`plot_synthesis_status` and :func:`animate`.

    Raises a ``ValueError`` if ``to_check`` contains any values that are not allowed.

    Parameters
    ----------
    to_check
        The variable to check. We ensure that it doesn't contain any extra (not
        allowed) values. If a list, we check its contents. If a dict, we check
        its keys.
    to_check_name
        Name of the ``to_check`` variable, used in the error message.

    Raises
    ------
    ValueError
        If ``to_check`` takes an illegal value.
    """  # numpydoc ignore=EX01
    allowed_vals = [
        "display_metamer",
        "plot_loss",
        "plot_representation_error",
        "plot_pixel_values",
        "misc",
    ]
    try:
        vals = to_check.keys()
    except AttributeError:
        vals = to_check
    not_allowed = [v for v in vals if v not in allowed_vals]
    if not_allowed:
        raise ValueError(
            f"{to_check_name} contained value(s) {not_allowed}! "
            f"Only {allowed_vals} are permissible!"
        )


def _setup_synthesis_fig(
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float, float] | None = None,
    included_plots: list[str] = [
        "display_metamer",
        "plot_loss",
        "plot_representation_error",
    ],
    display_metamer_width: float = 1,
    plot_loss_width: float = 1,
    plot_representation_error_width: float = 1,
    plot_pixel_values_width: float = 1,
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes], dict[str, int]]:
    """
    Set up figure for :func:`plot_synthesis_status`.

    Creates figure with enough axes for the all the plots you want. Will
    also create index in ``axes_idx`` for them if you haven't done so already.

    If ``fig=None``, all axes will be on the same row and have the same width.
    If you want them to be on different rows, will need to initialize ``fig``
    yourself and pass that in. For changing width, change the corresponding
    ``*_width`` arg, which gives width relative to other axes. So if you want
    the axis for the ``representation_error`` plot to be twice as wide as the
    others, set ``representation_error_width=2``.

    Parameters
    ----------
    fig
        The figure to plot on or ``None``. If ``None``, we create a new figure.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows for more
        fine-grained control of the resulting figure. Probably only helpful if fig is
        also defined. Possible keys: ``"loss"``, ``"representation_error"``,
        ``"pixel_values"``, ``"misc"``. Values should all be ints. If you tell this
        function to create a plot that doesn't have a corresponding key, we find the
        lowest int that is not already in the dict, so if you have
        axes that you want unchanged, place their idx in ``"misc"``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have relative width=1 correspond to 5.
    included_plots
        Which plots to include. Must be some subset of ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values'``.
    display_metamer_width
        Relative width of the axis for the synthesized metamer.
    plot_loss_width
        Relative width of the axis for loss plot.
    plot_representation_error_width
        Relative width of the axis for representation error plot.
    plot_pixel_values_width
        Relative width of the axis for image pixel intensities histograms.

    Returns
    -------
    fig
        The figure to plot on.
    axes
        List or array of axes contained in fig.
    axes_idx
        Dictionary identifying the idx for each plot type.
    """  # numpydoc ignore=EX01
    n_subplots = 0
    axes_idx = axes_idx.copy()
    width_ratios = []
    if "display_metamer" in included_plots:
        n_subplots += 1
        width_ratios.append(display_metamer_width)
        if "display_metamer" not in axes_idx:
            axes_idx["display_metamer"] = data._find_min_int(axes_idx.values())
    if "plot_loss" in included_plots:
        n_subplots += 1
        width_ratios.append(plot_loss_width)
        if "plot_loss" not in axes_idx:
            axes_idx["plot_loss"] = data._find_min_int(axes_idx.values())
    if "plot_representation_error" in included_plots:
        n_subplots += 1
        width_ratios.append(plot_representation_error_width)
        if "plot_representation_error" not in axes_idx:
            axes_idx["plot_representation_error"] = data._find_min_int(
                axes_idx.values()
            )
    if "plot_pixel_values" in included_plots:
        n_subplots += 1
        width_ratios.append(plot_pixel_values_width)
        if "plot_pixel_values" not in axes_idx:
            axes_idx["plot_pixel_values"] = data._find_min_int(axes_idx.values())
    if fig is None:
        width_ratios = np.array(width_ratios)
        if figsize is None:
            # we want (5, 5) for each subplot, with a bit of room between
            # each subplot
            figsize = ((width_ratios * 5).sum() + width_ratios.sum() - 1, 5)
        width_ratios = width_ratios / width_ratios.sum()
        fig, axes = plt.subplots(
            1,
            n_subplots,
            figsize=figsize,
            gridspec_kw={"width_ratios": width_ratios},
        )
        if n_subplots == 1:
            axes = [axes]
    else:
        axes = fig.axes
    # make sure misc contains all the empty axes
    misc_axes = axes_idx.get("misc", [])
    if not hasattr(misc_axes, "__iter__"):
        misc_axes = [misc_axes]
    all_axes = []
    for i in axes_idx.values():
        # so if it's a list of ints
        if hasattr(i, "__iter__"):
            all_axes.extend(i)
        else:
            all_axes.append(i)
    misc_axes += [i for i, _ in enumerate(fig.axes) if i not in all_axes]
    axes_idx["misc"] = misc_axes
    return fig, axes, axes_idx


def plot_synthesis_status(
    metamer: Metamer,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    iteration: int | None = None,
    ylim: tuple[float, float] | None | Literal[False] = None,
    vrange: tuple[float, float] | str = "indep1",
    zoom: float | None = None,
    plot_representation_error_as_rgb: bool = False,
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float, float] | None = None,
    included_plots: list[str] = [
        "display_metamer",
        "plot_loss",
        "plot_representation_error",
    ],
    width_ratios: dict[str, float] = {},
) -> tuple[mpl.figure.Figure, dict[str, int]]:
    r"""
    Make a plot showing synthesis status.

    We create several subplots to analyze this. The plots to include are
    specified by including their name in the ``included_plots`` list. All plots
    can be created separately using the method with the same name.

    Parameters
    ----------
    metamer
        Metamer object whose status we want to plot.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) image).
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``metamer.store_progress>1`` (that is, the metamer was not cached on every
        iteration), then we show the cached metamer from the nearest iteration.
    ylim
        The ylimit to use for the representation_error plot. We pass
        this value directly to ``plot_representation_error``.
    vrange
        The vrange option to pass to :func:`display_metamer()`. See
        docstring of :func:`~plenoptic.tools.display.imshow` for possible values.
    zoom
        How much to zoom in / enlarge the metamer, the ratio
        of display pixels to image pixels. If ``None``, we
        attempt to find the best value ourselves.
    plot_representation_error_as_rgb
        The representation can be image-like with multiple channels, and we
        have no way to determine whether it should be represented as an RGB
        image or not, so the user must set this flag to tell us. It will be
        ignored if the response doesn't look image-like or if the
        model has its own plot_representation_error() method. Else, it will
        be passed to :func:`~plenoptic.tools.display.imshow`, see that method's
        docstring for details.
    fig
        If ``None``, we create a new figure. otherwise we assume this is
        an empty figure that has the appropriate size and number of
        subplots.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values',
        'misc'``. Values should all be ints. If you tell this function to
        create a plot that doesn't have a corresponding key, we find the lowest
        int that is not already in the dict, so if you have axes that you want
        unchanged, place their idx in ``'misc'``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have each axis be of size ``(5, 5)``.
    included_plots
        Which plots to include. Must be some subset of ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values'``.
    width_ratios
        If ``width_ratios`` is an empty dictionary, all plots will have the
        same width. To change that, specify their relative widths; keys should
        be strings (possible values same as ``included_plots``) and values should
        be floats specifying their relative width.

    Returns
    -------
    fig
        The figure containing this plot.
    axes_idx
        Dictionary giving index of each plot.

    Raises
    ------
    ValueError
        If ``metamer.metamer`` object is not 3d or 4d.
    ValueError
        If the ``iteration is not None`` and the given ``metamer`` object was run
        with ``store_progress=False``.

    Warns
    -----
    UserWarning
        If the iteration used for ``saved_metamer`` is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``metamer.store_progress=2``).

    See Also
    --------
    display_metamer
        One of this function's axis-level component functions: display metamer at
        a given synthesis iteration.
    plot_loss
        One of this function's axis-level component functions: plot synthesis loss
        over iterations.
    plot_representation_error
        One of this function's axis-level component functions: plot error in model
        representation at a given synthesis iteration.
    plot_pixel_values
        One of this function's axis-level component functions: plot histogram of
        pixel values in target image and metamer at a given synthesis iteration.
    animate
        Create a video that animates this figure over synthesis iteration.

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import torch
      >>> img = po.data.einstein()
      >>> model = po.simul.Gaussian(30).eval()
      >>> po.tools.remove_grad(model)
      >>> met = po.synth.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> po.synth.metamer.plot_synthesis_status(met)
      (<Figure size ...>, {'display_metamer': 0, ...})

    If model has its own ``plot_representation`` method, this function will use it
    for plotting the representation error (see
    :func:`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli.plot_representation`
    ):

    .. plot::
      :context: close-figs

      >>> img = po.data.reptile_skin()
      >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
      >>> met = po.synth.MetamerCTF(img, model, po.tools.optim.l2_norm)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamerCTF_ps.pt"))
      >>> po.synth.metamer.plot_synthesis_status(met)
      (<Figure size ...>, {'display_metamer': 0, ...})

    Change the included plots:

    .. plot::
      :context: close-figs

      >>> included_plots = ["plot_loss", "plot_pixel_values"]
      >>> po.synth.metamer.plot_synthesis_status(met, included_plots=included_plots)
      (<Figure size ...>, {'plot_loss': 0, ...})

    Adjust width of included plots:

    .. plot::
      :context: close-figs

      >>> width_ratios = {"plot_representation_error": 3}
      >>> po.synth.metamer.plot_synthesis_status(met, width_ratios=width_ratios)
      (<Figure size ...>, {'display_metamer': 0, ...})

    Plot on existing figure, ignoring some axes and rearranging others:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 5, figsize=(16, 4))
      >>> axes_idx = {"misc": [0, 3], "plot_loss": 4}
      >>> po.synth.metamer.plot_synthesis_status(met, fig=fig, axes_idx=axes_idx)
      (<Figure size ...>, {'misc': [0, 3], ...})
    """
    if iteration is not None and not metamer.store_progress:
        raise ValueError(
            "synthesis() was run with store_progress=False, "
            "cannot specify which iteration to plot (only"
            " last one, with iteration=None)"
        )
    if metamer.metamer.ndim not in [3, 4]:
        raise ValueError(
            "plot_synthesis_status() expects 3 or 4d data;"
            "unexpected behavior will result otherwise!"
        )
    _check_included_plots(included_plots, "included_plots")
    _check_included_plots(width_ratios, "width_ratios")
    _check_included_plots(axes_idx, "axes_idx")
    width_ratios = {f"{k}_width": v for k, v in width_ratios.items()}
    fig, axes, axes_idx = _setup_synthesis_fig(
        fig, axes_idx, figsize, included_plots, **width_ratios
    )

    def check_iterables(i: int, vals: list | tuple) -> bool:
        """
        Determine whether i is in vals.

        Works with an iterable of iterables and iterable of non-iterables.

        Parameters
        ----------
        i
            The value we're looking for.
        vals
            The iterable it might be in.

        Returns
        -------
        contained
            Whether i is in vals.
        """  # numpydoc ignore=EX01
        for j in vals:
            try:
                # then it's an iterable
                if i in j:
                    return True
            except TypeError:
                # then it's not an iterable
                if i == j:
                    return True

    if "display_metamer" in included_plots:
        display_metamer(
            metamer,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=iteration,
            ax=axes[axes_idx["display_metamer"]],
            zoom=zoom,
            vrange=vrange,
        )
    if "plot_loss" in included_plots:
        plot_loss(metamer, iteration=iteration, ax=axes[axes_idx["plot_loss"]])
    if "plot_representation_error" in included_plots:
        plot_representation_error(
            metamer,
            batch_idx=batch_idx,
            iteration=iteration,
            ax=axes[axes_idx["plot_representation_error"]],
            ylim=ylim,
            as_rgb=plot_representation_error_as_rgb,
        )
        # this can add a bunch of axes, so this will try and figure
        # them out
        new_axes = [
            i
            for i, _ in enumerate(fig.axes)
            if not check_iterables(i, axes_idx.values())
        ] + [axes_idx["plot_representation_error"]]
        axes_idx["plot_representation_error"] = new_axes
    if "plot_pixel_values" in included_plots:
        plot_pixel_values(
            metamer,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=iteration,
            ax=axes[axes_idx["plot_pixel_values"]],
        )
    return fig, axes_idx


def animate(
    metamer: Metamer,
    framerate: int = 10,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    ylim: str | None | tuple[float, float] | Literal[False] = None,
    vrange: tuple[float, float] | str = (0, 1),
    zoom: float | None = None,
    plot_representation_error_as_rgb: bool = False,
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float, float] | None = None,
    included_plots: list[str] = [
        "display_metamer",
        "plot_loss",
        "plot_representation_error",
    ],
    width_ratios: dict[str, float] = {},
) -> mpl.animation.FuncAnimation:
    r"""
    Animate synthesis progress.

    This is essentially the figure produced by
    ``metamer.plot_synthesis_status`` animated over time, for each stored
    iteration.

    This functions returns a matplotlib FuncAnimation object. See our
    documentation (e.g., :ref:`quickstart-nb`) for examples on how to view it in
    a Jupyter notebook. In order to save, use ``anim.save(filename)``. In either
    case, this can take a while and you'll need the appropriate writer installed
    and on your path, e.g., ffmpeg, imagemagick, etc). See
    :doc:`matplotlib documentation <matplotlib:api/animation_api>` for more details.

    Parameters
    ----------
    metamer
        Metamer object whose synthesis we want to animate.
    framerate
        How many frames a second to display.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) image).
    ylim
        The y-limits of the representation_error plot:

        * If a tuple, then this is the ylim of all plots

        * If ``None``, then all plots have the same limits, all
          symmetric about 0 with a limit of
          ``np.abs(representation_error).max()`` (for the initial
          representation_error).

        * If ``False``, don't modify limits.

        * If a string, must be ``"rescale"`` or of the form ``"rescaleN"``,
          where N can be any integer. If ``"rescaleN"``, we rescale the
          limits every N frames (we rescale as if ``ylim=None``). If
          ``"rescale"``, then we do this 10 times over the course of the
          animation.

    vrange
        The vrange option to pass to :func:`display_metamer()`. See
        docstring of :func:`~plenoptic.tools.display.imshow` for possible values.
    zoom
        How much to zoom in / enlarge the metamer, the ratio
        of display pixels to image pixels. If ``None``, we
        attempt to find the best value ourselves.
    plot_representation_error_as_rgb
        The representation can be image-like with multiple channels, and we
        have no way to determine whether it should be represented as an RGB
        image or not, so the user must set this flag to tell us. It will be
        ignored if the representation doesn't look image-like or if the
        model has its own ``plot_representation_error()`` method. Else, it will
        be passed to :func:`~plenoptic.tools.display.imshow`, see that method's
        docstring for details.
    fig
        If ``None``, create the figure from scratch. Else, should be an empty
        figure with enough axes (the expected use here is have same-size
        movies with different plots).
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values',
        'misc'``. Values should all be ints. If you tell this function to
        create a plot that doesn't have a corresponding key, we find the lowest
        int that is not already in the dict, so if you have axes that you want
        unchanged, place their idx in ``'misc'``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have each axis be of size ``(5, 5)``.
    included_plots :
        Which plots to include. Must be some subset of ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values'``.
    width_ratios
        If ``width_ratios`` is an empty dictionary, all plots will have the
        same width. To change that, specify their relative widths; keys should
        be strings (possible values same as ``included_plots``) and values should
        be floats specifying their relative width.

    Returns
    -------
    anim
        The animation object. In order to view, must convert to HTML
        or save.

    Raises
    ------
    ValueError
        If the given ``metamer`` object was run with ``store_progress=False``.
    ValueError
        If ``metamer.metamer`` object is not 3d or 4d.
    ValueError
        If we do not know how to interpret the value of ``ylim``.

    See Also
    --------
    display_metamer
        One of this function's axis-level component functions: display metamer at
        a given synthesis iteration.
    plot_loss
        One of this function's axis-level component functions: plot synthesis loss
        over iterations.
    plot_representation_error
        One of this function's axis-level component functions: plot error in model
        representation at a given synthesis iteration.
    plot_pixel_values
        One of this function's axis-level component functions: plot histogram of
        pixel values in target image and metamer at a given synthesis iteration.
    plot_synthesis_status
        Create a figure that shows a frame from this movie: the synthesis status at
        a given iteration.

    Notes
    -----
    Unless specified, we use the ffmpeg backend, which requires that you have
    ffmpeg installed and on your path (https://ffmpeg.org/download.html). To use
    a different, use the matplotlib rcParams:
    ``matplotlib.rcParams['animation.writer'] = writer``, see `matplotlib
    documentation
    <https://matplotlib.org/stable/api/animation_api.html#writer-classes>`_ for
    more details.

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import torch
      >>> img = po.data.einstein()
      >>> model = po.simul.Gaussian(30).eval()
      >>> po.tools.remove_grad(model)
      >>> met = po.synth.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> ani = po.synth.metamer.animate(met)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-1.gif")

    .. image:: animate-example-1.gif

    This function can only be used if
    :meth:`~plenoptic.synthesize.metamer.Metamer.synthesize` was called with
    ``store_progress``.

    >>> import plenoptic as po
    >>> img = po.data.einstein()
    >>> model = po.simul.Gaussian(30).eval()
    >>> po.tools.remove_grad(model)
    >>> met = po.synth.Metamer(img, model)
    >>> met.to(torch.float64)
    >>> met.synthesize(5)
    >>> ani = po.synth.metamer.animate(met)
    Traceback (most recent call last):
    ValueError: synthesize() was run with store_progress=False...

    If model has its own ``plot_representation`` method, this function will use it
    for plotting the representation error (see
    :func:`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli.plot_representation`
    ):

    .. plot::
      :context: close-figs

      >>> img = po.data.reptile_skin()
      >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
      >>> met = po.synth.MetamerCTF(img, model, po.tools.optim.l2_norm)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamerCTF_ps.pt"))
      >>> ani = po.synth.metamer.animate(met)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-2.gif")

    .. image:: animate-example-2.gif

    Change the included plots:

    .. plot::
      :context: close-figs

      >>> included_plots = ["plot_loss", "plot_pixel_values"]
      >>> ani = po.synth.metamer.animate(met, included_plots=included_plots)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-3.gif")

    .. image:: animate-example-3.gif

    Adjust width of included plots:

    .. plot::
      :context: close-figs

      >>> width_ratios = {"plot_representation_error": 3}
      >>> ani = po.synth.metamer.animate(met, width_ratios=width_ratios)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-4.gif")

    .. image:: animate-example-4.gif

    Use an existing figure, ignoring some axes and rearranging others:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 5, figsize=(16, 4))
      >>> axes_idx = {"misc": [0, 3], "plot_loss": 4}
      >>> ani = po.synth.metamer.animate(met, fig=fig, axes_idx=axes_idx)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-5.gif")

    .. image:: animate-example-5.gif
    """
    if not metamer.store_progress:
        raise ValueError(
            "synthesize() was run with store_progress=False, cannot animate!"
        )
    if metamer.metamer.ndim not in [3, 4]:
        raise ValueError(
            "animate() expects 3 or 4d data; unexpected behavior will result otherwise!"
        )
    _check_included_plots(included_plots, "included_plots")
    _check_included_plots(width_ratios, "width_ratios")
    _check_included_plots(axes_idx, "axes_idx")
    if metamer.target_representation.ndimension() == 4:
        # we have to do this here so that we set the
        # ylim_rescale_interval such that we never rescale ylim
        # (rescaling ylim messes up an image axis)
        ylim = False
    try:
        if ylim.startswith("rescale"):
            try:
                ylim_rescale_interval = int(ylim.replace("rescale", ""))
            except ValueError:
                # then there's nothing we can convert to an int there
                ylim_rescale_interval = int((metamer.saved_metamer.shape[0] - 1) // 10)
                if ylim_rescale_interval == 0:
                    ylim_rescale_interval = int(metamer.saved_metamer.shape[0] - 1)
            ylim = None
        else:
            raise ValueError(f"Don't know how to handle ylim {ylim}!")
    except AttributeError:
        # this way we'll never rescale
        ylim_rescale_interval = len(metamer.saved_metamer) + 1
    # we run plot_synthesis_status to initialize the figure if either fig is
    # None or if there are no titles on any axes, which we assume means that
    # it's an empty figure
    if fig is None or not any([ax.get_title() for ax in fig.axes]):
        fig, axes_idx = plot_synthesis_status(
            metamer=metamer,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=0,
            figsize=figsize,
            ylim=ylim,
            vrange=vrange,
            zoom=zoom,
            fig=fig,
            axes_idx=axes_idx,
            included_plots=included_plots,
            plot_representation_error_as_rgb=plot_representation_error_as_rgb,
            width_ratios=width_ratios,
        )
    # grab the artist for the second plot (we don't need to do this for the
    # metamer or representation plot, because we use the update_plot
    # function for that)
    if "plot_loss" in included_plots:
        scat = fig.axes[axes_idx["plot_loss"]].collections[0]
    # can have multiple plots
    if "plot_representation_error" in included_plots:
        try:
            rep_error_axes = [
                fig.axes[i] for i in axes_idx["plot_representation_error"]
            ]
        except TypeError:
            # in this case, axes_idx['plot_representation_error'] is not iterable and
            # so is a single value
            rep_error_axes = [fig.axes[axes_idx["plot_representation_error"]]]
    else:
        rep_error_axes = []
    if "display_metamer" in included_plots:
        fig.axes[axes_idx["display_metamer"]].set_title("Metamer")

    if metamer.target_representation.ndimension() == 4:
        if "plot_representation_error" in included_plots:
            warnings.warn(
                "Looks like representation is image-like, haven't fully"
                " thought out how to best handle rescaling color ranges yet!"
            )
        # replace the bit of the title that specifies the range,
        # since we don't make any promises about that. we have to do
        # this here because we need the figure to have been created
        for ax in rep_error_axes:
            ax.set_title(re.sub(r"\n range: .* \n", "\n\n", ax.get_title()))

    def movie_plot(i: int) -> list[mpl.artist.Artist]:
        """
        Matplotlib function for animation.

        Update plots for frame ``i``.

        Parameters
        ----------
        i
            The frame to plot.

        Returns
        -------
        artists
            The updated matplotlib artists.
        """  # numpydoc ignore=EX01
        # this warning is not relevant for animate
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="loss iteration and iteration for"
            )
            artists = []
            if "display_metamer" in included_plots:
                artists.extend(
                    display.update_plot(
                        fig.axes[axes_idx["display_metamer"]],
                        data=metamer.saved_metamer[i],
                        batch_idx=batch_idx,
                    )
                )
            if "plot_representation_error" in included_plots:
                rep_error = _representation_error(
                    metamer,
                    iteration=min(i * metamer.store_progress, len(metamer.losses) - 1),
                )

                # we pass rep_error_axes to update, and we've grabbed
                # the right things above
                artists.extend(
                    display.update_plot(
                        rep_error_axes,
                        batch_idx=batch_idx,
                        model=metamer.model,
                        data=rep_error,
                    )
                )
                # again, we know that rep_error_axes contains all the axes
                # with the representation ratio info
                if (
                    (i + 1) % ylim_rescale_interval == 0
                    and metamer.target_representation.ndimension() == 3
                ):
                    display.rescale_ylim(rep_error_axes, rep_error)

            if "plot_pixel_values" in included_plots:
                # this is the dumbest way to do this, but it's simple --
                # clearing the axes can cause problems if the user has, for
                # example, changed the tick locator or formatter. not sure how
                # to handle this best right now
                fig.axes[axes_idx["plot_pixel_values"]].clear()
                plot_pixel_values(
                    metamer,
                    batch_idx=batch_idx,
                    channel_idx=channel_idx,
                    iteration=min(i * metamer.store_progress, len(metamer.losses) - 1),
                    ax=fig.axes[axes_idx["plot_pixel_values"]],
                )
            if "plot_loss" in included_plots:
                # loss always contains values from every iteration, but everything
                # else will be subsampled.
                x_val = metamer._convert_iteration(i, False)
                scat.set_offsets((x_val, metamer.losses[x_val]))
                artists.append(scat)
            # as long as blitting is True, need to return a sequence of artists
            return artists

    # don't need an init_func, since we handle initialization ourselves
    anim = mpl.animation.FuncAnimation(
        fig,
        movie_plot,
        frames=len(metamer.saved_metamer),
        blit=True,
        interval=1000.0 / framerate,
        repeat=False,
    )
    plt.close(fig)
    return anim
