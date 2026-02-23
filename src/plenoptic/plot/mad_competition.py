"""Plots for understanding MADCompetition objects."""

import warnings
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyrtools.tools.display import make_figure as pt_make_figure

from .. import tensors
from ..synthesize import MADCompetition
from . import display

__all__ = [
    "mad_loss",
    "mad_image",
    "mad_pixel_values",
    "mad_synthesis_status",
    "mad_animate",
    "mad_image_all",
    "mad_loss_all",
]


def __dir__() -> list[str]:
    return __all__


def mad_loss(
    mad: MADCompetition,
    iteration: int | None = None,
    axes: list[mpl.axes.Axes] | mpl.axes.Axes | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Plot metric losses.

    Plots ``mad.optimized_metric_loss`` and ``mad.reference_metric_loss`` on two
    separate axes, over all iterations. Also plots a red dot at ``iteration``,
    to highlight the loss there. If ``iteration=None``, then the dot will be at
    the final iteration.

    Parameters
    ----------
    mad
        MADCompetition object whose loss we want to plot.
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed.
    axes
        Pre-existing axes for plot. If a list of axes, must be the two axes to
        use for this plot. If a single axis, we'll split it in half
        horizontally. If ``None``, we call :func:`matplotlib.pyplot.gca()`.
    **kwargs
        Passed to :func:`matplotlib.pyplot.semilogy`.

    Returns
    -------
    axes :
        The matplotlib axes containing the plot.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.

    Notes
    -----
    We plot ``abs(mad.losses)`` because if we're maximizing the synthesis
    metric, we minimized its negative. By plotting the absolute value, we get
    them all on the same scale.
    """
    # this warning is not relevant for this plotting function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="loss iteration and iteration for")
        progress = mad.get_progress(iteration)

    if axes is None:
        axes = plt.gca()
    if not hasattr(axes, "__iter__"):
        axes = display.clean_up_axes(
            axes, False, ["top", "right", "bottom", "left"], ["x", "y"]
        )
        gs = axes.get_subplotspec().subgridspec(1, 2)
        fig = axes.figure
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    losses = [mad.reference_metric_loss, mad.optimized_metric_loss]
    names = ["reference_metric_loss", "optimized_metric_loss"]
    for ax, loss, name in zip(axes, losses, names):
        ax.plot(loss, **kwargs)
        ax.scatter(progress["iteration"], progress[name], c="r")
        ax.set(xlabel="Synthesis iteration", ylabel=name.capitalize().replace("_", " "))
    return ax


def mad_image(
    mad: MADCompetition,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    zoom: float | None = None,
    iteration: int | None = None,
    ax: mpl.axes.Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Display MAD image.

    We use :func:`~plenoptic.tools.display.imshow` to display the synthesized image and
    attempt to automatically find the most reasonable zoom value. You can override this
    value using the zoom arg, but remember that :func:`~plenoptic.tools.display.imshow`
    is opinionated about the size of the resulting image and will throw an Exception if
    the axis created is not big enough for the selected zoom.

    Parameters
    ----------
    mad
        MADCompetition object whose MAD image we want to display.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we assume
        image is RGB(A) and show all channels.
    zoom
        How much to zoom in / enlarge the synthesized image, the ratio
        of display pixels to image pixels. If ``None``, we
        attempt to find the best value ourselves.
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``mad.store_progress>1`` (that is, the MAD image was not cached on every
        iteration), then we show the cached MAD image from the nearest iteration.
    ax
        Pre-existing axes for plot. If ``None``, we call :func:`matplotlib.pyplot.gca`.
    title
        Title to add to axis. If ``None``, we use ``"MAD Image [iteration={iter}]"``,
        where ``iter`` gives the iteration corresponding to the displayed image.
    **kwargs
        Passed to :func:`~plenoptic.tools.display.imshow`.

    Returns
    -------
    ax :
        The matplotlib axes containing the plot.

    Raises
    ------
    ValueError
        If ``batch_idx`` is not an int.
    IndexError
        If ``iteration`` takes an illegal value.

    Warns
    -----
    UserWarning
        If the iteration used for ``saved_mad_image`` is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``mad.store_progress=2``).
    """
    progress = mad.get_progress(iteration)
    try:
        image = progress["saved_mad_image"]
        iter = progress["store_progress_iteration"]
    except KeyError:
        if iteration is not None:
            raise IndexError("When mad.store_progress=False, iteration must be None!")
        image = mad.mad_image
        # losses will always have one extra value, the current loss.
        iter = len(mad.losses) - 1

    if batch_idx is None:
        raise ValueError("batch_idx must be an integer!")
    # we're only plotting one image here, so if the user wants multiple
    # channels, they must be RGB
    as_rgb = bool(channel_idx is None and image.shape[1] > 1)
    if ax is None:
        ax = plt.gca()
    if title is None:
        title = f"MAD Image [iteration={iter}]"
    display.imshow(
        image,
        ax=ax,
        title=title,
        zoom=zoom,
        batch_idx=batch_idx,
        channel_idx=channel_idx,
        as_rgb=as_rgb,
        **kwargs,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax


def mad_pixel_values(
    mad: MADCompetition,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    iteration: int | None = None,
    ylim: tuple[float] | Literal[False] = False,
    ax: mpl.axes.Axes | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    r"""
    Plot histogram of pixel values of reference and MAD images.

    As a way to check the distributions of pixel intensities and see
    if there's any values outside the allowed range.

    Parameters
    ----------
    mad
        MADCompetition object with the images whose pixel values we want to compare.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) images).
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``mad.store_progress>1`` (that is, the MAD image was not cached on every
        iteration), then we use the cached MAD image from the nearest iteration.
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
    ax :
        Creates axes.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.

    Warns
    -----
    UserWarning
        If the iteration used for ``saved_mad_image`` is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``mad.store_progress=2``).
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
        """
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
    progress = mad.get_progress(iteration)
    try:
        mad_image = progress["saved_mad_image"]
        iter = progress["store_progress_iteration"]
    except KeyError:
        if iteration is not None:
            raise IndexError("When mad.store_progress=False, iteration must be None!")
        mad_image = mad.mad_image
        # losses will always have one extra value, the current loss.
        iter = len(mad.losses) - 1
    image = mad.image[batch_idx]
    if channel_idx is not None:
        image = image[channel_idx]
        mad_image = mad_image[channel_idx]
    image = tensors.to_numpy(image).flatten()
    mad_image = tensors.to_numpy(mad_image).flatten()

    if ax is None:
        ax = plt.gca()
    ax.hist(
        mad_image,
        bins=min(_freedman_diaconis_bins(image), 50),
        label=f"MAD image [iteration={iter}]",
        **kwargs,
    )
    ax.hist(
        image,
        bins=min(_freedman_diaconis_bins(image), 50),
        label="Reference image",
        **kwargs,
    )
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title("Histogram of pixel values")
    return ax


def _check_included_plots(to_check: list[str] | dict[str, int], to_check_name: str):
    """
    Check whether the user wanted us to create plots that we can't.

    Helper function for :func:`mad_synthesis_status` and :func:`animate`.

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
    """
    allowed_vals = [
        "mad_image",
        "mad_loss",
        "mad_pixel_values",
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
    figsize: tuple[float] | None = None,
    included_plots: list[str] = [
        "mad_image",
        "mad_loss",
        "mad_pixel_values",
    ],
    mad_image_width: float = 1,
    mad_loss_width: float = 2,
    mad_pixel_values_width: float = 1,
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes], dict[str, int]]:
    """
    Set up figure for :func:`mad_synthesis_status`.

    Creates figure with enough axes for the all the plots you want. Will
    also create index in ``axes_idx`` for them if you haven't done so already.

    If ``fig=None``, all axes will be on the same row and have the same width. If
    you want them to be on different rows, will need to initialize ``fig`` yourself
    and pass that in. For changing width, change the corresponding ``*_width`` arg,
    which gives width relative to other axes. So if you want the axis for the
    loss plot to be three times as wide as the others, set ``loss_width=3``.

    Parameters
    ----------
    fig
        The figure to plot on or ``None``. If ``None``, we create a new figure.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows for more
        fine-grained control of the resulting figure. Probably only helpful if fig is
        also defined. Possible keys: ``"loss"``, ``"pixel_values"``, ``"misc"``. Values
        should all be ints. If you tell this function to create a plot that doesn't have
        a corresponding key, we find the lowest int that is not already in the dict, so
        if you have axes that you want unchanged, place their idx in ``"misc"``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have relative width=1 correspond to 5.
    included_plots
        Which plots to include. Must be some subset of ``'mad_image',
        'mad_loss', 'mad_pixel_values'``.
    mad_image_width
        Relative width of the axis for the synthesized image.
    mad_loss_width
        Relative width of the axis for loss plot.
    mad_pixel_values_width
        Relative width of the axis for image pixel intensities histograms.

    Returns
    -------
    fig
        The figure to plot on.
    axes
        List or array of axes contained in fig.
    axes_idx
        Dictionary identifying the idx for each plot type.
    """
    n_subplots = 0
    axes_idx = axes_idx.copy()
    width_ratios = []
    if "mad_image" in included_plots:
        n_subplots += 1
        width_ratios.append(mad_image_width)
        if "mad_image" not in axes_idx:
            axes_idx["mad_image"] = tensors._find_min_int(axes_idx.values())
    if "mad_loss" in included_plots:
        n_subplots += 1
        width_ratios.append(mad_loss_width)
        if "mad_loss" not in axes_idx:
            axes_idx["mad_loss"] = tensors._find_min_int(axes_idx.values())
    if "mad_pixel_values" in included_plots:
        n_subplots += 1
        width_ratios.append(mad_pixel_values_width)
        if "mad_pixel_values" not in axes_idx:
            axes_idx["mad_pixel_values"] = tensors._find_min_int(axes_idx.values())
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


def mad_synthesis_status(
    mad: MADCompetition,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    iteration: int | None = None,
    vrange: tuple[float] | str = "indep1",
    zoom: float | None = None,
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float] | None = None,
    included_plots: list[str] = [
        "mad_image",
        "mad_loss",
        "mad_pixel_values",
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
    mad
        MADCompetition object whose status we want to plot.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) image).
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``mad.store_progress>1`` (that is, the MAD image was not cached on every
        iteration), then we use the cached MAD image from the nearest iteration.
    vrange
        The vrange option to pass to :func:`mad_image()`. See
        docstring of :func:`~plenoptic.tools.display.imshow` for possible values.
    zoom
        How much to zoom in / enlarge the synthesized image, the ratio
        of display pixels to image pixels. If ``None``, we
        attempt to find the best value ourselves.
    fig
        If ``None``, we create a new figure. otherwise we assume this is
        an empty figure that has the appropriate size and number of
        subplots.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'mad_image',
        'mad_loss', 'mad_pixel_values', 'misc'``. Values should all be ints. If you
        tell this function to create a plot that doesn't have a corresponding
        key, we find the lowest int that is not already in the dict, so if you
        have axes that you want unchanged, place their idx in ``'misc'``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have each axis be of size ``(5, 5)``.
    included_plots
        Which plots to include. Must be some subset of ``'mad_image',
        'mad_loss', 'mad_pixel_values'``.
    width_ratios
        If ``width_ratios`` is an empty dictionary, ``mad_loss`` will have
        double the width of the other plots. To change that, specify their
        relative widths using the keys: ['mad_image', 'mad_loss',
        'mad_pixel_values'] and floats specifying their relative width.

    Returns
    -------
    fig
        The figure containing this plot.
    axes_idx
        Dictionary giving index of each plot.

    Raises
    ------
    ValueError
        If ``mad.mad_image`` object is not 3d or 4d.
    ValueError
        If the ``iteration is not None`` and the given ``mad`` object was run
        with ``store_progress=False``.

    Warns
    -----
    UserWarning
        If the iteration used for ``saved_mad_image`` is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``mad.store_progress=2``).
    """
    if iteration is not None and not mad.store_progress:
        raise ValueError(
            "synthesis() was run with store_progress=False, "
            "cannot specify which iteration to plot (only"
            " last one, with iteration=None)"
        )
    if mad.mad_image.ndim not in [3, 4]:
        raise ValueError(
            "mad_synthesis_status() expects 3 or 4d data;"
            "unexpected behavior will result otherwise!"
        )
    _check_included_plots(included_plots, "included_plots")
    _check_included_plots(width_ratios, "width_ratios")
    _check_included_plots(axes_idx, "axes_idx")
    width_ratios = {f"{k}_width": v for k, v in width_ratios.items()}
    fig, axes, axes_idx = _setup_synthesis_fig(
        fig, axes_idx, figsize, included_plots, **width_ratios
    )

    if "mad_image" in included_plots:
        mad_image(
            mad,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=iteration,
            ax=axes[axes_idx["mad_image"]],
            zoom=zoom,
            vrange=vrange,
        )
    if "mad_loss" in included_plots:
        mad_loss(mad, iteration=iteration, axes=axes[axes_idx["mad_loss"]])
        # this function creates a single axis for loss, which mad_loss then
        # split into two. this makes sure the right two axes are present in the
        # dict
        all_axes = []
        for i in axes_idx.values():
            # so if it's a list of ints
            if hasattr(i, "__iter__"):
                all_axes.extend(i)
            else:
                all_axes.append(i)
        new_axes = [i for i, _ in enumerate(fig.axes) if i not in all_axes]
        axes_idx["mad_loss"] = new_axes
    if "mad_pixel_values" in included_plots:
        mad_pixel_values(
            mad,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=iteration,
            ax=axes[axes_idx["mad_pixel_values"]],
        )
    return fig, axes_idx


def mad_animate(
    mad: MADCompetition,
    framerate: int = 10,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    zoom: float | None = None,
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float] | None = None,
    included_plots: list[str] = [
        "mad_image",
        "mad_loss",
        "mad_pixel_values",
    ],
    width_ratios: dict[str, float] = {},
) -> mpl.animation.FuncAnimation:
    r"""
    Animate synthesis progress.

    This is essentially the figure produced by
    :func:`mad_synthesis_status` animated over time, for each stored
    iteration.

    This functions returns a matplotlib FuncAnimation object. See our
    documentation (e.g., :ref:`quickstart-nb`) for examples on how to view it in
    a Jupyter notebook. In order to save, use ``anim.save(filename)``. In either
    case, this can take a while and you'll need the appropriate writer installed
    and on your path, e.g., ffmpeg, imagemagick, etc). See
    :doc:`matplotlib documentation <matplotlib:api/animation_api>` for more details.

    Parameters
    ----------
    mad
        MADCompetition object whose synthesis we want to animate.
    framerate
        How many frames a second to display.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) image).
    zoom
        How much to zoom in / enlarge the synthesized image, the ratio
        of display pixels to image pixels. If ``None``, we
        attempt to find the best value ourselves.
    fig
        If ``None``, we create a new figure. otherwise we assume this is
        an empty figure that has the appropriate size and number of
        subplots.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'mad_image',
        'mad_loss', 'mad_pixel_values', 'misc'``. Values should all be ints. If you
        tell this function to create a plot that doesn't have a corresponding
        key, we find the lowest int that is not already in the dict, so if you
        have axes that you want unchanged, place their idx in ``'misc'``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have each axis be of size ``(5, 5)``.
    included_plots
        Which plots to include. Must be some subset of ``'mad_image',
        'mad_loss', 'mad_pixel_values'``.
    width_ratios
        If ``width_ratios`` is an empty dictionary, ``mad_loss`` will have
        double the width of the other plots. To change that, specify their
        relative widths using the keys: ['mad_image', 'mad_loss',
        'mad_pixel_values'] and floats specifying their relative width.

    Returns
    -------
    anim
        The animation object. In order to view, must convert to HTML
        or save.

    Raises
    ------
    ValueError
        If the given ``mad`` object was run with ``store_progress=False``.
    ValueError
        If ``mad.mage_image`` object is not 3d or 4d.
    ValueError
        If we do not know how to interpret the value of ``ylim``.

    Notes
    -----
    Unless specified, we use the ffmpeg backend, which requires that you have
    ffmpeg installed and on your path (https://ffmpeg.org/download.html).
    To use a different, use the matplotlib rcParams:
    ``matplotlib.rcParams['animation.writer'] = writer``, see
    `matplotlib documentation
    <https://matplotlib.org/stable/api/animation_api.html#writer-classes>`_ for more
    details.
    """
    if not mad.store_progress:
        raise ValueError(
            "synthesize() was run with store_progress=False, cannot animate!"
        )
    if mad.mad_image.ndim not in [3, 4]:
        raise ValueError(
            "animate() expects 3 or 4d data; unexpected behavior will result otherwise!"
        )
    _check_included_plots(included_plots, "included_plots")
    _check_included_plots(width_ratios, "width_ratios")
    _check_included_plots(axes_idx, "axes_idx")
    # we run mad_synthesis_status to initialize the figure if either fig is
    # None or if there are no titles on any axes, which we assume means that
    # it's an empty figure
    if fig is None or not any([ax.get_title() for ax in fig.axes]):
        fig, axes_idx = mad_synthesis_status(
            mad=mad,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=0,
            figsize=figsize,
            zoom=zoom,
            fig=fig,
            included_plots=included_plots,
            axes_idx=axes_idx,
            width_ratios=width_ratios,
        )
    # grab the artist for the second plot (we don't need to do this for the
    # MAD image plot, because we use the update_plot function for that)
    if "mad_loss" in included_plots:
        scat = [fig.axes[i].collections[0] for i in axes_idx["mad_loss"]]
    if "mad_image" in included_plots:
        fig.axes[axes_idx["mad_image"]].set_title("MAD Image")

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
        """
        # this warning is not relevant for animate
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="loss iteration and iteration for"
            )
            artists = []
            if "mad_image" in included_plots:
                artists.extend(
                    display.update_plot(
                        fig.axes[axes_idx["mad_image"]],
                        data=mad.saved_mad_image[i],
                        batch_idx=batch_idx,
                    )
                )
            if "mad_pixel_values" in included_plots:
                # this is the dumbest way to do this, but it's simple --
                # clearing the axes can cause problems if the user has, for
                # example, changed the tick locator or formatter. not sure how
                # to handle this best right now
                fig.axes[axes_idx["mad_pixel_values"]].clear()
                mad_pixel_values(
                    mad,
                    batch_idx=batch_idx,
                    channel_idx=channel_idx,
                    iteration=min(i * mad.store_progress, len(mad.losses) - 1),
                    ax=fig.axes[axes_idx["mad_pixel_values"]],
                )
            if "mad_loss" in included_plots:
                # loss always contains values from every iteration, but everything
                # else will be subsampled.
                x_val = mad._convert_iteration(i, False)
                scat[0].set_offsets((x_val, mad.reference_metric_loss[x_val]))
                scat[1].set_offsets((x_val, mad.optimized_metric_loss[x_val]))
                artists.extend(scat)
            # as long as blitting is True, need to return a sequence of artists
            return artists

    # don't need an init_func, since we handle initialization ourselves
    anim = mpl.animation.FuncAnimation(
        fig,
        movie_plot,
        frames=len(mad.saved_mad_image),
        blit=True,
        interval=1000.0 / framerate,
        repeat=False,
    )
    plt.close(fig)
    return anim


def mad_image_all(
    mad_metric1_min: MADCompetition,
    mad_metric2_min: MADCompetition,
    mad_metric1_max: MADCompetition,
    mad_metric2_max: MADCompetition,
    metric1_name: str | None = None,
    metric2_name: str | None = None,
    zoom: int | float = 1,
    **kwargs: Any,
) -> mpl.figure.Figure:
    """
    Display all MAD Competition images.

    To generate a full set of MAD Competition images, you need four instances:
    one for minimizing and maximizing each metric. This helper function creates
    a figure to display the full set of images.

    In addition to the four MAD Competition images, this also plots the initial
    image from ``mad_metric1_min``, for comparison.

    Parameters
    ----------
    mad_metric1_min
        ``MADCompetition`` object that minimized the first metric.
    mad_metric2_min
        ``MADCompetition`` object that minimized the second metric.
    mad_metric1_max
        ``MADCompetition`` object that maximized the first metric.
    mad_metric2_max
        ``MADCompetition`` object that maximized the second metric.
    metric1_name
        Name of the first metric. If ``None``, we use the name of the
        ``optimized_metric`` function from ``mad_metric1_min``.
    metric2_name
        Name of the second metric. If ``None``, we use the name of the
        ``optimized_metric`` function from ``mad_metric2_min``.
    zoom
        Ratio of display pixels to image pixels. See
        :func:`~plenoptic.tools.display.imshow` for details.
    **kwargs
        Passed to :func:`~plenoptic.tools.display.imshow`.

    Returns
    -------
    fig
        Figure containing the images.

    Raises
    ------
    ValueError
        If the four ``MADCompetition`` instances do not have the same ``image``
        attribute.
    """
    # this is a bit of a hack right now, because they don't all have same
    # initial image
    if not torch.allclose(mad_metric1_min.image, mad_metric2_min.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if not torch.allclose(mad_metric1_min.image, mad_metric1_max.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if not torch.allclose(mad_metric1_min.image, mad_metric2_max.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if metric1_name is None:
        metric1_name = mad_metric1_min.optimized_metric.__name__
    if metric2_name is None:
        metric2_name = mad_metric2_min.optimized_metric.__name__
    fig = pt_make_figure(3, 2, [zoom * i for i in mad_metric1_min.image.shape[-2:]])
    mads = [mad_metric1_min, mad_metric1_max, mad_metric2_min, mad_metric2_max]
    titles = [
        f"Minimize {metric1_name}",
        f"Maximize {metric1_name}",
        f"Minimize {metric2_name}",
        f"Maximize {metric2_name}",
    ]
    # we're only plotting one image here, so if the user wants multiple
    # channels, they must be RGB
    if kwargs.get("channel_idx") is None and mad_metric1_min.initial_image.shape[1] > 1:
        as_rgb = True
    else:
        as_rgb = False
    display.imshow(
        mad_metric1_min.image,
        ax=fig.axes[0],
        title="Reference image",
        zoom=zoom,
        as_rgb=as_rgb,
        **kwargs,
    )
    display.imshow(
        mad_metric1_min.initial_image,
        ax=fig.axes[1],
        title="Initial (noisy) image",
        zoom=zoom,
        as_rgb=as_rgb,
        **kwargs,
    )
    for ax, mad, title in zip(fig.axes[2:], mads, titles):
        mad_image(mad, zoom=zoom, ax=ax, title=title, **kwargs)
    return fig


def mad_loss_all(
    mad_metric1_min: MADCompetition,
    mad_metric2_min: MADCompetition,
    mad_metric1_max: MADCompetition,
    mad_metric2_max: MADCompetition,
    metric1_name: str | None = None,
    metric2_name: str | None = None,
    metric1_kwargs: dict = {"c": "C0"},
    metric2_kwargs: dict = {"c": "C1"},
    min_kwargs: dict = {"linestyle": "--"},
    max_kwargs: dict = {"linestyle": "-"},
    figsize: tuple[int, int] = (10, 5),
) -> mpl.figure.Figure:
    """
    Plot loss for full set of MAD Competiton instances.

    To generate a full set of MAD Competition images, you need four instances:
    one for minimizing and maximizing each metric. This helper function creates
    a two-axis figure to display the loss for this full set.

    Parameters
    ----------
    mad_metric1_min
        ``MADCompetition`` object that minimized the first metric.
    mad_metric2_min
        ``MADCompetition`` object that minimized the second metric.
    mad_metric1_max
        ``MADCompetition`` object that maximized the first metric.
    mad_metric2_max
        ``MADCompetition`` object that maximized the second metric.
    metric1_name
        Name of the first metric. If ``None``, we use the name of the
        ``optimized_metric`` function from ``mad_metric1_min``.
    metric2_name
        Name of the second metric. If ``None``, we use the name of the
        ``optimized_metric`` function from ``mad_metric2_min``.
    metric1_kwargs
        Dictionary of arguments to pass to :func:`matplotlib.pyplot.plot` to identify
        synthesis instance where the first metric was being optimized.
    metric2_kwargs
        Dictionary of arguments to pass to :func:`matplotlib.pyplot.plot` to identify
        synthesis instance where the second metric was being optimized.
    min_kwargs
        Dictionary of arguments to pass to :func:`matplotlib.pyplot.plot` to identify
        synthesis instance where ``optimized_metric`` was being minimized.
    max_kwargs
        Dictionary of arguments to pass to :func:`matplotlib.pyplot.plot` to identify
        synthesis instance where ``optimized_metric`` was being maximized.
    figsize
        Size of the figure we create.

    Returns
    -------
    fig
        Figure containing the plot.

    Raises
    ------
    ValueError
        If the four ``MADCompetition`` instances do not have the same ``image``
        attribute.
    """
    if not torch.allclose(mad_metric1_min.image, mad_metric2_min.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if not torch.allclose(mad_metric1_min.image, mad_metric1_max.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if not torch.allclose(mad_metric1_min.image, mad_metric2_max.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if metric1_name is None:
        metric1_name = mad_metric1_min.optimized_metric.__name__
    if metric2_name is None:
        metric2_name = mad_metric2_min.optimized_metric.__name__
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    mad_loss(
        mad_metric1_min,
        axes=axes,
        label=f"Minimize {metric1_name}",
        **metric1_kwargs,
        **min_kwargs,
    )
    mad_loss(
        mad_metric1_max,
        axes=axes,
        label=f"Maximize {metric1_name}",
        **metric1_kwargs,
        **max_kwargs,
    )
    # we pass the axes backwards here because the fixed and synthesis metrics are
    # the opposite as they are in the instances above.
    mad_loss(
        mad_metric2_min,
        axes=axes[::-1],
        label=f"Minimize {metric2_name}",
        **metric2_kwargs,
        **min_kwargs,
    )
    mad_loss(
        mad_metric2_max,
        axes=axes[::-1],
        label=f"Maximize {metric2_name}",
        **metric2_kwargs,
        **max_kwargs,
    )
    axes[0].set(ylabel="Loss", title=metric2_name)
    axes[1].set(ylabel="Loss", title=metric1_name)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.1, 0.5))
    return fig
