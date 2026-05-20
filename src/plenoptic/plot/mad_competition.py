"""Plots for understanding MADCompetition objects."""

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from pyrtools.tools.display import make_figure as pt_make_figure

from .._synthesize import MADCompetition
from . import display
from .synthesis import synthesis_imshow, synthesis_loss

__all__ = [
    "mad_imshow_all",
    "mad_loss_all",
]


def __dir__() -> list[str]:
    return __all__


def mad_imshow_all(
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
        :func:`~plenoptic.plot.synthesis_imshow` for details.
    **kwargs
        Passed to :func:`~plenoptic.plot.synthesis_imshow`.

    Returns
    -------
    fig
        Figure containing the images.

    Raises
    ------
    ValueError
        If the four ``MADCompetition`` instances do not have the same ``image``
        attribute.

    See Also
    --------
    :func:`~plenoptic.plot.synthesis_imshow`
        Display the image from a single :class:`~plenoptic.MADCompetition` instance.
    :func:`~plenoptic.plot.synthesis_status`
        Create a composite plot showing synthesis status of a single
        :class:`~plenoptic.MADCompetition` instance.

    Examples
    --------
    See the `MAD Competition <mad-nb>`_ `tutorial notebooks <mad-concept>`_ in the User
    Guide of documentation for examples.
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
        synthesis_imshow(mad, zoom=zoom, ax=ax, title=title, **kwargs)
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

    See Also
    --------
    :func:`~plenoptic.plot.synthesis_loss`
        Display the loss from a single :class:`~plenoptic.MADCompetition` instance.
    :func:`~plenoptic.plot.synthesis_status`
        Create a composite plot showing synthesis status of a single
        :class:`~plenoptic.MADCompetition` instance.

    Examples
    --------
    See the `MAD Competition <mad-nb>`_ `tutorial notebooks <mad-concept>`_ in the User
    Guide of documentation for examples.
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
    synthesis_loss(
        mad_metric1_min,
        axes=axes,
        label=f"Minimize {metric1_name}",
        **metric1_kwargs,
        **min_kwargs,
    )
    synthesis_loss(
        mad_metric1_max,
        axes=axes,
        label=f"Maximize {metric1_name}",
        **metric1_kwargs,
        **max_kwargs,
    )
    # we pass the axes backwards here because the fixed and synthesis metrics are
    # the opposite as they are in the instances above.
    synthesis_loss(
        mad_metric2_min,
        axes=axes[::-1],
        label=f"Minimize {metric2_name}",
        **metric2_kwargs,
        **min_kwargs,
    )
    synthesis_loss(
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
