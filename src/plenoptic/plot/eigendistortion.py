"""Plots for understanding Eigendistortion objects."""

import warnings
from collections.abc import Callable
from typing import Any

import torch
from matplotlib.figure import Figure
from torch import Tensor

from .._synthesize import Eigendistortion
from .display import imshow

__all__ = [
    "eigendistortion_imshow_all",
]


def __dir__() -> list[str]:
    return __all__


def eigendistortion_imshow_all(
    eigendistortion: Eigendistortion,
    eigenindex: int | list[int] = [0, -1],
    channel_idx: int | None = None,
    distortion_scale: float | list[float] = 5.0,
    process_image: Callable[[Tensor], Tensor] | None = None,
    plot_complex: str = "rectangular",
    as_rgb: bool = False,
    suptitle: str = "Eigendistortions",
    suptitle_kwargs: dict | None = None,
    **kwargs: Any,
) -> Figure:
    r"""
    Display base image, eigendistortions alone, and eigendistortions added to the image.

    If image or eigendistortions have 3 channels, then it is assumed to be a color
    image and it is converted to grayscale. This is merely for display convenience
    and may change in the future.

    Parameters
    ----------
    eigendistortion
        Eigendistortion object whose synthesized eigendistortion we want to display.
    eigenindex
        Index of eigendistortion to plot. E.g. If there are 10 eigenvectors, 0 will
        index the first one, and -1 or 9 will index the last one. Note that this is
        the same as the ``batch_idx`` (i.e., the index in the first dimension).
    channel_idx
        Which index to take from the channel dimension. If ``None``, we assume
        image is RGB(A) and show all channels.
    distortion_scale
        Amount by which to scale eigendistortion for ``image + (distortion_scale *
        eigendistortion)`` for display. If a list, must be the same length as
        ``eigenindex`` and will multiply each distortion by the corresponding
        ``distortion_scale`` value.
    process_image
        A function to process the plotted image. E.g., multiplying by the stdev ImageNet
        then adding the mean of ImageNet to undo image preprocessing or clamping between
        0 and 1. If ``None``, then no processing is performed.
    plot_complex
        Parameter for :func:`~plenoptic.plot.imshow` determining how to handle
        complex values. See that method's docstring for details.
    as_rgb
        Whether to consider the channels as encoding RGB(A) values. If ``True``, we
        attempt to plot the image in color, so your tensor must have 3 (or 4 if
        you want the alpha channel) elements in the channel dimension. If ``False``,
        we plot each channel as a separate grayscale image.
    suptitle
        Super title to plot above all axes.
    suptitle_kwargs
        Additional arguments for :func:`matplotlib.pyplot.suptitle`.
    **kwargs
        Additional arguments for :func:`~plenoptic.plot.imshow`.

    Returns
    -------
    fig
        Figure containing the displayed images.

    Raises
    ------
    ValueError
        If ``distortion_scale`` is not a single value and
        ``len(distortion_scale) != len(eigenindex)``.
    ValueError
        If a value of ``eigenindex`` doesn't correspond to one of the
        synthesized eigendistortions.

    Warns
    -----
    UserWarning
        If ``process_image=None`` and ``as_rgb=True``, because we are adding 0.5
        to the distortion.

    See Also
    --------
    :func:`~plenoptic.plot.synthesis_imshow`
        Display single eigendistortion added to image.

    Examples
    --------
    .. plot::

      >>> import plenoptic as po
      >>> import torch
      >>> img = po.data.einstein().to(torch.float64)
      >>> lg = po.models.LuminanceGainControl(
      ...     (31, 31), pad_mode="circular", pretrained=True, cache_filt=True
      ... ).eval()
      >>> lg = lg.to(torch.float64)
      >>> po.remove_grad(lg)
      >>> eig = po.Eigendistortion(img, lg)
      >>> # grab saved example eigendistortion, which runs the above to completion.
      >>> eig.load(
      ...     po.data.fetch_data("example_eigendistortion.pt"),
      ...     map_location="cpu",
      ...     tensor_equality_atol=1e-7,
      ... )
      >>> po.plot.eigendistortion_imshow_all(eig)
      <PyrFigure size ...>
    """
    # reshape so channel dim is last
    im_shape = eigendistortion._image_shape
    image = eigendistortion.image.detach().view(1, *im_shape).cpu()
    if not hasattr(eigenindex, "__iter__"):
        eigenindex = [eigenindex]
    if not hasattr(distortion_scale, "__iter__"):
        distortion_scale = [distortion_scale] * len(eigenindex)
    if len(distortion_scale) != len(eigenindex):
        raise ValueError(
            "If distortion_scale is a list, it must have the same number of values "
            f"as eigenindex! {len(distortion_scale)=} vs. {len(eigenindex)=}"
        )
    distortions = [torch.ones_like(image)]
    distortion_titles = [""]
    img_titles = ["Original image"]
    dist_suffix = ""
    process_dist = None
    if process_image is None:

        def process_image(x: Tensor) -> Tensor:
            return x

        if as_rgb:

            def process_dist(x: Tensor) -> Tensor:
                return x + 0.5

            warnings.warn(
                "Adding 0.5 to distortion to plot as RGB image, else matplotlib"
                " clipping will result in a strange looking image..."
            )
            dist_suffix = " + 0.5"
    # the only situation in which the following is not true is if process_image
    # was None and as_rgb=True
    if process_dist is None:
        process_dist = process_image

    img_processed = [process_image(image)]

    for a, idx in zip(distortion_scale, eigenindex):
        dist = (
            eigendistortion.eigendistortions[eigendistortion._indexer(idx)]
            .unsqueeze(0)
            .cpu()
        )
        img_processed.append(torch.clamp(process_image(image + a * dist), 0, 1))
        img_titles.append(f"{a} * Eigendistortion[{idx}]")
        distortion_titles.append(f"Eigendistortion[{idx}]{dist_suffix}")
        distortions.append(process_dist(dist))

    fig = imshow(
        distortions + img_processed,
        plot_complex=plot_complex,
        title=distortion_titles + img_titles,
        col_wrap=len(distortions),
        as_rgb=as_rgb,
        **kwargs,
    )
    fig.axes[0].set_visible(False)

    if suptitle_kwargs is None:
        suptitle_kwargs = {}
    va = suptitle_kwargs.pop("verticalalignment", None)
    va = suptitle_kwargs.pop("va", "bottom") if va is None else "bottom"
    y = suptitle_kwargs.pop("y", 1.05)
    fig.suptitle(suptitle, y=y, va=va, **suptitle_kwargs)

    return fig
