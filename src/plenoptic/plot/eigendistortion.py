"""Plots for understanding Eigendistortion objects."""

import warnings
from collections.abc import Callable
from typing import Any

import torch
from matplotlib.figure import Figure
from torch import Tensor

from .._synthesize import Eigendistortion
from .display import imshow
from .synthesis import _get_synthesis_image

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
    process_distortion: Callable[[Tensor], Tensor] | None = None,
    suptitle: str = "Eigendistortions",
    suptitle_kwargs: dict | None = None,
    **kwargs: Any,
) -> Figure:
    r"""
    Display base image, eigendistortions alone, and eigendistortions added to the image.

    This function creates a figure with 2 rows and ``len(eigenindex)+1`` columns. The
    first row shows the eigendistortions alone, while the second shows the base image
    in the first column and then that image plus the eigendistortions (scaled by
    ``distortion_scale``) in the remaining columns.

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
        A function to process images in the second row. E.g., multiplying by the stdev
        ImageNet then adding the mean of ImageNet to undo image preprocessing or
        clamping between 0 and 1. If ``None``, then no processing is performed.
    process_distortion
        A function to process images in the first row, the eigendistortions alone. If
        ``None`` and the images are grayscale then no processing is performed. If
        ``None`` and the images are color (i.e., ``channel_idx is None`` and they have
        more than 1 channel), then we add 0.5 to the eigendistortions alone. This is
        because matplotlib will clip RGB(A) images to lie between 0 and 1, and
        eigendistortion values are typically centered around 0.
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
        If ``process_distortion=None`` and we're plotting images in color (i.e.,
        because ``channel_idx is None`` and the image has more than one channel),
        because we are adding 0.5 to the distortion.

    See Also
    --------
    :func:`~plenoptic.plot.synthesis_imshow`
        Display single eigendistortion added to image.

    Examples
    --------
    .. plot::
      :context: reset

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

    You can process the images before plotting by using the ``process_image``
    argument. Here, we clip the images to lie between 0 and 1:

    .. plot::
      :context: close-figs

      >>> clip = lambda x: x.clip(0, 1)
      >>> po.plot.eigendistortion_imshow_all(eig, process_image=clip)
      <PyrFigure size ...>

    Note that we only apply ``process_image`` to the images on the bottom row (the base
    image and that image plus the scaled distortions), not the top row (eigendistortions
    alone). If you wish to also apply some processing to the top row, use the
    ``process_distortions`` argument (see below for more examples).

    When the base image and distortions are RGB(A) images, if ``process_distortions`` is
    not specified, we will add 0.5 to the top row (the distortions alone). This is
    because matplotlib clips RGB(A) images to lie between 0 and 1. Since most
    eigendistortions are centered around 0, this would result in a unrepresentative
    visualization.

    .. plot::
      :context: close-figs

      >>> class ColorModel(torch.nn.Module):
      ...     "Simple model that takes color image as input and outputs 2d conv."
      ...
      ...     def __init__(self):
      ...         super().__init__()
      ...         self.conv = torch.nn.Conv2d(3, 4, 3, 1)
      ...
      ...     def forward(self, x):
      ...         return self.conv(x)
      >>> img = po.data.color_wheel().to(torch.float64)
      >>> img = po.process.center_crop(img, 20)
      >>> # Set seed for random initialization of model filters
      >>> po.set_seed(0)
      >>> model = ColorModel().to(torch.float64)
      >>> po.remove_grad(model)
      >>> model.eval()
      ColorModel(...)
      >>> eig_rgb = po.Eigendistortion(img, model)
      >>> eig_rgb.load(
      ...     po.data.fetch_data("example_eigendistortion_color.pt"),
      ...     map_location="cpu",
      ... )
      >>> po.plot.eigendistortion_imshow_all(eig_rgb, zoom=10)
      <PyrFigure size ...>

    Note that the titles in the figure above specify that 0.5 has been added to the
    distortions.

    Another common use case for ``process_image`` is in a situation like the following.
    Here, we are visualizing eigendistortions generated for VGG16, trained on ImageNet.
    As is common when using ImageNet, the images were z-scored during training. We
    should thus similarly normalize our base image when generating eigendistortions.
    However, when visualizing our distortions, we should un-normalize them, to transform
    them back into regular pixel space. ``process_image`` allows us to do just that.
    (See the :ref:`demo-eigendistortions` notebook for more details about this result.)

    .. plot::
      :context: close-figs

      >>> from torchvision import models
      >>> from torchvision.models import feature_extraction
      >>> class TorchVision(torch.nn.Module):
      ...     def __init__(self, model, return_node: str):
      ...         super().__init__()
      ...         self.extractor = feature_extraction.create_feature_extractor(
      ...             model, return_nodes=[return_node]
      ...         )
      ...         self.model = model
      ...         self.return_node = return_node
      ...
      ...     def forward(self, x):
      ...         return self.extractor(x)[self.return_node]
      >>> def normalize(img_tensor):
      ...     "Standardize the image for vgg16."
      ...     return (img_tensor - img_tensor.mean()) / img_tensor.std()
      >>> img = po.process.center_crop(po.data.parrot(False).to(torch.float64), 254)
      >>> orig_mean = img.mean().detach()
      >>> orig_std = img.std().detach()
      >>> img = normalize(img)
      >>> vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
      >>> vgg = TorchVision(vgg, "features.11").to(torch.float64)
      >>> po.remove_grad(vgg)
      >>> vgg.eval()
      TorchVision(...)
      >>> eig_vgg = po.Eigendistortion(img, vgg)
      >>> eig_vgg.load(
      ...     po.data.fetch_data("berardino_vgg16.pt"),
      ...     map_location="cpu",
      ... )
      >>> def unnormalize(x):
      ...     return x * orig_std + orig_mean
      >>> po.plot.eigendistortion_imshow_all(eig_vgg, process_image=unnormalize)
      <PyrFigure size ...>

    In the above figure, as mentioned above, ``process_image`` is only applied to the
    bottom row of images, whereas the bottom row has just had 0.5 added to it. We can
    similarly apply this un-normalization to the bottom row:

    .. plot::
      :context: close-figs

      >>> po.plot.eigendistortion_imshow_all(
      ...     eig_vgg, process_image=unnormalize, process_distortion=unnormalize
      ... )
      <PyrFigure size ...>

    Note that we are now *only* applying ``unnormalize`` (not adding 0.5 and applying
    ``unnormalize``). If we wished, we could compose them ourselves (though in this
    specific case, that doesn't make much sense):

    >>> process_distortion = lambda x: unnormalize(x + 0.5)
    """
    if not hasattr(eigenindex, "__iter__"):
        eigenindex = [eigenindex]
    if not hasattr(distortion_scale, "__iter__"):
        distortion_scale = [distortion_scale] * len(eigenindex)
    if len(distortion_scale) != len(eigenindex):
        raise ValueError(
            "If distortion_scale is a list, it must have the same number of values "
            f"as eigenindex! {len(distortion_scale)=} vs. {len(eigenindex)=}"
        )
    image = eigendistortion.image
    # this is a dummy image that will be plotted on an axis we turn off, just to make
    # sure we have the same number of images per row, which makes imshow happy.
    distortions = [torch.ones_like(image)]
    distortion_titles = [""]
    as_rgb = bool(channel_idx is None and image.shape[1] > 1)

    def identity_map(x: Tensor) -> Tensor:
        # default behavior for process_image/distortion
        # numpydoc ignore=GL08
        return x

    if process_image is None:
        process_image = identity_map
        image_title = "{scale} * Eigendistortion[{idx}]"
        img_titles = ["Original image"]
    else:
        image_title = "Processed({scale} * Eigendistortion[{idx}])"
        img_titles = ["Processed(Original image)"]
    if process_distortion is None:
        distortion_title = "Eigendistortion[{idx}]"
        if as_rgb:

            def process_distortion(x: Tensor) -> Tensor:
                # return process_image(x + 0.5)
                return x + 0.5

            distortion_title += " + 0.5"
            warnings.warn(
                "Adding 0.5 to distortions to plot as RGB image, else matplotlib"
                " clipping will result in a strange looking image..."
            )
        else:
            process_distortion = identity_map
    else:
        distortion_title = "Processed(Eigendistortion[{idx}])"

    img_processed = [process_image(image)]

    dist, batch_idx = _get_synthesis_image(eigendistortion, eigenindex)
    for scale, idx, img_idx in zip(distortion_scale, eigenindex, batch_idx):
        img_processed.append(process_image(image + scale * dist[img_idx].unsqueeze(0)))
        img_titles.append(image_title.format(scale=scale, idx=idx))
        distortion_titles.append(distortion_title.format(idx=idx))
        distortions.append(process_distortion(dist[img_idx].unsqueeze(0)))

    fig = imshow(
        distortions + img_processed,
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
