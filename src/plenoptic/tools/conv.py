"""Convolution-related utility functions."""  # numpydoc ignore=ES01

import math
from typing import Literal

import numpy as np
import pyrtools as pt
import torch
import torch.nn.functional as F
from torch import Tensor


def correlate_downsample(
    image: Tensor,
    filt: Tensor,
    padding_mode: Literal["constant", "reflect", "replicate", "circular"] = "reflect",
) -> Tensor:
    """
    Correlate with a filter and downsample by a factor of 2.

    This operation allows one to downsample in an alias-resistant manner, removing the
    high frequencies that would result in aliasing in a smaller image.

    Parameters
    ----------
    image
        Image, or batch of images, of shape (batch, channel, height, width).
        Batches and channels are handled independently.
    filt
        2D tensor defining the filter to correlate with the input ``image``.
    padding_mode
        How to pad the image, so that we return an image of the appropriate size. The
        option ``"constant"`` means padding with zeros.

    Returns
    -------
    downsampled_image
        The downsampled image.

    Raises
    ------
    ValueError
        If ``filt`` or ``image`` has the wrong number of dimensions.

    See Also
    --------
    blur_downsample
        Perform this operation a user-specified number of times using a named filter.
    upsample_convolve
        Perform the inverse operation, upsampling and convolving with a filter.
    """
    if image.ndim != 4:
        raise ValueError(f"image must be 4d but has {image.ndim} dimensions instead!")
    if filt.ndim != 2:
        raise ValueError(f"filt must be 2d but has {filt.ndim} dimensions instead!")
    assert image.ndim == 4 and filt.ndim == 2
    n_channels = image.shape[1]
    image_padded = same_padding(image, kernel_size=filt.shape, pad_mode=padding_mode)
    return F.conv2d(
        image_padded,
        filt.repeat(n_channels, 1, 1, 1),
        stride=2,
        groups=n_channels,
    )


def upsample_convolve(
    image: Tensor,
    odd: tuple[int, int],
    filt: Tensor,
    padding_mode: Literal["constant", "reflect", "replicate", "circular"] = "reflect",
) -> Tensor:
    """
    Upsample by 2 and convolve with a filter.

    When upsampling an image, we need some way to estimate the new pixels; convolving
    with a filter allows us to interpolate these pixels from their neighbors.

    Parameters
    ----------
    image
        Image, or batch of images, of shape (batch, channel, height, width).
        Batches and channels are handled independently.
    odd
        This should contain two integers of value 0 or 1, which determines whether
        the output height and width should be even (0) or odd (1).
    filt
        2D tensor defining the filter to correlate with the input ``image``.
    padding_mode
        How to pad the image, so that we return an image of the appropriate size. The
        option ``"constant"`` means padding with zeros.

    Returns
    -------
    upsampled_image
        The upsampled image.

    Raises
    ------
    ValueError
        If ``filt`` or ``image`` has the wrong number of dimensions.

    See Also
    --------
    upsample_blur
        Perform this operation a user-specified number of times using a named filter.
    correlate_downsample
        Perform the inverse operation, correlating and downsampling an image.
    """
    if image.ndim != 4:
        raise ValueError(f"image must be 4d but has {image.ndim} dimensions instead!")
    if filt.ndim != 2:
        raise ValueError(f"filt must be 2d but has {filt.ndim} dimensions instead!")
    filt = filt.flip((0, 1))

    n_channels = image.shape[1]
    pad_start = torch.as_tensor(filt.shape) // 2
    pad_end = torch.as_tensor(filt.shape) - torch.as_tensor(odd) - pad_start
    pad = torch.as_tensor([pad_start[1], pad_end[1], pad_start[0], pad_end[0]])
    image_prepad = F.pad(image, tuple(pad // 2), mode=padding_mode)
    image_upsample = F.conv_transpose2d(
        image_prepad,
        weight=torch.ones(
            (n_channels, 1, 1, 1), device=image.device, dtype=image.dtype
        ),
        stride=2,
        groups=n_channels,
    )
    image_postpad = F.pad(image_upsample, tuple(pad % 2))
    return F.conv2d(image_postpad, filt.repeat(n_channels, 1, 1, 1), groups=n_channels)


def blur_downsample(
    image: Tensor,
    n_scales: int = 1,
    filtname: str = "binom5",
    scale_filter: bool = True,
) -> Tensor:
    """
    Correlate with a named filter and downsample by 2.

    This operation allows one to downsample in an alias-resistant manner, removing the
    high frequencies that would result in aliasing in a smaller image.

    Parameters
    ----------
    image
        Image, or batch of images, of shape (batch, channel, height, width).
        Batches and channels are handled independently.
    n_scales
        Apply the blur and downsample procedure recursively ``n_scales`` times.
        Must be positive.
    filtname
        Name of the filter. See ``pyrtools.named_filter`` for options.
    scale_filter
        If ``True``, the filter sums to 1 (i.e., it does not affect the DC component of
        the signal and the output's mean will approximately match that of the input). If
        ``False``, the filter sums to 2 (and the output's mean will be roughly double
        that of the input).

    Returns
    -------
    downsampled_image
        The downsampled image.

    Raises
    ------
    ValueError
        If ``n_scales`` is not positive.

    See Also
    --------
    correlate_downsample
        Perform this operation once using a user-specified filter.
    upsample_blur
        Perform the inverse operation, upsampling and convolving a user-specified number
        of times using a named filter.
    """
    if n_scales < 1:
        raise ValueError("n_scales must be positive!")
    f = pt.named_filter(filtname)
    filt = torch.as_tensor(np.outer(f, f), dtype=image.dtype, device=image.device)
    if scale_filter:
        filt = filt / 2
    for _ in range(n_scales):
        image = correlate_downsample(image, filt)
    return image


def upsample_blur(
    image: Tensor,
    odd: tuple[int, int],
    n_scales: int = 1,
    filtname: str = "binom5",
    scale_filter: bool = True,
) -> Tensor:
    """
    Upsample by 2 and convolve with named filter.

    When upsampling an image, we need some way to estimate the new pixels; convolving
    with a filter allows us to interpolate these pixels from their neighbors.

    Parameters
    ----------
    image
        Image, or batch of images, of shape (batch, channel, height, width).
        Batches and channels are handled independently.
    odd
        This should contain two integers of value 0 or 1, which determines whether
        the output height and width should be even (0) or odd (1).
    n_scales
        Apply the blur and downsample procedure recursively ``n_scales`` times.
        Must be positive.
    filtname
        Name of the filter. See ``pyrtools.named_filter`` for options.
    scale_filter
        If ``True``, the filter sums to 4 (i.e., it does not affect the DC component of
        the signal and the output's mean will approximately match that of the input). If
        ``False``, the filter sums to 2 (and the output's mean will be roughly half
        that of the input).

    Returns
    -------
    upsampled_image
        The upsampled image.

    Raises
    ------
    ValueError
        If ``n_scales`` is not positive.

    See Also
    --------
    upsample_convolve
        Perform this operation once using a user-specified filter.
    blur_downsample
        Perform the inverse operation, correlating and downsampling a user-specified
        number of times using a named filter.
    """
    if n_scales < 1:
        raise ValueError("n_scales must be positive!")
    f = pt.named_filter(filtname)
    filt = torch.as_tensor(np.outer(f, f), dtype=image.dtype, device=image.device)
    if scale_filter:
        filt = filt * 2
    for _ in range(n_scales):
        image = upsample_convolve(image, odd, filt)
    return image


def _get_same_padding(x: int, kernel_size: int, stride: int, dilation: int) -> int:
    """Determine integer padding for F.pad() given img and kernel."""  # noqa: DOC201
    # numpydoc ignore=ES01,PR01,RT01
    pad = (math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x
    pad = max(pad, 0)
    return pad


def same_padding(
    image: Tensor,
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = (1, 1),
    dilation: int | tuple[int, int] = (1, 1),
    pad_mode: str = "circular",
) -> Tensor:
    """
    Pad a tensor so that 2D convolution will result in output with same dims.

    Parameters
    ----------
    image
        Image, or batch of images, with at least 2 dimensions (height and width).
        Any additional dimensions are handled independently.
    kernel_size
        Size of the kernel that ``image`` will be convolved with.
    stride
        Stride argument that will be passed to the convolution function.
    dilation
        Dilation argument that will be passed to the convolution function.
    pad_mode
        How to pad ``image``. See :func:`torch.nn.functional.pad` for possible
        values.

    Returns
    -------
    padded_image
        The padded tensor.

    Raises
    ------
    ValueError
        If ``image`` is not 4d.
    """  # numpydoc ignore=ES01
    if len(image.shape) < 2:
        raise ValueError("Input must be tensor whose last dims are height x width")
    ih, iw = image.shape[-2:]
    pad_h = _get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = _get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    if pad_h > 0 or pad_w > 0:
        image = F.pad(
            image,
            [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            mode=pad_mode,
        )
    return image
