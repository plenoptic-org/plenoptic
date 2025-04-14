import math

import numpy as np
import pyrtools as pt
import torch
import torch.nn.functional as F
from torch import Tensor


def correlate_downsample(image, filt, padding_mode="reflect"):
    """Correlate with a filter and downsample by 2.

    Parameters
    ----------
    image: torch.Tensor of shape (batch, channel, height, width)
        Image, or batch of images. Channels are treated in the same way as batches.
    filt: 2-D torch.Tensor
        The filter to correlate with the input image
    padding_mode: string, optional
        One of "constant", "reflect", "replicate", "circular". The option "constant"
        means padding with zeros.
    """
    assert isinstance(image, torch.Tensor) and isinstance(filt, torch.Tensor)
    assert image.ndim == 4 and filt.ndim == 2
    n_channels = image.shape[1]
    image_padded = same_padding(image, kernel_size=filt.shape, pad_mode=padding_mode)
    return F.conv2d(
        image_padded,
        filt.repeat(n_channels, 1, 1, 1),
        stride=2,
        groups=n_channels,
    )


def upsample_convolve(image, odd, filt, padding_mode="reflect"):
    """Upsample by 2 and convolve with a filter.

    Parameters
    ----------
    image: torch.Tensor of shape (batch, channel, height, width)
        Image, or batch of images. Channels are treated in the same way as
        batches.
    odd: tuple, list or numpy.ndarray
        This should contain two integers of value 0 or 1, which determines whether
        the output height and width should be even (0) or odd (1).
    filt: 2-D torch.Tensor
        The filter to convolve with the upsampled image
    padding_mode: string, optional
        One of "constant", "reflect", "replicate", "circular". The option "constant"
        means padding with zeros.
    """
    assert isinstance(image, torch.Tensor) and isinstance(filt, torch.Tensor)
    assert image.ndim == 4 and filt.ndim == 2
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


def blur_downsample(x, n_scales=1, filtname="binom5", scale_filter=True):
    """Correlate with a binomial coefficient filter and downsample by 2.

    Parameters
    ----------
    x: torch.Tensor of shape (batch, channel, height, width)
        Image, or batch of images. Channels are treated in the same way as batches.
    n_scales: int, optional. Should be non-negative.
        Apply the blur and downsample procedure recursively `n_scales` times. Default to
        1.
    filtname: str, optional
        Name of the filter. See `pt.named_filter` for options. Default to "binom5".
    scale_filter: bool, optional
        If true (default), the filter sums to 1 (ie. it does not affect the DC
        component of the signal). If false, the filter sums to 2.
    """
    f = pt.named_filter(filtname)
    filt = torch.as_tensor(np.outer(f, f), dtype=x.dtype, device=x.device)
    if scale_filter:
        filt = filt / 2
    for _ in range(n_scales):
        x = correlate_downsample(x, filt)
    return x


def upsample_blur(x, odd, filtname="binom5", scale_filter=True):
    """Upsample by 2 and convolve with a binomial coefficient filter.

    Parameters
    ----------
    x: torch.Tensor of shape (batch, channel, height, width)
        Image, or batch of images. Channels are treated in the same way as batches.
    odd: tuple, list or numpy.ndarray
        This should contain two integers of value 0 or 1, which determines whether
        the output height and width should be even (0) or odd (1).
    filtname: str, optional
        Name of the filter. See `pt.named_filter` for options. Default to "binom5".
    scale_filter: bool, optional
        If true (default), the filter sums to 4 (ie. it multiplies the signal
        by 4 before the blurring operation). If false, the filter sums to 2.
    """
    f = pt.named_filter(filtname)
    filt = torch.as_tensor(np.outer(f, f), dtype=x.dtype, device=x.device)
    if scale_filter:
        filt = filt * 2
    return upsample_convolve(x, odd, filt)


def _get_same_padding(x: int, kernel_size: int, stride: int, dilation: int) -> int:
    """Helper function to determine integer padding for F.pad() given img and kernel."""
    pad = (math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x
    pad = max(pad, 0)
    return pad


def same_padding(
    x: Tensor,
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = (1, 1),
    dilation: int | tuple[int, int] = (1, 1),
    pad_mode: str = "circular",
) -> Tensor:
    """Pad a tensor so that 2D convolution will result in output with same dims."""
    assert len(x.shape) > 2, "Input must be tensor whose last dims are height x width"
    ih, iw = x.shape[-2:]
    pad_h = _get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = _get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            mode=pad_mode,
        )
    return x
