import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import pyrtools as pt
from typing import Union, Tuple
import math

# TODO
# documentation
# test that it does the right thing for multiple channels
# handle batch dimension and infer dimension 1,2,3
# faster implementation with separable 1d conv
# fft - circular


def correlate_downsample(image, filt, padding_mode="reflect"):
    if padding_mode == "zero":
        padding_mode = "constant"
    assert isinstance(image, torch.Tensor) and isinstance(filt, torch.Tensor)

    n_channels = image.shape[1]
    image_padded = same_padding(image, kernel_size=filt.shape, pad_mode=padding_mode)
    return F.conv2d(image_padded, filt.repeat(n_channels, 1, 1, 1), stride=2, groups=n_channels)


def upsample_convolve(image, odd, filt, padding_mode="reflect"):
    """Upsample by 2 and convolve with a filter

    Parameters
    ----------
    image: torch.Tensor of shape (batch, channel, height, width)
        Image, or batch of images. Channels are also treated as batches.
    odd: tuple, list, numpy.ndarray
        This should contain two integers of value 0 or 1, which determines whether
        the output height and width should be even (0) or odd (1).
    filt: 2-D torch.Tensor
        The filter to convolve with the upsampled image
    padding_mode: string
        One of "constant", "reflect", "replicate", "circular" or "zero" (same as "constant")

    """

    if padding_mode == "zero":
        padding_mode = "constant"
    assert isinstance(image, torch.Tensor) and isinstance(filt, torch.Tensor)
    filt = filt.flip((0, 1))

    n_channels = image.shape[1]
    pad_size = np.array(filt.shape) - np.array(odd) + 1
    pad_rule = np.array([[0, -1, 0, 1],
                         [0, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 1, 1]])
    pad_plan = pad_rule[pad_size % 4] + np.outer(pad_size // 4, np.array([1, 1, 0, 0]))
    image_prepad = F.pad(image, [pad_plan[1, 0], pad_plan[1, 1], pad_plan[0, 0], pad_plan[0, 1]], mode=padding_mode)
    image_upsample = F.conv_transpose2d(image_prepad, weight=torch.ones((n_channels, 1, 1, 1), device=image.device),
                                        stride=2, groups=n_channels)
    image_postpad = F.pad(image_upsample, [pad_plan[1, 2], pad_plan[1, 3], pad_plan[0, 2], pad_plan[0, 3]])
    return F.conv2d(image_postpad, filt.repeat(n_channels, 1, 1, 1), groups=n_channels)


def blur_downsample(x, filtname='binom5', filter_norm_one=False):
    f = pt.named_filter(filtname)
    filt = torch.tensor(np.outer(f, f), dtype=torch.float32, device=x.device)
    if filter_norm_one:
        filt = filt / 2
    return correlate_downsample(x, filt)


def upsample_blur(x, odd, filtname='binom5', filter_norm_one=False):
    f = pt.named_filter(filtname)
    filt = torch.tensor(np.outer(f, f), dtype=torch.float32, device=x.device)
    if filter_norm_one:
        filt = filt * 2
    return upsample_convolve(x, odd, filt)


def _get_same_padding(
        x: int,
        kernel_size: int,
        stride: int,
        dilation: int
) -> int:
    """Helper function to determine integer padding for F.pad() given img and kernel"""
    pad = (math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x
    pad = max(pad, 0)
    return pad


def same_padding(
        x: Tensor,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = (1, 1),
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        pad_mode: str = "circular",
) -> Tensor:
    """Pad a tensor so that 2D convolution will result in output with same dims."""
    assert len(x.shape) > 2, "Input must be tensor whose last dims are height x width"
    ih, iw = x.shape[-2:]
    pad_h = _get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = _get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x,
                  [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                  mode=pad_mode)
    return x
