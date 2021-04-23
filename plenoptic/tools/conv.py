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


def correlate_downsample(image, filt, edges="reflect1", step=2, start=(0, 0), stop=None):

    n_channels = image.shape[1]

    if len(image.shape) == 3:

        if isinstance(filt, np.ndarray) or filt.shape[0] != n_channels:
            filt = torch.tensor(filt, dtype=torch.float32).repeat(n_channels,  1, 1).to(image.device)

        if edges == 'zero':
            return nn.functional.conv1d(image, filt, bias=None, stride=step,
                                        padding=(filt.shape[-1] // 2),
                                        dilation=1, groups=n_channels)

        elif edges == 'reflect1':
            pad = nn.ReflectionPad1d(filt.shape[-1]//2)
            return nn.functional.conv1d(pad(image), filt, bias=None, stride=step,
                                        padding=0, dilation=1, groups=n_channels)

    if len(image.shape) == 4:

        if isinstance(filt, np.ndarray) or filt.shape[0] != n_channels:
            filt = torch.tensor(filt, dtype=torch.float32).repeat(n_channels,  1, 1, 1).to(image.device)

        if edges == 'zero':
            return nn.functional.conv2d(image, filt, bias=None, stride=step,
                                        padding=(filt.shape[-2] // 2, filt.shape[-1] // 2),
                                        dilation=1, groups=n_channels)

        elif edges == 'reflect1':
            pad = nn.ReflectionPad2d(filt.shape[-1]//2)
            return nn.functional.conv2d(pad(image), filt, bias=None, stride=step,
                                        padding=0, dilation=1, groups=n_channels)

    if len(image.shape) == 5:

        edges = 'zero'
        step = (2, 2, 2)

        if isinstance(filt, np.ndarray) or filt.shape[0] != n_channels:
            filt = torch.tensor(filt, dtype=torch.float32).repeat(n_channels, 1, 1, 1, 1)

        if edges == 'zero':
            return nn.functional.conv3d(image, filt, bias=None, stride=step,
                                        padding=(filt.shape[-3] // 2, filt.shape[-2] // 2, filt.shape[-1] // 2),
                                        dilation=1, groups=n_channels)

        # elif edges == 'reflect1':
        #     pad = nn.ReflectionPad3d(filt.shape[-1] // 2)
        #     return nn.functional.conv3d(pad(image), filt, bias=None, stride=step,
        #                                 padding=0, dilation=1, groups=n_channels)


def upsample_convolve(image, odd, filt, edges="reflect1", step=(2, 2), start=(0, 0), stop=None):

    n_channels = image.shape[1]

    if isinstance(filt, np.ndarray) or filt.shape[0] != n_channels:
        filt = torch.tensor(filt, dtype=torch.float32, device=image.device).repeat(n_channels,  1, 1, 1)

    if edges == 'zero':
        return F.conv_transpose2d(image, filt, stride=step, padding=(filt.shape[-2] // 2, filt.shape[-1] // 2),
                                  output_padding=(1 - odd).tolist(), groups=n_channels)

    if edges == 'reflect1':
        # TODO - generalize to other image / filt sizes!
        # this solution is specific to power of two images and filt [5 x 5]
        # need start and stop arguments, two tuples of boolean values, even / odd
        pad = nn.ReflectionPad2d(1)
        return F.conv_transpose2d(pad(image), filt, stride=step, padding=4,
                                  output_padding=(1 - odd).tolist(), groups=n_channels)


def blur_downsample(x, filtname='binom5', step=(2, 2)):
    f = pt.named_filter(filtname)
    return correlate_downsample(x, filt=np.outer(f, f), step=step)


def upsample_blur(x, odd, filtname='binom5', step=(2, 2)):
    f = pt.named_filter(filtname)
    return upsample_convolve(x, odd, filt=np.outer(f, f), step=step)


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
