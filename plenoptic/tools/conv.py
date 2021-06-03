import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import pyrtools as pt
from typing import Union, Tuple
import math

# TODO under development
# needs documentation and testing
# handle multiple channels
# handle batch dimension
# handle signal of dimension 1,2,3
# faster implementation with separable 1d conv


def correlate_downsample(signal, filt, edges="reflect1",
                         step=2, start=(0, 0), stop=None):
    """compute the correlation of `signal` with `filter`

    Args:
        signal ([type]): [description]
        filt ([type]): [description]
        edges (str, optional): [description]. Defaults to "reflect1".
        step (int, optional): [description]. Defaults to 2.
        start (tuple, optional): [description]. Defaults to (0, 0).
        stop ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    n_channels = signal.shape[1]

    if len(signal.shape) == 3:

        if isinstance(filt, np.ndarray) or filt.shape[0] != n_channels:
            filt = torch.tensor(filt, dtype=torch.float32)
            filt = filt.repeat(n_channels, 1, 1).to(signal.device)

        if edges == 'zero':
            return nn.functional.conv1d(signal, filt, bias=None, stride=step,
                                        padding=(filt.shape[-1] // 2),
                                        dilation=1, groups=n_channels)

        elif edges == 'reflect1':
            pad = nn.ReflectionPad1d(filt.shape[-1]//2)
            return nn.functional.conv1d(pad(signal), filt, bias=None,
                                        stride=step, padding=0, dilation=1,
                                        groups=n_channels)

    if len(signal.shape) == 4:

        if isinstance(filt, np.ndarray) or filt.shape[0] != n_channels:
            filt = torch.tensor(filt, dtype=torch.float32)
            filt = filt.repeat(n_channels,  1, 1, 1).to(signal.device)

        if edges == 'zero':
            return nn.functional.conv2d(signal, filt, bias=None, stride=step,
                            padding=(filt.shape[-2] // 2, filt.shape[-1] // 2),
                                        dilation=1, groups=n_channels)

        elif edges == 'reflect1':
            pad = nn.ReflectionPad2d(filt.shape[-1]//2)
            return nn.functional.conv2d(pad(signal), filt, bias=None,
                                        stride=step, padding=0, dilation=1,
                                        groups=n_channels)

    if len(signal.shape) == 5:

        edges = 'zero'
        step = (2, 2, 2)

        if isinstance(filt, np.ndarray) or filt.shape[0] != n_channels:
            filt = torch.tensor(filt, dtype=torch.float32)
            filt = filt.repeat(n_channels, 1, 1, 1, 1)

        if edges == 'zero':
            return nn.functional.conv3d(signal, filt, bias=None, stride=step,
            padding=(filt.shape[-3] // 2, filt.shape[-2] // 2, filt.shape[-1] // 2),
                                        dilation=1, groups=n_channels)

        # elif edges == 'reflect1':
        #     pad = nn.ReflectionPad3d(filt.shape[-1] // 2)
        #     return nn.functional.conv3d(pad(signal), filt, bias=None, stride=step,
        #                                 padding=0, dilation=1, groups=n_channels)


def upsample_convolve(signal, filt, edges="reflect1",
                      step=(2, 2), start=(0, 0), stop=None):

    n_channels = signal.shape[1]

    if isinstance(filt, np.ndarray) or filt.shape[0] != n_channels:
        filt = torch.tensor(filt, dtype=torch.float32)
        filt = filt.repeat(n_channels,  1, 1, 1)

    if edges == 'zero':
        upsample_convolve = nn.functional.conv_transpose2d(signal, filt,
                            bias=None, stride=step,
                            padding=(filt.shape[-2] // 2, filt.shape[-1] // 2),
                            output_padding=1, groups=n_channels, dilation=1)

    if edges == 'reflect1':
        # TODO - generalize to other signal / filt sizes!
        # this solution is specific to power of two signals and filt [5 x 5]
        # need start and stop arguments, two tuples of boolean values,
        # even / odd
        pad = nn.ReflectionPad2d(1)
        return nn.functional.conv_transpose2d(pad(signal), filt, bias=None,
                                              stride=step, padding=4,
                                              output_padding=1,
                                              groups=n_channels,
                                              dilation=1)


def blur_downsample(x, filtname='binom5', step=(2, 2)):
    """""[summary]""

    Args:
        x ([type]): [description]
        filtname (str, optional): [description]. Defaults to 'binom5'.
        step (tuple, optional): [description]. Defaults to (2, 2).

    Returns:
        [type]: [description]

    TODO
    ----
    make it separable for efficiency and clarity
    """
    f = pt.named_filter(filtname)
    return correlate_downsample(x, filt=np.outer(f, f), step=step)


def upsample_blur(x, step=(2, 2)):
    f = pt.named_filter('binom5')
    return upsample_convolve(x, filt=np.outer(f, f), step=step)


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
