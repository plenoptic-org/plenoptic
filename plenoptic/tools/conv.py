import numpy as np
import torch
from torch import nn
from torchvision import transforms
import pyrtools as pt

# TODO
# tests
# faster implementation with separable 1d conv
# handle batch dimension and infer dimension 1,2,3
# fft - circular
# test that it does the right thing for multiple channels

def correlate_downsample(image, filt, edges="reflect1", step=(2, 2), start=(0, 0), stop=None):

    n_channels = image.shape[1]

    if isinstance(filt, np.ndarray) or filt.shape[0] != n_channels:
        filt = torch.tensor(filt, dtype=torch.float32).repeat(n_channels,  1, 1, 1)

    if edges == 'zero':
        return nn.functional.conv2d(image, filt, bias=None, stride=step, padding=(filt.shape[-2] // 2, filt.shape[-1] // 2), dilation=1, groups=n_channels)

    elif edges == 'reflect1':
        pad = nn.ReflectionPad2d(filt.shape[-1]//2)
        return nn.functional.conv2d(pad(image), filt, bias=None, stride=step, padding=0, dilation=1, groups=n_channels)


def upsample_convolve(image, filt, edges="reflect1", step=(2, 2), start=(0, 0), stop=None):

    n_channels = image.shape[1]

    if isinstance(filt, np.ndarray) or filt.shape[0] != n_channels:
        filt = torch.tensor(filt, dtype=torch.float32).repeat(n_channels,  1, 1, 1)

    if edges == 'zero':
        upsample_convolve = nn.functional.conv_transpose2d(image, filt, bias=None, stride=step, padding=(filt.shape[-2] // 2, filt.shape[-1] // 2), output_padding=1, groups=n_channels, dilation=1)

    if edges == 'reflect1':
        # TODO - generalize to other image / filt sizes!
        # this solution is specific to power of two images and filt [5 x 5]
        # need start and stop arguments, two tuples of boolean values, even / odd
        pad = nn.ReflectionPad2d(1)
        return nn.functional.conv_transpose2d(pad(image), filt, bias=None, stride=step, padding=4, output_padding=1, groups=n_channels, dilation=1)

def blur_downsample(x):
    f = pt.named_filter('binom5')
    return correlate_downsample(x, filt=np.outer(f, f))

def upsample_blur(x):
    f = pt.named_filter('binom5')
    return upsample_convolve(x, filt=np.outer(f, f))
