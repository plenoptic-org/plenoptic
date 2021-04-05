from typing import Union, Tuple
import torch
from torch import nn, nn as nn, Tensor
from torch import Tensor
import numpy as np
from torch.nn import functional as F

from plenoptic.tools.conv import same_padding
from plenoptic.simulate.canonical_computations.filters import circular_gaussian2d

__all__ = ["Identity", "Linear", "Gaussian", "CenterSurround"]


class Identity(torch.nn.Module):
    r"""simple class that just returns a copy of the image

    We use this as a "dummy model" for metrics that we don't have the
    representation for. We use this as the model and then just change
    the objective function.
    """

    def __init__(self, name=None):
        super().__init__()
        if name is not None:
            self.name = name

    def forward(self, img):
        """Return a copy of the image

        Parameters
        ----------
        img : torch.Tensor
            The image to return

        Returns
        -------
        img : torch.Tensor
            a clone of the input image

        """
        y = 1 * img
        return y


class Linear(nn.Module):
    r"""Simplistic linear convolutional model:
    It splits the input greyscale image into low and high frequencies.

    Parameters
    ----------
    kernel_size:
        Convolutional kernel size.
    pad_mode:
        Mode with which to pad image using `nn.functional.pad()`.
    default_filters:
        Initialize the filters to a low-pass and a band-pass.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        pad_mode: str = "circular",
        default_filters: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.pad_mode = pad_mode

        self.conv = nn.Conv2d(1, 2, kernel_size, bias=False)

        if default_filters:
            variance = 3
            f1 = circular_gaussian2d(kernel_size, std=np.sqrt(variance))

            f2 = circular_gaussian2d(kernel_size, std=np.sqrt(variance/3))
            f2 = f2 - f1
            f2 = f2 / f2.sum()

            self.conv.weight.data = torch.cat([f1, f2], dim=0)

    def forward(self, x: Tensor) -> Tensor:
        y = same_padding(x, self.kernel_size, pad_mode=self.pad_mode)
        h = self.conv(y)
        return h


class Gaussian(nn.Module):
    """Isotropic Gaussian single-channel convolutional filter.
    Kernel elements are normalized and sum to one.

    Parameters
    ----------
    kernel_size:
        Size of convolutional kernel.
    std:
        Standard deviation of circularly symmtric Gaussian kernel.
    pad_mode:
        Padding mode argument to pass to `torch.nn.functional.pad`.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        std: float = 3.0,
        pad_mode: str = "circular",
    ):
        super().__init__()
        assert std > 0, "Gaussian standard deviation must be positive"
        self.std = nn.Parameter(torch.tensor(std))
        self.kernel_size = kernel_size
        self.pad_mode = pad_mode

    @property
    def filt(self):
        filt = circular_gaussian2d(self.kernel_size, self.std)
        return filt

    def forward(self, x: Tensor) -> Tensor:
        self.std.data = self.std.data.abs()  # ensure stdev is positive

        x = same_padding(x, self.kernel_size, pad_mode=self.pad_mode)
        y = F.conv2d(x, self.filt)
        return y


class CenterSurround(nn.Module):
    """Center-Surround, Difference of Gaussians (DoG) filter model. Can be either
    on-center/off-surround, or vice versa.

    Filter is constructed as:
    .. math::
        f &= amplitude_ratio * center - surround \\
        f &= f/f.sum()

    The signs of center and surround are determined by `center` argument. The standard
    deviation of the surround Gaussian is constrained to be at least `width_ratio_limit`
    times that of the center Gaussian.

    Parameters
    ----------
    kernel_size:
        Shape of convolutional kernel.
    on_center:
        Dictates whether center is on or off; surround will be the opposite of center
        (i.e. on-off or off-on).
    width_ratio_limit:
        Sets a lower bound on the ratio of `surround_std` over `center_std`.
        The surround Gaussian must be wider than the center Gaussian in order to be a
        proper Difference of Gaussians. `surround_std` will be clamped to `ratio_limit`
        times `center_std`.
    amplitude_ratio:
        Ratio of center/surround amplitude. Applied before filter normalization.
    center_std:
        Standard deviation of circular Gaussian for center.
    surround_std:
        Standard deviation of circular Gaussian for surround. Must be at least
        `ratio_limit` times `center_std`.
    pad_mode:
        Padding for convolution, defaults to "circular".
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        on_center: bool = True,
        width_ratio_limit: float = 4.0,
        amplitude_ratio: float = 1.25,
        center_std: float = 1.0,
        surround_std: float = 4.0,
        pad_mode: str = "circular",
    ):
        super().__init__()

        assert width_ratio_limit > 1.0, "stdev of surround must be greater than center"
        assert amplitude_ratio >= 1.0, "ratio of amplitudes must at least be 1."

        self.on_center = on_center
        self.kernel_size = kernel_size
        self.width_ratio_limit = width_ratio_limit
        self.register_buffer("amplitude_ratio", torch.tensor(amplitude_ratio))

        self.center_std = nn.Parameter(torch.tensor(center_std))
        self.surround_std = nn.Parameter(torch.tensor(surround_std))

        self.pad_mode = pad_mode

    @property
    def filt(self) -> Tensor:
        """Creates an on center/off surround, or off center/on surround conv filter"""
        filt_center = circular_gaussian2d(self.kernel_size, self.center_std)
        filt_surround = circular_gaussian2d(self.kernel_size, self.surround_std)
        on_amp = self.amplitude_ratio

        if self.on_center:  # on center, off surround
            filt = on_amp * filt_center - filt_surround  # on center, off surround
        else:  # off center, on surround
            filt = on_amp * filt_surround - filt_center

        filt = filt / filt.sum()

        return filt

    def _clamp_surround_std(self) -> Tensor:
        """Clamps surround standard deviation to ratio_limit times center_std"""
        return self.surround_std.clamp(
            min=self.width_ratio_limit * float(self.center_std), max=None
        )

    def forward(self, x: Tensor) -> Tensor:
        x = same_padding(x, self.kernel_size, pad_mode=self.pad_mode)
        self._clamp_surround_std()  # clip the surround stdev
        y = F.conv2d(x, self.filt, bias=None)
        return y
