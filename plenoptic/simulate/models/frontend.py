"""
Model architectures in this file are found in [1].

[1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions of hierarchical
    representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
"""

from typing import Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...tools.conv import same_padding
from ...tools.display import imshow
from ...tools.signal import make_disk
from collections import OrderedDict

__all__ = ["Gaussian", "CenterSurround", "LinearNonlinear", "LuminanceGainControl",
           "LuminanceContrastGainControl", "OnOff"]


def circular_gaussian(
    size: Union[int, Tuple[int, int]],
    std: Tensor,
) -> Tensor:
    """Creates normalized, centered circular 2D gaussian tensor with which to convolve.
    Parameters
    ----------
    size:
        Filter kernel size.
    std:
        Standard deviation of 2D circular Gaussian.

    Returns
    -------
    filt: Tensor
    """
    assert std > 0.0, "stdev must be positive"

    device = std.device

    if isinstance(size, int):
        size = (size, size)

    origin = torch.tensor(((size[0] + 1) / 2.0, (size[1] + 1) / 2.0), device=device)

    shift_y = torch.arange(1, size[1] + 1, device=device) - origin[1]
    shift_x = torch.arange(1, size[0] + 1, device=device) - origin[0]

    (xramp, yramp) = torch.meshgrid(shift_y, shift_x)

    log_filt = ((xramp ** 2) + (yramp ** 2)) / (-2.0 * std ** 2)

    filt = torch.exp(log_filt)
    filt = filt / filt.sum()  # normalize

    return filt


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
        filt = circular_gaussian(self.kernel_size, self.std)
        return filt.view(1, 1, *filt.shape)

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
    on_center:
        Dictates whether center is on or off; surround will be the opposite of center
        (i.e. on-off or off-on).
    kernel_size: Union[int, Tuple[int, int]], optional
    width_ratio_limit:
        Ratio of surround stdev over center stdev. Surround stdev will be clamped to
        ratio_limit times center_std.
    amplitude_ratio:
        Ratio of center/surround amplitude. Applied before filter normalization.
    center_std:
        Standard deviation of circular Gaussian for center.
    surround_std:
        Standard deviation of circular Gaussian for surround. Must be at least
        ratio_limit times center_std.
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
        filt_center = circular_gaussian(self.kernel_size, self.center_std)
        filt_surround = circular_gaussian(self.kernel_size, self.surround_std)
        on_amp = self.amplitude_ratio

        if self.on_center:  # on center, off surround
            filt = on_amp * filt_center - filt_surround  # on center, off surround
        else:  # off center, on surround
            filt = on_amp * filt_surround - filt_center

        filt = filt / filt.sum()

        return filt.view(1, 1, *filt.shape)

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


class LinearNonlinear(nn.Module):
    """Linear-Nonlinear model, applies a difference of Gaussians filter followed by an
    activation function.

    Parameters
    ----------
    activation:
        Activation function following linear convolution.

    Attributes
    ----------
    center_surround: nn.Module
        `CenterSurround` difference of Gaussians filter.

    Notes
    -----
    See `CenterSurround` class for full Parameter docstring.

    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        on_center: bool = True,
        width_ratio_limit: float = 4.0,
        amplitude_ratio: float = 1.25,
        pad_mode: str = "circular",
        activation: Callable[[Tensor], Tensor] = F.softplus,
    ):
        super().__init__()
        self.center_surround = CenterSurround(
            kernel_size,
            on_center,
            width_ratio_limit,
            amplitude_ratio,
            pad_mode=pad_mode,
        )
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        y = self.activation(self.center_surround(x))
        return y

    def display_filters(self, zoom=5.0, **kwargs):
        """Displays convolutional filters of model

        Parameters
        ----------
        zoom: float
            Magnification factor for po.imshow()
        **kwargs:
            Keyword args for po.imshow
        Returns
        -------
        fig: PyrFigure
        """

        weights = self.center_surround.filt.detach()
        title = "linear"
        fig = imshow(
            weights, title=title, zoom=zoom, vrange="indep0", **kwargs
        )

        return fig


class LuminanceGainControl(nn.Module):
    """ Linear center-surround followed by luminance gain control and activation.
    Parameters
    ----------
    See `CenterSurround` class for full Parameter docstring.

    Attributes
    ----------
    center_surround: nn.Module
        Difference of Gaussians linear filter.
    luminance: nn.Module
        Gaussian convolutional kernel used to normalize signal by local luminance.
    luminance_scalar: nn.Parameter
        Scale factor for luminance normalization.
    """
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        on_center: bool = True,
        width_ratio_limit: float = 4.0,
        amplitude_ratio: float = 1.25,
        pad_mode: str = "circular",
        activation: Callable[[Tensor], Tensor] = F.softplus,
    ):
        super().__init__()
        self.center_surround = CenterSurround(
            kernel_size,
            on_center,
            width_ratio_limit,
            amplitude_ratio,
            pad_mode=pad_mode,
        )
        self.luminance = Gaussian(kernel_size=kernel_size)
        self.luminance_scalar = nn.Parameter(torch.rand(1) * 10)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        linear = self.center_surround(x)
        lum = self.luminance(x)
        lum_normed = linear / (1 + self.luminance_scalar * lum)
        y = self.activation(lum_normed)
        return y

    def display_filters(self, zoom=5.0, **kwargs):
        """Displays convolutional filters of model

        Parameters
        ----------
        zoom: float
            Magnification factor for po.imshow()
        **kwargs:
            Keyword args for po.imshow
        Returns
        -------
        fig: PyrFigure
        """

        weights = torch.cat(
            [
                self.center_surround.filt,
                self.luminance.filt,
            ],
            dim=0,
        ).detach()

        title = ["linear", "luminance norm",]

        fig = imshow(
            weights, title=title, col_wrap=2, zoom=zoom, vrange="indep0", **kwargs
        )

        return fig


class LuminanceContrastGainControl(nn.Module):
    """ Linear center-surround followed by luminance and contrast gain control,
    and activation function.

    Parameters
    ----------
    See `CenterSurround` class for full Parameter docstring.

    Attributes
    ----------
    center_surround: nn.Module
        Difference of Gaussians linear filter.
    luminance: nn.Module
        Gaussian convolutional kernel used to normalize signal by local luminance.
    contrast: nn.Module
        Gaussian convolutional kernel used to normalize signal by local contrast.
    luminance_scalar: nn.Parameter
        Scale factor for luminance normalization.
    contrast_scalar: nn.Parameter
        Scale factor for contrast normalization.

    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        on_center: bool = True,
        width_ratio_limit: float = 4.0,
        amplitude_ratio: float = 1.25,
        pad_mode: str = "circular",
        activation: Callable[[Tensor], Tensor] = F.softplus,
    ):
        super().__init__()

        self.center_surround = CenterSurround(
            kernel_size,
            on_center,
            width_ratio_limit,
            amplitude_ratio,
            pad_mode=pad_mode,
        )
        self.luminance = Gaussian(kernel_size)
        self.contrast = Gaussian(kernel_size)

        self.luminance_scalar = nn.Parameter(torch.rand(1) * 10)
        self.contrast_scalar = nn.Parameter(torch.rand(1) * 10)

        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        linear = self.center_surround(x)
        lum = self.luminance(x)
        lum_normed = linear / (1 + self.luminance_scalar * lum)

        con = self.contrast(lum_normed.pow(2)).sqrt() + 1E-6  # avoid div by zero
        con_normed = lum_normed / (1 + self.contrast_scalar * con)
        y = self.activation(con_normed)
        return y

    def display_filters(self, zoom=5.0, **kwargs):
        """Displays convolutional filters of model

        Parameters
        ----------
        zoom: float
            Magnification factor for po.imshow()
        **kwargs:
            Keyword args for po.imshow
        Returns
        -------
        fig: PyrFigure
        """

        weights = torch.cat(
            [
                self.center_surround.filt,
                self.luminance.filt,
                self.contrast.filt,
            ],
            dim=0,
        ).detach()

        title = ["linear", "luminance norm", "contrast norm"]

        fig = imshow(
            weights, title=title, col_wrap=3, zoom=zoom, vrange="indep0", **kwargs
        )

        return fig


class OnOff(nn.Module):
    """Two-channel on-off and off-on center-surround model with local contrast and
    luminance gain control.

    This model is called OnOff in Berardino et al 2017.

    Parameters
    ----------
    apply_mask:
    activation:
        Activation function following linear and gain control operations.

    Notes
    -----
    See `CenterSurround` class for full Parameter docstring.

    These 12 parameters (standard deviations & scalar constants) were reverse-engineered
    from model from [1]. Please use at your own discretion.

    [1] Berardino et al., Eigen-Distortions of Hierarchical Representations (2017)
        http://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        width_ratio_limit: float = 4.0,
        amplitude_ratio: float = 1.25,
        pad_mode: str = "circular",
        pretrained=False,
        activation: Callable[[Tensor], Tensor] = F.softplus,
        apply_mask: bool = False,
    ):
        super().__init__()
        kernel_size = (31, 31) if pretrained else kernel_size

        self.on = LuminanceContrastGainControl(
            kernel_size,
            True,  # on_center = True
            width_ratio_limit,
            amplitude_ratio,
            pad_mode,
            activation
        )

        self.off = LuminanceContrastGainControl(
            kernel_size,
            False,  # on_center = False
            width_ratio_limit,
            amplitude_ratio,
            pad_mode,
            activation
        )

        if pretrained:
            self.load_state_dict(self._pretrained_state_dict())

        self.apply_mask = apply_mask
        self._disk = None  # cached disk to apply to image

    def forward(self, x: Tensor) -> Tensor:
        y = torch.cat((self.on(x), self.off(x)), dim=1)

        if self.apply_mask:
            im_shape = x.shape[-2:]
            if self._disk is None or self._disk.shape != im_shape:  # cache new mask
                self._disk = make_disk(im_shape).to(x.device)

            y = self._disk * y  # apply the mask

        return y

    def display_filters(self, zoom=5.0, **kwargs):
        """Displays convolutional filters of model

        Parameters
        ----------
        zoom: float
            Magnification factor for po.imshow()
        **kwargs:
            Keyword args for po.imshow
        Returns
        -------
        fig: PyrFigure
        """

        weights = torch.cat(
            [
                self.on.center_surround.filt,
                self.off.center_surround.filt,
                self.on.luminance.filt,
                self.off.luminance.filt,
                self.on.contrast.filt,
                self.off.contrast.filt,
            ],
            dim=0,
        ).detach()

        title = [
            "linear on",
            "linear off",
            "luminance norm on",
            "luminance norm off",
            "contrast norm on",
            "contrast norm off",
        ]

        fig = imshow(
            weights, title=title, col_wrap=2, zoom=zoom, vrange="indep0", **kwargs
        )

        return fig

    @staticmethod
    def _pretrained_state_dict() -> OrderedDict:
        """Roughly interpreted from trained weights in Berardino et al 2017"""
        state_dict = OrderedDict(
            [
                ("on.luminance_scalar", torch.tensor([3.2637])),
                ("on.contrast_scalar", torch.tensor([7.3405])),
                ("on.center_surround.center_std", torch.tensor([1.15])),
                ("on.center_surround.surround_std", torch.tensor([5.0])),
                ("on.center_surround.amplitude_ratio", torch.tensor([1.25])),
                ("on.luminance.std", torch.tensor([8.7366])),
                ("on.contrast.std", torch.tensor([2.7353])),
                ("off.luminance_scalar", torch.tensor([14.3961])),
                ("off.contrast_scalar", torch.tensor([16.7423])),
                ("off.center_surround.center_std", torch.tensor([0.56])),
                ("off.center_surround.surround_std", torch.tensor([1.6])),
                ("off.center_surround.amplitude_ratio", torch.tensor([1.25])),
                ("off.luminance.std", torch.tensor([1.4751])),
                ("off.contrast.std", torch.tensor([1.5583])),
            ]
        )
        return state_dict
