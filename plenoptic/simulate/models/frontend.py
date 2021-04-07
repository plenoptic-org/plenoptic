"""
Model architectures in this file are found in [1]_, [2]_. `frontend.OnOff()` has
optional pretrained filters that were reverse-engineered from a previously-trained model
and should be used at your own discretion.

References
----------
.. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions of hierarchical
    representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
.. [2] http://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
"""

from typing import Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .naive import Gaussian, CenterSurround
from ...tools.display import imshow
from ...tools.signal import make_disk
from collections import OrderedDict

__all__ = ["LinearNonlinear", "LuminanceGainControl",
           "LuminanceContrastGainControl", "OnOff"]


class LinearNonlinear(nn.Module):
    """Linear-Nonlinear model, applies a difference of Gaussians filter followed by an
    activation function. Model is described in [1]_ and [2]_.

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
    pad_mode:
        Padding for convolution, defaults to "circular".
    activation:
        Activation function following linear convolution.

    Attributes
    ----------
    center_surround: nn.Module
        `CenterSurround` difference of Gaussians filter.

    References
    ----------
    .. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions of hierarchical
        representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
    .. [2] http://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
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
        title = "linear filt"
        fig = imshow(
            weights, title=title, zoom=zoom, vrange="indep0", **kwargs
        )

        return fig


class LuminanceGainControl(nn.Module):
    """ Linear center-surround followed by luminance gain control and activation.
    Model is described in [1]_ and [2]_.

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
    pad_mode:
        Padding for convolution, defaults to "circular".
    activation:
        Activation function following linear convolution.

    Attributes
    ----------
    center_surround: nn.Module
        Difference of Gaussians linear filter.
    luminance: nn.Module
        Gaussian convolutional kernel used to normalize signal by local luminance.
    luminance_scalar: nn.Parameter
        Scale factor for luminance normalization.

    References
    ----------
    .. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions of hierarchical
        representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
    .. [2] http://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
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

        title = ["linear filt", "luminance filt",]

        fig = imshow(
            weights, title=title, col_wrap=2, zoom=zoom, vrange="indep0", **kwargs
        )

        return fig


class LuminanceContrastGainControl(nn.Module):
    """ Linear center-surround followed by luminance and contrast gain control,
    and activation function. Model is described in [1]_ and [2]_.

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
    pad_mode:
        Padding for convolution, defaults to "circular".
    activation:
        Activation function following linear convolution.

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

    References
    ----------
    .. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions of hierarchical
        representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
    .. [2] http://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
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

        title = ["linear filt", "luminance filt", "contrast filt"]

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
    kernel_size:
        Shape of convolutional kernel.
    width_ratio_limit:
        Sets a lower bound on the ratio of `surround_std` over `center_std`.
        The surround Gaussian must be wider than the center Gaussian in order to be a
        proper Difference of Gaussians. `surround_std` will be clamped to `ratio_limit`
        times `center_std`.
    amplitude_ratio:
        Ratio of center/surround amplitude. Applied before filter normalization.
    pad_mode:
        Padding for convolution, defaults to "circular".
    pretrained:
        Whether or not to load model params estimated from [1]_. See Notes for details.
    activation:
        Activation function following linear and gain control operations.
    apply_mask:
        Whether or not to apply circular disk mask centered on the input image. This is
        useful for synthesis methods like Eigendistortions to ensure that the
        synthesized distortion will not appear in the periphery. See
        `plenoptic.tools.signal.make_disk()` for details on how mask is created.

    Notes
    -----
    These 12 parameters (standard deviations & scalar constants) were reverse-engineered
    from model from [1]_, [2]_. Please use these pretrained weights at your own
    discretion.

    References
    ----------
    .. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions of
        hierarchical representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
    .. [2] http://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
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
        cache_filt: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if pretrained:
            assert kernel_size == (31, 31), "pretrained model has kernel_size (31, 31)"
            assert cache_filt is False

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
            if self._disk.device != x.device:
                self._disk = self._disk.to(x.device)

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
            "linear filt on",
            "linear filt off",
            "luminance filt on",
            "luminance filt off",
            "contrast filt on",
            "contrast filt off",
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
