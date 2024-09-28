"""
Model architectures in this file are found in [1]_, [2]_. `frontend.OnOff()` has
optional pretrained filters that were reverse-engineered from a previously-trained model
and should be used at your own discretion.

References
----------
.. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions of hierarchical
    representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
.. [2] https://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
"""

from collections import OrderedDict
from collections.abc import Callable
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...tools.display import imshow
from ...tools.signal import make_disk
from .naive import CenterSurround, Gaussian

__all__ = [
    "LinearNonlinear",
    "LuminanceGainControl",
    "LuminanceContrastGainControl",
    "OnOff",
]


class LinearNonlinear(nn.Module):
    """Linear-Nonlinear model, applies a difference of Gaussians filter followed by an
    activation function. Model is described in [1]_ and [2]_.

    This model is called LN in Berardino et al. 2017 [1]_.

    Parameters
    ----------
    kernel_size:
        Shape of convolutional kernel.
    on_center:
        Dictates whether center is on or off; surround will be the opposite of center
        (i.e. on-off or off-on).
    amplitude_ratio:
        Ratio of center/surround amplitude. Applied before filter normalization.
    pad_mode:
        Padding for convolution, defaults to "reflect".
    pretrained:
        Whether or not to load model params from [3]_. See Notes for details.
    activation:
        Activation function following linear convolution.
    cache_filt:
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass.

    Attributes
    ----------
    center_surround: nn.Module
        `CenterSurround` difference of Gaussians filter.

    Notes
    -----
    These 2 parameters (standard deviations) were taken from Table 2, page 149
    from [3]_ and are the values used [1]_. Please use these pretrained weights
    at your own discretion.

    References
    ----------
    .. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions
       of hierarchical representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
    .. [2] http://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
    .. [3] A Berardino, Hierarchically normalized models of visual distortion
       sensitivity: Physiology, perception, and application; Ph.D. Thesis,
       2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        on_center: bool = True,
        amplitude_ratio: float = 1.25,
        pad_mode: str = "reflect",
        pretrained: bool = False,
        activation: Callable[[Tensor], Tensor] = F.softplus,
        cache_filt: bool = False,
    ):
        super().__init__()
        if pretrained:
            assert kernel_size in [
                31,
                (31, 31),
            ], "pretrained model has kernel_size (31, 31)"
            if cache_filt is False:
                warn(
                    "pretrained is True but cache_filt is False. Set cache_filt to "
                    "True for efficiency unless you are fine-tuning."
                )
        self.center_surround = CenterSurround(
            kernel_size,
            on_center,
            amplitude_ratio,
            pad_mode=pad_mode,
            cache_filt=cache_filt,
        )
        if pretrained:
            self.load_state_dict(self._pretrained_state_dict())
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
        fig = imshow(weights, title=title, zoom=zoom, vrange="indep0", **kwargs)

        return fig

    @staticmethod
    def _pretrained_state_dict() -> OrderedDict:
        """Copied from Table 2 in Berardino, 2018"""
        state_dict = OrderedDict(
            [
                ("center_surround.center_std", torch.as_tensor([0.5339])),
                ("center_surround.surround_std", torch.as_tensor([6.148])),
                ("center_surround.amplitude_ratio", torch.as_tensor([1.25])),
            ]
        )
        return state_dict


class LuminanceGainControl(nn.Module):
    """Linear center-surround followed by luminance gain control and activation.
    Model is described in [1]_ and [2]_.

    This model is called LG in Berardino et al. 2017 [1]_.

    Parameters
    ----------
    kernel_size:
        Shape of convolutional kernel.
    on_center:
        Dictates whether center is on or off; surround will be the opposite of center
        (i.e. on-off or off-on).
    amplitude_ratio:
        Ratio of center/surround amplitude. Applied before filter normalization.
    pad_mode:
        Padding for convolution, defaults to "reflect".
    pretrained:
        Whether or not to load model params from [3]_. See Notes for details.
    activation:
        Activation function following linear convolution.
    cache_filt:
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass.

    Attributes
    ----------
    center_surround: nn.Module
        Difference of Gaussians linear filter.
    luminance: nn.Module
        Gaussian convolutional kernel used to normalize signal by local luminance.
    luminance_scalar: nn.Parameter
        Scale factor for luminance normalization.

    Notes
    -----
    These 4 parameters (standard deviations and scalar constants) were taken
    from Table 2, page 149 from [3]_ and are the values used [1]_. Please use
    these pretrained weights at your own discretion.

    References
    ----------
    .. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions of
       hierarchical representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
    .. [2] http://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
    .. [3] A Berardino, Hierarchically normalized models of visual distortion
       sensitivity: Physiology, perception, and application; Ph.D. Thesis,
       2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        on_center: bool = True,
        amplitude_ratio: float = 1.25,
        pad_mode: str = "reflect",
        pretrained: bool = False,
        activation: Callable[[Tensor], Tensor] = F.softplus,
        cache_filt: bool = False,
    ):
        super().__init__()
        if pretrained:
            assert kernel_size in [
                31,
                (31, 31),
            ], "pretrained model has kernel_size (31, 31)"
            if cache_filt is False:
                warn(
                    "pretrained is True but cache_filt is False. Set cache_filt to "
                    "True for efficiency unless you are fine-tuning."
                )
        self.center_surround = CenterSurround(
            kernel_size,
            on_center,
            amplitude_ratio,
            pad_mode=pad_mode,
            cache_filt=cache_filt,
        )
        self.luminance = Gaussian(
            kernel_size=kernel_size,
            pad_mode=pad_mode,
            cache_filt=cache_filt,
        )
        self.luminance_scalar = nn.Parameter(torch.rand(1) * 10)
        if pretrained:
            self.load_state_dict(self._pretrained_state_dict())
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

        title = [
            "linear filt",
            "luminance filt",
        ]

        fig = imshow(
            weights,
            title=title,
            col_wrap=2,
            zoom=zoom,
            vrange="indep0",
            **kwargs,
        )

        return fig

    @staticmethod
    def _pretrained_state_dict() -> OrderedDict:
        """Copied from Table 2 in Berardino, 2018"""
        state_dict = OrderedDict(
            [
                ("luminance_scalar", torch.as_tensor([14.95])),
                ("center_surround.center_std", torch.as_tensor([1.962])),
                ("center_surround.surround_std", torch.as_tensor([4.235])),
                ("center_surround.amplitude_ratio", torch.as_tensor([1.25])),
                ("luminance.std", torch.as_tensor([4.235])),
            ]
        )
        return state_dict


class LuminanceContrastGainControl(nn.Module):
    """Linear center-surround followed by luminance and contrast gain control,
    and activation function. Model is described in [1]_ and [2]_.

    This model is called LGG in Berardino et al. 2017 [1]_.

    Parameters
    ----------
    kernel_size:
        Shape of convolutional kernel.
    on_center:
        Dictates whether center is on or off; surround will be the opposite of center
        (i.e. on-off or off-on).
    amplitude_ratio:
        Ratio of center/surround amplitude. Applied before filter normalization.
    pad_mode:
        Padding for convolution, defaults to "reflect".
    pretrained:
        Whether or not to load model params from [3]_. See Notes for details.
    activation:
        Activation function following linear convolution.
    cache_filt:
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass.

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

    Notes
    -----
    These 6 parameters (standard deviations and constants) were taken from
    Table 2, page 149 from [3]_ and are the values used [1]_. Please use these
    pretrained weights at your own discretion.

    References
    ----------
    .. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions of
       hierarchical representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
    .. [2] http://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
    .. [3] A Berardino, Hierarchically normalized models of visual distortion
       sensitivity: Physiology, perception, and application; Ph.D. Thesis,
       2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        on_center: bool = True,
        amplitude_ratio: float = 1.25,
        pad_mode: str = "reflect",
        pretrained: bool = False,
        activation: Callable[[Tensor], Tensor] = F.softplus,
        cache_filt: bool = False,
    ):
        super().__init__()
        if pretrained:
            assert kernel_size in [
                31,
                (31, 31),
            ], "pretrained model has kernel_size (31, 31)"
            if cache_filt is False:
                warn(
                    "pretrained is True but cache_filt is False. Set cache_filt to "
                    "True for efficiency unless you are fine-tuning."
                )
        self.center_surround = CenterSurround(
            kernel_size,
            on_center,
            amplitude_ratio,
            pad_mode=pad_mode,
            cache_filt=cache_filt,
        )
        self.luminance = Gaussian(
            kernel_size=kernel_size,
            pad_mode=pad_mode,
            cache_filt=cache_filt,
        )
        self.contrast = Gaussian(
            kernel_size=kernel_size,
            pad_mode=pad_mode,
            cache_filt=cache_filt,
        )

        self.luminance_scalar = nn.Parameter(torch.rand(1) * 10)
        self.contrast_scalar = nn.Parameter(torch.rand(1) * 10)
        if pretrained:
            self.load_state_dict(self._pretrained_state_dict())
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        linear = self.center_surround(x)
        lum = self.luminance(x)
        lum_normed = linear / (1 + self.luminance_scalar * lum)

        con = self.contrast(lum_normed.pow(2)).sqrt() + 1e-6  # avoid div by zero
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
            weights,
            title=title,
            col_wrap=3,
            zoom=zoom,
            vrange="indep0",
            **kwargs,
        )

        return fig

    @staticmethod
    def _pretrained_state_dict() -> OrderedDict:
        """Copied from Table 2 in Berardino, 2018"""
        state_dict = OrderedDict(
            [
                ("luminance_scalar", torch.as_tensor([2.94])),
                ("contrast_scalar", torch.as_tensor([34.03])),
                ("center_surround.center_std", torch.as_tensor([0.7363])),
                ("center_surround.surround_std", torch.as_tensor([48.37])),
                ("center_surround.amplitude_ratio", torch.as_tensor([1.25])),
                ("luminance.std", torch.as_tensor([170.99])),
                ("contrast.std", torch.as_tensor([2.658])),
            ]
        )
        return state_dict


class OnOff(nn.Module):
    """Two-channel on-off and off-on center-surround model with local contrast and
    luminance gain control.

    This model is called OnOff in Berardino et al 2017 [1]_.

    Parameters
    ----------
    kernel_size:
        Shape of convolutional kernel.
    amplitude_ratio:
        Ratio of center/surround amplitude. Applied before filter normalization.
    pad_mode:
        Padding for convolution, defaults to "reflect".
    pretrained:
        Whether or not to load model params estimated from [1]_. See Notes for details.
    activation:
        Activation function following linear and gain control operations.
    apply_mask:
        Whether or not to apply circular disk mask centered on the input image. This is
        useful for synthesis methods like Eigendistortions to ensure that the
        synthesized distortion will not appear in the periphery. See
        `plenoptic.tools.signal.make_disk()` for details on how mask is created.
    cache_filt:
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass. Cached to `self._filt`.

    Notes
    -----
    These 12 parameters (standard deviations & scalar constants) were taken
    from Table 2, page 149 from [3]_ and are the values used [1]_. Please use
    these pretrained weights at your own discretion.

    References
    ----------
    .. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions of
       hierarchical representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
    .. [2] https://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
    .. [3] A Berardino, Hierarchically normalized models of visual distortion
       sensitivity: Physiology, perception, and application; Ph.D. Thesis,
       2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        amplitude_ratio: float = 1.25,
        pad_mode: str = "reflect",
        pretrained: bool = False,
        activation: Callable[[Tensor], Tensor] = F.softplus,
        apply_mask: bool = False,
        cache_filt: bool = False,
    ):
        super().__init__()
        if pretrained:
            assert kernel_size in [
                31,
                (31, 31),
            ], "pretrained model has kernel_size (31, 31)"
            if cache_filt is False:
                warn(
                    "pretrained is True but cache_filt is False. Set"
                    " cache_filt to True for efficiency unless you are"
                    " fine-tuning."
                )

        self.center_surround = CenterSurround(
            kernel_size=kernel_size,
            on_center=[True, False],
            amplitude_ratio=amplitude_ratio,
            out_channels=2,
            pad_mode=pad_mode,
            cache_filt=cache_filt,
        )

        self.luminance = Gaussian(
            kernel_size=kernel_size,
            out_channels=2,
            pad_mode=pad_mode,
            cache_filt=cache_filt,
        )

        self.contrast = Gaussian(
            kernel_size=kernel_size,
            out_channels=2,
            pad_mode=pad_mode,
            cache_filt=cache_filt,
        )

        # init scalar values around fitted parameters found in Berardino et al 2017
        self.luminance_scalar = nn.Parameter(torch.rand(2) * 10)
        self.contrast_scalar = nn.Parameter(torch.rand(2) * 10)

        if pretrained:
            self.load_state_dict(self._pretrained_state_dict())

        self.apply_mask = apply_mask
        self._disk = None  # cached disk to apply to image
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        linear = self.center_surround(x)
        lum = self.luminance(x)
        lum_normed = linear / (1 + self.luminance_scalar.view(1, 2, 1, 1) * lum)

        con = self.contrast(lum_normed.pow(2), groups=2).sqrt() + 1e-6  # avoid div by 0
        con_normed = lum_normed / (1 + self.contrast_scalar.view(1, 2, 1, 1) * con)
        y = self.activation(con_normed)

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
                self.center_surround.filt,
                self.luminance.filt,
                self.contrast.filt,
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
            weights,
            title=title,
            col_wrap=2,
            zoom=zoom,
            vrange="indep0",
            **kwargs,
        )

        return fig

    @staticmethod
    def _pretrained_state_dict() -> OrderedDict:
        """Copied from Table 2 in Berardino, 2018"""
        state_dict = OrderedDict(
            [
                ("luminance_scalar", torch.as_tensor([3.2637, 14.3961])),
                ("contrast_scalar", torch.as_tensor([7.3405, 16.7423])),
                ("center_surround.center_std", torch.as_tensor([1.237, 0.3233])),
                ("center_surround.surround_std", torch.as_tensor([30.12, 2.184])),
                ("center_surround.amplitude_ratio", torch.as_tensor([1.25])),
                ("luminance.std", torch.as_tensor([76.4, 2.184])),
                ("contrast.std", torch.as_tensor([7.49, 2.43])),
            ]
        )
        return state_dict
