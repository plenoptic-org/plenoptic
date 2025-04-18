"""
Simple convolutional models of the visual system's front-end.

All models are some combination of linear filtering, non-linear activation, and
(optionally) gain control. Model architectures in this file are found in [1]_, [2]_,
pretrained parameters from [3]_.

References
----------
.. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions of hierarchical
    representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
.. [2] https://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
.. [3] A Berardino, Hierarchically normalized models of visual distortion
   sensitivity: Physiology, perception, and application; Ph.D. Thesis,
   2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf
"""

from collections import OrderedDict
from collections.abc import Callable
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyrtools.tools.display import PyrFigure
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
    """
    Linear-Nonlinear model.

    This model applies a difference of Gaussians filter followed by an activation
    function.

    Model is described in [1]_ and [2]_, where it is called LN.

    Parameters
    ----------
    kernel_size
        Shape of convolutional kernel.
    on_center
        Dictates whether center is on or off; surround will be the opposite of center
        (i.e. on-off or off-on).
    amplitude_ratio
        Ratio of center/surround amplitude. Applied before filter normalization.
    pad_mode
        Padding for convolution.
    pretrained
        Whether or not to load model params from [3]_. See Notes for details.
    activation
        Activation function following linear convolution.
    cache_filt
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass.

    Attributes
    ----------
    center_surround: ~plenoptic.simulate.models.naive.CenterSurround
        Difference of Gaussians filter.

    Notes
    -----
    These 2 parameters (standard deviations) were taken from Table 2, page 149
    from [3]_ and are the values used [1]_. Please use these pretrained weights
    at your own discretion.

    References
    ----------
    .. [1] A Berardino, J Ballé, V Laparra, EP Simoncelli, Eigen-distortions
       of hierarchical representations, NeurIPS 2017; https://arxiv.org/abs/1710.02266
    .. [2] https://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
    .. [3] A Berardino, Hierarchically normalized models of visual distortion
       sensitivity: Physiology, perception, and application; Ph.D. Thesis,
       2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf

    Examples
    --------
    >>> import plenoptic as po
    >>> ln_model = po.simul.LinearNonlinear(31, pretrained=True, cache_filt=True)
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
            if kernel_size not in [31, (31, 31)]:
                raise ValueError("pretrained model has kernel_size (31, 31)")
            if cache_filt is False:
                warn(
                    "pretrained is True but cache_filt is False. Set cache_filt to "
                    "True for efficiency unless you are fine-tuning."
                )
            if not on_center:
                warn(
                    "pretrained model had on_center=True, so on_center=False might "
                    "not make sense"
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
        """
        Compute model response on input tensor.

        We use same-padding to ensure that the output and input shapes are matched.

        Parameters
        ----------
        x
            The input tensor, should be 4d (batch, channel, height, width).

        Returns
        -------
        y
            Model response to input.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> ln_model = po.simul.LinearNonlinear(31, pretrained=True, cache_filt=True)
          >>> img = po.data.einstein()
          >>> y = ln_model.forward(img)
          >>> titles = ["Input image", "Output"]
          >>> po.imshow([img, y], title=titles)  # doctest: +ELLIPSIS
          <PyrFigure size...>
        """
        y = self.activation(self.center_surround(x))
        return y

    def display_filters(
        self,
        vrange: tuple[float, float] | str = "indep0",
        zoom: float | None = 5.0,
        title: str | list[str] | None = "linear filter",
        **kwargs,
    ) -> PyrFigure:
        """
        Display convolutional filter of model.

        Parameters
        ----------
        vrange, zoom, title
            Arguments for :func:`plenoptic.imshow`, see its docstrings for details.
        **kwargs
            Keyword args for :func:`plenoptic.imshow`.

        Returns
        -------
        fig:
            The figure containing the image.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> ln_model = po.simul.LinearNonlinear(31, pretrained=True, cache_filt=True)
          >>> ln_model.display_filters()  # doctest: +ELLIPSIS
          <PyrFigure ...>
        """  # numpydoc ignore=ES01
        weights = self.center_surround.filt.detach()
        fig = imshow(weights, title=title, zoom=zoom, vrange=vrange, **kwargs)

        return fig

    @staticmethod
    def _pretrained_state_dict() -> OrderedDict:
        """
        Return parameters fit to human distortion judgments.

        Values copied from Table 2 in [1]_.

        Returns
        -------
        state_dict
            Dictionary of parameters, to pass to :func:`load_state_dict`.

        References
        ----------
        .. [1] A Berardino, Hierarchically normalized models of visual distortion
           sensitivity: Physiology, perception, and application; Ph.D. Thesis,
           2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf
        """
        state_dict = OrderedDict(
            [
                ("center_surround.center_std", torch.as_tensor([0.5339])),
                ("center_surround.surround_std", torch.as_tensor([6.148])),
                ("center_surround.amplitude_ratio", torch.as_tensor([1.25])),
            ]
        )
        return state_dict


class LuminanceGainControl(nn.Module):
    """
    Linear center-surround followed by luminance gain control and activation.

    Model is described in [1]_ and [2]_, where it is called LG.

    Parameters
    ----------
    kernel_size
        Shape of convolutional kernel.
    on_center
        Dictates whether center is on or off; surround will be the opposite of center
        (i.e. on-off or off-on).
    amplitude_ratio
        Ratio of center/surround amplitude. Applied before filter normalization.
    pad_mode
        Padding for convolution.
    pretrained
        Whether or not to load model params from [3]_. See Notes for details.
    activation
        Activation function following linear convolution.
    cache_filt
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass.

    Attributes
    ----------
    center_surround: ~plenoptic.simulate.models.naive.CenterSurround
        Difference of Gaussians linear filter.
    luminance: ~plenoptic.simulate.models.naive.Gaussian
        Gaussian convolutional kernel used to normalize signal by local luminance.
    luminance_scalar: torch.nn.parameter.Parameter
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
    .. [2] https://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
    .. [3] A Berardino, Hierarchically normalized models of visual distortion
       sensitivity: Physiology, perception, and application; Ph.D. Thesis,
       2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf

    Examples
    --------
    >>> import plenoptic as po
    >>> lg_model = po.simul.LuminanceGainControl(31, pretrained=True, cache_filt=True)
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
            if kernel_size not in [31, (31, 31)]:
                raise ValueError("pretrained model has kernel_size (31, 31)")
            if cache_filt is False:
                warn(
                    "pretrained is True but cache_filt is False. Set cache_filt to "
                    "True for efficiency unless you are fine-tuning."
                )
            if not on_center:
                warn(
                    "pretrained model had on_center=True, so on_center=False might "
                    "not make sense"
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
        """
        Compute model response on input tensor.

        We use same-padding to ensure that the output and input shapes are matched.

        Parameters
        ----------
        x
            The input tensor, should be 4d (batch, channel, height, width).

        Returns
        -------
        y
            Model response to input.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> lg_model = po.simul.LuminanceGainControl(
          ...     31, pretrained=True, cache_filt=True
          ... )
          >>> img = po.data.einstein()
          >>> y = lg_model.forward(img)
          >>> titles = ["Input image", "Output"]
          >>> po.imshow([img, y], title=titles)  # doctest: +ELLIPSIS
          <PyrFigure size...>
        """
        linear = self.center_surround(x)
        lum = self.luminance(x)
        lum_normed = linear / (1 + self.luminance_scalar * lum)
        y = self.activation(lum_normed)
        return y

    def display_filters(
        self,
        vrange: tuple[float, float] | str = "indep0",
        zoom: float | None = 5.0,
        title: str | list[str] | None = ["linear filt", "luminance filt"],
        col_wrap: int | None = 2,
        **kwargs,
    ) -> PyrFigure:
        """
        Display convolutional filters of model.

        Parameters
        ----------
        vrange, zoom, title, col_wrap
            Arguments for :func:`plenoptic.imshow`, see its docstrings for details.
        **kwargs
            Keyword args for :func:`plenoptic.imshow`.

        Returns
        -------
        fig:
            The figure containing the displayed filters.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> lg_model = po.simul.LuminanceGainControl(
          ...     31, pretrained=True, cache_filt=True
          ... )
          >>> lg_model.display_filters()  # doctest: +ELLIPSIS
          <PyrFigure ...>
        """  # numpydoc ignore=ES01
        weights = torch.cat(
            [
                self.center_surround.filt,
                self.luminance.filt,
            ],
            dim=0,
        ).detach()

        fig = imshow(
            weights,
            title=title,
            col_wrap=col_wrap,
            zoom=zoom,
            vrange=vrange,
            **kwargs,
        )

        return fig

    @staticmethod
    def _pretrained_state_dict() -> OrderedDict:
        """
        Return parameters fit to human distortion judgments.

        Values copied from Table 2 in [1]_.

        Returns
        -------
        state_dict
            Dictionary of parameters, to pass to :func:`load_state_dict`.

        References
        ----------
        .. [1] A Berardino, Hierarchically normalized models of visual distortion
           sensitivity: Physiology, perception, and application; Ph.D. Thesis,
           2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf
        """
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
    """
    Center-surround followed by luminance and contrast gain control, then activation.

    Model is described in [1]_ and [2]_, where it is called LGG.

    Parameters
    ----------
    kernel_size
        Shape of convolutional kernel.
    on_center
        Dictates whether center is on or off; surround will be the opposite of center
        (i.e. on-off or off-on).
    amplitude_ratio
        Ratio of center/surround amplitude. Applied before filter normalization.
    pad_mode
        Padding for convolution.
    pretrained
        Whether or not to load model params from [3]_. See Notes for details.
    activation
        Activation function following linear convolution.
    cache_filt
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass.

    Attributes
    ----------
    center_surround: ~plenoptic.simulate.models.naive.CenterSurround
        Difference of Gaussians linear filter.
    luminance: ~plenoptic.simulate.models.naive.Gaussian
        Gaussian convolutional kernel used to normalize signal by local luminance.
    contrast: ~plenoptic.simulate.models.naive.Gaussian
        Gaussian convolutional kernel used to normalize signal by local contrast.
    luminance_scalar: torch.nn.parameter.Parameter
        Scale factor for luminance normalization.
    contrast_scalar: torch.nn.parameter.Parameter
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
    .. [2] https://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
    .. [3] A Berardino, Hierarchically normalized models of visual distortion
       sensitivity: Physiology, perception, and application; Ph.D. Thesis,
       2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf

    Examples
    --------
    >>> import plenoptic as po
    >>> lgg_model = po.simul.LuminanceContrastGainControl(
    ...     31, pretrained=True, cache_filt=True
    ... )
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
            if kernel_size not in [31, (31, 31)]:
                raise ValueError("pretrained model has kernel_size (31, 31)")
            if cache_filt is False:
                warn(
                    "pretrained is True but cache_filt is False. Set cache_filt to "
                    "True for efficiency unless you are fine-tuning."
                )
            if not on_center:
                warn(
                    "pretrained model had on_center=True, so on_center=False might "
                    "not make sense"
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
        """
        Compute model response on input tensor.

        We use same-padding to ensure that the output and input shapes are matched.

        Parameters
        ----------
        x
            The input tensor, should be 4d (batch, channel, height, width).

        Returns
        -------
        y
            Model response to input.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> lgg_model = po.simul.LuminanceContrastGainControl(
          ...     31, pretrained=True, cache_filt=True
          ... )
          >>> img = po.data.einstein()
          >>> y = lgg_model.forward(img)
          >>> titles = ["Input image", "Output"]
          >>> po.imshow([img, y], title=titles)  # doctest: +ELLIPSIS
          <PyrFigure size...>
        """
        linear = self.center_surround(x)
        lum = self.luminance(x)
        lum_normed = linear / (1 + self.luminance_scalar * lum)

        con = self.contrast(lum_normed.pow(2)).sqrt() + 1e-6  # avoid div by zero
        con_normed = lum_normed / (1 + self.contrast_scalar * con)
        y = self.activation(con_normed)
        return y

    def display_filters(
        self,
        vrange: tuple[float, float] | str = "indep0",
        zoom: float | None = 5.0,
        title: str | list[str] | None = [
            "linear filt",
            "luminance filt",
            "contrast filt",
        ],
        col_wrap: int | None = 3,
        **kwargs,
    ) -> PyrFigure:
        """
        Display convolutional filters of model.

        Parameters
        ----------
        vrange, zoom, title, col_wrap
            Arguments for :func:`plenoptic.imshow`, see its docstrings for details.
        **kwargs
            Keyword args for :func:`plenoptic.imshow`.

        Returns
        -------
        fig:
            The figure containing the displayed filters.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> lgg_model = po.simul.LuminanceContrastGainControl(
          ...     31, pretrained=True, cache_filt=True
          ... )
          >>> lgg_model.display_filters()  # doctest: +ELLIPSIS
          <PyrFigure ...>
        """  # numpydoc ignore=ES01
        weights = torch.cat(
            [
                self.center_surround.filt,
                self.luminance.filt,
                self.contrast.filt,
            ],
            dim=0,
        ).detach()

        fig = imshow(
            weights,
            title=title,
            col_wrap=col_wrap,
            zoom=zoom,
            vrange=vrange,
            **kwargs,
        )

        return fig

    @staticmethod
    def _pretrained_state_dict() -> OrderedDict:
        """
        Return parameters fit to human distortion judgments.

        Values copied from Table 2 in [1]_.

        Returns
        -------
        state_dict
            Dictionary of parameters, to pass to :func:`load_state_dict`.

        References
        ----------
        .. [1] A Berardino, Hierarchically normalized models of visual distortion
           sensitivity: Physiology, perception, and application; Ph.D. Thesis,
           2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf
        """
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
    """
    On-off and off-on center-surround with contrast and luminance gain control.

    Model is described in [1]_ and [2]_, where it is called OnOff.

    Parameters
    ----------
    kernel_size
        Shape of convolutional kernel.
    amplitude_ratio
        Ratio of center/surround amplitude. Applied before filter normalization.
    pad_mode
        Padding for convolution.
    pretrained
        Whether or not to load model params estimated from [1]_. See Notes for details.
    activation
        Activation function following linear and gain control operations.
    apply_mask
        Whether or not to apply circular disk mask centered on the input image. This is
        useful for synthesis methods like Eigendistortions to ensure that the
        synthesized distortion will not appear in the periphery. See
        :func:`plenoptic.tools.signal.make_disk()` for details on how mask is created.
    cache_filt
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass.

    Attributes
    ----------
    center_surround: ~plenoptic.simulate.models.naive.CenterSurround
        2-channel (on-off and off-on) difference of Gaussians linear filter.
    luminance: ~plenoptic.simulate.models.naive.Gaussian
        2-channel Gaussian convolutional kernel used to normalize signal by local
        luminance.
    contrast: ~plenoptic.simulate.models.naive.Gaussian
        2-channel Gaussian convolutional kernel used to normalize signal by local
        contrast.
    luminance_scalar: torch.nn.parameter.Parameter
        Scale factor for luminance normalization.
    contrast_scalar: torch.nn.parameter.Parameter
        Scale factor for contrast normalization.

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

    Examples
    --------
    >>> import plenoptic as po
    >>> onoff_model = po.simul.OnOff(31, pretrained=True, cache_filt=True)
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
            if kernel_size not in [31, (31, 31)]:
                raise ValueError("pretrained model has kernel_size (31, 31)")
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
        """
        Compute model response on input tensor.

        We use same-padding to ensure that the output and input shapes are matched.

        Parameters
        ----------
        x
            The input tensor, should be 4d (batch, channel, height, width).

        Returns
        -------
        y
            Model response to input.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> onoff_model = po.simul.OnOff(31, pretrained=True, cache_filt=True)
          >>> img = po.data.einstein()
          >>> y = onoff_model.forward(img)
          >>> titles = ["Input image", "Output channel 0", "Output channel 1"]
          >>> po.imshow([img, y], title=titles)  # doctest: +ELLIPSIS
          <PyrFigure size...>
        """
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

    def display_filters(
        self,
        vrange: tuple[float, float] | str = "indep0",
        zoom: float | None = 5.0,
        title: str | list[str] | None = [
            "linear filt on",
            "linear filt off",
            "luminance filt on",
            "luminance filt off",
            "contrast filt on",
            "contrast filt off",
        ],
        col_wrap: int | None = 2,
        **kwargs,
    ) -> PyrFigure:
        """
        Display convolutional filters of model.

        Parameters
        ----------
        vrange, zoom, title, col_wrap
            Arguments for :func:`plenoptic.imshow`, see its docstrings for
            details.
        **kwargs
            Keyword args for :func:`plenoptic.imshow`.

        Returns
        -------
        fig:
            The figure containing the displayed filters.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> onoff_model = po.simul.OnOff(31, pretrained=True, cache_filt=True)
          >>> onoff_model.display_filters()  # doctest: +ELLIPSIS
          <PyrFigure ...>
        """  # numpydoc ignore=ES01
        weights = torch.cat(
            [
                self.center_surround.filt,
                self.luminance.filt,
                self.contrast.filt,
            ],
            dim=0,
        ).detach()

        fig = imshow(
            weights,
            title=title,
            col_wrap=col_wrap,
            zoom=zoom,
            vrange=vrange,
            **kwargs,
        )

        return fig

    @staticmethod
    def _pretrained_state_dict() -> OrderedDict:
        """
        Return parameters fit to human distortion judgments.

        Values copied from Table 2 in [1]_.

        Returns
        -------
        state_dict
            Dictionary of parameters, to pass to :func:`load_state_dict`.

        References
        ----------
        .. [1] A Berardino, Hierarchically normalized models of visual distortion
           sensitivity: Physiology, perception, and application; Ph.D. Thesis,
           2018; https://www.cns.nyu.edu/pub/lcv/berardino-phd.pdf
        """
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
