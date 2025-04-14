import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from ...tools.conv import same_padding
from ..canonical_computations.filters import _validate_filter_args, circular_gaussian2d

__all__ = ["Identity", "Linear", "Gaussian", "CenterSurround"]


class Identity(torch.nn.Module):
    r"""simple class that just returns a copy of the image

    We use this as a "dummy model" for metrics that we don't have the
    representation for. We use this as the model and then just change
    the objective function.

    Examples
    --------
    >>> import plenoptic as po
    >>> identity_model = po.simul.Identity()
    >>> identity_model
    Identity()
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Return a copy of the tensor

        Parameters
        ----------
        x : torch.Tensor
            The tensor to return

        Returns
        -------
        x : torch.Tensor
            a clone of the input tensor

        Examples
        --------
        .. plot::

           >>> import plenoptic as po
           >>> identity_model = po.simul.Identity()
           >>> img = po.data.curie()
           >>> y = identity_model.forward(img)
           >>> titles = ["Input", "Output (identical)"]
           >>> po.imshow([img, y], title=titles) #doctest: +ELLIPSIS
           <PyrFigure ...>

        """
        y = 1 * x
        return y


class Linear(nn.Module):
    r"""Simplistic linear convolutional model.

    If ``default_filters=True``, this model splits the input image into low
    and high frequencies.

    Parameters
    ----------
    kernel_size:
        Convolutional kernel size.
    pad_mode:
        Mode with which to pad image using `nn.functional.pad()`.
    default_filters:
        Initialize the filters to a low-pass and a band-pass. If False, filters are
        randomly initialized.

    Raises
    ------
    ValueError:
        If kernel_size is not one or two positive integers.

    Examples
    --------
    >>> import plenoptic as po
    >>> linear_model = po.simul.Linear()
    >>> linear_model
    Linear(
      (conv): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), bias=False)
    )

    To specify the kernel size :

    >>> linear_model = po.simul.Linear(kernel_size=(5, 5))
    >>> linear_model
    Linear(
      (conv): Conv2d(1, 2, kernel_size=(5, 5), stride=(1, 1), bias=False)
    )

    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = (3, 3),
        pad_mode: str = "circular",
        default_filters: bool = True,
    ):
        super().__init__()

        self.pad_mode = pad_mode
        # std and out_channels are not used by Linear, so set to values we know will
        # pass
        self.kernel_size, _, _ = _validate_filter_args(kernel_size, 1, 1)

        self.conv = nn.Conv2d(1, 2, kernel_size, bias=False)

        if default_filters:
            var = torch.as_tensor(3.0)
            f1 = circular_gaussian2d(kernel_size, std=torch.sqrt(var))

            f2 = circular_gaussian2d(kernel_size, std=torch.sqrt(var / 3))

            f2 = f2 - f1
            f2 = f2 / f2.sum()

            self.conv.weight.data = torch.cat([f1, f2], dim=0)

    def forward(self, x: Tensor) -> Tensor:
        """Convolve filter with input tensor.

        We use same-padding to ensure that the output and input shapes are matched.

        Parameters
        ----------
        x :
            The input tensor, should be 4d (batch, channel, height, width)

        Returns
        -------
        y :
            a linear convolution of the input image, of same shape as the input.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> linear_model = po.simul.Linear()
          >>> img = po.data.curie()
          >>> y = linear_model.forward(img)
          >>> po.imshow([img, y], title=["Input image", "Lowpass channel output",
          ...                            "Bandpass channel output"]) #doctest: +ELLIPSIS
          <PyrFigure size...>

        """
        y = same_padding(x, self.kernel_size, pad_mode=self.pad_mode)
        h = self.conv(y)
        return h


class Gaussian(nn.Module):
    """Isotropic Gaussian convolutional filter.
    Kernel elements are normalized and sum to one.

    Parameters
    ----------
    kernel_size:
        Size of convolutional kernel.
    std:
        Standard deviation of circularly symmtric Gaussian kernel.
    pad_mode:
        Padding mode argument to pass to `torch.nn.functional.pad`.
    out_channels:
        Number of filters. If None, inferred from shape of ``std``.
    cache_filt:
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass. Cached to `self._filt`.

    Raises
    ------
    ValueError:
        If out_channels is not a positive integer.
    ValueError:
        If kernel_size is not a positive integer.
    ValueError:
        If std is not positive.
    ValueError:
        If std is non-scalar and ``len(std) != out_channels``

    Examples
    --------
    >>> import plenoptic as po
    >>> gaussian_model = po.simul.Gaussian(kernel_size=10)
    >>> gaussian_model
    Gaussian()

    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        std: int | list[int] | float | list[float] | Tensor = 3.0,
        pad_mode: str = "reflect",
        out_channels: int | None = None,
        cache_filt: bool = False,
    ):
        super().__init__()
        self.kernel_size, std, out_channels = _validate_filter_args(
            kernel_size, std, out_channels
        )
        self.std = nn.Parameter(std)

        self.pad_mode = pad_mode
        self.out_channels = out_channels

        self.cache_filt = cache_filt
        self.register_buffer("_filt", None)

    @property
    def filt(self):
        if self._filt is not None:  # use old filter
            return self._filt
        else:  # create new filter, optionally cache it
            filt = circular_gaussian2d(self.kernel_size, self.std, self.out_channels)

            if self.cache_filt:
                self.register_buffer("_filt", filt)
            return filt

    def forward(self, x: Tensor, **conv2d_kwargs) -> Tensor:
        """Convolve Gaussian filter with input tensor

        We use same-padding to ensure that the output and input shapes are matched.

        Parameters
        ----------
        x :
            The input tensor, should be 4d (batch, channel, height, width)
        conv2d_kwargs :
            Passed to [](torch.nn.functional.conv2d).

        Returns
        -------
        y :
            a linear convolution of the input image, of same shape as the input.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> gaussian_model = po.simul.Gaussian(kernel_size=10)
          >>> img = po.data.curie()
          >>> y = gaussian_model.forward(img)
          >>> po.imshow([img, y], title=["Input image", "Output"]) #doctest: +ELLIPSIS
          <PyrFigure size...>

        Multiple output channels with different standard deviations.

        .. plot::

          >>> import plenoptic as po
          >>> gaussian_model = po.simul.Gaussian(kernel_size=10, std=[2, 5],
          ...                                    out_channels=2)
          >>> img = po.data.curie()
          >>> y = gaussian_model.forward(img)
          >>> po.imshow([img, y], title=["Input image", "Output Channel 0",
          ...                            "Output Channel 1"]) #doctest: +ELLIPSIS
          <PyrFigure ...>

        """
        self.std.data = self.std.data.abs()  # ensure stdev is positive

        x = same_padding(x, self.kernel_size, pad_mode=self.pad_mode)
        y = F.conv2d(x, self.filt, **conv2d_kwargs)

        return y


class CenterSurround(nn.Module):
    """Center-Surround, Difference of Gaussians (DoG) filter model. Can be either
    on-center/off-surround, or vice versa.

    Filter is constructed as:
    .. math::
        f &= amplitude_ratio * center - surround \\
        f &= f/f.sum()

    The signs of center and surround are determined by ``on_center`` argument.

    Parameters
    ----------
    kernel_size:
        Shape of convolutional kernel.
    on_center:
        Dictates whether center is on or off; surround will be the opposite of center
        (i.e. on-off or off-on). If List of bools, then list length must equal
        `out_channels`, if just a single bool, then all `out_channels` will be assumed
        to be all on-off or off-on.
    amplitude_ratio:
        Ratio of center/surround amplitude. Applied before filter normalization. Must be
        greater than or equal to 1.
    center_std:
        Standard deviation of circular Gaussian for center.
    surround_std:
        Standard deviation of circular Gaussian for surround.
    out_channels:
        Number of filters. If None, inferred from shape of ``center_std``.
    pad_mode:
        Padding for convolution, defaults to "circular".
    cache_filt:
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass. Cached to `self._filt`

    Raises
    ------
    ValueError:
        If out_channels is not a positive integer.
    ValueError:
        If kernel_size is not a positive integer.
    ValueError:
        If center_std or surround_std are not positive.
    ValueError:
        If center_std and surround_std do not have the same number of values.
    ValueError:
        If center_std or surround_std are non-scalar and their lengths do not
        equal ``out_channels``

    Examples
    --------
    >>> import plenoptic as po
    >>> cs_model = po.simul.CenterSurround(kernel_size=10)
    >>> cs_model
    CenterSurround()

    Model with both on-center/off-surround and off-center/on-surround:

    >>> import plenoptic as po
    >>> cs_model = po.simul.CenterSurround(10, [True, False])
    >>> cs_model
    CenterSurround()

    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        on_center: bool | list[bool] = True,
        amplitude_ratio: float = 1.25,
        center_std: int | list[int] | float | list[float] | Tensor = 1.0,
        surround_std: int | list[int] | float | list[float] | Tensor = 4.0,
        out_channels: int | None = None,
        pad_mode: str = "reflect",
        cache_filt: bool = False,
    ):
        super().__init__()

        on_center = torch.as_tensor(on_center)
        if out_channels is None and on_center.numel() != 1:
            out_channels = len(on_center)

        self.kernel_size, center_std, out_channels = _validate_filter_args(
            kernel_size,
            center_std,
            out_channels,
            "center_std",
        )
        _, surround_std, _ = _validate_filter_args(
            kernel_size, surround_std, out_channels, "surround_std", "len(center_std)"
        )

        self.center_std = nn.Parameter(center_std)
        self.surround_std = nn.Parameter(surround_std)

        # make sure each channel is on-off or off-on
        if on_center.numel() == 1:
            on_center = on_center.repeat(out_channels)
        if len(on_center) != out_channels:
            raise ValueError("len(on_center) must equal out_channels")
        self.on_center = on_center

        amplitude_ratio = torch.as_tensor(amplitude_ratio)
        if amplitude_ratio.nelement() > 1:
            raise ValueError("amplitude_ratio must be a scalar")
        if amplitude_ratio < 1.0:
            raise ValueError("amplitude_ratio must at least be 1.")
        self.register_buffer("amplitude_ratio", amplitude_ratio)

        self.out_channels = out_channels
        self.pad_mode = pad_mode

        self.cache_filt = cache_filt
        self.register_buffer("_filt", None)

    @property
    def filt(self) -> Tensor:
        """Creates an on center/off surround, or off center/on surround conv filter"""
        if self._filt is not None:
            # use cached filt
            return self._filt
        else:
            # generate new filt and optionally cache

            filt_center = circular_gaussian2d(
                self.kernel_size, self.center_std, self.out_channels
            )
            filt_surround = circular_gaussian2d(
                self.kernel_size, self.surround_std, self.out_channels
            )

            # sign is + or - depending on center is on or off
            sign = torch.as_tensor(
                [1.0 if x else -1.0 for x in self.on_center],
                device=self.amplitude_ratio.device,
            )
            sign = sign.view(self.out_channels, 1, 1, 1)
            filt = self.amplitude_ratio * (sign * (filt_center - filt_surround))

            if self.cache_filt:
                self.register_buffer("_filt", filt)
        return filt

    def forward(self, x: Tensor) -> Tensor:
        """Convolve center-surround filter with input tensor.

        We use same-padding to ensure that the output and input shapes are matched.

        Parameters
        ----------
        x :
            The input tensor, should be 4d (batch, channel, height, width)

        Returns
        -------
        y :
            a linear convolution of the input image, of same shape as the input.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> cs_model = po.simul.CenterSurround(kernel_size=10)
          >>> img = po.data.curie()
          >>> y = cs_model.forward(img)
          >>> po.imshow([img, y], title=["Input image", "Output"]) #doctest: +ELLIPSIS
          <PyrFigure size...>

        Model with both on-center/off-surround and off-center/on-surround:

        .. plot::

          >>> import plenoptic as po
          >>> cs_model = po.simul.CenterSurround(10, [True, False])
          >>> img = po.data.curie()
          >>> y = cs_model.forward(img)
          >>> titles = ["Input image", "On-center/off-surround",
          ...           "Off-center/on-surround"]
          >>> po.imshow([img, y], title=titles) #doctest: +ELLIPSIS
          <PyrFigure size...>

        """
        x = same_padding(x, self.kernel_size, pad_mode=self.pad_mode)
        y = F.conv2d(x, self.filt, bias=None)
        return y
