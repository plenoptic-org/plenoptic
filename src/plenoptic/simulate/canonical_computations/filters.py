"""
Simple filters for visual models.

The functions in this module only create tensors, they do not perform convolution.
"""

import torch
from deprecated.sphinx import deprecated
from torch import Tensor

__all__ = ["gaussian1d", "circular_gaussian2d"]


@deprecated(
    "gaussian1d will be removed soon.",  # noqa: E501
    "1.2.0",
)
def gaussian1d(kernel_size: int = 11, std: int | float | Tensor = 1.5) -> Tensor:
    """
    Create normalized 1D Gaussian.

    1d Gaussian of size ``kernel_size``, centered half-way, with variable std
    deviation, and sum of 1.

    Parameters
    ----------
    kernel_size
        Size of Gaussian. Recommended to be odd so that kernel is properly centered.
    std
        Standard deviation of Gaussian.

    Returns
    -------
    filt:
        1d Gaussian with ``Size([kernel_size])``.

    Raises
    ------
    ValueError
        If ``std`` non-scalar.
    ValueError
        If ``std`` non-positive.

    Examples
    --------
    .. plot::

      >>> from plenoptic.simulate import gaussian1d
      >>> from torch.nn.functional import conv1d
      >>> import torch
      >>> import matplotlib.pyplot as plt
      >>> # define a filter
      >>> kernel_size = 21
      >>> filt = gaussian1d(kernel_size=kernel_size, std=2).reshape(1, 1, kernel_size)
      >>> # define a sinusoid + noise
      >>> sin_plus_noise = torch.sin(
      ...     torch.linspace(0, 5 * torch.pi, 500)
      ... ) + 0.5 * torch.randn(500)
      >>> sin_plus_noise = sin_plus_noise.reshape(1, 1, 500)
      >>> # convolve signal with the Gaussian filter
      >>> smooth_sin = conv1d(sin_plus_noise, filt, padding="same")
      >>> # plot filter, signal and convolved signal
      >>> f, axs = plt.subplots(3, 1)
      >>> # plot filter and convolution results
      >>> axs[0].plot(filt.flatten())
      [...
      >>> axs[0].set_title("1D Gaussian filter")
      Text(0.5, 1.0, '1D Gaussian filter')
      >>> axs[1].plot(sin_plus_noise.flatten())
      [...
      >>> axs[1].set_title("Sin + Noise")
      Text(0.5, 1.0, 'Sin + Noise')
      >>> axs[2].plot(smooth_sin.flatten())
      [...
      >>> axs[2].set_title("Convolved Sin + Noise")
      Text(0.5, 1.0, 'Convolved Sin + Noise')
      >>> plt.tight_layout()
    """
    try:
        dtype = std.dtype
    except AttributeError:
        dtype = torch.float32
    std = torch.as_tensor(std, dtype=dtype)
    if std.numel() != 1:
        raise ValueError("std must have only one element!")
    if std <= 0:
        raise ValueError("std must be positive!")
    device = std.device

    x = torch.arange(kernel_size).to(device)
    mu = kernel_size // 2
    gauss = torch.exp(-((x - mu) ** 2) / (2 * std**2))
    filt = gauss / gauss.sum()  # normalize
    return filt


def circular_gaussian2d(
    kernel_size: int | tuple[int, int],
    std: int | list[int] | float | list[float] | Tensor,
    out_channels: int | None = None,
) -> Tensor:
    """
    Create normalized, centered circular 2D gaussian tensor with which to convolve.

    The filter is normalized by total pixel-sum (*not* by ``2*pi*std``) and has shape
    ``(out_channels, 1, height, width)``. For 2d convolutions in torch, the first
    dimensions of the filter tensor corresponds to ``out_channels`` and the second to
    ``in_channels``, see :class:`torch.nn.Conv2d` for more details.

    Parameters
    ----------
    kernel_size
        Filter kernel size. Recommended to be odd so that kernel is properly centered.
        If you use same-padding, convolution with an odd-length kernel will be faster,
        see :func:`torch.nn.functional.conv2d`.
    std
        Standard deviation of 2D circular Gaussian. If a scalar and ``out_channels`` is
        not ``None``, all out channels will have the same value. If not a scalar and
        ``out_channels`` is not ``None``, ``len(std)`` must equal ``out_channels``.
    out_channels
        Number of output channels. If ``None``, inferred from shape of ``std``.

    Returns
    -------
    filt:
        Circular gaussian kernel.

    Raises
    ------
    ValueError:
        If out_channels is not a positive integer.
    ValueError:
        If kernel_size is not one or two positive integers.
    ValueError:
        If std is not positive.
    ValueError:
        If std is non-scalar and ``len(std) != out_channels``

    See Also
    --------
    :class:`~plenoptic.simulate.models.naive.Gaussian`
        Torch Module to perform this convolution.

    Examples
    --------
    Single output channel.

    .. plot::

      >>> import plenoptic as po
      >>> from plenoptic.simulate import circular_gaussian2d
      >>> from torch.nn.functional import conv2d
      >>> import torch
      >>> import matplotlib.pyplot as plt
      >>> kernel_size = 32
      >>> filt_2d = circular_gaussian2d(kernel_size=kernel_size, std=2)
      >>> filt_2d.shape
      torch.Size([1, 1, 32, 32])
      >>> einstein_img = po.data.einstein()
      >>> blurred_einstein = conv2d(einstein_img, filt_2d, padding="same")
      >>> po.imshow(
      ...     [einstein_img, filt_2d, blurred_einstein],
      ...     title=["Einstein", "2D Gaussian Filter", "Blurred Einstein"],
      ... )
      <PyrFigure ...>

    Multiple output channels with different standard deviations.

    .. plot::

      >>> import plenoptic as po
      >>> from plenoptic.simulate import circular_gaussian2d
      >>> from torch.nn.functional import conv2d
      >>> import torch
      >>> import matplotlib.pyplot as plt
      >>> kernel_size = 32
      >>> filt_2d = circular_gaussian2d(
      ...     kernel_size=kernel_size, std=[2, 5.5], out_channels=2
      ... )
      >>> filt_2d.shape
      torch.Size([2, 1, 32, 32])
      >>> einstein_img = po.data.einstein()
      >>> blurred_einstein = conv2d(einstein_img, filt_2d, padding="same")
      >>> titles = [
      ...     "Einstein",
      ...     "2D Gaussian Filter",
      ...     "Larger 2D Gaussian Filter",
      ...     "Blurred Einstein",
      ...     "Blurrier Einstein",
      ... ]
      >>> po.imshow([einstein_img, filt_2d, blurred_einstein], title=titles)
      <PyrFigure ...>

    Multiple input and output channels, convolved independently. See
    :func:`torch.nn.functional.conv2d` to understand the behavior below:

    .. plot::

      >>> import plenoptic as po
      >>> from plenoptic.simulate import circular_gaussian2d
      >>> from torch.nn.functional import conv2d
      >>> import torch
      >>> import matplotlib.pyplot as plt
      >>> kernel_size = 32
      >>> filt_2d = circular_gaussian2d(
      ...     kernel_size=kernel_size, std=[2, 5.5], out_channels=2
      ... ).repeat(3, 1, 1, 1)
      >>> filt_2d.shape
      torch.Size([6, 1, 32, 32])
      >>> wheel = po.data.color_wheel(as_gray=False)
      >>> blurred_wheel = conv2d(wheel, filt_2d, groups=3, padding="same")
      >>> titles = ["Wheel", "Blurred Wheel", "Blurrier Wheel"]
      >>> # note that the order of channels: the first two correspond to the first
      >>> # channel of the input image, convolved with the each of the two gaussians,
      >>> # and so on.
      >>> po.imshow(
      ...     [wheel, blurred_wheel[:, ::2], blurred_wheel[:, 1::2]],
      ...     title=titles,
      ...     as_rgb=True,
      ... )
      <PyrFigure ...>
    """
    kernel_size, std, out_channels = _validate_filter_args(
        kernel_size, std, out_channels
    )

    origin = (kernel_size + 1) / 2

    shift_y = torch.arange(1, kernel_size[0] + 1, device=std.device) - origin[0]
    shift_x = torch.arange(1, kernel_size[1] + 1, device=std.device) - origin[1]

    (xramp, yramp) = torch.meshgrid(shift_x, shift_y, indexing="xy")

    log_filt = (xramp**2) + (yramp**2)
    log_filt = log_filt.repeat(out_channels, 1, 1, 1)
    log_filt = log_filt / (-2.0 * std**2).view(out_channels, 1, 1, 1)

    filt = torch.exp(log_filt)
    # normalize
    filt = filt / torch.sum(filt, dim=[1, 2, 3], keepdim=True)

    return filt


def _validate_filter_args(
    kernel_size: int | tuple[int, int],
    std: int | list[int] | float | list[float] | Tensor,
    out_channels: int | None,
    std_name: str = "std",
    out_channels_name: str = "out_channels",
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Validate common filter args.

    Checks that:

    - kernel_size is positive, integer-valued, and has 1 or 2 values

    - std is positive and either a single value (i.e., an int, float, or scalar tensor)
      or ``len(std) == out_channels``

    - out_channels must be a positive integer.

    Does the following and then returns the three values

    - if ``out_channels`` is ``None``, then infer from shape of ``std``.

    - makes ``kernel_size`` a 1d tensor of size 2

    - makes ``std`` a float32 1d tensor of size ``out_channels``

    Parameters
    ----------
    kernel_size
        Filter kernel size.
    std
        Standard deviation of 2D circular Gaussian. If a scalar and ``out_channels`` is
        not ``None``, all out channels will have the same value. If not a scalar and
        ``out_channels`` is not ``None``, ``len(std)`` must equal ``out_channels``.
    out_channels
        Number of output channels. If ``None``, inferred from ``len(std)``.
    std_name, out_channels_name
        Names of these variables to raise more informative error messages (when e.g.,
        calling from ``CenterSurround``, which uses this function to validate different
        std arguments).

    Returns
    -------
    kernel_size, std, out_channels
        The validated tensors.

    Raises
    ------
    ValueError
        If out_channels is not a positive integer.
    ValueError
        If kernel_size is not one or two positive integers.
    ValueError
        If std is not positive.
    ValueError
        If std is non-scalar and ``len(std) != out_channels``
    """
    std = torch.as_tensor(std)
    if not torch.is_floating_point(std):
        std = std.to(torch.float32)

    if out_channels is None:
        out_channels = len(std) if std.ndim != 0 else 1

    if out_channels < 1 or isinstance(out_channels, float):
        raise ValueError(f"{out_channels_name} must be positive integer")

    if std.ndim == 0:
        std = std.repeat(out_channels)

    kernel_size = torch.as_tensor(kernel_size).to(std.device)
    if kernel_size.numel() == 1:
        kernel_size = kernel_size.repeat(2)
    if torch.is_floating_point(kernel_size):
        raise ValueError("kernel_size must be integer-valued")
    if torch.any(kernel_size < 1):
        raise ValueError("kernel_size must be positive")

    if torch.any(std <= 0):
        raise ValueError(f"{std_name} must be positive")
    if len(std) != out_channels:
        raise ValueError(
            f"If non-scalar, len({std_name}) must equal {out_channels_name}"
        )

    return kernel_size, std, out_channels
