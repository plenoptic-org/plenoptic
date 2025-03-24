import torch
from torch import Tensor

__all__ = ["gaussian1d", "circular_gaussian2d"]


def gaussian1d(kernel_size: int = 11, std: int | float | Tensor = 1.5) -> Tensor:
    """Normalized 1D Gaussian.

    1d Gaussian of size `kernel_size`, centered half-way, with variable std
    deviation, and sum of 1.

    With default values, this is the 1d Gaussian used to generate the windows
    for SSIM

    Parameters
    ----------
    kernel_size:
        Size of Gaussian. Recommended to be odd so that kernel is properly centered.
    std:
        Standard deviation of Gaussian.

    Returns
    -------
    filt:
        1d Gaussian with `Size([kernel_size])`.

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
      >>> sin_plus_noise = (
      ... torch.sin(torch.linspace(0, 5 * torch.pi, 500)) +
      ... 0.5 * torch.randn(500)
      ... )
      >>> sin_plus_noise = sin_plus_noise.reshape(1, 1, 500)
      >>> # convolve signal with the Gaussian filter
      >>> smooth_sin = conv1d(sin_plus_noise, filt, padding="same")
      >>> # plot filter, signal and convolved signal
      >>> f, axs = plt.subplots(3, 1)
      >>> # plot filter and convolution results
      >>> axs[0].plot(filt.flatten())  #doctest: +ELLIPSIS
      [...
      >>> axs[0].set_title("1D Gaussian filter") #doctest: +ELLIPSIS
      Text(0.5, 1.0, '1D Gaussian filter')
      >>> axs[1].plot(sin_plus_noise.flatten()) #doctest: +ELLIPSIS
      [...
      >>> axs[1].set_title("Sin + Noise") #doctest: +ELLIPSIS
      Text(0.5, 1.0, 'Sin + Noise')
      >>> axs[2].plot(smooth_sin.flatten()) #doctest: +ELLIPSIS
      [...
      >>> axs[2].set_title("Convolved Sin + Noise") #doctest: +ELLIPSIS
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
    std: float | Tensor,
    out_channels: int = 1,
) -> Tensor:
    """Creates normalized, centered circular 2D gaussian tensor with which to convolve.

    Parameters
    ----------
    kernel_size:
        Filter kernel size. Recommended to be odd so that kernel is properly centered.
        If you use same-padding, convolution with an odd-length kernel will also be
        faster, see
        https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    std:
        Standard deviation of 2D circular Gaussian.
    out_channels:
        Number of channels with same kernel repeated along channel dim.

    Returns
    -------
    filt:
        Circular gaussian kernel, normalized by total pixel-sum (_not_ by 2pi*std).
        ``filt`` has shape ``(out_channels=n_channels, in_channels=1, height, width)``

    Examples
    --------
    .. plot::

      >>> import plenoptic as po
      >>> from plenoptic.simulate import circular_gaussian2d
      >>> from torch.nn.functional import conv2d
      >>> import torch
      >>> import matplotlib.pyplot as plt
      >>> kernel_size = 32
      >>> filt_2d = circular_gaussian2d(kernel_size=kernel_size, std=2.)
      >>> einstein_img = po.data.einstein()
      >>> blurred_einstein = conv2d(einstein_img, filt_2d, padding="same")
      >>> po.imshow(
      ...     [filt_2d, einstein_img, blurred_einstein],
      ...     title=["2D Gaussian Filter", "Einstein", "Blurred Einstein"]
      ... )  #doctest: +ELLIPSIS
      <PyrFigure ...>

    """
    device = torch.device("cpu") if isinstance(std, float) else std.device

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(std, float) or std.shape == torch.Size([]):
        std = torch.ones(out_channels, device=device) * std

    assert out_channels >= 1, "number of filters must be positive integer"
    assert torch.all(std > 0.0), "stdev must be positive"
    assert len(std) == out_channels, "Number of stds must equal out_channels"
    origin = torch.as_tensor(((kernel_size[0] + 1) / 2.0, (kernel_size[1] + 1) / 2.0))
    origin = origin.to(device)

    shift_y = torch.arange(1, kernel_size[0] + 1, device=device) - origin[0]  # height
    shift_x = torch.arange(1, kernel_size[1] + 1, device=device) - origin[1]  # width

    (xramp, yramp) = torch.meshgrid(shift_x, shift_y, indexing="xy")

    log_filt = (xramp**2) + (yramp**2)
    log_filt = log_filt.repeat(out_channels, 1, 1, 1)  # 4D
    log_filt = log_filt / (-2.0 * std**2).view(out_channels, 1, 1, 1)

    filt = torch.exp(log_filt)
    filt = filt / torch.sum(filt, dim=[1, 2, 3], keepdim=True)  # normalize

    return filt
