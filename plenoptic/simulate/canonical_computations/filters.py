from typing import Union, Tuple

import torch
from torch import Tensor

__all__ = ["gaussian1d", "circular_gaussian2d"]


def gaussian1d(kernel_size: int = 11, std: Union[float, Tensor] = 1.5) -> Tensor:
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
    """
    assert std > 0.0, "std must be positive"
    if isinstance(std, float):
        std = torch.tensor(std)
    device = std.device

    x = torch.arange(kernel_size).to(device)
    mu = kernel_size // 2
    gauss = torch.exp(-((x - mu) ** 2) / (2 * std ** 2))
    filt = gauss / gauss.sum()  # normalize
    return filt


def circular_gaussian2d(
    kernel_size: Union[int, Tuple[int, int]],
    std: Union[float, Tensor],
    n_channels: int = 1,
) -> Tensor:
    """Creates normalized, centered circular 2D gaussian tensor with which to convolve.

    Parameters
    ----------
    kernel_size:
        Filter kernel size. Recommended to be odd so that kernel is properly centered.
    std:
        Standard deviation of 2D circular Gaussian.
    n_channels:
        Number of channels with same kernel repeated along channel dim.

    Returns
    -------
    filt:
        Circular gaussian kernel, normalized by total pixel-sum (_not_ by 2pi*std).
        `filt` has `Size([out_channels=n_channels, in_channels=1, height, width])`.
    """
    assert std > 0.0, "stdev must be positive"
    if isinstance(std, float):
        std = torch.tensor(std)

    device = std.device

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    origin = torch.tensor(((kernel_size[0] + 1) / 2.0, (kernel_size[1] + 1) / 2.0))
    origin = origin.to(device)

    shift_y = torch.arange(1, kernel_size[0] + 1, device=device) - origin[0]
    shift_x = torch.arange(1, kernel_size[1] + 1, device=device) - origin[1]

    (xramp, yramp) = torch.meshgrid(shift_y, shift_x)

    log_filt = ((xramp ** 2) + (yramp ** 2)) / (-2.0 * std ** 2)

    filt = torch.exp(log_filt)
    filt = filt / filt.sum()  # normalize
    filt = torch.stack([filt] * n_channels, dim=0).unsqueeze(1)

    return filt
