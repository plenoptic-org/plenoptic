import torch
from torch import Tensor
import torch.nn as nn

from typing import Union, Tuple
from ...tools.conv import same_padding

__all__ = ["Linear", "LinearNonlinear"]


class Linear(nn.Module):
    """Simplistic linear convolutional model:
    It splits the input greyscale image into low and high frequencies.

    Parameters
    ----------
    kernel_size:
        Convolutional kernel size.
    pad_mode:
        Mode with which to pad image using `nn.functional.pad()`
    default_filters :
        Initialize the first three filters to: low pass, high pass on / off.
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
            xs = torch.linspace(-2, 2, kernel_size[0])
            g1 = torch.exp(-(xs ** 2) / (2 * variance))
            f1 = torch.outer(g1, g1)
            f1 = f1 / f1.sum()

            g2 = torch.exp(-(xs ** 2) / (variance / 3))
            f2 = torch.outer(g2, g2)
            f2 = f2 / f2.sum() - f1
            f2 = f2 - f2.sum()

            self.conv.weight.data[0, 0] = nn.Parameter(f1)
            self.conv.weight.data[1, 0] = nn.Parameter(f2)

    def forward(self, x: Tensor) -> Tensor:
        y = same_padding(x, self.kernel_size, pad_mode=self.pad_mode)
        h = self.conv(y)
        return h


class LinearNonlinear(nn.Module):
    """Canonical functional model of early visual processing.
    Parameters
    ----------
    n_channels: int, optional
        number of convolutional channels
    kernel_size : tuple of two ints
        size of the receptive fields (odd size supported), [3x3] by default
    default_filters : bool, optional
        Initialize the first three filters to: low pass, high pass on / off.

    Notes
    -----
    The idea of "Threshold Logic Unit" goes at least as far back as:
    McCulloch, W.S. and Pitts, W., 1943. A logical calculus of the ideas
    immanent in nervous activity. The bulletin of mathematical biophysics

    (cascades of) linear filters and simple nonlinearities are
    surprisingly effective at performing visual tasks and at capturing
    cell responses in the early visual system:
    - the linear filters capture basic cell properties (tuning/invariance)
    - the ReLU captures the point (a.k.a. memoryless or static) nonlinearity
        of cells typicall input-output curve. To capture modulatory phenomenon,
        use other nonlinearities like Gain control / Divisive normalization.

    """

    def __init__(
            self,
            n_channels: int = 3,
            kernel_size: Union[int, Tuple[int, int]]=(3, 3),
            default_filters: bool = True,
            pad_mode: str = "circular",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.pad_mode = pad_mode

        self.conv = nn.Conv2d(1, n_channels, kernel_size, bias=False)

        if default_filters and n_channels >= 3:
            variance = 3
            xs = torch.linspace(-2, 2, kernel_size[0])

            g1 = torch.exp(-(xs ** 2) / (2 * variance))
            f1 = torch.outer(g1, g1)
            f1 = f1 / f1.sum()

            g2 = torch.exp(-(xs ** 2) / (variance / 3))
            f2 = torch.outer(g2, g2)
            f2 = f2 / f2.sum() - f1
            f2 = f2 - f2.sum()

            self.conv.weight.data[0, 0] = nn.Parameter(f1)
            self.conv.weight.data[1, 0] = nn.Parameter(f2)
            self.conv.weight.data[2, 0] = nn.Parameter(-f2)

        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        y = same_padding(x, self.kernel_size, pad_mode=self.pad_mode)
        h = self.relu(self.conv(y))
        return h
