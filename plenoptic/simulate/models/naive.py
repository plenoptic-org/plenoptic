from typing import Union, Tuple
import torch
from torch import nn
from torch import Tensor
from plenoptic.tools.conv import same_padding


__all__ = ["Identity", "Linear"]


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
        return img.clone()


class Linear(nn.Module):
    r"""Simplistic linear convolutional model:
    It splits the input greyscale image into low and high frequencies.

    Parameters
    ----------
    kernel_size:
        Convolutional kernel size.
    pad_mode:
        Mode with which to pad image using `nn.functional.pad()`
    default_filters:
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
