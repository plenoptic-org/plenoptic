import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Tuple, Union, Optional
import numpy as np
import os
from ...tools.signal import make_disk
from ...tools.display import imshow


__all__ = []


def circular_gaussian(size: Union[int, Tuple[int, int]],
                      std: torch.Tensor) -> torch.Tensor:
    """Creates normalized, centered circular 2D gaussian tensor with which to convolve.
    Parameters
    ----------
    size:
        Filter kernel size.
    std:
        Standard deviation of 2D circular Gaussian.

    Returns
    -------
    filt: torch.Tensor
    """
    assert std > 0, "stdev must be positive"

    device = std.device

    if isinstance(size, int):
        size = (size, size)

    origin = torch.tensor(((size[0] + 1) / 2., (size[1] + 1) / 2.), device=device)

    shift_y = torch.arange(1, size[1] + 1, device=device) - origin[1]
    shift_x = torch.arange(1, size[0] + 1, device=device) - origin[0]

    (xramp, yramp) = torch.meshgrid(shift_y, shift_x)

    amp = 1/(2*np.pi * std)  # normalized amplitude

    log_filt = ((xramp ** 2) + (yramp ** 2)) / (-2. * std**2)

    filt = amp * torch.exp(log_filt)

    return filt


class Gaussian(nn.Module):

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]] = (7, 7),
                 std: float = 3.):
        super().__init__()
        assert std > 0, "Gaussian standard deviation must be positive"
        self.std = nn.Parameter(torch.tensor(std))
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.std.data = self.std.data.abs()
        filt = circular_gaussian(self.kernel_size, self.std)
        y = F.conv2d(x, filt.view(1, 1, *filt.shape))
        return y


class CenterSurround(nn.Module):
    """ Center-Surround, Difference of Gaussians (DoG) filter model. Can be either on-center/off-surround, or vice versa.
    Parameters
    ----------
    center: str, Optional
        Dictates whether center is on or off. The surround always will be the opposite of the center.
        Must be either ['on', 'off']; default is 'on' center.
    kernel_size: Union[int, Tuple[int, int]], optional
    ratio_limit: float, Optional
        Ratio of surround stdev over center stdev. Surround stdev will be clamped to ratio_limit times center_std.
    center_std: float, Optional
        Standard deviation of circular Gaussian for center.
    surround_std: float, Optional
        Standard deviation of circular Gaussian for surround. Must be at least ratio_limit times center_std.
    """

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]] = (7, 7),
                 center: str = 'on',
                 ratio_limit: float = 4.,
                 center_std: float = 1.,
                 surround_std: float = 4.) -> None:
        super().__init__()

        assert center in ['on', 'off'], "center must be 'on' or 'off'"

        self.center = center
        self.kernel_size = kernel_size
        self.ratio_limit = ratio_limit

        self.center_std = nn.Parameter(torch.tensor(center_std))
        self.surround_std = nn.Parameter(torch.tensor(surround_std))

    def _center_surround(self) -> torch.Tensor:
        """Creates an on center/off surround, or off center/on surround conv filter"""
        filt_center = circular_gaussian(self.kernel_size, self.center_std)
        filt_surround = circular_gaussian(self.kernel_size, self.surround_std)
        filt = filt_center - filt_surround  # on center, off surround

        if self.center == 'off':  # off center, on surround
            filt = filt * -1

        return filt.view(1, 1, *filt.shape)

    def _clamp_surround_std(self) -> torch.Tensor:
        """Clamps surround standard deviation to ratio_limit times center_std"""
        return self.surround_std.clamp(min=self.ratio_limit * float(self.center_std), max=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._clamp_surround_std()  # clip the surround stdev
        filt = self._center_surround()
        y = F.conv2d(x, filt, bias=None)
        return y

    def display_filters(self, zoom: float = 5., **kwargs):
        """Displays convolutional filter
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

        weights = self._center_surround()
        weights = weights
        title = 'on center, off surround' if self.center == 'on' else 'off center, on surround'

        fig = imshow(weights, title=title, zoom=zoom, **kwargs)

        return fig


class LN(nn.Module):
    def __init__(self, kernel_size, center='on'):
        super().__init__()
        self.center_surround = CenterSurround(kernel_size=kernel_size, center=center)

    def forward(self, x):
        y = F.softplus(self.center_surround(x))
        return y


class LG(nn.Module):
    def __init__(self, kernel_size, center='on'):
        super().__init__()
        self.center_surround = CenterSurround(kernel_size=kernel_size, center=center)
        self.luminance = Gaussian(kernel_size=kernel_size)

    def forward(self, x):
        lum = self.luminance(x)
        luminance_normalized = self.center_surround(x)/lum
        y = F.softplus(luminance_normalized)
        return y


class LGG(nn.Module):
    def __init__(self, kernel_size, center='on'):
        super().__init__()
        self.center_surround = CenterSurround(kernel_size=kernel_size, center=center)
        self.luminance = Gaussian(kernel_size=kernel_size)
        self.contrast = Gaussian(kernel_size=kernel_size)

    def forward(self, x):
        lum = self.luminance(x)
        luminance_normalized = self.center_surround(x)/lum
        contrast = (self.contrast(luminance_normalized.pow(2)) + 1E-6).sqrt()
        contrast_normalized = luminance_normalized/contrast
        y = F.softplus(contrast_normalized)
        return y


class OnOff(nn.Module):
    def __init__(self, kernel_size, pretrained=False):
        super().__init__()
        kernel_size = (31, 31) if pretrained else kernel_size
        self.on = LGG(kernel_size, 'on')
        self.off = LGG(kernel_size, 'off')

    def forward(self, x):
        y = torch.cat((self.on(x),
                       self.off(x)), dim=1)
        return y


class FrontEnd(nn.Module):
    """Luminance and contrast gain control, modeling retina and LGN

    Parameters
    ----------
    disk_mask: boolean, optional
        Apply circular Gaussian mask to center of image. The mask itself is square.
    pretrained: bool
        Load weights from Berardino et al. 2017. These are 31x31 convolutional filters. When Pretrained is False,
        filters will still be 31x31. Major changes in the future will allow users to specify kernel size.
    requires_grad: bool
        Whether or not model is trainable.
    Returns
    -------
    y: torch.Tensor
        representation (B, 2, H, W)

    Notes
    -----
    Berardino et al., Eigen-Distortions of Hierarchical Representations (2017)
    http://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html
    """

    def __init__(self,
                 disk_mask: bool = False,
                 pretrained: bool = False,
                 requires_grad: bool = True):
        super().__init__()

        # convolutional weights
        self.linear = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=31, bias=False)
        self.luminance = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=31, bias=False)
        self.contrast = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=31, groups=2, bias=False)

        # contrast and luminance normalization scaling
        self.luminance_scalars = nn.Parameter(torch.rand((1, 2, 1, 1)))
        self.contrast_scalars = nn.Parameter(torch.rand((1, 2, 1, 1)))

        # pad all transforms for convolution
        pad = nn.ReflectionPad2d(self.linear.weight.shape[-1]//2)
        self.linear_pad = transforms.Compose([pad, self.linear])
        self.luminance_pad = transforms.Compose([pad, self.luminance])
        self.contrast_pad = transforms.Compose([pad, self.contrast])
        self.softplus = nn.Softplus()

        self.disk_mask = disk_mask
        self._disk = None  # cached disk to apply to image

        if pretrained:
            self._load_pretrained()

        if not requires_grad:  # turn off gradient
            [p.requires_grad_(False) for p in self.parameters()]

    def _load_pretrained(self):
        """Load FrontEnd model weights used from Berardino et al (2017)"""
        state_dict = torch.load(os.path.join(os.path.dirname(__file__), 'weights/FrontEnd.pt'))
        self.load_state_dict(state_dict)

    def _luminance_normalization(self, x):
        s = self.luminance_scalars
        return torch.div(self.linear_pad(x), (1 + s * self.luminance_pad(x)))

    def _contrast_normalization(self, x):
        s = self.contrast_scalars
        return torch.div(x, 1 + s * torch.sqrt(1e-10 + self.contrast_pad(x**2)))

    def forward(self, x):
        x = self._luminance_normalization(x)
        x = self._contrast_normalization(x)
        x = self.softplus(x)

        if self._disk is not None and self._disk.shape == x.shape[-2:]:  # uses cached disk_mask if size matches
            x = self._disk * x

        elif ((self._disk is not None and self._disk.shape != x.shape[-2:])  # create new disk if disk size mismatch
                or (self._disk is None and self.disk_mask)):  # or if disk does not yet exist

            self._disk = make_disk(x.shape[-1]).to(x.device)
            x = self._disk * x

        return x

    def display_filters(self, zoom=5., **kwargs):
        """Displays convolutional filters of FrontEnd model
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

        weights = torch.cat([self.linear.weight.detach(),
                             self.luminance.weight.detach(),
                             self.contrast.weight.detach()], dim=0)

        title = ['linear on', 'linear off',
                'luminance norm on', 'luminance norm off',
                'contrast norm on', 'contrast norm off']

        fig = imshow(weights, title=title, col_wrap=2, zoom=zoom, **kwargs)

        return fig
