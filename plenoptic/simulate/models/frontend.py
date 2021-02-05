import torch
import torch.nn as nn
from torchvision import transforms
from ...tools.signal import make_disk
import os
import plenoptic as po


class FrontEnd(nn.Module):
    """Luminance and contrast gain control, modeling retina and LGN

    Parameters
    ----------
    disk_mask: boolean, optional
        crop the result to a disk at the center of the image
    x: torch.Tensor
        greyscale image (B, 1, H, W), requires H and W to be equal and greater than 31

    Returns
    -------
    y: torch.Tensor
        representation (B, 2, H, W)

    Notes
    -----
    based on code by Alex Berardino
    Eigen-Distortions of Hierarchical Representations
    Image Quality Assessment Models
    http://www.cns.nyu.edu/~lcv/eigendistortions/ModelsIQA.html

    TODO
    -----
    Parameterize Conv2D as Difference of Gaussians -- the model should have 13 params total
    videos
        optional argument for video
        (B, 1, T, X, Y)
        Conv3d
        Four channels
        midget on    fine space coarse time
        midget off
        parasol on   fine time coarse space
        parasol off
    """

    def __init__(self, disk_mask=False, pretrained=False, requires_grad=True):
        """
        Parameters
        ----------
        disk_mask: bool
            Apply circular Gaussian mask to center of image. The mask itself is square.
        pretrained: bool
            Load weights from Berardino et al. 2017. These are 31x31 convolutional filters.
        requires_grad: bool
            Whether or not model is trainable.
        """
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

        fig = po.imshow(weights, title=title, col_wrap=2, zoom=zoom, **kwargs)

        return fig
