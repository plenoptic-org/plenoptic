import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
from ...tools.signal import make_disk

import os
dirname = os.path.dirname(__file__)


class Front_End(nn.Module):
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
    rerange input in [0,1]
    make 12 parameters explicitely learnable
    adapt to image size
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
    def __init__(self, disk_mask=False):
        super().__init__()

        filts = np.load(dirname + '/front_end_filters.npy')
        scals = np.load(dirname + '/front_end_scalars.npy')

        linear = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=31, bias=False)
        linear.weight = nn.Parameter(torch.tensor((filts[0], filts[3]),
                                     dtype=torch.float32).unsqueeze(1))

        luminance = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=31,
                              bias=False)
        luminance.weight = nn.Parameter(torch.tensor((filts[1], filts[4]),
                                        dtype=torch.float32).unsqueeze(1))

        contrast = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=31,
                             groups=2, bias=False)
        contrast.weight = nn.Parameter(torch.tensor((filts[2], filts[5]),
                                       dtype=torch.float32).unsqueeze(1))

        pad = nn.ReflectionPad2d(filts.shape[-1]//2)

        self.luminance_scals = torch.tensor((scals[0], scals[2]), dtype=torch.float32)
        self.luminance_scals = self.luminance_scals.view((1, 2, 1, 1))

        self.contrast_scals = torch.tensor((scals[1], scals[3]), dtype=torch.float32)
        self.contrast_scals = self.luminance_scals.view((1, 2, 1, 1))

        del filts, scals

        self.linear = transforms.Compose([pad, linear])
        self.luminance = transforms.Compose([pad, luminance])
        self.contrast = transforms.Compose([pad, contrast])
        self.softplus = nn.Softplus()

        self.disk_mask = disk_mask

    def luminance_normalization(self, x):

        s = self.luminance_scals
        return torch.div(self.linear(x), (1 + s * self.luminance(x)))

    def contrast_normalization(self, x):

        s = self.contrast_scals
        return torch.div(x, 1 + s * torch.sqrt(1e-10 + self.contrast(x ** 2)))

    def forward(self, x):

        image_size = x.shape[-1]
        x = self.luminance_normalization(x)
        x = self.contrast_normalization(x)
        x = self.softplus(x)

        if self.disk_mask:
            self.disk = make_disk(image_size)

            x = self.disk * x

        return x
