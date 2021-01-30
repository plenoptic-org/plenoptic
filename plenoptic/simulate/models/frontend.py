import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
from ...tools.signal import make_disk

import os
dirname = os.path.dirname(__file__)


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
    def __init__(self, disk_mask=False, requires_grad=False, dtype=torch.float32):
        super().__init__()
        filts = np.load(dirname + '/front_end_filters.npy')
        scals = np.load(dirname + '/front_end_scalars.npy')

        self.linear = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=31, bias=False)
        self.linear.weight = nn.Parameter(torch.tensor((filts[0], filts[3]), dtype=dtype).unsqueeze(1))

        self.luminance = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=31, bias=False)
        self.luminance.weight.data = torch.tensor((filts[1], filts[4]), dtype=dtype).unsqueeze(1)

        self.contrast = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=31, groups=2, bias=False)
        self.contrast.weight.data = torch.tensor((filts[2], filts[5]), dtype=dtype).unsqueeze(1)

        # contrast and luminance scaling
        self.luminance_scale = nn.Parameter(torch.rand((1,2,1,1), dtype=dtype))
        self.luminance_scale.data = torch.tensor((scals[0], scals[2]), dtype=dtype).view((1,2,1,1))

        self.contrast_scale = nn.Parameter(torch.rand((1, 2, 1, 1), dtype=dtype))
        self.contrast_scale.data = torch.tensor((scals[1], scals[3]), dtype=dtype).view((1, 2, 1, 1))
        # self.contrast_scale = nn.Parameter(torch.tensor((scals[1], scals[3]))).view((1, 2, 1, 1))

        # pad all transforms for convolution
        pad = nn.ReflectionPad2d(filts.shape[-1]//2)
        self.linear_pad = transforms.Compose([pad, self.linear])
        self.luminance_pad = transforms.Compose([pad, self.luminance])
        self.contrast_pad = transforms.Compose([pad, self.contrast])
        self.softplus = nn.Softplus()

        self.disk_mask = disk_mask
        self.disk = None

    def luminance_normalization(self, x):
        s = self.luminance_scale
        return torch.div(self.linear_pad(x), (1 + s * self.luminance_pad(x)))

    def contrast_normalization(self, x):
        s = self.contrast_scale
        return torch.div(x, 1 + s * torch.sqrt(1e-10 + self.contrast_pad(x**2)))

    def forward(self, x):
        x = self.luminance_normalization(x)
        x = self.contrast_normalization(x)
        x = self.softplus(x)

        if self.disk_mask:
            self.disk = make_disk(x.shape[-1]).to(x.device)
            x = self.disk * x

        return x
