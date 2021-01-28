import torch
import torch.nn as nn
from einops import rearrange

from ..canonical_computations.steerable_pyramid_freq import \
    Steerable_Pyramid_Freq
from ...tools.signal import rectangular_to_polar, polar_to_rectangular


class Polar_Pyramid(nn.Module):
    """
       non linear invertible transform

       TODO:
       multi channel input (eg. from front end)
       is_complex=False
       downsample=True
    """

    def __init__(self, image_size, n_ori=4, n_scale='auto'):
        super().__init__()

        self.n_scale = n_scale
        self.n_ori = n_ori

        self.pyr = Steerable_Pyramid_Freq(image_size, height=n_scale,
                                          is_complex=True, order=n_ori-1,
                                          downsample=False)

    def analysis(self, x):
        b, c, h, w = x.shape
        assert c == 1
        # TODO handle multichannel

        y = self.pyr(x)
        self.low_pass = y[:, 0:1]
        self.high_pass = y[:, -1:]

        real = y[:, 1:-1:2]
        imag = y[:, 2:-1:2]
        amplitude, phase = rectangular_to_polar(real, imag)
        z = torch.cat((amplitude, phase), dim=1)
        return z

    def synthesis(self, z):
        b, c, h, w = z.shape
        amplitude, phase = (z[:, :c//2], z[:, c//2:])
        real, imag = polar_to_rectangular(amplitude, phase)
        y = rearrange(torch.stack((real, imag), dim=1),
                      'b r i h w -> b (i r) h w')
        y = torch.cat((self.low_pass, y, self.high_pass), dim=1)
        self.pyr.pyr_coeffs = self.pyr.convert_tensor_to_pyr(y)
        x = self.pyr.recon_pyr()
        return x

    def forward(self, x):
        return self.analysis(x)
