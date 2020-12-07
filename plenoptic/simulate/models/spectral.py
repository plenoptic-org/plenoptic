import torch
import torch.nn as nn
from ..canonical_computations.steerable_pyramid_freq \
    import Steerable_Pyramid_Freq
from ...tools.signal import rectangular_to_polar
from ...tools.conv import blur_downsample


class Spectral(nn.Module):
    """This model computes the local energy in each pyramid band,
    that is to say the local spectral amplitude of the signal.
    """

    def __init__(self, image_size, n_ori=6, n_scale=4):
        super().__init__()

        self.complex_pyr = Steerable_Pyramid_Freq(image_size,
                                                  height=n_scale,
                                                  is_complex=True,
                                                  order=n_ori-1,
                                                  downsample=False)

    def forward(self, x, vectorize=False):
        assert x.ndim == 4

        y = self.complex_pyr(x)
        energy, phase = rectangular_to_polar(y[:, 1:-1:2], y[:, 2:-1:2])

        # local spatial averaging
        stats = torch.cat([blur_downsample(torch.sqrt(y[:, 0:1]**2)),
                           blur_downsample(energy),
                           blur_downsample(torch.sqrt(y[:, -1:]**2))],
                          dim=1)
          
        if vectorize:
            stats  = stats.view(x.shape[0], -1)

        # global spatial averaging
        # stats = torch.cat([torch.sqrt(y[:, 0:1]**2).mean(dim=(2, 3)),
        #                    energy.mean(dim=(2, 3)),
        #                    torch.sqrt(y[:, -1:]**2).mean(dim=(2, 3))],
        #                   dim=1).view(x.shape[0], -1)

        return stats
