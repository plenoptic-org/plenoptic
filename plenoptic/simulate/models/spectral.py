import torch
import torch.nn as nn
from ..canonical_computations.steerable_pyramid_freq \
    import Steerable_Pyramid_Freq
from ...tools.signal import rectangular_to_polar
from ...tools.conv import blur_downsample


class Spectral(nn.Module):
    """Compute the local spectral amplitude of the input.

    This is done by representing the input in a pyramid,
    and computing the local energy in each band.

    TODO
    ----
    - add a parameter to control local / global pooling
    - extra statistic (mean) on highpass / lowpass
    """

    def __init__(self, image_size, n_ori=6, n_scale=4, is_complex=True):
        super().__init__()

        self.pyr = Steerable_Pyramid_Freq(image_size,
                                          height=n_scale,
                                          is_complex=is_complex,
                                          order=n_ori-1,
                                          downsample=False)

    def forward(self, x, vectorize=False):
        assert x.ndim == 4

        y = self.pyr(x)
        energy, _ = rectangular_to_polar(y[:, 1:-1:2], y[:, 2:-1:2])


        y = torch.cat([blur_downsample(torch.sqrt(y[:, 0:1]**2)),
                       blur_downsample(energy),
                       blur_downsample(torch.sqrt(y[:, -1:]**2))],
                      dim=1)

        # TODO ready to go complex when pyr is
        # if torch.is_complex(y):
        #     y = torch.abs(y)
        # else:
        #     y = torch.sqrt(y**2)
        # local spatial averaging
        # y = blur_downsample(y)

        if vectorize:
            y = y.view(x.shape[0], -1)

        return y
