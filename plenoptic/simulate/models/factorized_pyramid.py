import torch.nn as nn
from plenoptic.simulate.canonical_computations.non_linearities import (
    local_gain_control, local_gain_control_dict, local_gain_release,
    local_gain_release_dict, polar_to_rectangular_dict,
    rectangular_to_polar_dict)
from plenoptic.simulate.canonical_computations.steerable_pyramid_freq import \
    Steerable_Pyramid_Freq
from plenoptic.tools.signal import polar_to_rectangular, rectangular_to_polar


class Factorized_Pyramid(nn.Module):
    """
    non linear invertible transform

    factorize, parition / expand, multiply

    residuals in amplitude automatically

    handle:
        is_complex=False
        downsample=True
        multi channel input (eg. from front end)

    recursive - generalization of scattering
    """

    def __init__(self, image_size, n_ori=4, n_scale='auto',
                 downsample=True, is_complex=True):
        super().__init__()

        self.n_scale = n_scale
        self.n_ori = n_ori

        self.pyr = Steerable_Pyramid_Freq(image_size, height=n_scale,
                                          is_complex=is_complex, order=n_ori-1,
                                          downsample=downsample)
        if downsample:
            if is_complex:
                self.decomposition = rectangular_to_polar_dict
                self.recomposition = polar_to_rectangular_dict
            else:
                self.decomposition = local_gain_control_dict
                self.recomposition = local_gain_release_dict
        else:
            if is_complex:
                self.decomposition = rectangular_to_polar
                self.recomposition = polar_to_rectangular
            else:
                self.decomposition = local_gain_control
                self.recomposition = local_gain_release

    def analysis(self, x):
        b, c, h, w = x.shape
        assert c == 1

        y = self.pyr(x)
        amplitude, phase = self.decomposition(y)
        # self.low_pass = y["residual_lowpass"]
        # self.high_pass = y["residual_highpass"]

        return amplitude, phase

    def synthesis(self, amplitude, phase):

        y = self.recomposition(amplitude, phase)
        # y["residual_lowpass"] = self.low_pass
        # y["residual_highpass"] = self.high_pass
        x = self.pyr.recon_pyr(y)
        return x

    def forward(self, x):
        return self.analysis(x)
