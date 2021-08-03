import torch.nn as nn
from plenoptic.simulate.canonical_computations.non_linearities import (
    local_gain_control, local_gain_control_dict, local_gain_release,
    local_gain_release_dict, polar_to_rectangular_dict,
    rectangular_to_polar_dict)
from plenoptic.simulate.canonical_computations.steerable_pyramid_freq import \
    Steerable_Pyramid_Freq
from plenoptic.tools.signal import polar_to_rectangular, rectangular_to_polar


class FactorizedPyramid(nn.Module):
    """
    An non-linear transform which factorizes signal and is exactely invertible.

    Loosely partitions things and stuff.

    Analogous to Fourier amplitude and phase for a localized multiscale
    and oriented transform.

    Notes
    -----
    residuals are stored in amplitude

    by default the not downsampled version also returns a tensor,
    which allows easy further processing
        eg. recursive Factorized Pyr
        (analogous to the scattering transform)

    TODO
    ----
    flesh out the relationship btw real and complex cases

    handle multi channel input
        eg. from front end, or from recursive calls
        hack: fold channels into batch dim and then back out

    cross channel processing - thats next level
    """

    def __init__(self, image_size, n_ori=4, n_scale='auto',
                 downsample_dict=True, is_complex=True):
        super().__init__()

        self.downsample_dict = downsample_dict
        self.is_complex = is_complex

        self.pyr = Steerable_Pyramid_Freq(image_size,
                                          order=n_ori-1,
                                          height=n_scale,
                                          is_complex=is_complex,
                                          downsample=downsample_dict)
        self.pyr_info = None
        self.n_ori = self.pyr.num_orientations
        self.n_scale = self.pyr.num_scales

        if downsample_dict:
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
        y = self.pyr.forward(x)
        if not self.downsample_dict:
            y, self.pyr_info = self.pyr.convert_pyr_to_tensor(y)
        energy, state = self.decomposition(y)
        return energy, state

    def synthesis(self, energy, state):
        y = self.recomposition(energy, state)
        if not self.downsample_dict:
            assert self.pyr_info is not None
            y = self.pyr.convert_tensor_to_pyr(y, *self.pyr_info)
        x = self.pyr.recon_pyr(y)
        return x

    def forward(self, x):
        return self.analysis(x)
