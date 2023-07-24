import torch.nn as nn
from plenoptic.simulate.canonical_computations.non_linearities import (
    local_gain_control, local_gain_control_dict, local_gain_release,
    local_gain_release_dict, polar_to_rectangular_dict,
    rectangular_to_polar_dict)
from plenoptic.simulate.canonical_computations.steerable_pyramid_freq import SteerablePyramidFreq
from plenoptic.tools.signal import polar_to_rectangular, rectangular_to_polar


class Factorized_Pyramid(nn.Module):
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
                 downsample_dict=True, is_complex=True,
                 tight_frame=True):
        super().__init__()

        self.downsample_dict = downsample_dict
        self.is_complex = is_complex

        pyr = SteerablePyramidFreq(image_size,
                                     order=n_ori-1,
                                     height=n_scale,
                                     is_complex=is_complex,
                                     downsample=downsample_dict,
                                     tight_frame=tight_frame)
        self.n_ori = pyr.num_orientations
        self.n_scale = pyr.num_scales

        if downsample_dict:
            self.pyramid_analysis  = lambda x: pyr.forward(x)
            self.pyramid_synthesis = lambda y: pyr.recon_pyr(y)
            if is_complex:
                self.decomposition = rectangular_to_polar_dict
                self.recomposition = polar_to_rectangular_dict
            else:
                self.decomposition = local_gain_control_dict
                self.recomposition = local_gain_release_dict
        else:
            self.pyramid_analysis  = lambda x: pyr.convert_pyr_to_tensor(
                                                           pyr.forward(x))
            self.pyramid_synthesis = lambda y: pyr.recon_pyr(
                                 pyr.convert_tensor_to_pyr(y))
            if is_complex:
                self.decomposition = rectangular_to_polar
                self.recomposition = polar_to_rectangular
            else:
                self.decomposition = local_gain_control
                self.recomposition = local_gain_release

    def analysis(self, x):
        y = self.pyramid_analysis(x)
        energy, state = self.decomposition(y)
        return energy, state

    def synthesis(self, energy, state):
        y = self.recomposition(energy, state)
        x = self.pyramid_synthesis(y)
        return x

    def forward(self, x):
        return self.analysis(x)
