import torch
import torch.nn as nn
from .frontend import Front_End
from ..canonical_computations import Steerable_Pyramid_Freq
from ..canonical_computations.non_linearities import rectangular_to_polar_dict, local_gain_control


class Steerable_GainControl(nn.Module):
    """steerable pyramid followed by local gain control"""
    def __init__(self, shape, is_complex=False, n_ori=4, n_scale=3, residuals=True, front_end=False):
        super().__init__()

        self.residuals = residuals

        if front_end:
            self.front_end = Front_End()

        if is_complex:
            self.steerable_pyramid = Steerable_Pyramid_Freq(shape, order=n_ori-1, height=n_scale, is_complex=True)
            self.non_linearity = rectangular_to_polar_dict
        else:
            self.steerable_pyramid = Steerable_Pyramid_Freq(shape, order=n_ori-1, height=n_scale, is_complex=False)
            self.non_linearity = local_gain_control

    def forward(self, x, vectorize=True):

        if hasattr(self, 'front_end'):
            x = self.front_end(x)

        pyr = self.steerable_pyramid(x)
        ampl, state = self.non_linearity(pyr, residuals=self.residuals)

        if vectorize:
            return torch.cat([s.view(x.shape[0], -1) for s in ampl.values()], dim=1)
        else:
            return ampl
