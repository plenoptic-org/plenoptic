import torch
import torch.nn as nn
from ..canonical_computations import Steerable_Pyramid_Freq
from ..canonical_computations.non_linearities import rectangular_to_polar_dict, local_gain_control


class Steerable_GainControl(nn.Module):
    """steerable pyramid followed by local gain control"""
    def __init__(self, shape, is_complex=False):
        super().__init__()

        if is_complex:
            self.SPF = Steerable_Pyramid_Freq(shape, order=2, height=2, is_complex=True)
            self.nl = rectangular_to_polar_dict
        else:
            self.SPF = Steerable_Pyramid_Freq(shape, order=2, height=2, is_complex=False)
            self.nl = local_gain_control

    def forward(self, x):

        pyr = self.SPF(x)
        _, state = self.nl(pyr)

        return state
