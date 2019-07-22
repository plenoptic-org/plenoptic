import torch
import torch.nn as nn
import numpy as np

from ..tools.conv import upsample_blur # blur_downsample

from .frontend import Front_End
from .steerable_pyramid_freq import Steerable_Pyramid_Freq
from .non_linearities import local_gain_control


def steer(pyr_coeffs, residuals=True):
    S = np.max(np.array([k for k in pyr_coeffs.keys() if isinstance(k, tuple)])[:, 0]) + 1

    pyr_coeffs_steered = {}

    for s in range(S):
        pyr_coeffs_steered[s, 0] = pyr_coeffs[s, 0]
        pyr_coeffs_steered[s, 1] = (pyr_coeffs[s, 0] + pyr_coeffs[s, 1]) / np.sqrt(2)
        pyr_coeffs_steered[s, 2] = pyr_coeffs[s, 1]
        pyr_coeffs_steered[s, 3] = (pyr_coeffs[s, 1] - pyr_coeffs[s, 0]) / np.sqrt(2)

    if residuals:
        pyr_coeffs_steered['residual_lowpass'] = pyr_coeffs['residual_lowpass']
        pyr_coeffs_steered['residual_highpass'] = pyr_coeffs['residual_highpass']

    return pyr_coeffs_steered


def standardize(X, epsilon=1e-16):
    return (X - torch.mean(X)) / (torch.std(X) + epsilon)


def global_gain_control(coeffs):
    coeffs_normalized = {}
    for k in coeffs.keys():
        coeffs_normalized[k] = standardize(coeffs[k])

    return coeffs_normalized


def ori_diff(coeffs):
    S = np.max(np.array([k for k in coeffs.keys() if isinstance(k, tuple)])[:, 0]) + 1
    B = np.max(np.array([k for k in coeffs.keys() if isinstance(k, tuple)])[:, 1]) + 1

    oridiff = {}
    for s in range(S):
        for b in range(B):
            oridiff[s, b] = coeffs[s, (b + 1) % 4] - coeffs[s, b]

    return oridiff


def scale_diff(coeff):
    S = np.max(np.array([k for k in coeff.keys() if isinstance(k, tuple)])[:, 0]) + 1
    B = np.max(np.array([k for k in coeff.keys() if isinstance(k, tuple)])[:, 1]) + 1

    scalediff = {}
    j = 0
    for s in range(1, S):
        for b in range(B):
            fine = coeff[s - 1, b]  # standardize(coeff[s-1, b])
            coarse = upsample_blur(coeff[s, b])  # standardize(po.upsample_blur(coeff[s, b]))
            scalediff[j, b] = (fine - coarse) / np.sqrt(2)
        #             scalediff[j+1, b] = (fine + coarse) / np.sqrt(2)
        #         j += 2
        j += 1

    return scalediff


class V2(nn.Module):
    """


    """

    def __init__(self, frontend=False, steer=False):
        super(V2, self).__init__()

        self.frontend = frontend
        if self.frontend:
            self.F = Front_End()

        self.steer = steer
        if self.steer:
            self.L1 = Steerable_Pyramid_Freq([256, 256], order=1, height=5, is_complex=False)
        else:
            self.L1 = Steerable_Pyramid_Freq([256, 256], order=3, height=5, is_complex=False)

        self.L2 = ori_diff
        self.L3 = scale_diff

        self.GG = global_gain_control
        self.LG = local_gain_control

    def forward(self, x):

        if self.frontend:
            x = self.F(x)

        if self.steer:
            activation1 = steer(self.L1(x))
        else:
            activation1 = self.L1(x)

        activation_standard = self.GG(activation1)
        energy1, state1 = self.LG(activation_standard, (2, 2))

        o = self.GG(self.L2(energy1))
        s = self.GG(self.L3(energy1))

        energy2, state2 = self.LG(o, residuals=False)
        energy3, state3 = self.LG(s, residuals=False)

        return [k for k in energy1.values()] + [k for k in energy2.values()] + [k for k in energy3.values()]
#         return [k for k in state1.values()] + [k for k in state2.values()] + [k for k in state3.values()]
#         return [k for k in energy2.values()] + [k for k in energy3.values()]