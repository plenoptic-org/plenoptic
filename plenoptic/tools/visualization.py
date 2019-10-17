import numpy as np
import torch
from .data import to_numpy

def convert_pyrshow(pyr_coeffs, image_index, channel):
    pyr_coeffvis = pyr_coeffs.copy()
    for k in pyr_coeffvis.keys():
        im = to_numpy(pyr_coeffvis[k][image_index,channel, ...])
        #imag and real component exist
        if im.shape[-1] == 2:
            pyr_coeffvis[k] = im[..., 0] + 1j * im[..., 1]
        else:
            pyr_coeffvis[k] = im

    return pyr_coeffvis
