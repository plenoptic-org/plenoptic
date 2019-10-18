import numpy as np
import torch
from .data import torch_complex_to_numpy, to_numpy

def convert_pyrshow(pyr_coeffs, image_index, channel):
    pyr_coeffvis = pyr_coeffs.copy()
    for k in pyr_coeffvis.keys():
        im = pyr_coeffvis[k][image_index,channel, ...]
        #imag and real component exist
        if im.shape[-1] == 2:
            pyr_coeffvis[k] = torch_complex_to_numpy(im)
        else:
            pyr_coeffvis[k] = to_numpy(im)

    return pyr_coeffvis
