import numpy as np
import torch

def convert_pyrshow(pyr_coeffs, image_index):
    pyr_coeffvis = pyr_coeffs.copy()
    for k in pyr_coeffvis.keys():
        im = pyr_coeffvis[k][image_index, ...].detach().numpy()
        #imag and real component exist
        if im.shape[0] == 2:
            pyr_coeffvis[k] = im[0,...] + 1j * im[1,...]
        else:
            pyr_coeffvis[k] = im

    return pyr_coeffvis
