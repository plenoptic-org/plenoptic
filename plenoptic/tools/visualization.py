import numpy as np
import torch

def convert_pyrshow(pyr_coeffs, image_index):
    for k in pyr_coeffs.keys():
        im = pyr_coeffs[k][image_index,:,:,:]
        pyr_coeffs[k] = im.reshape([im.shape[1], im.shape[2]])

    return pyr_coeffs
