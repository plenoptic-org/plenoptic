#!/usr/bin/env python3
import os.path as op
import torch
import plenoptic as po
import matplotlib.pyplot as plt
import pytest
import pyrtools as pt
import numpy as np
import itertools
from plenoptic.tools.data import to_numpy, torch_complex_to_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')
print("On device %s" % device)


def check_pyr_coeffs(coeff_np, coeff_torch, rtol=1e-3, atol=1e-3):
    '''
    function that checks if two sets of pyramid coefficients (one numpy  and one torch) are the same
    We set an absolute and relative tolerance and the following function checks if
    abs(coeff1-coeff2) <= atol + rtol*abs(coeff1)
    Inputs:
    coeff1: numpy pyramid coefficients
    coeff2: torch pyramid coefficients
    Both coeffs must obviously have the same number of scales, orientations etc.
    '''

    for k in coeff_np.keys():
        coeff_np_k = coeff_np[k]
        coeff_torch_k  = coeff_torch[k].squeeze()
        if coeff_torch_k.shape[-1] == 2:
            coeff_torch_k = torch_complex_to_numpy(coeff_torch_k)
        else:
            coeff_torch_k = to_numpy(coeff_torch_k)

        np.testing.assert_allclose(coeff_np_k, coeff_torch_k, rtol=rtol, atol=atol)


class TestSteerablePyramid(object):

    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize('is_complex', [True, False])
    @pytest.mark.parametrize("im_shape", [None, (255, 255), (256, 128), (128, 256), (255, 256),
                                          (256, 255)])
    def test_pyramid(self, height, order, is_complex, im_shape):
        x = po.make_basic_stimuli()
        if im_shape is not None:
            x = x[..., :im_shape[0], :im_shape[1]]
        spc = po.simul.Steerable_Pyramid_Freq(x.shape[-2:], height=height, order=order,
                                              is_complex=is_complex)
        spc(x)

    @pytest.mark.parametrize("height", [3,4,5])
    @pytest.mark.parametrize("order", [1,2,3])
    @pytest.mark.parametrize("is_complex", [False, True])
    def test_torch_vs_numpy_pyr(self, height, order, is_complex):
        x = plt.imread(op.join(DATA_DIR, 'curie.pgm'))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex)
        torch_sp.to(device)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)

        #Check with non-square image
        x = pt.synthetic_images.ramp((256,128))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex)
        torch_sp.to(device)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)

        #check non-powers-of-2 images
        x = pt.synthetic_images.ramp((200,200))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex)
        torch_sp.to(device)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize("im_shape", [None, (255, 255), (256, 128), (128, 256), (255, 256),
                                          (256, 255)])
    def test_complete_recon(self, im, is_complex, height, order, im_shape):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im = torch.tensor(im, dtype=dtype).unsqueeze(0).unsqueeze(0)
        pyr = po.simul.Steerable_Pyramid_Freq(im.shape[-2:], height, order, is_complex=is_complex)
        pyr(im)
        recon = pyr.recon_pyr()
        torch.allclose(recon, im)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize("im_shape", [None, (255, 255), (256, 128), (255, 256)])
    def test_partial_recon(self, im, is_complex, height, order, im_shape):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im_tensor = torch.tensor(im, dtype=dtype).unsqueeze(0).unsqueeze(0)
        po_pyr = po.simul.Steerable_Pyramid_Freq(im.shape, height, order, is_complex=is_complex)
        po_pyr(im_tensor)
        pt_pyr = pt.pyramids.SteerablePyramidFreq(im, height, order, is_complex=is_complex)
        # this is almost certainly over-kill: we're checking every
        # possible combination of reconstructing bands and levels
        recon_levels = []
        for i in range(po_pyr.num_scales):
            recon_levels.extend(list(itertools.combinations(range(po_pyr.num_scales), i)))
        recon_bands = []
        for i in range(po_pyr.num_orientations):
            recon_bands.extend(list(itertools.combinations(range(po_pyr.num_orientations), i)))
        for levels, bands in itertools.product(['all'] + recon_levels, ['all'] + recon_bands):
            po_recon = po.to_numpy(po_pyr.recon_pyr(levels, bands))
            pt_recon = pt_pyr.recon_pyr(levels, bands)
            np.allclose(po_recon, pt_recon)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize("im_shape", [None, (255, 255), (256, 128), (128, 256), (255, 256),
                                          (256, 255)])
    def test_recon_match_pyrtools(self, im, is_complex, height, order, im_shape):
        # this should fail if and only if test_complete_recon does, but
        # may as well include it just in case
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im_tensor = torch.tensor(im, dtype=dtype).unsqueeze(0).unsqueeze(0)
        po_pyr = po.simul.Steerable_Pyramid_Freq(im.shape, height, order, is_complex=is_complex)
        po_pyr(im_tensor)
        pt_pyr = pt.pyramids.SteerablePyramidFreq(im, height, order, is_complex=is_complex)
        po_recon = po.to_numpy(po_pyr.recon_pyr())
        pt_recon = pt_pyr.recon_pyr()
        np.allclose(po_recon, pt_recon)
