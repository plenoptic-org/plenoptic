#!/usr/bin/env python3
import os.path as op
import imageio
import torch
import plenoptic as po
import matplotlib.pyplot as plt
import pytest
import pyrtools as pt
import numpy as np
import itertools
from plenoptic.tools.data import to_numpy, torch_complex_to_numpy
from conftest import DEVICE, DATA_DIR, DTYPE


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
        coeff_torch_k  = coeff_torch[k]
        if coeff_torch_k.shape[-1] == 2:
            coeff_torch_k = torch_complex_to_numpy(coeff_torch_k)
        else:
            coeff_torch_k = to_numpy(coeff_torch_k)
        coeff_torch_k = coeff_torch_k.squeeze()
        np.testing.assert_allclose(coeff_torch_k, coeff_np_k, rtol=rtol, atol=atol)

def check_band_energies(coeff_1, coeff_2, rtol=1e-4, atol=1e-4):
    '''
    function that checks if the energy in each band of two pyramids are the same.
    We set an absolute and relative tolerance and the function checks for each band if
    abs(coeff_1-coeff_2) <= atol + rtol*abs(coeff_1)
    Args:
    coeff_1: first dictionary of torch tensors corresponding to each band
    coeff_2: second dictionary of torch tensors corresponding to each band
    '''

    for i in range(len(coeff_1.items())):
        k1 = list(coeff_1.keys())[i]
        k2 = list(coeff_2.keys())[i]
        band_1 = coeff_1[k1]
        band_2 = coeff_2[k2]
        if band_1.shape[-1] == 2:
            band_1 = torch_complex_to_numpy(band_1)
            band_2 = torch_complex_to_numpy(band_2)
        else:
            band_1 = to_numpy(band_1)
            band_2 = to_numpy(band_2)
        band_1 = band_1.squeeze()
        band_2 = band_2.squeeze()

        np.testing.assert_allclose(np.sum(np.abs(band_1)**2),np.sum(np.abs(band_2)**2), rtol=rtol, atol=atol)

def check_parseval(im ,coeff, rtol=1e-4, atol=0):
    '''
    function that checks if the pyramid is parseval, i.e. energy of coeffs is
    the same as the energy in the original image.
    Args:
    input image: image stimulus as torch.Tensor
    coeff: dictionary of torch tensors corresponding to each band
    '''
    total_band_energy = 0
    im_energy = np.sum(to_numpy(im)**2)
    for k,v in coeff.items():
        band = coeff[k]
        if band.shape[-1] == 2:
            band = torch_complex_to_numpy(band)
        else:
            band = to_numpy(band)
        band = band.squeeze()

        total_band_energy += np.sum(np.abs(band)**2)

    np.testing.assert_allclose(total_band_energy, im_energy, rtol=rtol, atol=atol)


class TestSteerablePyramid(object):

    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [1, 2, 3])
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

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("height", ['auto', 1, 2, 3])
    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("downsample", [False, True])
    @pytest.mark.parametrize('is_complex', [True, False])
    @pytest.mark.parametrize("im_shape", [None, (224,224),(256, 128), (128, 256)])
    def test_tight_frame(self, im, height, order, is_complex, downsample, im_shape):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]

        im = im / 255
        im = torch.tensor(im, dtype=DTYPE).unsqueeze(0).unsqueeze(0)

        pyr = po.simul.Steerable_Pyramid_Freq(im.shape[-2:], height, order, is_complex=is_complex, downsample=downsample, tight_frame = True)
        pyr.forward(im)
        check_parseval(im, pyr.pyr_coeffs)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("height", [3,4,5])
    @pytest.mark.parametrize("order", [1,2,3])
    @pytest.mark.parametrize("is_complex", [False, True])
    @pytest.mark.parametrize("im_shape", [None, (224,224),(256, 128), (128, 256)])
    def test_not_downsample(self, im, height, order, is_complex, im_shape):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]

        im = im / 255
        sp_downsample = po.simul.Steerable_Pyramid_Freq(image_shape = im.shape, height = height, order = order,
                                                        is_complex = is_complex, downsample = False, tight_frame=True)
        sp_notdownsample = po.simul.Steerable_Pyramid_Freq(image_shape = im.shape, height = height, order = order,
                                                            is_complex = is_complex, downsample = True, tight_frame=True)
        sp_downsample.to(DEVICE)
        sp_notdownsample.to(DEVICE)

        im_t = torch.tensor(im, dtype=DTYPE).unsqueeze(0).unsqueeze(0).to(DEVICE)
        sp_downsample.forward(im_t)
        sp_notdownsample.forward(im_t)

        check_band_energies(sp_notdownsample.pyr_coeffs, sp_downsample.pyr_coeffs)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("height", [3,4,5])
    @pytest.mark.parametrize("order", [1,2,3])
    @pytest.mark.parametrize("is_complex", [False, True])
    @pytest.mark.parametrize("im_shape", [None, (224,224),(256, 128), (128, 256)])
    @pytest.mark.parametrize("scales", [[0], [1], [0, 1, 2], [2], [], ['residual_highpass', 'residual_lowpass'],
                                        ['residual_highpass', 0, 1, 'residual_lowpass']])
    def test_pyr_to_tensor(self, im, height, order, is_complex, im_shape, scales, rtol=1e-12, atol=1e-12):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]

        im = im / 255
        sp_notdownsample = po.simul.Steerable_Pyramid_Freq(image_shape = im.shape, height = height, order = order,
                                                                is_complex = is_complex, downsample = False)
        sp_notdownsample.to(DEVICE)
        im_t = torch.tensor(im, dtype=DTYPE).unsqueeze(0).unsqueeze(0).to(DEVICE)

        pyr_tensor = sp_notdownsample.forward(im_t, scales = scales)
        pyr_coeff_dict = sp_notdownsample.convert_tensor_to_pyr(pyr_tensor)
        for i in range(len(pyr_coeff_dict.keys())):
            k1 = list(pyr_coeff_dict.keys())[i]
            k2 = list(sp_notdownsample.pyr_coeffs.keys())[i]
            np.testing.assert_allclose(to_numpy(pyr_coeff_dict[k1]), to_numpy(sp_notdownsample.pyr_coeffs[k2]), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("height", [3,4,5])
    @pytest.mark.parametrize("order", [1,2,3])
    @pytest.mark.parametrize("is_complex", [False, True])
    def test_torch_vs_numpy_pyr(self, height, order, is_complex):
        x = plt.imread(op.join(DATA_DIR, 'curie.pgm'))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype=DTYPE).unsqueeze(0).unsqueeze(0).to(DEVICE)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex, tight_frame=False,downsample=True)
        torch_sp.to(DEVICE)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)

        #Check with non-square image
        x = pt.synthetic_images.ramp((256,128))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype=DTYPE).unsqueeze(0).unsqueeze(0).to(DEVICE)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex, tight_frame = False, downsample = True)
        torch_sp.to(DEVICE)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)

        #check non-powers-of-2 images
        x = pt.synthetic_images.ramp((200,200))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype=DTYPE).unsqueeze(0).unsqueeze(0).to(DEVICE)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex, tight_frame = False, downsample = True)
        torch_sp.to(DEVICE)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)


    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("tight_frame", [True, False])
    @pytest.mark.parametrize("downsample", [False, True])
    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("im_shape", [None, (224,224),(256, 128), (128, 256)])
    def test_complete_recon(self, im, is_complex, tight_frame, downsample, height, order, im_shape):
        print(im,is_complex, tight_frame, downsample, height, order, im_shape)

        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im = torch.tensor(im, dtype=DTYPE).unsqueeze(0).unsqueeze(0)
        pyr = po.simul.Steerable_Pyramid_Freq(im.shape[-2:], height, order, is_complex=is_complex, downsample=downsample, tight_frame = tight_frame)
        pyr.forward(im)
        recon = to_numpy(pyr.recon_pyr())
        np.testing.assert_allclose(recon, im.data.cpu().numpy(), rtol=1e-4, atol=1e-4)


    @pytest.mark.parametrize("im", ['einstein','curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("tight_frame", [True, False])
    @pytest.mark.parametrize("downsample", [False, True])
    @pytest.mark.parametrize("height", ['auto'])
    @pytest.mark.parametrize("order", [3])
    @pytest.mark.parametrize("im_shape", [None, (224,224), (256, 128), (128,256)])
    def test_partial_recon(self, im, is_complex, tight_frame, downsample, height, order, im_shape):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im_tensor = torch.tensor(im, dtype=DTYPE).unsqueeze(0).unsqueeze(0)
        po_pyr = po.simul.Steerable_Pyramid_Freq(im.shape, height, order, is_complex=is_complex, downsample=downsample, tight_frame=tight_frame)
        po_pyr.forward(im_tensor)
        pt_pyr = pt.pyramids.SteerablePyramidFreq(im, height, order, is_complex=is_complex)

        recon_levels = [[0], [1,3], [1,3,4]]
        #for i in range(po_pyr.num_scales):
        #    recon_levels.extend(list(itertools.combinations(range(po_pyr.num_scales), i)))
        recon_bands = [[1],[1,3]]
        #for i in range(po_pyr.num_orientations):
        #    recon_bands.extend(list(itertools.combinations(range(po_pyr.num_orientations), i)))
        for levels, bands in itertools.product(['all'] + recon_levels, ['all'] + recon_bands):
            po_recon = po.to_numpy(po_pyr.recon_pyr(levels, bands).squeeze())
            pt_recon = pt_pyr.recon_pyr(levels, bands)
            np.testing.assert_allclose(po_recon, pt_recon,rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("height", ['auto', 1, 3, 4])
    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("im_shape", [None, (224,224),  (256, 128), (128, 256)])
    def test_recon_match_pyrtools(self, im, is_complex, height, order, im_shape, rtol=1e-6, atol=1e-6):
        # this should fail if and only if test_complete_recon does, but
        # may as well include it just in case
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im_tensor = torch.tensor(im, dtype=DTYPE).unsqueeze(0).unsqueeze(0)
        po_pyr = po.simul.Steerable_Pyramid_Freq(im.shape, height, order, is_complex=is_complex, tight_frame=False)
        po_pyr.forward(im_tensor)
        pt_pyr = pt.pyramids.SteerablePyramidFreq(im, height, order, is_complex=is_complex)
        po_recon = po.to_numpy(po_pyr.recon_pyr().squeeze())
        pt_recon = pt_pyr.recon_pyr()
        np.testing.assert_allclose(po_recon, pt_recon, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("downsample", [True, False])
    @pytest.mark.parametrize("scales", [[0], [5], [0, 1, 2], [0, 3, 5],
                                        ['residual_highpass', 'residual_lowpass'],
                                        ['residual_highpass', 0, 1, 'residual_lowpass']])
    def test_scales_arg(self, is_complex, downsample, scales):
        img = imageio.imread(op.join(DATA_DIR, 'einstein.pgm'))
        img = torch.tensor(img / 255, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pyr = po.simul.Steerable_Pyramid_Freq(img.shape[-2:], is_complex=is_complex, downsample=downsample)
        pyr.forward(img)
        pyr_coeffs = pyr.pyr_coeffs.copy()
        pyr.forward(img, scales)
        reduced_pyr_coeffs = pyr.pyr_coeffs.copy()
        for k, v in reduced_pyr_coeffs.items():
            if (v != pyr_coeffs[k]).any():
                raise Exception("Reduced pyr_coeffs should be same as original, but at least key "
                                f"{k} is not")

        # recon_pyr should always fail
        with pytest.raises(Exception):
            pyr.recon_pyr()
        with pytest.raises(Exception):
            pyr.recon_pyr(scales)
