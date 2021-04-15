#!/usr/bin/env python3
import os.path as op
from _pytest.fixtures import fixture

import matplotlib.pyplot as plt
import plenoptic as po
import pytest
import torch
from torchvision.transforms.functional import center_crop

from conftest import DATA_DIR, DEVICE, DTYPE

imageA = po.load_images(DATA_DIR + '/256/reptil_skin.pgm')
imgA = center_crop(imageA, 64).to(DEVICE).to(DEVICE)

class TestSequences(object):

    def test_deviation_from_line_and_brownian_bridge(self):
        torch.manual_seed(0)
        t = 2**6
        d = 2**15
        b = po.sample_brownian_bridge(torch.randn(1, d),
                                      torch.randn(1, d),
                                      t, d**.5)
        a, f = po.deviation_from_line(b)
        assert torch.abs(a[t//2] - .5) < 1e-2, f"{a[t//2]}"
        assert torch.abs(f[t//2] - 2**.5/2) < 1e-2, f"{f[t//2]}"


class TestGeodesic(object):

        # @pytest.mark.parametrize("model", [OnOff], fixture=True)
        # model = po.simul.Texture_Statistics(imgA.shape[-2:],
        #                                     n_ori=4, n_scale=3,
        #                                     n_shifts=n_shifts)

    @pytest.mark.parametrize('n_steps', [5, 10])
    @pytest.mark.parametrize("init", ['straight', 'bridge'])
    @pytest.mark.parametrize("nu", [0, .1])
    def test_geodesic_texture(self, n_steps, init, nu):

        model = po.simul.OnOff(kernel_size=(31, 31), pretrained=True)
        sequence = po.translation_sequence(imgA[0], n_steps)
        moog = po.synth.Geodesic(sequence[0:1], sequence[-1:],
                                 model, n_steps, init)
        moog.synthesize(max_iter=5, learning_rate=0.001, nu=nu)
        moog.plot_loss()
        moog.plot_deviation_from_line(video=sequence)

    # def test_geodesic_OnOff(self):
    #     try:
    #         from adabelief_pytorch import AdaBelief
    #         import adabelief_pytorch
    #         print(adabelief_pytorch.__version__)
    #         optimizer = AdaBelief([moog.x], lr=0.001, eps=1e-16, betas=(0.9,0.999),
    #                             weight_decouple=True, rectify=False, print_change_log=False)
    #     except:
    #         optimizer = 'Adam'

    #     moog = po.synth.Geodesic(imgA, imgB, model, n_steps, init='straight')
    #     moog.synthesize(optimizer=optimizer)

    # def test_geodesic_spectral(self):

    #     model = po.simul.Spectral(imgA.shape[-2:])
    #     n_steps = 11
    #     moog = po.synth.Geodesic(imgA, imgB, model, n_steps, init='straight')
    #     moog.synthesize(max_iter=5, learning_rate=0.005, lmbda=.1)

    # def test_geodesic_polarpyr(self):
    #     image_size = 64
    #     einstein = po.tools.make_synthetic_stimuli(image_size,
    #                                            requires_grad=False)[-1]
    #     vid = po.translation_sequence(einstein)
    #     from torchvision.transforms.functional import center_crop
    #     vid = center_crop(vid, image_size // 2)
    #     vid = po.tools.rescale(vid, 0, 1)

    #     imgA = vid[0:1]
    #     imgB = vid[-1:]

    #     model = po.simul.Polar_Pyramid(imgB.shape[-2:])
    #     n_steps = len(vid)
    #     moog = po.synth.Geodesic(imgA, imgB, model, n_steps,
    #                              init='straight')
    #     moog.synthesize(max_iter=5, lmbda=.1)

