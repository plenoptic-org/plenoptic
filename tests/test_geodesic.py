#!/usr/bin/env python3
import os.path as op
import torch
import plenoptic as po
import matplotlib.pyplot as plt
import pytest
from test_plenoptic import DEVICE, DATA_DIR, DTYPE


class TestGeodesic(object):

    def test_geodesic_spectral(self):
        imageA = plt.imread(op.join(DATA_DIR, 'reptil_skin.pgm')) / 255.
        imageB = plt.imread(op.join(DATA_DIR, 'metal.pgm')) / 255.
        c = 64 + 32
        imageA = imageA[c:-c, c:-c]
        imageB = imageB[c:-c, c:-c]
        imgA = torch.tensor(imageA, dtype=DTYPE, device=DEVICE
                            ).unsqueeze(0).unsqueeze(0)
        imgB = torch.tensor(imageB, dtype=DTYPE, device=DEVICE
                            ).unsqueeze(0).unsqueeze(0)

        model = po.simul.Spectral(imgA.shape[-2:])
        n_steps = 11
        moog = po.synth.Geodesic(imgA, imgB, model, n_steps, init='straight',
                                 lmbda=.1)
        moog.synthesize(5, learning_rate=0.005)

    def test_geodesic_polarpyr(self):
        image_size = 64
        einstein = po.make_basic_stimuli(image_size, requires_grad=False)[-1]
        vid = po.translation_sequence(einstein)
        from torchvision.transforms.functional import center_crop
        vid = center_crop(vid, image_size // 2)
        vid = po.rescale(vid, 0, 1)

        imgA = vid[0:1]
        imgB = vid[-1:]

        model = po.simul.Polar_Pyramid(imgB.shape[-2:])
        n_steps = len(vid)
        moog = po.synth.Geodesic(imgA, imgB, model, n_steps,
                                 init='straight', lmbda=.1)
        moog.synthesize(5)

    @pytest.mark.parametrize('n_shifts', [3, 7, 9])
    def test_geodesic_texture(self, n_shifts):
        imageA = plt.imread(op.join(DATA_DIR, 'reptil_skin.pgm')) / 255.
        imageB = plt.imread(op.join(DATA_DIR, 'metal.pgm')) / 255.
        c = 64 + 32
        imageA = imageA[c:-c, c:-c]
        imageB = imageB[c:-c, c:-c]
        imgA = torch.tensor(imageA, dtype=DTYPE, device=DEVICE
                            ).unsqueeze(0).unsqueeze(0)
        imgB = torch.tensor(imageB, dtype=DTYPE, device=DEVICE
                            ).unsqueeze(0).unsqueeze(0)

        model = po.simul.Texture_Statistics(imgA.shape[-2:],
                                            n_ori=4, n_scale='auto',
                                            n_shifts=n_shifts)
        n_steps = 11
        moog = po.synth.Geodesic(imgA, imgB, model, n_steps, init='straight',
                                 lmbda=.1)
        moog.synthesize(5, learning_rate=0.005)
