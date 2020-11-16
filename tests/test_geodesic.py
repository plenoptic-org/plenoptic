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
                                            n_ori=4, n_scale=4,
                                            n_shifts=n_shifts)
        n_steps = 11
        moog = po.synth.Geodesic(imgA, imgB, model, n_steps, init='straight',
                                 lmbda=.1)
        moog.synthesize(5, learning_rate=0.005)
