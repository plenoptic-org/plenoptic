#!/usr/bin/env python3
import os.path as op

import matplotlib.pyplot as plt
import plenoptic as po
import pytest
import torch

from test_plenoptic import DATA_DIR, DEVICE, DTYPE


class TestGeodesic(object):

    def test_brownian_bridge(self):
        s = 64
        n_steps = 100
        max_norm = 4

        x0 = torch.randn(1, 1, s, s)
        x1 = torch.randn(1, 1, s, s)

        N = 200
        bridges = torch.zeros(n_steps+1, N, s, s)
        for n in range(N):
            bridges[:, n] = po.tools.sample_brownian_bridge(x0, x1,
                                                            n_steps,
                                                            max_norm)[:, 0]

        # check pylons
        assert (bridges[0] - x0[0]).pow(2).sum() < 1e-6
        assert (bridges[-1] - x1[0]).pow(2).sum() < 1e-6
        # check max l2 norm from straight line
        # NOTE distance from line does not support batch
        assert (torch.max(po.tools.distance_from_line(bridges[:, 0:1], x0, x1))
                - max_norm).abs() < .1
        # check max std of bridge coordinate
        assert (torch.max((bridges - po.tools.make_straight_line(x0, x1,
                                                                 n_steps)
                           ).std((1)).mean((1, 2)) * s) - max_norm).abs() < .01

    def test_distance_from_line(self):

        s = 64
        n_steps = 100
        max_norm = 3
        y0 = torch.randn(1, 1, s, s)
        y1 = torch.randn(1, 1, s, s)
        y = po.tools.sample_brownian_bridge(y0, y1, n_steps, max_norm)
        dist = po.tools.distance_from_line(y, y0, y1)

        line = (y1 - y0).flatten()
        u = line / torch.norm(line)
        y_ = (y - y0).view(len(y), -1)  # center
        d = torch.norm(y_ - (y_ @ u)[:, None]*u[None, :], dim=1)

        assert (dist - d).pow(2).mean() < 1e-6

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
        moog = po.synth.Geodesic(imgA, imgB, model, n_steps, init='straight')
        moog.synthesize(max_iter=5, learning_rate=0.005, lmbda=.1)

    # def test_geodesic_polarpyr(self):
    #     image_size = 64
    #     einstein = po.make_basic_stimuli(image_size, requires_grad=False)[-1]
    #     vid = po.translation_sequence(einstein)
    #     from torchvision.transforms.functional import center_crop
    #     vid = center_crop(vid, image_size // 2)
    #     vid = po.rescale(vid, 0, 1)

    #     imgA = vid[0:1]
    #     imgB = vid[-1:]

    #     model = po.simul.Polar_Pyramid(imgB.shape[-2:])
    #     n_steps = len(vid)
    #     moog = po.synth.Geodesic(imgA, imgB, model, n_steps,
    #                              init='straight')
    #     moog.synthesize(max_iter=5, lmbda=.1)

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
                                            n_ori=4, n_scale=3,
                                            n_shifts=n_shifts)
        n_steps = 11
        moog = po.synth.Geodesic(imgA, imgB, model, n_steps, init='straight')
        moog.synthesize(max_iter=5, learning_rate=0.005, lmbda=.1)
