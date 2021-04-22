#!/usr/bin/env python3
import os.path as op
# from _pytest.fixtures import fixture

import plenoptic as po
import pytest
import torch
from torchvision.transforms.functional import center_crop

from conftest import DATA_DIR, DEVICE, DTYPE

imageA = po.load_images(DATA_DIR + '/256/reptil_skin.pgm')
imgA = center_crop(imageA, [64]).to(DEVICE).to(DEVICE)


class TestSequences(object):

    def test_deviation_from_line_and_brownian_bridge(self):
        """this probabilistic test passes with high probability
        in high dimensions, but for reproducibility we
        set the seed manually."""
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
        # .savefig(f"./logs/geodesicloss-n_steps{n_steps}-init{init}-nu{nu}-model{model.__class__.__name__}.pdf")
        moog.plot_deviation_from_line(video=sequence)
        # .savefig(f"./logs/geodesicdevfromline-n_steps{n_steps}-init{init}-nu{nu}-model{model.__class__.__name__}.pdf")
