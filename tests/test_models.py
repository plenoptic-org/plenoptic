#!/usr/bin/env python3
import os.path as op
import torch
import plenoptic as po
import matplotlib.pyplot as plt
import pytest
from test_plenoptic import DEVICE, DATA_DIR, DTYPE


class TestLinear(object):

    def test_linear(self):
        model = po.simul.Linear().to(DEVICE)
        x = po.make_basic_stimuli().to(DEVICE)
        assert model(x).requires_grad

    def test_linear_metamer(self):
        model = po.simul.Linear().to(DEVICE)
        image = plt.imread(op.join(DATA_DIR, 'nuts.pgm')).astype(float) / 255.
        im0 = torch.tensor(image, requires_grad=True, dtype=DTYPE, device=DEVICE).squeeze().unsqueeze(0).unsqueeze(0)
        M = po.synth.Metamer(im0, model)
        synthesized_signal, synthesized_representation = M.synthesize(max_iter=3, learning_rate=1, seed=1)


class TestLinearNonlinear(object):

    def test_linear_nonlinear(self):
        model = po.simul.Linear_Nonlinear().to(DEVICE)
        x = po.make_basic_stimuli().to(DEVICE)
        assert model(x).requires_grad

    def test_linear_nonlinear_metamer(self):
        model = po.simul.Linear_Nonlinear().to(DEVICE)
        image = plt.imread(op.join(DATA_DIR, 'metal.pgm')).astype(float) / 255.
        im0 = torch.tensor(image, requires_grad=True, dtype=torch.float32,
                           device=DEVICE).squeeze().unsqueeze(0).unsqueeze(0)
        M = po.synth.Metamer(im0, model)
        synthesized_signal, synthesized_representation = M.synthesize(max_iter=3, learning_rate=1, seed=0)


# class TestConv(object):
# TODO expand, arbitrary shapes, dim


class TestLaplacianPyramid(object):

    def test_grad(self):
        L = po.simul.Laplacian_Pyramid().to(DEVICE)
        y = L.analysis(po.make_basic_stimuli().to(DEVICE))
        assert y[0].requires_grad
