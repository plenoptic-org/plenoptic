#!/usr/bin/env python3
import os.path as op
import torch
import plenoptic as po
import matplotlib.pyplot as plt
import pytest
from conftest import DATA_DIR, DTYPE


class TestLinear(object):

    def test_linear(self):
        model = po.simul.Linear()
        x = po.make_basic_stimuli()
        assert model(x).requires_grad

    def test_linear_metamer(self):
        model = po.simul.Linear()
        image = plt.imread(op.join(DATA_DIR, 'nuts.pgm')).astype(float) / 255.
        im0 = torch.tensor(image, requires_grad=True, dtype=DTYPE).squeeze().unsqueeze(0).unsqueeze(0)
        M = po.synth.Metamer(im0, model)
        synthesized_signal, synthesized_representation = M.synthesize(max_iter=3, learning_rate=1, seed=1)


class TestLinearNonlinear(object):

    def test_linear_nonlinear(self):
        model = po.simul.Linear_Nonlinear()
        x = po.make_basic_stimuli()
        assert model(x).requires_grad

    def test_linear_nonlinear_metamer(self):
        model = po.simul.Linear_Nonlinear()
        image = plt.imread(op.join(DATA_DIR, 'metal.pgm')).astype(float) / 255.
        im0 = torch.tensor(image,requires_grad=True,dtype = torch.float32).squeeze().unsqueeze(0).unsqueeze(0)
        M = po.synth.Metamer(im0, model)
        synthesized_signal, synthesized_representation = M.synthesize(max_iter=3, learning_rate=1,seed=0)


# class TestConv(object):
# TODO expand, arbitrary shapes, dim


class TestLaplacianPyramid(object):

    def test_grad(self):
        L = po.simul.Laplacian_Pyramid()
        y = L.analysis(po.make_basic_stimuli())
        assert y[0].requires_grad
