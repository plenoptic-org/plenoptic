#!/usr/bin/env python3
import os.path as op
import torch
import plenoptic as po
from plenoptic.tools import to_numpy
import matplotlib.pyplot as plt
import pytest
from test_plenoptic import DEVICE, DATA_DIR, DTYPE


x = po.tools.make_synthetic_stimuli()
image = po.load_images(DATA_DIR + '/256/nuts.pgm')
im0 = torch.tensor(image, requires_grad=True, dtype=DTYPE)

class TestLinear(object):

    def test_linear(self):
        model = po.simul.Linear()
        assert model(x).requires_grad

    def test_linear_metamer(self):
        model = po.simul.Linear()
        M = po.synth.Metamer(im0, model)
        synthesized_signal, synthesized_representation = M.synthesize(max_iter=3,
         learning_rate=1, seed=1)


class TestLinearNonlinear(object):

    def test_linear_nonlinear(self):
        model = po.simul.Linear_Nonlinear()
        assert model(x).requires_grad

    def test_linear_nonlinear_metamer(self):
        model = po.simul.Linear_Nonlinear()
        M = po.synth.Metamer(im0, model)
        m_image, m_representation = M.synthesize(max_iter=3, learning_rate=1,
                                                 seed=0)

# class TestConv(object):
# TODO expand, arbitrary shapes, dim


class TestLaplacianPyramid(object):

    def test_grad(self):
        L = po.simul.Laplacian_Pyramid()
        y = L.analysis(x)
        assert y[0].requires_grad
