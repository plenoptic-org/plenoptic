#!/usr/bin/env python3
import plenoptic as po


class TestLinear(object):

    def test_linear(self, basic_stim):
        model = po.simul.Linear()
        assert model(basic_stim).requires_grad

    def test_linear_metamer(self, einstein_img):
        model = po.simul.Linear()
        M = po.synth.Metamer(einstein_img, model)
        M.synthesize(max_iter=3, learning_rate=1, seed=1)


class TestLinearNonlinear(object):

    def test_linear_nonlinear(self, basic_stim):
        model = po.simul.Linear_Nonlinear()
        assert model(basic_stim).requires_grad

    def test_linear_nonlinear_metamer(self, einstein_img):
        model = po.simul.Linear_Nonlinear()
        M = po.synth.Metamer(einstein_img, model)
        M.synthesize(max_iter=3, learning_rate=1, seed=0)


class TestLaplacianPyramid(object):

    def test_grad(self, basic_stim):
        L = po.simul.Laplacian_Pyramid()
        y = L.analysis(basic_stim)
        assert y[0].requires_grad
