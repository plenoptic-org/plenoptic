#!/usr/bin/env python3
import plenoptic as po


class TestLinear(object):

    def test_linear(self):
        model = po.simul.Linear()
        x = po.make_basic_stimuli()
        assert model(x).requires_grad

    def test_linear_metamer(self, einstein_img):
        model = po.simul.Linear()
        M = po.synth.Metamer(einstein_img, model)
        M.synthesize(max_iter=3, learning_rate=1, seed=1)


class TestLinearNonlinear(object):

    def test_linear_nonlinear(self):
        model = po.simul.Linear_Nonlinear()
        x = po.make_basic_stimuli()
        assert model(x).requires_grad

    def test_linear_nonlinear_metamer(self, einstein_img):
        model = po.simul.Linear_Nonlinear()
        M = po.synth.Metamer(einstein_img, model)
        M.synthesize(max_iter=3, learning_rate=1, seed=0)


class TestLaplacianPyramid(object):

    def test_grad(self):
        L = po.simul.Laplacian_Pyramid()
        y = L.analysis(po.make_basic_stimuli())
        assert y[0].requires_grad
