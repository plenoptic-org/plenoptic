#!/usr/bin/env python3
import plenoptic as po
import pytest
import matplotlib.pyplot as plt
import torch
from conftest import DEVICE


@pytest.fixture()
def image_input():
    return torch.rand(1, 1, 100, 100)


class TestFrontEnd:

    all_models = [
        "frontend.CenterSurround",
        "frontend.Gaussian",
        "frontend.LN",
        "frontend.LG",
        "frontend.LGG",
        "frontend.OnOff",
    ]

    @pytest.mark.parametrize("model", all_models[:-1], indirect=True)
    def test_output_shape(self, model):
        model = model.to(DEVICE)
        img = torch.ones(1, 1, 100, 100).to(DEVICE)
        assert model(img).shape == img.shape

    @pytest.mark.parametrize("model", all_models, indirect=True)
    def test_gradient_flow(self, model):
        model = model.to(DEVICE)
        img = torch.ones(1, 1, 100, 100).to(DEVICE)
        y = model(img)
        y.sum().backward()

    def test_onoff(self):
        mdl = po.simul.OnOff(7, pretrained=False).to(DEVICE)

    def test_pretrained_onoff(self):
        mdl = po.simul.OnOff(7, pretrained=True).to(DEVICE)

    def test_frontend_display_filters(self):
        mdl = po.simul.OnOff((31, 31), pretrained=True)
        fig = mdl.display_filters()
        plt.close(fig)


class TestLinear(object):

    def test_linear(self, basic_stim):
        model = po.simul.Linear().to(DEVICE)
        assert model(basic_stim).requires_grad

    def test_linear_metamer(self, einstein_img):
        model = po.simul.Linear().to(DEVICE)
        M = po.synth.Metamer(einstein_img, model)
        M.synthesize(max_iter=3, learning_rate=1, seed=1)


class TestLinearNonlinear(object):

    def test_linear_nonlinear(self, basic_stim):

        model = po.simul.LinearNonlinear().to(DEVICE)
        assert model(basic_stim).requires_grad

    def test_linear_nonlinear_metamer(self, einstein_img):
        model = po.simul.LinearNonlinear().to(DEVICE)
        M = po.synth.Metamer(einstein_img, model)
        M.synthesize(max_iter=3, learning_rate=1, seed=0)


class TestLaplacianPyramid(object):

    def test_grad(self, basic_stim):
        L = po.simul.Laplacian_Pyramid().to(DEVICE)
        y = L.analysis(basic_stim)
        assert y[0].requires_grad
