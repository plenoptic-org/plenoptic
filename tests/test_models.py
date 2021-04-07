#!/usr/bin/env python3
import plenoptic as po
import pytest
import matplotlib.pyplot as plt
import torch
from torch import Tensor

import plenoptic
from plenoptic.simulate.canonical_computations import gaussian1d, circular_gaussian2d
from conftest import DEVICE


@pytest.fixture()
def image_input():
    return torch.rand(1, 1, 100, 100)


class TestFrontEnd:

    all_models = [
        "frontend.LinearNonlinear",
        "frontend.LuminanceGainControl",
        "frontend.LuminanceContrastGainControl",
        "frontend.OnOff",
    ]

    @pytest.mark.parametrize("model", all_models[:-1], indirect=True)
    def test_output_shape(self, model):
        img = torch.ones(1, 1, 100, 100).to(DEVICE)
        assert model(img).shape == img.shape

    @pytest.mark.parametrize("model", all_models, indirect=True)
    def test_gradient_flow(self, model):
        img = torch.ones(1, 1, 100, 100).to(DEVICE)
        y = model(img)
        assert y.requires_grad

    def test_onoff(self):
        mdl = po.simul.OnOff(7, pretrained=False).to(DEVICE)

    @pytest.mark.parametrize("kernel_size", [7, 31])
    def test_pretrained_onoff(self, kernel_size):
        if kernel_size != 31:
            with pytest.raises(AssertionError):
                mdl = po.simul.OnOff(kernel_size, pretrained=True).to(DEVICE)
        else:
            mdl = po.simul.OnOff(kernel_size, pretrained=True).to(DEVICE)

    @pytest.mark.parametrize("model", all_models, indirect=True)
    def test_frontend_display_filters(self, model):
        fig = model.display_filters()
        plt.close(fig)


class TestNaive(object):

    all_models = [
        "naive.Identity",
        "naive.Linear",
        "naive.Gaussian",
        "naive.CenterSurround",
    ]

    @pytest.mark.parametrize("model", all_models, indirect=True)
    def test_gradient_flow(self, model):
        img = torch.ones(1, 1, 100, 100).to(DEVICE).requires_grad_()
        y = model(img)
        assert y.requires_grad

    def test_linear(self, basic_stim):
        model = plenoptic.simul.Linear().to(DEVICE)
        assert model(basic_stim).requires_grad

    def test_linear_metamer(self, einstein_img):
        model = plenoptic.simul.Linear().to(DEVICE)
        M = po.synth.Metamer(einstein_img, model)
        M.synthesize(max_iter=3, learning_rate=1, seed=1)


class TestLaplacianPyramid(object):

    def test_grad(self, basic_stim):
        L = po.simul.Laplacian_Pyramid().to(DEVICE)
        y = L.analysis(basic_stim)
        assert y[0].requires_grad


class TestFilters:
    @pytest.mark.parametrize("std", [5., torch.tensor(1.), -1., 0.])
    @pytest.mark.parametrize("kernel_size", [(31, 31), (3, 2), (7, 7), 5])
    @pytest.mark.parametrize("out_channels", [1, 3, 10])
    def test_circular_gaussian2d_shape(self, std, kernel_size, out_channels):
        if std <=0.:
            with pytest.raises(AssertionError):
                circular_gaussian2d((7, 7), std)
        else:
            filt = circular_gaussian2d(kernel_size, std, out_channels)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            assert filt.shape == (out_channels, 1, *kernel_size)
            assert filt.sum().isclose(torch.ones(1) * out_channels)

    def test_circular_gaussian2d_wrong_std_length(self):
        std = torch.tensor([1., 2.])
        out_channels = 3
        with pytest.raises(AssertionError):
            circular_gaussian2d((7, 7), std, out_channels)

    @pytest.mark.parametrize("kernel_size", [5, 11, 20])
    @pytest.mark.parametrize("std", [1., 20., 0.])
    def test_gaussian1d(self, kernel_size, std):
        if std <=0:
            with pytest.raises(AssertionError):
                gaussian1d(kernel_size, std)
        else:
            filt = gaussian1d(kernel_size, std)
            assert filt.sum().isclose(torch.ones(1))
            assert filt.shape == torch.Size([kernel_size])
