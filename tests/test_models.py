#!/usr/bin/env python3
import os.path as op
import torch
import plenoptic as po
import matplotlib.pyplot as plt
import pytest
import numpy as np
import pyrtools as pt
from test_plenoptic import DEVICE, DATA_DIR, DTYPE


@pytest.fixture()
def image_input():
    return torch.rand(1, 1, 100, 100)


class TestFrontEnd:

    kernel_size = (7, 7)
    all_models = [
        po.simul.CenterSurround(kernel_size),
        po.simul.Gaussian(kernel_size),
        po.simul.LN(kernel_size),
        po.simul.LG(kernel_size),
        po.simul.LGG(kernel_size),
        po.simul.OnOff(kernel_size)
    ]

    @pytest.mark.parametrize("model", all_models[:-1])  # shape of onoff is different
    def test_output_shape(self, model):
        img = torch.ones(1, 1, 100, 100)
        assert model(img).shape == img.shape

    @pytest.mark.parametrize("model", all_models)
    def test_gradient_flow(self, model):
        img = torch.ones(1, 1, 100, 100)
        y = model(img)
        y.sum().backward()

    def test_onoff(self):
        mdl = po.simul.OnOff(7, pretrained=False)

    def test_pretrained_onoff(self):
        mdl = po.simul.OnOff(7, pretrained=True)

    def test_frontend_display_filters(self):
        mdl = po.simul.OnOff((31, 31), pretrained=True)
        fig = mdl.display_filters()
        plt.close(fig)


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
