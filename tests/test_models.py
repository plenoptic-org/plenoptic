#!/usr/bin/env python3
import plenoptic as po


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
