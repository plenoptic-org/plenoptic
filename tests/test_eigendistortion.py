import plenoptic.synthesize.autodiff as autodiff
import pytest
import torch
from torch import nn
from plenoptic.simulate.models.frontend import FrontEnd
from plenoptic.synthesize.eigendistortion import Eigendistortion
from conftest import get_model, DEVICE

# to be used for default model instantiation
SMALL_DIM = 20
LARGE_DIM = 100


class TestEigendistortionSynthesis:

    @pytest.mark.parametrize('model', ['FrontEnd'], indirect=True)
    def test_input_dimensionality(self, model):
        with pytest.raises(AssertionError) as e_info:
            e = Eigendistortion(torch.zeros((1, 1, 1), device=DEVICE), model)  # should be 4D

        with pytest.raises(AssertionError) as e_info:
            e = Eigendistortion(torch.zeros((2, 1, 1, 1), device=DEVICE), model)  # batch dim must be 1

    @pytest.mark.parametrize('model', ['FrontEnd'], indirect=True)
    def test_method_assertion(self, einstein_img, model):
        einstein_img = einstein_img[..., :SMALL_DIM, :SMALL_DIM]
        ed = Eigendistortion(einstein_img, model)
        with pytest.raises(AssertionError) as e_info:
            ed.synthesize(method='asdfsdfasf')

    @pytest.mark.parametrize('model', ['FrontEnd', 'ColorModel'], indirect=True)
    def test_method_exact(self, model, einstein_img, color_img):
        # in this case, we're working with grayscale images
        if model.__class__ == FrontEnd:
            n_chans = 1
            img = einstein_img
        else:
            img = color_img
            n_chans = 3
        img = img[..., :SMALL_DIM, :SMALL_DIM]

        ed = Eigendistortion(img, model)
        # invert matrix explicitly
        ed.synthesize(method='exact')

        assert len(ed.synthesized_eigenvalues) == n_chans*SMALL_DIM**2
        assert len(ed.synthesized_signal) == n_chans*SMALL_DIM**2
        assert len(ed.synthesized_eigenindex) == n_chans*SMALL_DIM**2

        # test that each eigenvector returned is original img shape
        assert ed.synthesized_signal.shape[-3:] == (n_chans, SMALL_DIM, SMALL_DIM)

    @pytest.mark.parametrize('model', ['FrontEnd', 'ColorModel'], indirect=True)
    def test_method_power(self, model, einstein_img, color_img):
        if model.__class__ == FrontEnd:
            n_chans = 1
            img = einstein_img
        else:
            img = color_img
            n_chans = 3
        img = img[..., :LARGE_DIM, :LARGE_DIM]
        ed = Eigendistortion(img, model)
        ed.synthesize(method='power', max_steps=3)

        # test it should only return two eigenvectors and values
        assert len(ed.synthesized_eigenvalues) == 2
        assert len(ed.synthesized_signal) == 2
        assert len(ed.synthesized_eigenindex) == 2

        assert ed.synthesized_signal.shape[-3:] == (n_chans, LARGE_DIM, LARGE_DIM)

    @pytest.mark.parametrize('model', ['FrontEnd'], indirect=True)
    def test_orthog_iter(self, model, einstein_img):
        n, k = 30, 10
        n_chans = 1  # TODO color
        einstein_img = einstein_img[..., :n, :n]
        ed = Eigendistortion(einstein_img, model)
        ed.synthesize(k=k, method='power', max_steps=10)

        assert ed.synthesized_signal.shape == (k*2, n_chans, n, n)
        assert ed.synthesized_eigenindex.allclose(torch.cat((torch.arange(k), torch.arange(n**2 - k, n**2))))
        assert len(ed.synthesized_eigenvalues) == 2*k

    @pytest.mark.parametrize('model', ['FrontEnd'], indirect=True)
    def test_method_randomized_svd(self, model, einstein_img):
        n, k = 30, 10
        n_chans = 1  # TODO color
        einstein_img = einstein_img[..., :n, :n]
        ed = Eigendistortion(einstein_img, model)
        ed.synthesize(k=k, method='randomized_svd')
        assert ed.synthesized_signal.shape == (k, n_chans, n, n)
        assert ed.synthesized_eigenindex.allclose(torch.arange(k))
        assert len(ed.synthesized_eigenvalues) == k

    @pytest.mark.parametrize('model', ['FrontEnd'], indirect=True)
    def test_method_accuracy(self, model, einstein_img):
        # test pow and svd against ground-truth jacobian (exact) method
        einstein_img = einstein_img[..., :SMALL_DIM, :SMALL_DIM]
        e_jac = Eigendistortion(einstein_img, model)
        e_pow = Eigendistortion(einstein_img, model)
        e_svd = Eigendistortion(einstein_img, model)

        k_pow, k_svd = 10, 75
        e_jac.synthesize(method='exact')
        e_pow.synthesize(k=k_pow, method='power', max_steps=300)
        e_svd.synthesize(k=k_svd, method='randomized_svd')

        assert e_pow.synthesized_eigenvalues[0].isclose(e_jac.synthesized_eigenvalues[0], atol=1e-2)
        assert e_pow.synthesized_eigenvalues[-1].isclose(e_jac.synthesized_eigenvalues[-1], atol=1e-2)
        assert e_svd.synthesized_eigenvalues[0].isclose(e_jac.synthesized_eigenvalues[0], atol=1e-2)

    @pytest.mark.parametrize("model", ['FrontEnd', 'ColorModel'], indirect=True)
    @pytest.mark.parametrize("method", ['power', 'randomized_svd'])
    @pytest.mark.parametrize("k", [2, 3])
    def test_display(self, model, einstein_img, color_img, method, k):
        # in this case, we're working with grayscale images
        if model.__class__ == FrontEnd:
            img = einstein_img
        else:
            img = color_img
        img = img[..., :SMALL_DIM, :SMALL_DIM]
        e_pow = Eigendistortion(img, model)
        e_pow.synthesize(k=k, method=method, max_steps=10)
        e_pow.plot_distorted_image(eigen_index=0)
        e_pow.plot_distorted_image(eigen_index=-1)


class TestAutodiffFunctions:

    @pytest.fixture(scope='class')
    def state(self, einstein_img):
        """variables to be reused across tests in this class"""
        torch.manual_seed(0)

        k = 2  # num vectors with which to compute vjp, jvp, Fv
        einstein_img = einstein_img[..., :16, :16]  # reduce image size

        # eigendistortion object
        ed = Eigendistortion(einstein_img, get_model('FrontEnd'))

        x, y = ed._input_flat, ed._representation_flat

        x_dim = x.flatten().shape[0]
        y_dim = y.flatten().shape[0]

        return x, y, x_dim, y_dim, k

    def test_jacobian(self, state):
        x, y, x_dim, y_dim, k = state

        jac = autodiff.jacobian(y, x)
        assert jac.shape == (y_dim, x_dim)
        assert jac.requires_grad is False

    @pytest.mark.parametrize('detach', [False, True])
    def test_vec_jac_prod(self, state, detach):
        x, y, x_dim, y_dim, k = state

        U = torch.randn((y_dim, k), device=DEVICE)
        U = U / U.norm(dim=0, p=2)

        vjp = autodiff.vector_jacobian_product(y, x, U, detach=detach)
        assert vjp.shape == (x_dim, k)
        assert vjp.requires_grad != detach

    def test_jac_vec_prod(self, state):
        x, y, x_dim, y_dim, k = state

        V = torch.randn((x_dim, k), device=DEVICE)
        V = V / V.norm(dim=0, p=2)
        jvp = autodiff.jacobian_vector_product(y, x, V)
        assert jvp.shape == (y_dim, k)
        assert x.requires_grad and y.requires_grad
        assert jvp.requires_grad is False

    def test_fisher_vec_prod(self, state):
        x, y, x_dim, y_dim, k = state

        V = torch.randn((x_dim, k), device=DEVICE)
        Jv = autodiff.jacobian_vector_product(y, x, V)
        Fv = autodiff.vector_jacobian_product(y, x, Jv)

        jac = autodiff.jacobian(y, x)

        Fv2 = jac.T @ jac @ V  # manually compute product to compare accuracy

        assert Fv.shape == (x_dim, k)
        assert Fv2.allclose(Fv, rtol=1E-2)

    def test_simple_model_eigenvalues(self):
        """Test if Jacobian is constant in all directions for linear model"""
        singular_value = torch.ones(1, device=DEVICE) * 3.

        class LM(nn.Module):
            """Simple y = Mx where M=3"""

            def __init__(self):
                super().__init__()
                self.M = nn.Linear(1, 1, bias=False)
                self.M.weight.data = singular_value

            def forward(self, x):
                y = self.M(x)
                return y

        x0 = torch.randn((1, 1, 5, 1), requires_grad=True, device=DEVICE)
        x0 = x0 / x0.norm()
        mdl = LM().to(DEVICE)

        k = 10
        x_dim = x0.numel()
        V = torch.randn((x_dim, k), device=DEVICE)  # random directions
        V = V / V.norm(dim=0, p=2)

        e = Eigendistortion(x0, mdl)
        x, y = e._input_flat, e._representation_flat
        Jv = autodiff.jacobian_vector_product(y, x, V)
        Fv = autodiff.vector_jacobian_product(y, x, Jv)

        assert torch.diag(V.T @ Fv).sqrt().allclose(singular_value)
