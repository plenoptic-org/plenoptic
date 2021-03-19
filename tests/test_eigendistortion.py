import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import plenoptic.synthesize.autodiff as autodiff
import pytest
import torch
from torch import nn
from plenoptic.simulate.models.frontend import FrontEnd
from plenoptic.synthesize.eigendistortion import Eigendistortion
from conftest import DEVICE, DATA_DIR, DTYPE

# to be used for default model instantiation
SMALL_DIM = 20
LARGE_DIM = 100


class ColorModel(nn.Module):
    """Simple model that takes color image as input and outputs 2d conv."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, 1)

    def forward(self, x):
        return self.conv(x)


def get_synthesis_object(im_dim=20, color=False):
    r""" Helper for Pytests. Instantiates Eigendistortion object for FrontEnd model.

    Parameters
    ----------
    im_dim : int
        number of pixels of one side of small square image to be used with Jacobian explicit solver.
    color: bool
        Whether or not img and model are color.
    Returns
    -------
    ed: Eigendistortion
        Eigendistortion object to be used in tests.
    """
    torch.manual_seed(0)

    if not color:
        mdl = FrontEnd(pretrained=True, requires_grad=False).to(DEVICE)
        img = plt.imread(op.join(DATA_DIR, 'einstein.pgm'))
        img_np = img[:im_dim, :im_dim] / np.max(img)
        img = torch.Tensor(img_np).view([1, 1, im_dim, im_dim]).to(DEVICE)

    else:
        img0 = plt.imread(op.join(DATA_DIR, 'color_wheel.jpg'))
        n = img0.shape[0]
        skip = n//im_dim
        img_np = img0[::skip, ::skip].copy()/np.max(img0)
        img = torch.as_tensor(img_np, device=DEVICE, dtype=torch.float).permute((2,0,1)).unsqueeze(0)
        mdl = ColorModel()

    ed = Eigendistortion(img, mdl)

    return ed


class TestEigendistortionSynthesis:

    def test_input_dimensionality(self):
        mdl = FrontEnd().to(DEVICE)
        with pytest.raises(AssertionError) as e_info:
            e = Eigendistortion(torch.zeros(1, 1, 1), mdl)  # should be 4D

        with pytest.raises(AssertionError) as e_info:
            e = Eigendistortion(torch.zeros(2, 1, 1, 1), mdl)  # batch dim must be 1

    def test_method_assertion(self):
        ed = get_synthesis_object(im_dim=SMALL_DIM)
        with pytest.raises(AssertionError) as e_info:
            ed.synthesize(method='asdfsdfasf')

    @pytest.mark.parametrize('color', [False, True])
    def test_method_exact(self, color):
        # invert matrix explicitly
        n_chans = 3 if color else 1
        ed = get_synthesis_object(im_dim=SMALL_DIM, color=color)
        ed.synthesize(method='exact')

        assert len(ed.synthesized_eigenvalues) == n_chans*SMALL_DIM**2
        assert len(ed.synthesized_signal) == n_chans*SMALL_DIM**2
        assert len(ed.synthesized_eigenindex) == n_chans*SMALL_DIM**2

        # test that each eigenvector returned is original img shape
        assert ed.synthesized_signal.shape[-3:] == (n_chans, SMALL_DIM, SMALL_DIM)

    @pytest.mark.parametrize('color', [False, True])
    def test_method_power(self, color):
        n_chans = 3 if color else 1
        n_steps = 3
        ed = get_synthesis_object(im_dim=LARGE_DIM, color=color)
        ed.synthesize(method='power', max_steps=n_steps)

        # test it should only return two eigenvectors and values
        assert len(ed.synthesized_eigenvalues) == 2
        assert len(ed.synthesized_signal) == 2
        assert len(ed.synthesized_eigenindex) == 2

        assert ed.synthesized_signal.shape[-3:] == (n_chans, LARGE_DIM, LARGE_DIM)

    def test_orthog_iter(self):
        n, k = 30, 10
        n_chans = 1  # TODO color
        ed = get_synthesis_object(im_dim=n)
        ed.synthesize(k=k, method='power', max_steps=10)

        assert ed.synthesized_signal.shape == (k*2, n_chans, n, n)
        assert ed.synthesized_eigenindex.allclose(torch.cat((torch.arange(k), torch.arange(n**2 - k, n**2))))
        assert len(ed.synthesized_eigenvalues) == 2*k

    def test_method_randomized_svd(self):
        n, k = 30, 10
        n_chans = 1  # TODO color
        ed = get_synthesis_object(im_dim=n)
        ed.synthesize(k=k, method='randomized_svd')
        assert ed.synthesized_signal.shape == (k, n_chans, n, n)
        assert ed.synthesized_eigenindex.allclose(torch.arange(k))
        assert len(ed.synthesized_eigenvalues) == k

    def test_method_accuracy(self):
        # test pow and svd against ground-truth jacobian (exact) method
        e_jac = get_synthesis_object(im_dim=SMALL_DIM)
        e_pow = get_synthesis_object(im_dim=SMALL_DIM)
        e_svd = get_synthesis_object(im_dim=SMALL_DIM)

        k_pow, k_svd = 10, 75
        e_jac.synthesize(method='exact')
        e_pow.synthesize(k=k_pow, method='power', max_steps=300)
        e_svd.synthesize(k=k_svd, method='randomized_svd')

        assert e_pow.synthesized_eigenvalues[0].isclose(e_jac.synthesized_eigenvalues[0], atol=1e-2)
        assert e_pow.synthesized_eigenvalues[-1].isclose(e_jac.synthesized_eigenvalues[-1], atol=1e-2)
        assert e_svd.synthesized_eigenvalues[0].isclose(e_jac.synthesized_eigenvalues[0], atol=1e-2)

    @pytest.mark.parametrize("color", [False, True])
    @pytest.mark.parametrize("method", ['power', 'randomized_svd'])
    @pytest.mark.parametrize("k", [2, 3])
    def test_display(self, color, method, k):
        e_pow = get_synthesis_object(im_dim=SMALL_DIM, color=color)
        e_pow.synthesize(k=k, method=method, max_steps=10)
        e_pow.plot_distorted_image(eigen_index=0)
        e_pow.plot_distorted_image(eigen_index=-1)


class TestAutodiffFunctions:

    @staticmethod
    def _state():
        """variables to be reused across tests in this class"""
        torch.manual_seed(0)

        k = 2  # num vectors with which to compute vjp, jvp, Fv

        ed = get_synthesis_object(im_dim=SMALL_DIM)  # eigendistortion object

        x, y = ed._input_flat, ed._representation_flat

        x_dim = x.flatten().shape[0]
        y_dim = y.flatten().shape[0]

        return x, y, x_dim, y_dim, k

    def test_jacobian(self):
        x, y, x_dim, y_dim, k = self._state()

        jac = autodiff.jacobian(y, x)
        assert jac.shape == (y_dim, x_dim)
        assert jac.requires_grad is False

    @pytest.mark.parametrize('detach', [False, True])
    def test_vec_jac_prod(self, detach):
        x, y, x_dim, y_dim, k = self._state()

        U = torch.randn(y_dim, k)
        U = U / U.norm(dim=0, p=2)

        vjp = autodiff.vector_jacobian_product(y, x, U, detach=detach)
        assert vjp.shape == (x_dim, k)
        assert vjp.requires_grad != detach

    def test_jac_vec_prod(self):
        x, y, x_dim, y_dim, k = self._state()

        V = torch.randn(x_dim, k)
        V = V / V.norm(dim=0, p=2)
        jvp = autodiff.jacobian_vector_product(y, x, V)
        assert jvp.shape == (y_dim, k)
        assert x.requires_grad and y.requires_grad
        assert jvp.requires_grad is False

    def test_fisher_vec_prod(self):
        x, y, x_dim, y_dim, k = self._state()

        V = torch.randn(x_dim, k)
        Jv = autodiff.jacobian_vector_product(y, x, V)
        Fv = autodiff.vector_jacobian_product(y, x, Jv)

        jac = autodiff.jacobian(y, x)

        Fv2 = jac.T @ jac @ V  # manually compute product to compare accuracy

        assert Fv.shape == (x_dim, k)
        assert Fv2.allclose(Fv, rtol=1E-2)

    def test_simple_model_eigenvalues(self):
        """Test if Jacobian is constant in all directions for linear model"""
        singular_value = torch.ones(1)*3.

        class LM(nn.Module):
            """Simple y = Mx where M=3"""
            def __init__(self):
                super().__init__()
                self.M = nn.Linear(1, 1, bias=False)
                self.M.weight.data = singular_value

            def forward(self, x):
                y = self.M(x)
                return y

        x0 = torch.randn((1, 1, 5, 1), requires_grad=True)
        x0 = x0 / x0.norm()
        mdl = LM()

        k = 10
        x_dim = x0.numel()
        V = torch.randn(x_dim, k)  # random directions
        V = V / V.norm(dim=0, p=2)

        e = Eigendistortion(x0, mdl)
        x, y = e._input_flat, e._representation_flat
        Jv = autodiff.jacobian_vector_product(y, x, V)
        Fv = autodiff.vector_jacobian_product(y, x, Jv)

        assert torch.diag(V.T @ Fv).sqrt().allclose(singular_value)

