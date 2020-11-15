import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import plenoptic.synthesize.autodiff as autodiff
import pytest
import torch
from torch import nn
from plenoptic.simulate.models.frontend import Front_End
from plenoptic.synthesize.eigendistortion import Eigendistortion
from test_plenoptic import DEVICE, DATA_DIR, DTYPE

# to be used for default model instantiation
SMALL_DIM = 20
LARGE_DIM = 100


def get_synthesis_object(im_dim=20):
    r""" Helper for Pytests. Instantiates Eigendistortion object for FrontEnd model.

    Parameters
    ----------
    im_dim : int
        number of pixels of one side of small square image to be used with Jacobian explicit solver.
    Returns
    -------
    ed: Eigendistortion
        Eigendistortion object to be used in tests.
    """
    torch.manual_seed(0)
    mdl = Front_End().to(DEVICE)  # initialize simple model with which to compute eigendistortions

    img = plt.imread(op.join(DATA_DIR, 'einstein.pgm'))
    img_np = img[:im_dim, :im_dim] / np.max(img)
    img = torch.Tensor(img_np).view([1, 1, im_dim, im_dim]).to(DEVICE)

    ed = Eigendistortion(img, mdl, dtype=DTYPE)

    return ed


class TestEigendistortionSynthesis:

    def test_input_dimensionality(self):
        mdl = Front_End().to(DEVICE)
        with pytest.raises(AssertionError) as e_info:
            e = Eigendistortion(torch.zeros(1, 1, 1), mdl)  # should be 4D

        with pytest.raises(AssertionError) as e_info:
            e = Eigendistortion(torch.zeros(2, 1, 1, 1), mdl)  # batch dim must be 1

    def test_method_assertion(self):
        ed = get_synthesis_object(im_dim=SMALL_DIM)
        with pytest.raises(AssertionError) as e_info:
            ed.synthesize(method='asdfsdfasf')

    def test_method_exact(self):
        # invert matrix explicitly
        ed = get_synthesis_object(im_dim=SMALL_DIM)
        ed.synthesize(method='exact')

        assert len(ed.distortions['eigenvalues']) == SMALL_DIM**2
        assert len(ed.distortions['eigenvectors']) == SMALL_DIM**2
        assert len(ed.distortions['eigenvector_index']) == SMALL_DIM**2

        # test that each eigenvector returned is original img shape
        assert ed.distortions['eigenvectors'][0].shape == (SMALL_DIM, SMALL_DIM)

    def test_method_power(self):
        n_steps = 3
        ed = get_synthesis_object(im_dim=LARGE_DIM)
        ed.synthesize(method='power', n_steps=n_steps)

        # test it should only return two eigenvectors and values
        assert len(ed.distortions['eigenvalues']) == 2
        assert len(ed.distortions['eigenvectors']) == 2
        assert len(ed.distortions['eigenvector_index']) == 2

        assert ed.distortions['eigenvectors'][0].shape == (LARGE_DIM, LARGE_DIM)

    @pytest.mark.parametrize("e_vecs", [[0, 1, -2, -1], []])
    def test_method_lanczos(self, e_vecs):
        # return first and last two eigenvectors
        n_steps = 5

        # run once with e_vecs specified
        ed = get_synthesis_object(im_dim=LARGE_DIM)

        if n_steps < len(e_vecs) * 2:
            with pytest.warns(RuntimeWarning) as not_enough_iter_warning:
                ed.synthesize(method='lanczos', n_steps=n_steps, e_vecs=e_vecs)
        else:
            with pytest.warns(UserWarning) as lanczos_experimental_warning:
                ed.synthesize(method='lanczos', n_steps=n_steps, e_vecs=e_vecs)

        if len(e_vecs) > 0:
            assert len(ed.distortions['eigenvalues']) == len(e_vecs)
        else:
            assert len(ed.distortions['eigenvalues']) == n_steps

        assert len(ed.distortions['eigenvectors']) == len(e_vecs)
        assert len(ed.distortions['eigenvector_index']) == len(e_vecs)

        if len(e_vecs) > 0:
            assert ed.distortions['eigenvectors'][0].shape == (LARGE_DIM, LARGE_DIM)

    def test_lanczos_accuracy(self):
        n = 30
        e_vals = (torch.randn(n**2)**2).sort(descending=True)[0]
        eigen_test_matrix = torch.diag(e_vals)
        ed = get_synthesis_object(im_dim=n)
        with pytest.warns(UserWarning) as lanczos_experimental_warning:
            ed.synthesize(method='lanczos', n_steps=eigen_test_matrix.shape[-1], debug_A=eigen_test_matrix)

        assert (e_vals[0]-ed.distortions['eigenvalues'][0]) < 1e-2

    def test_method_equivalence(self):

        e_jac = get_synthesis_object(im_dim=SMALL_DIM)
        e_pow = get_synthesis_object(im_dim=SMALL_DIM)

        e_jac.synthesize(method='exact')
        e_pow.synthesize(method='power', n_steps=500, verbose=False)

        print(e_pow.distortions['eigenvalues'].shape)
        print(e_pow.distortions['eigenvalues'][0], e_pow.distortions['eigenvalues'][1])
        print(e_jac.distortions['eigenvalues'][0], e_jac.distortions['eigenvalues'][-1])

        assert e_pow.distortions['eigenvalues'][0].isclose(e_jac.distortions['eigenvalues'][0], atol=1e-3)
        assert e_pow.distortions['eigenvalues'][1].isclose(e_jac.distortions['eigenvalues'][-1], atol=1e-3)


class TestAutodiffFunctions:

    @staticmethod
    def _state():
        """variables to be reused across tests in this class"""
        torch.manual_seed(0)

        k = 2  # num vectors with which to compute vjp, jvp, Fv

        ed = get_synthesis_object(im_dim=SMALL_DIM)  # eigendistortion object

        x, y = ed.input_flat, ed.representation_flat

        x_dim = x.flatten().shape[0]
        y_dim = y.flatten().shape[0]

        return x, y, x_dim, y_dim, k

    def test_jacobian(self):
        x, y, x_dim, y_dim, k = self._state()

        jac = autodiff.jacobian(y, x)
        assert jac.shape == (y_dim, x_dim)
        assert jac.requires_grad == False

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
        assert jvp.requires_grad == False

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
        x, y = e.input_flat, e.representation_flat
        Jv = autodiff.jacobian_vector_product(y, x, V)
        Fv = autodiff.vector_jacobian_product(y, x, Jv)

        assert torch.diag(V.T @ Fv).sqrt().allclose(singular_value)
