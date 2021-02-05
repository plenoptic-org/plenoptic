import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import plenoptic.synthesize.autodiff as autodiff
import pytest
import torch
from torch import nn
from plenoptic.simulate.models.frontend import Front_End
import plenoptic as po
from plenoptic.synthesize.eigendistortion import Eigendistortion
from test_plenoptic import DEVICE, DATA_DIR, DTYPE

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
        mdl = Front_End().to(DEVICE)  # initialize simple model with which to compute eigendistortions
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
        mdl = Front_End().to(DEVICE)
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
        ed.synthesize(method='power', n_steps=n_steps)

        # test it should only return two eigenvectors and values
        assert len(ed.synthesized_eigenvalues) == 2
        assert len(ed.synthesized_signal) == 2
        assert len(ed.synthesized_eigenindex) == 2

        assert ed.synthesized_signal.shape[-3:] == (n_chans, LARGE_DIM, LARGE_DIM)

    @pytest.mark.parametrize("e_vecs", [[0, 1, -2, -1], []])
    @pytest.mark.parametrize("color", [False, True])
    def test_method_lanczos(self, e_vecs, color):
        # return first and last two eigenvectors
        n_steps = 5
        n_chans = 3 if color else 1

        # run once with e_vecs specified
        ed = get_synthesis_object(im_dim=LARGE_DIM, color=color)

        if n_steps < len(e_vecs) * 2:
            with pytest.warns(RuntimeWarning) as not_enough_iter_warning:
                ed.synthesize(method='lanczos', n_steps=n_steps, e_vecs=e_vecs)
        else:
            with pytest.warns(UserWarning) as lanczos_experimental_warning:
                ed.synthesize(method='lanczos', n_steps=n_steps, e_vecs=e_vecs)

        if len(e_vecs) > 0:
            assert len(ed.synthesized_eigenvalues) == len(e_vecs)
        else:
            assert len(ed.synthesized_eigenvalues) == n_steps

        assert len(ed.synthesized_signal) == len(e_vecs)
        assert len(ed.synthesized_eigenindex) == len(e_vecs)

        if len(e_vecs) > 0:
            assert ed.synthesized_signal.shape[-3:] == (n_chans, LARGE_DIM, LARGE_DIM)

    def test_lanczos_accuracy(self):
        n = 30
        e_vals = (torch.randn(n**2)**2).sort(descending=True)[0]
        eigen_test_matrix = torch.diag(e_vals)
        ed = get_synthesis_object(im_dim=n)
        with pytest.warns(UserWarning) as lanczos_experimental_warning:
            ed.synthesize(method='lanczos', n_steps=eigen_test_matrix.shape[-1], debug_A=eigen_test_matrix)

        assert (e_vals[0]-ed.synthesized_eigenvalues[0]) < 1e-2

    def test_method_equivalence(self):

        e_jac = get_synthesis_object(im_dim=SMALL_DIM)
        e_pow = get_synthesis_object(im_dim=SMALL_DIM)

        e_jac.synthesize(method='exact', store_progress=True)
        e_pow.synthesize(method='power', n_steps=500, store_progress=True)

        print(e_pow.synthesized_eigenvalues.shape)
        print(e_pow.synthesized_eigenvalues[0], e_pow.synthesized_eigenvalues[1])
        print(e_jac.synthesized_eigenvalues[0], e_jac.synthesized_eigenvalues[-1])

        assert e_pow.synthesized_eigenvalues[0].isclose(e_jac.synthesized_eigenvalues[0], atol=1e-3)
        assert e_pow.synthesized_eigenvalues[1].isclose(e_jac.synthesized_eigenvalues[-1], atol=1e-2)

        fig_max = e_pow.plot_loss(0)
        # fig_max.show()

        fig_min = e_pow.plot_loss(-1)
        # fig_min.show()

    @pytest.mark.parametrize("color", [False, True])
    def test_display(self, color):
        e_pow = get_synthesis_object(im_dim=SMALL_DIM, color=color)
        e_pow.synthesize(method='power', n_steps=50, store_progress=True)

        e_pow.display_first_and_last()
        e_pow.plot_synthesized_image(0)


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
        x, y = e._input_flat, e._representation_flat
        Jv = autodiff.jacobian_vector_product(y, x, V)
        Fv = autodiff.vector_jacobian_product(y, x, Jv)

        assert torch.diag(V.T @ Fv).sqrt().allclose(singular_value)


if __name__ == '__main__':
    tmp = TestEigendistortionSynthesis()
    tmp.test_method_equivalence()
    ed = get_synthesis_object(20, True)
    ed.synthesize('power', n_steps=3)
    print(ed.synthesized_signal.shape)
