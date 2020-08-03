import pytest
import torch
import requests
import math
import tqdm
import tarfile
import os
import numpy as np
import pyrtools as pt
import plenoptic as po
import os.path as op
import matplotlib.pyplot as plt
import matplotlib
from plenoptic.simulate.models.frontend import Front_End
from plenoptic.synthesize.eigendistortion import Eigendistortion
import plenoptic.synthesize.autodiff as autodiff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')

def get_synthesis_object(small_dim=None, large_dim=None):
    """ Instantiates Eigendistortion objects for small and large image for FrontEnd model.

    Parameters
    ----------
    small_dim : int
        number of pixels of one side of small square image to be used with Jacobian explicit solver
    large_dim : int
        number of pixels of one side of square image to be used with iterative methods

    """
    torch.manual_seed(0)
    mdl = Front_End().to(device)  # initialize simple model with which to compute eigendistortions
    e_small = None
    e_large = None

    if small_dim is not None:
        img_small = torch.randn(1, 1, small_dim, small_dim).to(device)
        e_small = Eigendistortion(img_small, mdl, dtype=torch.float32).to(device)

    if large_dim is not None:
        img = plt.imread(op.join(DATA_DIR, 'einstein.pgm'))
        img_np = img[:large_dim, :large_dim] / np.max(img)
        img_large = torch.Tensor(img_np).view([1, 1, large_dim, large_dim]).to(device)

        e_large = Eigendistortion(img_large, mdl, dtype=torch.float32).to(device)

    return e_small, e_large


class TestEigendistortionSynthesis(object):

    # Numerical eigenvector approximations

    # Test explicit solver
    def test_solve_jacobian(self):
        # invert matrix explicitly
        small_dim = 20
        e_small, _ = get_synthesis_object(small_dim=small_dim, large_dim=None)
        e_small.synthesize(method='jacobian')

        assert len(e_small.distortions['eigenvalues']) == small_dim ** 2
        assert len(e_small.distortions['eigenvectors']) == small_dim ** 2
        assert len(e_small.distortions['eigenvector_index']) == small_dim ** 2

        # test that each eigenvector returned is original img shape
        assert e_small.distortions['eigenvectors'][0].shape == (small_dim, small_dim)

    # Test iterative methods
    def test_power_method(self):
        large_dim = 100
        n_steps = 3
        _, e_large = get_synthesis_object(small_dim=None, large_dim=large_dim)
        e_large.synthesize(method='power', n_steps=n_steps)

        # test it should only return two eigenvectors and values
        assert len(e_large.distortions['eigenvalues']) == 2
        assert len(e_large.distortions['eigenvectors']) == 2
        assert len(e_large.distortions['eigenvector_index']) == 2

        assert e_large.distortions['eigenvectors'][0].shape == (large_dim, large_dim)

    def test_lanczos(self):
        # return first and last two eigenvectors
        # full re-orthogonalization
        large_dim = 100
        n_steps = 5
        e_vecs = [0, 1, -2, -1]

        # run once with e_vecs specified
        _, e_large = get_synthesis_object(small_dim=None, large_dim=large_dim)
        e_large.synthesize(method='lanczos', orthogonalize='full', n_steps=n_steps, e_vecs=e_vecs)

        assert len(e_large.distortions['eigenvalues']) == len(e_vecs)
        assert len(e_large.distortions['eigenvectors']) == len(e_vecs)
        assert len(e_large.distortions['eigenvector_index']) == len(e_vecs)
        assert e_large.distortions['eigenvectors'][0].shape == (large_dim, large_dim)
        del e_large

        # run again with e_vecs as None
        # should return no eigenvectors or index, and return n_steps eigenvalues
        _, e_large = get_synthesis_object(small_dim=None, large_dim=large_dim)
        e_large.synthesize(method='lanczos', orthogonalize='full', n_steps=n_steps, e_vecs=None)

        assert len(e_large.distortions['eigenvalues']) == n_steps
        assert len(e_large.distortions['eigenvectors']) == 0
        assert len(e_large.distortions['eigenvector_index']) == 0

    def test_method_equivalence(self):

        large_dim = 20
        _, e_large = get_synthesis_object(small_dim=None, large_dim=large_dim)

        dist_jac = e_large.synthesize(method='jacobian').clone()
        dist_power = e_large.synthesize(method='power', n_steps=100, verbose=False).clone()
        dist_lanczos = e_large.synthesize(method='lanczos', n_steps=300, e_vecs=[0], verbose=False).clone()
        pass
        # assert (dist_jac['eigenvectors'][0] - dist_power['eigenvectors'][0]).norm() < 1e-6
        # assert (dist_jac['eigenvectors'][0] - dist_lanczos['eigenvectors'][0]).norm() < 1e-6


class TestAutodiffFunctions(object):

    def _state(self):
        """variables to be reused across tests in this class"""
        torch.manual_seed(0)

        im_dim = 20  # 50x50 image
        k = 2  # num vectors with which to compute vjp, jvp, Fv

        ed, _ = get_synthesis_object(small_dim=im_dim, large_dim=None)  # eigendistortion object

        x, y = ed.image_flattensor, ed.out_flattensor

        x_dim = x.flatten().shape[0]
        y_dim = y.flatten().shape[0]

        return x, y, x_dim, y_dim, k


    def test_jacobian(self):
        x, y, x_dim, y_dim, k = self._state()

        jac = autodiff.jacobian(y, x)
        assert jac.shape == (y_dim, x_dim)

    def test_vec_jac_prod(self):
        x, y, x_dim, y_dim, k = self._state()

        U = torch.randn(y_dim, k)

        vjp = autodiff.vector_jacobian_product(y, x, U)
        assert vjp.shape == (x_dim, k)

    def test_jac_vec_prod(self):
        x, y, x_dim, y_dim, k = self._state()

        V = torch.randn(x_dim, k)
        jvp = autodiff.jacobian_vector_product(y, x, V)
        assert jvp.shape == (y_dim, k)

    def test_fisher_vec_prod(self):
        x, y, x_dim, y_dim, k = self._state()

        V = torch.randn(x_dim, k)
        Jv = autodiff.jacobian_vector_product(y, x, V)
        Fv = autodiff.vector_jacobian_product(y, x, Jv)

        jac = autodiff.jacobian(y, x)

        Fv2 = jac.T @ jac @ V  # manually compute product to compare accuracy

        assert Fv.shape == (x_dim, k)
        assert Fv2.allclose(Fv, rtol=1E-2)
