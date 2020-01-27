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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')

def get_synthesis_object(small_dim=20, large_dim=100):
    """ Instantiates Eigendistortion objects for small and large image for FrontEnd model.

    Parameters
    ----------
    small_dim : int
        number of pixels of one side of small square image to be used with Jacobian explicit solver
    large_dim : int
        number of pixels of one side of square image to be used with iterative methods

    """
    img_small = torch.randn(1, 1, small_dim, small_dim).to(device)
    torch.manual_seed(0)

    img = matplotlib.image.imread(op.join(DATA_DIR, 'einstein.png'))
    img_np = img[:large_dim, :large_dim, 0] / np.max(img)
    img_large = torch.Tensor(img_np).view([1, 1, large_dim, large_dim]).to(device)

    mdl = Front_End().to(device)  # initialize simple model with which to compute eigendistortions
    e_small = Eigendistortion(img_small, mdl, dtype=torch.float32).to(device)
    e_large = Eigendistortion(img_large, mdl, dtype=torch.float32).to(device)

    return e_small, e_large


class TestEigendistortions(object):

    # Numerical eigenvector approximations

    # Test explicit solver
    def test_jacobian(self):
        # invert matrix explicitly
        small_dim = 20
        e_small, _ = get_synthesis_object(small_dim=small_dim)
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
        _, e_large = get_synthesis_object(large_dim=large_dim)
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
        _, e_large = get_synthesis_object(large_dim=large_dim)
        e_large.synthesize(method='lanczos', orthogonalize='full', n_steps=n_steps, e_vecs=e_vecs)

        assert len(e_large.distortions['eigenvalues']) == len(e_vecs)
        assert len(e_large.distortions['eigenvectors']) == len(e_vecs)
        assert len(e_large.distortions['eigenvector_index']) == len(e_vecs)
        assert e_large.distortions['eigenvectors'][0].shape == (large_dim, large_dim)

        # run again with e_vecs as None
        # should return no eigenvectors or index, and return n_steps eigenvalues
        _, e_large = get_synthesis_object(large_dim=large_dim)
        e_large.synthesize(method='lanczos', orthogonalize='full', n_steps=n_steps, e_vecs=None)

        assert len(e_large.distortions['eigenvalues']) == n_steps
        assert len(e_large.distortions['eigenvectors']) == 0
        assert len(e_large.distortions['eigenvector_index']) == 0