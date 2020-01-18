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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')


class TestEigendistortions(object):
    # Could include

    from plenoptic.synthesize.eigendistortion import Eigendistortion
    img_small = torch.randn(1, 1, 20, 20).to(device)
    torch.manual_seed(0)

    img = matplotlib.image.imread(op.join(DATA_DIR, 'einstein.png'))
    img_np = img[..., 0] / np.max(img)
    img_large = torch.Tensor(img_np).view([1, 1, 256, 256]).to(device)

    @staticmethod
    def get_synthesis_object():

        mdl = Front_End().to(device)  # initialize simple model with which to compute eigendistortions
        e_small = Eigendistortion(img_small, mdl, dtype=torch.float32).to(device)
        e_large = Eigendistortion(img_large, mdl, dtype=torch.float32).to(device)

        return e_small, e_large

    # Numerical eigenvector approximations
    def test_jacobian(self):
        # invert matrix explicitly
        e_small, _ = get_synthesis_object()
        e_small.synthesize(method='jacobian');

    def test_power_method(self):
        e_small, e_large = get_synthesis_object()
        e_small.synthesize(method='power', n_steps=5)
        e_large.synthesize(method='power', n_steps=5);

    def test_lanczos(self):
        # return first and last two eigenvectors
        # full re-orthogonalization
        e_small, e_large = get_synthesis_object()
        e_small.synthesize(method='lanczos', orthogonalize='full', n_steps=20, e_vecs=[0, 1, -2, -1])
        e_large.synthesize(method='lanczos', orthogonalize='full', n_steps=20, e_vecs=[0, 1, -2, -1]);
