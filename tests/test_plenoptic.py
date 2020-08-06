import pytest
import torch
import requests
import math
import tqdm
import itertools
import tarfile
import os
import numpy as np
import pyrtools as pt
import plenoptic as po
import os.path as op
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')
print("On device %s" % device)

@pytest.fixture
def test_files_dir():
    path = op.join(op.dirname(op.realpath(__file__)), '..', 'data', 'plenoptic-test-files')
    if not op.exists(path):
        print("matfiles required for testing not found, downloading now...")
        # Streaming, so we can iterate over the response.
        r = requests.get("https://osf.io/q9kn8/download", stream=True)

        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024*1024
        wrote = 0
        with open(path + ".tar.gz", 'wb') as f:
            for data in tqdm.tqdm(r.iter_content(block_size), unit='MB', unit_scale=True,
                                  total=math.ceil(total_size//block_size)):
                wrote += len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            raise Exception("Error downloading test files!")
        with tarfile.open(path + ".tar.gz") as f:
            f.extractall(op.dirname(path))
        os.remove(path + ".tar.gz")
    return path


class TestNonLinearities(object):

    def test_polar_amplitude_zero(self):
        a = torch.rand(10)*-1
        b = po.rescale(torch.randn(10), -np.pi / 2, np.pi / 2)

        with pytest.raises(ValueError) as e:
            _, _ = po.polar_to_rectangular(a, b)

    def test_coordinate_identity_transform_rectangular(self):
        dims = (10, 5, 256, 256)
        x = torch.randn(dims)
        y = torch.randn(dims)

        X, Y = po.polar_to_rectangular(*po.rectangular_to_polar(x, y))

        assert torch.norm(x - X) < 1e-3
        assert torch.norm(y - Y) < 1e-3

    def test_coordinate_identity_transform_polar(self):
        dims = (10, 5, 256, 256)

        # ensure vec len a is non-zero by adding .1 and then re-normalizing
        a = torch.rand(dims) + 0.1
        a = a / a.max()
        b = po.rescale(torch.randn(dims), -np.pi / 2, np.pi / 2)

        A, B = po.rectangular_to_polar(*po.polar_to_rectangular(a, b))

        assert torch.norm(a - A) < 1e-3
        assert torch.norm(b - B) < 1e-3

    def test_rectangular_to_polar_dict(self):
        x = po.make_basic_stimuli()
        spc = po.simul.Steerable_Pyramid_Freq(x.shape[-2:], height=5, order=1, is_complex=True)
        y = spc(x)
        energy, state = po.simul.non_linearities.rectangular_to_polar_dict(y)

    def test_rectangular_to_polar_real(self):
        x = torch.randn(10, 1, 256, 256)
        po.simul.non_linearities.rectangular_to_polar_real(x)

    def test_local_gain_control(self):
        x = po.make_basic_stimuli()
        spc = po.simul.Steerable_Pyramid_Freq(x.shape[-2:], height=5, order=1, is_complex=False)
        y = spc(x)
        energy, state = po.simul.non_linearities.local_gain_control(y)

    def test_normalize(self):
        x = po.make_basic_stimuli()
        # should operate on both of these, though it will do different
        # things
        po.simul.non_linearities.normalize(x[0].flatten())
        po.simul.non_linearities.normalize(x[0].flatten(), 1)
        po.simul.non_linearities.normalize(x[0])
        po.simul.non_linearities.normalize(x[0], 1)
        po.simul.non_linearities.normalize(x[0], sum_dim=1)

    def test_normalize_dict(self):
        x = po.make_basic_stimuli()
        v1 = po.simul.PrimaryVisualCortex(1, x.shape[-2:])
        v1(x[0])
        po.simul.non_linearities.normalize_dict(v1.representation)


def test_find_files(test_files_dir):
    assert op.exists(op.join(test_files_dir, 'buildSCFpyr0.mat'))


class TestPerceptualMetrics(object):

    im1 = po.rescale(plt.imread(op.join(DATA_DIR, 'einstein.png')).astype(float)[:, :, 0])
    im1 = torch.tensor(im1, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
    im2 = torch.rand_like(im1, requires_grad=True, device=device)

    @pytest.mark.parametrize("im1, im2", [(im1, im2)])
    def test_ssim(self, im1, im2):
        assert po.metric.ssim(im1, im2).requires_grad

    @pytest.mark.parametrize("im1, im2", [(im1, im2)])
    def test_msssim(self, im1, im2):
        assert po.metric.msssim(im1, im2).requires_grad

    @pytest.mark.parametrize("im1, im2", [(im1, im2)])
    def test_nlpd(self, im1, im2):
        assert po.metric.nlpd(im1, im2).requires_grad

    @pytest.mark.parametrize("im1, im2", [(im1, im2)])
    def test_nspd(self, im1, im2):
        assert po.metric.nspd(im1, im2).requires_grad

    @pytest.mark.parametrize("im1, im2", [(im1, im2)])
    def test_nspd2(self, im1, im2):
        assert po.metric.nspd(im1, im2, O=3, S=5, complex=True).requires_grad

    @pytest.mark.parametrize("im1, im2", [(im1, im2)])
    def test_nspd3(self, im1, im2):
        assert po.metric.nspd(im1, im2, O=1, S=5, complex=False).requires_grad

    @pytest.mark.parametrize("im1, im2", [(im1, im2)])
    def test_model_metric(self, im1, im2):
        model = po.simul.Front_End(disk_mask=True)
        assert po.metric.model_metric(im1, im2, model).requires_grad
