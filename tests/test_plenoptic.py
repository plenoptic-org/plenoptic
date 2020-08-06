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
import scipy.io as sio


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')
OSF_URL = {'plenoptic-test-files.tar.gz': 'q9kn8', 'ssim_images.tar.gz': 'j65tw',
           'ssim_analysis.mat': 'ndtc7'}
print("On device %s" % DEVICE)

def osf_download(filename):
    path = op.join(op.dirname(op.realpath(__file__)), '..', 'data', filename)
    if not op.exists(path.replace('.tar.gz', '')):
        print("matfiles required for testing not found, downloading now...")
        # Streaming, so we can iterate over the response.
        r = requests.get(f"https://osf.io/{OSF_URL[filename]}/download",
                         stream=True)

        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024*1024
        wrote = 0
        with open(path, 'wb') as f:
            for data in tqdm.tqdm(r.iter_content(block_size), unit='MB',
                                  unit_scale=True,
                                  total=math.ceil(total_size//block_size)):
                wrote += len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            raise Exception(f"Error downloading {filename}!")
        if filename.endswith('.tar.gz'):
            with tarfile.open(path) as f:
                f.extractall(op.dirname(path))
            os.remove(path)
    return path.replace('.tar.gz', '')


@pytest.fixture()
def test_files_dir():
    return osf_download('plenoptic-test-files.tar.gz')


@pytest.fixture()
def ssim_images():
    return osf_download('ssim_images.tar.gz')


@pytest.fixture()
def ssim_analysis():
    return osf_download('ssim_analysis.mat')


class TestNonLinearities(object):

    def test_coordinatetransform(self):
        a = torch.randn(10, 5, 256, 256)
        b = torch.randn(10, 5, 256, 256)

        A, B = po.polar_to_rectangular(*po.rectangular_to_polar(a, b))

        assert torch.norm(a - A) < 1e-3
        assert torch.norm(b - B) < 1e-3

        a = torch.rand(10, 5, 256, 256)
        b = po.rescale(torch.randn(10, 5, 256, 256), -np.pi / 2, np.pi / 2)

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
    im1 = torch.tensor(im1, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
    im2 = torch.rand_like(im1, requires_grad=True, device=DEVICE)

    @pytest.mark.parametrize('weighted', [True, False])
    def test_ssim(self, im1, im2, weighted):
        assert po.metric.ssim(im1, im2).requires_grad

    @pytest.mark.parametrize('func_name', ['noise', 'mse', 'ssim'])
    @pytest.mark.parametrize('size_A', [1, 3])
    @pytest.mark.parametrize('size_B', [1, 2, 3])
    @pytest.mark.parametrize('im1, im2', [(im1, im2)])
    def test_batch_handling(self, func_name, size_A, size_B, im1, im2):
        if func_name == 'noise':
            func = po.add_noise
            A = im1.repeat(size_A, 1, 1, 1)
            B = size_B * [4]
        elif func_name == 'mse':
            func = po.metric.mse
            A = im1.repeat(size_A, 1, 1, 1)
            B = im2.repeat(size_B, 1, 1, 1)
        elif func_name == 'ssim':
            func = po.metric.ssim
            A = im1.repeat(size_A, 1, 1, 1)
            B = im2.repeat(size_B, 1, 1, 1)
        if size_A != size_B and size_A != 1 and size_B != 1:
            with pytest.raises(Exception):
                func(A, B)
        else:
            if size_A > size_B:
                tgt_size = size_A
            else:
                tgt_size = size_B
            assert func(A, B).shape[0] == tgt_size

    @pytest.mark.parametrize('mode', ['many-to-one', 'one-to-many'])
    @pytest.mark.parametrize('im1', im1)
    def test_noise_independence(self, mode, im1):
        # this makes sure that we are drawing the noise independently in the
        # two cases here
        if mode == 'many-to-one':
            img = im1.repeat(2, 1, 1, 1)
            noise_lvl = 1
        elif mode == 'one-to-many':
            img = im1
            noise_lvl = [1, 1]
        noisy = po.add_noise(img, noise_lvl)
        assert not torch.equal(*noisy)

    @pytest.mark.parametrize('img', [im1, im2])
    @pytest.mark.parametrize('noise_lvl', [1, 128, [2, 4], [2, 4, 8], 0])
    @pytest.mark.parametrize('noise_as_tensor', [True, False])
    def test_add_noise(self, img, noise_lvl, noise_as_tensor):
        if noise_as_tensor:
            noise_lvl = torch.tensor(noise_lvl, dtype=torch.float32).unsqueeze(1)
        noisy = po.add_noise(img, noise_lvl)
        if not noise_as_tensor:
            # always needs to be a tensor to properly check with allclose
            noise_lvl = torch.tensor(noise_lvl, dtype=torch.float32).unsqueeze(1)
        assert torch.allclose(po.metric.mse(img, noisy), noise_lvl)

    @pytest.mark.parametrize('weighted', [True, False])
    @pytest.mark.parametrize('other_img', np.arange(1, 11))
    def test_ssim_analysis(self, weighted, other_img, ssim_images, ssim_analysis):
        analysis = sio.loadmat(ssim_analysis, squeeze_me=True)
        print(ssim_analysis)
        mat_type = {True: 'weighted', False: 'standard'}[weighted]
        base_img = po.load_images(op.join(ssim_images, analysis['base_img']))
        other = po.load_images(op.join(ssim_images, f"samp{other_img}.tif"))
        # dynamic range is 1 for these images, because po.load_images
        # automatically re-ranges them. They were comptued with
        # dynamic_range=255 in MATLAB, and by correctly setting this value,
        # that should be corrected for
        plen_val = po.metric.ssim(base_img, other, weighted)
        mat_val = torch.tensor(analysis[mat_type][f'samp{other_img}'].astype(np.float32))
        # float32 precision is ~1e-6 (see `np.finfo(np.float32)`), and the
        # errors increase through multiplication and other operations.
        print(plen_val-mat_val, plen_val, mat_val)
        assert torch.allclose(plen_val, mat_val.view_as(plen_val), atol=1e-5)

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
