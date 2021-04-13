import pytest
import torch
import requests
import math
import tqdm
import tarfile
import os
import numpy as np
import plenoptic as po
import os.path as op
import scipy.io as sio
from conftest import DATA_DIR


# If you add anything here, remember to update the docstring in osf_download!
OSF_URL = {'plenoptic-test-files.tar.gz': 'q9kn8', 'ssim_images.tar.gz': 'j65tw',
           'ssim_analysis.mat': 'ndtc7', 'MAD_results.tar.gz': 'jwcsr', 
           'portilla_simoncelli_matlab_test_vectors.tar.gz':'qtn5y',
           'portilla_simoncelli_test_vectors.tar.gz': '8r2gq',
           'portilla_simoncelli_synthesize.npy': '7aw82'}

# print("On device %s" % DEVICE)



def osf_download(filename):
    f"""Download file from plenoptic OSF page.

    From the OSF project at https://osf.io/ts37w/.

    Downloads the specified file to `plenoptic/data`, extracts and deletes the
    the .tar.gz file (if applicable), and returns the path.

    Parameters
    ----------
    filename : {'plenoptic-test-files.tar.gz', 'ssim_images.tar.gz',
                'ssim_analysis.mat', 'MAD_results.tar.gz'}
        Which file to download.

    Returns
    -------
    path : str
        The path to the downloaded directory or file.

    """
    path = op.join(op.dirname(op.realpath(__file__)), '..', 'data', filename)
    if not op.exists(path.replace('.tar.gz', '')):
        print(f"{filename} not found, downloading now...")
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
        print("DONE")
    return path.replace('.tar.gz', '')


@pytest.fixture()
def test_files_dir():
    return osf_download('plenoptic-test-files.tar.gz')


@pytest.fixture()
def ssim_images():
    return osf_download('ssim_images.tar.gz')


@pytest.fixture()
def ssim_analysis():
    ssim_analysis = osf_download('ssim_analysis.mat')
    return sio.loadmat(ssim_analysis, squeeze_me=True)


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

    def test_rectangular_to_polar_dict(self, basic_stim):
        spc = po.simul.Steerable_Pyramid_Freq(basic_stim.shape[-2:], height=5,
                                              order=1, is_complex=True)
        y = spc(basic_stim)
        energy, state = po.simul.non_linearities.rectangular_to_polar_dict(y)

    def test_rectangular_to_polar_real(self):
        x = torch.randn(10, 1, 256, 256)
        po.simul.non_linearities.rectangular_to_polar_real(x)

    def test_local_gain_control(self, basic_stim):
        spc = po.simul.Steerable_Pyramid_Freq(basic_stim.shape[-2:], height=5, order=1, is_complex=False)
        y = spc(basic_stim)
        energy, state = po.simul.non_linearities.local_gain_control(y)

    def test_normalize(self, basic_stim):
        # should operate on both of these, though it will do different
        # things
        po.simul.non_linearities.normalize(basic_stim[0].flatten())
        po.simul.non_linearities.normalize(basic_stim[0].flatten(), 1)
        po.simul.non_linearities.normalize(basic_stim[0])
        po.simul.non_linearities.normalize(basic_stim[0], 1)
        po.simul.non_linearities.normalize(basic_stim[0], sum_dim=1)

    def test_normalize_dict(self, basic_stim):
        spyr = po.simul.Steerable_Pyramid_Freq(basic_stim.shape[-2:])
        po.simul.non_linearities.normalize_dict(spyr(basic_stim))


def test_find_files(test_files_dir):
    assert op.exists(op.join(test_files_dir, 'buildSCFpyr0.mat'))


@pytest.mark.parametrize('paths', [DATA_DIR, op.join(DATA_DIR, 'einstein.png'),
                                   op.join(DATA_DIR, '256x256'),
                                   [op.join(DATA_DIR, 'einstein.png'),
                                    op.join(DATA_DIR, 'curie.pgm')]])
@pytest.mark.parametrize('as_gray', [True, False])
def test_load_images(paths, as_gray):
    if paths == DATA_DIR:
        # there's a 512 by 512 image here, which means we should raise
        # an Exception
        with pytest.raises(Exception):
            images = po.tools.data.load_images(paths, as_gray)
    else:
        images = po.tools.data.load_images(paths, as_gray)
        assert images.ndimension() == 4, "load_images did not return a 4d tensor!"


class TestPerceptualMetrics(object):

    @pytest.mark.parametrize('weighted', [True, False])
    def test_ssim(self, einstein_img, curie_img, weighted):
        curie_img.requires_grad_()
        assert po.metric.ssim(einstein_img, curie_img, weighted=weighted).requires_grad

    @pytest.mark.parametrize('func_name', ['noise', 'mse', 'ssim'])
    @pytest.mark.parametrize('size_A', [1, 3])
    @pytest.mark.parametrize('size_B', [1, 2, 3])
    def test_batch_handling(self, einstein_img, curie_img, func_name, size_A, size_B):
        if func_name == 'noise':
            func = po.add_noise
            A = einstein_img.repeat(size_A, 1, 1, 1)
            B = size_B * [4]
        elif func_name == 'mse':
            func = po.metric.mse
            A = einstein_img.repeat(size_A, 1, 1, 1)
            B = curie_img.repeat(size_B, 1, 1, 1)
        elif func_name == 'ssim':
            func = po.metric.ssim
            A = einstein_img.repeat(size_A, 1, 1, 1)
            B = curie_img.repeat(size_B, 1, 1, 1)
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
    def test_noise_independence(self, einstein_img, mode):
        # this makes sure that we are drawing the noise independently in the
        # two cases here
        if mode == 'many-to-one':
            einstein_img = einstein_img.repeat(2, 1, 1, 1)
            noise_lvl = 1
        elif mode == 'one-to-many':
            noise_lvl = [1, 1]
        noisy = po.add_noise(einstein_img, noise_lvl)
        assert not torch.equal(*noisy)

    @pytest.mark.parametrize('noise_lvl', [[1], [128], [2, 4], [2, 4, 8], [0]])
    @pytest.mark.parametrize('noise_as_tensor', [True, False])
    def test_add_noise(self, einstein_img, noise_lvl, noise_as_tensor):
        if noise_as_tensor:
            noise_lvl = torch.tensor(noise_lvl, dtype=torch.float32).unsqueeze(1)
        noisy = po.add_noise(einstein_img, noise_lvl)
        if not noise_as_tensor:
            # always needs to be a tensor to properly check with allclose
            noise_lvl = torch.tensor(noise_lvl, dtype=torch.float32).unsqueeze(1)
        assert torch.allclose(po.metric.mse(einstein_img, noisy), noise_lvl)

    @pytest.fixture
    def ssim_base_img(self, ssim_images, ssim_analysis):
        return po.load_images(op.join(ssim_images, ssim_analysis['base_img']))

    @pytest.mark.parametrize('weighted', [True, False])
    @pytest.mark.parametrize('other_img', np.arange(1, 11))
    def test_ssim_analysis(self, weighted, other_img, ssim_images,
                           ssim_analysis, ssim_base_img):
        mat_type = {True: 'weighted', False: 'standard'}[weighted]
        other = po.load_images(op.join(ssim_images, f"samp{other_img}.tif"))
        # dynamic range is 1 for these images, because po.load_images
        # automatically re-ranges them. They were comptued with
        # dynamic_range=255 in MATLAB, and by correctly setting this value,
        # that should be corrected for
        plen_val = po.metric.ssim(ssim_base_img, other, weighted)
        mat_val = torch.tensor(ssim_analysis[mat_type][f'samp{other_img}'].astype(np.float32))
        # float32 precision is ~1e-6 (see `np.finfo(np.float32)`), and the
        # errors increase through multiplication and other operations.
        print(plen_val-mat_val, plen_val, mat_val)
        assert torch.allclose(plen_val, mat_val.view_as(plen_val), atol=1e-5)

    def test_nlpd(self, einstein_img, curie_img):
        curie_img.requires_grad_()
        assert po.metric.nlpd(einstein_img, curie_img).requires_grad

    def test_nspd(self, einstein_img, curie_img):
        curie_img.requires_grad_()
        assert po.metric.nspd(einstein_img, curie_img).requires_grad

    def test_nspd2(self, einstein_img, curie_img):
        curie_img.requires_grad_()
        assert po.metric.nspd(einstein_img, curie_img, O=3, S=5, complex=True).requires_grad

    def test_nspd3(self, einstein_img, curie_img):
        curie_img.requires_grad_()
        assert po.metric.nspd(einstein_img, curie_img, O=1, S=5, complex=False).requires_grad

    @pytest.mark.parametrize('model', ['FrontEnd'], indirect=True)
    def test_model_metric(self, einstein_img, curie_img, model):
        curie_img.requires_grad_()
        assert po.metric.model_metric(einstein_img, curie_img, model).requires_grad
