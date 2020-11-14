import pytest
import torch
import requests
import math
import tqdm
import tarfile
import os
import numpy as np
import plenoptic as po
import pyrtools as pt
import os.path as op
import scipy.io as sio
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')
# If you add anything here, remember to update the docstring in osf_download!
OSF_URL = {'plenoptic-test-files.tar.gz': 'q9kn8', 'ssim_images.tar.gz': 'j65tw',
           'ssim_analysis.mat': 'ndtc7', 'MAD_results.tar.gz': 'jwcsr'}
print("On device %s" % DEVICE)


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
    return osf_download('ssim_analysis.mat')


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
        v1 = po.simul.PooledV1(1, x.shape[-2:])
        v1(x[0])
        po.simul.non_linearities.normalize_dict(v1.representation)


def test_find_files(test_files_dir):
    assert op.exists(op.join(test_files_dir, 'buildSCFpyr0.mat'))


class TestPerceptualMetrics(object):

    @pytest.mark.parametrize('weighted', [True, False])
    def test_ssim(self, weighted):
        im1 = po.load_images(op.join(DATA_DIR, 'einstein.pgm'))
        im2 = torch.randn_like(im1, requires_grad=True)
        assert po.metric.ssim(im1, im2).requires_grad

    @pytest.mark.parametrize('func_name', ['noise', 'mse', 'ssim'])
    @pytest.mark.parametrize('size_A', [1, 3])
    @pytest.mark.parametrize('size_B', [1, 2, 3])
    def test_batch_handling(self, func_name, size_A, size_B):
        im1 = po.load_images(op.join(DATA_DIR, 'einstein.pgm'))
        im2 = torch.randn_like(im1)
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
    def test_noise_independence(self, mode):
        # this makes sure that we are drawing the noise independently in the
        # two cases here
        img = po.load_images(op.join(DATA_DIR, 'einstein.pgm'))
        if mode == 'many-to-one':
            img = img.repeat(2, 1, 1, 1)
            noise_lvl = 1
        elif mode == 'one-to-many':
            noise_lvl = [1, 1]
        noisy = po.add_noise(img, noise_lvl)
        assert not torch.equal(*noisy)

    @pytest.mark.parametrize('noise_lvl', [[1], [128], [2, 4], [2, 4, 8], [0]])
    @pytest.mark.parametrize('noise_as_tensor', [True, False])
    def test_add_noise(self, noise_lvl, noise_as_tensor):
        img = po.load_images(op.join(DATA_DIR, 'einstein.pgm'))
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

    def test_nlpd(self):
        im1 = po.load_images(op.join(DATA_DIR, 'einstein.pgm'))
        im2 = torch.randn_like(im1, requires_grad=True)
        assert po.metric.nlpd(im1, im2).requires_grad

    def test_nspd(self):
        im1 = po.load_images(op.join(DATA_DIR, 'einstein.pgm'))
        im2 = torch.randn_like(im1, requires_grad=True)
        assert po.metric.nspd(im1, im2).requires_grad

    def test_nspd2(self):
        im1 = po.load_images(op.join(DATA_DIR, 'einstein.pgm'))
        im2 = torch.randn_like(im1, requires_grad=True)
        assert po.metric.nspd(im1, im2, O=3, S=5, complex=True).requires_grad

    def test_nspd3(self):
        im1 = po.load_images(op.join(DATA_DIR, 'einstein.pgm'))
        im2 = torch.randn_like(im1, requires_grad=True)
        assert po.metric.nspd(im1, im2, O=1, S=5, complex=False).requires_grad

    def test_model_metric(self):
        im1 = po.load_images(op.join(DATA_DIR, 'einstein.pgm'))
        im2 = torch.randn_like(im1, requires_grad=True)
        model = po.simul.Front_End(disk_mask=True)
        assert po.metric.model_metric(im1, im2, model).requires_grad


class TestDisplay(object):

    def test_update_plot_line(self):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(*x.shape)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, y1, '-o', label='hi')
        po.update_plot(ax, torch.tensor(y2).reshape(1, 1, len(x)))
        assert len(ax.lines) == 1, "Too many lines were plotted!"
        _, ax_y = ax.lines[0].get_data()
        if not np.allclose(ax_y, y2):
            raise Exception("Didn't update line correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict', 'tensor'])
    def test_update_plot_line_multi_axes(self, how):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(2, *y1.shape)
        if how == 'tensor':
            y2 = torch.tensor(y2).reshape(1, 2, *y1.shape)
        elif how == 'dict':
            y2 = {i: torch.tensor(y2[i]).reshape(1, 1, *y1.shape) for i in range(2)}
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.plot(x, y1, '-o', label='hi')
        po.update_plot(axes, y2)
        for i, ax in enumerate(axes):
            assert len(ax.lines) == 1, "Too many lines were plotted!"
            _, ax_y = ax.lines[0].get_data()
            if how == 'tensor':
                y_check = y2[0, i]
            else:
                y_check = y2[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update line correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict-single', 'dict-multi', 'tensor'])
    def test_update_plot_line_multi_channel(self, how):
        if how == 'dict-single':
            n_data = 1
        else:
            n_data = 2
        x = np.linspace(0, 100)
        y1 = np.random.rand(2, *x.shape)
        y2 = np.random.rand(n_data, *x.shape)
        if how == 'tensor':
            y2 = torch.tensor(y2).reshape(1, 2, len(x))
        elif how == 'dict-multi':
            y2 = {i: torch.tensor(y2[i]).reshape(1, 1, len(x)) for i in range(2)}
        elif how == 'dict-single':
            y2 = {0: torch.tensor(y2[0]).reshape(1, 1, len(x))}
        fig, ax = plt.subplots(1, 1)
        for i in range(2):
            ax.plot(x, y1[i], label=i)
        po.update_plot(ax, y2)
        assert len(ax.lines) == 2, "Too many lines were plotted!"
        for i in range(2):
            _, ax_y = ax.lines[i].get_data()
            if how == 'tensor':
                y_check = y2[0, i]
            elif how == 'dict-multi':
                y_check = y2[i]
            elif how == 'dict-single':
                y_check = {0: y2[0], 1: y1[1]}[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update line correctly!")
        plt.close('all')

    def test_update_plot_stem(self):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(*x.shape)
        fig, ax = plt.subplots(1, 1)
        ax.stem(x, y1, '-o', label='hi', use_line_collection=True)
        po.update_plot(ax, torch.tensor(y2).reshape(1, 1, len(x)))
        assert len(ax.containers) == 1, "Too many stems were plotted!"
        ax_y = ax.containers[0].markerline.get_ydata()
        if not np.allclose(ax_y, y2):
            raise Exception("Didn't update stems correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict', 'tensor'])
    def test_update_plot_stem_multi_axes(self, how):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(2, *y1.shape)
        if how == 'tensor':
            y2 = torch.tensor(y2).reshape(1, 2, *y1.shape)
        elif how == 'dict':
            y2 = {i: torch.tensor(y2[i]).reshape(1, 1, *y1.shape) for i in range(2)}
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.stem(x, y1, label='hi', use_line_collection=True)
        po.update_plot(axes, y2)
        for i, ax in enumerate(axes):
            assert len(ax.containers) == 1, "Too many stems were plotted!"
            ax_y = ax.containers[0].markerline.get_ydata()
            if how == 'tensor':
                y_check = y2[0, i]
            else:
                y_check = y2[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update stem correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict-single', 'dict-multi', 'tensor'])
    def test_update_plot_stem_multi_channel(self, how):
        if how == 'dict-single':
            n_data = 1
        else:
            n_data = 2
        x = np.linspace(0, 100)
        y1 = np.random.rand(2, *x.shape)
        y2 = np.random.rand(n_data, *x.shape)
        if how == 'tensor':
            y2 = torch.tensor(y2).reshape(1, 2, len(x))
        elif how == 'dict-multi':
            y2 = {i: torch.tensor(y2[i]).reshape(1, 1, len(x)) for i in range(2)}
        elif how == 'dict-single':
            y2 = {0: torch.tensor(y2[0]).reshape(1, 1, len(x))}
        fig, ax = plt.subplots(1, 1)
        for i in range(2):
            ax.stem(x, y1[i], label=i, use_line_collection=True)
        po.update_plot(ax, y2)
        assert len(ax.containers) == 2, "Too many lines were plotted!"
        for i in range(2):
            ax_y = ax.containers[i].markerline.get_ydata()
            if how == 'tensor':
                y_check = y2[0, i]
            elif how == 'dict-multi':
                y_check = y2[i]
            elif how == 'dict-single':
                y_check = {0: y2[0], 1: y1[1]}[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update line correctly!")
        plt.close('all')

    def test_update_plot_image(self):
        y1 = np.random.rand(1, 1, 100, 100)
        y2 = np.random.rand(*y1.shape)
        fig = pt.imshow(y1.squeeze())
        ax = fig.axes[0]
        po.update_plot(ax, torch.tensor(y2))
        assert len(ax.images) == 1, "Too many images were plotted!"
        ax_y = ax.images[0].get_array().data
        if not np.allclose(ax_y, y2):
            raise Exception("Didn't update image correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict', 'tensor'])
    def test_update_plot_image_multi_axes(self, how):
        y1 = np.random.rand(1, 2, 100, 100)
        y2 = np.random.rand(1, 2, 100, 100)
        if how == 'tensor':
            y2 = torch.tensor(y2)
        elif how == 'dict':
            y2 = {i: torch.tensor(y2[0, i]).reshape(1, 1, 100, 100) for i in range(2)}
        fig = pt.imshow([y for y in y1.squeeze()])
        po.update_plot(fig.axes, y2)
        for i, ax in enumerate(fig.axes):
            assert len(ax.images) == 1, "Too many lines were plotted!"
            ax_y = ax.images[0].get_array().data
            if how == 'tensor':
                y_check = y2[0, i]
            else:
                y_check = y2[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update image correctly!")
        plt.close('all')

    @pytest.mark.parametrize('as_rgb', [True, False])
    @pytest.mark.parametrize('channel_idx', [None, 0, [0, 1]])
    @pytest.mark.parametrize('batch_idx', [None, 0, [0, 1]])
    @pytest.mark.parametrize('is_complex', [False, 'logpolar', 'rectangular', 'polar'])
    @pytest.mark.parametrize('mini_im', [True, False])
    def test_imshow(self, as_rgb, channel_idx, batch_idx, is_complex, mini_im):
        fails = False
        if is_complex:
            im = torch.rand((2, 4, 10, 10, 2))
            # this is 2 (the two complex components) * 4 (the four channels) *
            # 2 (the two batches)
            n_axes = 16
        else:
            im = torch.rand((2, 4, 10, 10))
            # this is 4 (the four channels) * 2 (the two batches)
            n_axes = 8
        if mini_im:
            # n_axes here follows the same logic as above
            if is_complex:
                shape = [2, 4, 5, 5, 2]
                n_axes += 16
            else:
                shape = [2, 4, 5, 5]
                n_axes += 8
            im = [im, torch.rand(shape)]
        if not is_complex:
            # need to change this to one of the acceptable strings
            is_complex = 'rectangular'
        if batch_idx is None and channel_idx is None and not as_rgb:
            # then we'd have a 4d array we want to plot in grayscale -- don't
            # know how to do that
            fails = True
        else:
            if batch_idx is not None:
                # then we're only plotting one of the two batches
                n_axes /= 2
            if channel_idx is not None:
                # then we're only plotting one of the four channels
                n_axes /= 4
                # if channel_idx is not None, then we don't have all the
                # channels necessary for plotting RGB, so this will fail
                if as_rgb:
                    fails = True
            # when channel_idx=0, as_rgb does nothing, so don't want to
            # double-count
            elif as_rgb:
                # if we're plotting as_rgb, the four channels just specify
                # RGBA, so we only have one image for them
                n_axes /= 4
        if isinstance(batch_idx, list) or isinstance(channel_idx, list):
            # neither of these are supported
            fails = True
        if not fails:
            fig = po.imshow(im, as_rgb=as_rgb, channel_idx=channel_idx,
                            batch_idx=batch_idx, plot_complex=is_complex)
            assert len(fig.axes) == n_axes, f"Created {len(fig.axes)} axes, but expected {n_axes}! Probably plotting color as grayscale or vice versa"
            plt.close('all')
        if fails:
            with pytest.raises(Exception):
                po.imshow(im, as_rgb=as_rgb, channel_idx=channel_idx,
                          batch_idx=batch_idx, plot_complex=is_complex)

    @pytest.mark.parametrize('as_rgb', [True, False])
    @pytest.mark.parametrize('channel_idx', [None, 0, [0, 1]])
    @pytest.mark.parametrize('batch_idx', [None, 0, [0, 1]])
    @pytest.mark.parametrize('is_complex', [False, 'logpolar', 'rectangular', 'polar'])
    @pytest.mark.parametrize('mini_vid', [True, False])
    def test_animshow(self, as_rgb, channel_idx, batch_idx, is_complex, mini_vid):
        fails = False
        if is_complex:
            vid = torch.rand((2, 4, 10, 10, 10, 2))
            # this is 2 (the two complex components) * 4 (the four channels) *
            # 2 (the two batches)
            n_axes = 16
        else:
            vid = torch.rand((2, 4, 10, 10, 10))
            # this is 4 (the four channels) * 2 (the two batches)
            n_axes = 8
        if mini_vid:
            # n_axes here follows the same logic as above
            if is_complex:
                shape = [2, 4, 10, 5, 5, 2]
                n_axes += 16
            else:
                shape = [2, 4, 10, 5, 5]
                n_axes += 8
            vid = [vid, torch.rand(shape)]
        if not is_complex:
            # need to change this to one of the acceptable strings
            is_complex = 'rectangular'
        if batch_idx is None and channel_idx is None and not as_rgb:
            # then we'd have a 4d array we want to plot in grayscale -- don't
            # know how to do that
            fails = True
        else:
            if batch_idx is not None:
                # then we're only plotting one of the two batches
                n_axes /= 2
            if channel_idx is not None:
                # then we're only plotting one of the four channels
                n_axes /= 4
                # if channel_idx is not None, then we don't have all the
                # channels necessary for plotting RGB, so this will fail
                if as_rgb:
                    fails = True
            # when channel_idx=0, as_rgb does nothing, so don't want to
            # double-count
            elif as_rgb:
                # if we're plotting as_rgb, the four channels just specify
                # RGBA, so we only have one video for them
                n_axes /= 4
        if isinstance(batch_idx, list) or isinstance(channel_idx, list):
            # neither of these are supported
            fails = True
        if not fails:
            anim = po.animshow(vid, as_rgb=as_rgb, channel_idx=channel_idx,
                               batch_idx=batch_idx, plot_complex=is_complex)
            fig = anim._fig
            assert len(fig.axes) == n_axes, f"Created {len(fig.axes)} axes, but expected {n_axes}! Probably plotting color as grayscale or vice versa"
            plt.close('all')
        if fails:
            with pytest.raises(Exception):
                po.animshow(vid, as_rgb=as_rgb, channel_idx=channel_idx,
                            batch_idx=batch_idx, plot_complex=is_complex)
