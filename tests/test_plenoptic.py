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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')


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


def to_numpy(x):
    """helper function to detach tensor, get it on cpu, and convert it to numpy array

    """
    return x.detach().cpu().numpy().squeeze()


class TestLinear(object):

    def test_linear(self):
        model = po.simul.Linear()
        x = po.make_basic_stimuli()
        assert model(x).requires_grad

    def test_linear_metamer(self):
        model = po.simul.Linear()
        image = plt.imread(op.join(DATA_DIR, 'nuts.pgm')).astype(float) / 255.
        im0 = torch.tensor(image, requires_grad=True, dtype=dtype).squeeze().unsqueeze(0).unsqueeze(0)
        M = po.synth.Metamer(im0, model)
        matched_image, matched_representation = M.synthesize(max_iter=3, learning_rate=1, seed=1)

class TestLinearNonlinear(object):

    def test_linear_nonlinear(self):
        model = po.simul.Linear_Nonlinear()
        x = po.make_basic_stimuli()
        assert model(x).requires_grad

    def test_linear_nonlinear_metamer(self):
        model = po.simul.Linear_Nonlinear()
        image = plt.imread(op.join(DATA_DIR, 'metal.pgm')).astype(float) / 255.
        im0 = torch.tensor(image,requires_grad=True,dtype = torch.float32).squeeze().unsqueeze(0).unsqueeze(0)
        M = po.synth.Metamer(im0, model)
        matched_image, matched_representation = M.synthesize(max_iter=3, learning_rate=1,seed=0)


# class TestConv(object):
# TODO expand, arbitrary shapes, dim


class TestLaplacianPyramid(object):

    def test_grad(self):
        L = po.simul.Laplacian_Pyramid()
        y = L.analysis(po.make_basic_stimuli())
        assert y[0].requires_grad


class TestSteerablePyramid(object):

    @pytest.mark.parametrize("height", [3, 4, 5])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_real(self, height, order):
        x = po.make_basic_stimuli()
        spc = po.simul.Steerable_Pyramid_Freq(x.shape[-2:], height=height, order=order, is_complex=False)
        y = spc(x)

    @pytest.mark.parametrize("height", [3,4,5])
    @pytest.mark.parametrize("order", [1,2,3])
    def test_complex(self, height, order):
        x = po.make_basic_stimuli()
        spc = po.simul.Steerable_Pyramid_Freq(x.shape[-2:], height=height, order=order, is_complex=True)
        y = spc(x)

    # TODO reconstruction


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


def test_find_files(test_files_dir):
    assert op.exists(op.join(test_files_dir, 'buildSCFpyr0.mat'))


class TestPooling(object):

    def test_creation(self):
        windows, theta, ecc = po.simul.pooling.create_pooling_windows(.87)

    def test_creation_args(self):
        with pytest.raises(Exception):
            # we can't create these with transition_region_width != .5
            windows, theta, ecc = po.simul.pooling.create_pooling_windows(.87, .2, 30, 1.2, .7,
                                                                          100, 100)
        windows, theta, ecc = po.simul.pooling.create_pooling_windows(.87, .2, 30, 1.2, .5, 100,
                                                                      100)

    def test_ecc_windows(self):
        ecc, windows = po.simul.pooling.log_eccentricity_windows(n_windows=4)
        ecc, windows = po.simul.pooling.log_eccentricity_windows(n_windows=4.5)
        ecc, windows = po.simul.pooling.log_eccentricity_windows(window_width=.5)
        ecc, windows = po.simul.pooling.log_eccentricity_windows(window_width=1)

    def test_angle_windows(self):
        theta, windows = po.simul.pooling.polar_angle_windows(4)
        theta, windows = po.simul.pooling.polar_angle_windows(4, 1000)
        with pytest.raises(Exception):
            theta, windows = po.simul.pooling.polar_angle_windows(1.5)
        with pytest.raises(Exception):
            theta, windows = po.simul.pooling.polar_angle_windows(1)

    def test_calculations(self):
        # these really shouldn't change, but just in case...
        assert po.simul.pooling.calc_angular_window_width(2) == np.pi
        assert po.simul.pooling.calc_angular_n_windows(2) == np.pi
        with pytest.raises(Exception):
            po.simul.pooling.calc_eccentricity_window_width()
        assert po.simul.pooling.calc_eccentricity_window_width(n_windows=4) == 0.8502993454155389
        assert po.simul.pooling.calc_eccentricity_window_width(scaling=.87) == 0.8446653390527211
        assert po.simul.pooling.calc_eccentricity_window_width(5, 10, scaling=.87) == 0.8446653390527211
        assert po.simul.pooling.calc_eccentricity_window_width(5, 10, n_windows=4) == 0.1732867951399864
        assert po.simul.pooling.calc_eccentricity_n_windows(0.8502993454155389) == 4
        assert po.simul.pooling.calc_eccentricity_n_windows(0.1732867951399864, 5, 10) == 4
        assert po.simul.pooling.calc_scaling(4) == 0.8761474337786708
        assert po.simul.pooling.calc_scaling(4, 5, 10) == 0.17350368946058647
        assert np.isinf(po.simul.pooling.calc_scaling(4, 0))


# class TestSpectral(object):
#


class TestVentralStream(object):

    def test_rgc(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape)
        rgc(im)
        _ = rgc.plot_window_sizes('degrees')
        _ = rgc.plot_window_sizes('degrees', jitter=0)
        _ = rgc.plot_window_sizes('pixels')
        fig = pt.imshow(im.detach())
        _ = rgc.plot_windows(fig.axes[0])
        rgc.plot_representation()
        fig, axes = plt.subplots(2, 1)
        rgc.plot_representation(ax=axes[1])

    def test_rgc_metamer(self):
        # literally just testing that it runs
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3)

    def test_rgc_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        rgc = po.simul.PrimaryVisualCortex(.5, im.shape)
        rgc(im)
        rgc.save_reduced(op.join(tmp_path, 'test_rgc_save_load.pt'))
        rgc_copy = po.simul.RetinalGanglionCells.load_reduced(op.join(tmp_path,
                                                                      'test_rgc_save_load.pt'))
        if not len(rgc.PoolingWindows.windows) == len(rgc_copy.PoolingWindows.windows):
            raise Exception("Something went wrong saving and loading, the lists of windows are"
                            " not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(rgc.PoolingWindows.windows)):
            if not rgc.PoolingWindows.windows[i].allclose(rgc_copy.PoolingWindows.windows[i]):
                raise Exception("Something went wrong saving and loading, the windows %d are"
                                " not identical!" % i)

    def test_frontend(self):
        im = po.make_basic_stimuli()
        frontend = po.simul.Front_End()
        frontend(im)
        
    def test_frontend_eigendistortion(self):
        im = plt.imread(op.join(DATA_DIR, 'einstein.png'))[:,:,0]
        im = torch.tensor(im, dtype=dtype, device=device, requires_grad=True).unsqueeze(0).unsqueeze(0)
        frontend = po.simul.Front_End()
        edist = po.synth.Eigendistortion(im, frontend)
        edist.synthesize(jac=False, n_steps=5)

    def test_v1(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        v1(im)
        _ = v1.plot_window_sizes('pixels')
        for i in range(v1.num_scales):
            _ = v1.plot_window_sizes('pixels', i)
        v1.plot_representation()
        fig, axes = plt.subplots(2, 1)
        v1.plot_representation(ax=axes[1])

    def test_v1_mean_luminance(self):
        for fname in ['nuts', 'einstein']:
            im = plt.imread(op.join(DATA_DIR, fname+'.pgm'))
            im = torch.tensor(im, dtype=torch.float32, device=device)
            v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
            v1_rep = v1(im)
            rgc = po.simul.RetinalGanglionCells(.5, im.shape)
            rgc_rep = rgc(im)
            if not torch.allclose(rgc_rep, v1.mean_luminance):
                raise Exception("Somehow RGC and V1 mean luminance representations are not the "
                                "same for image %s!" % fname)
            if not torch.allclose(rgc_rep, v1_rep[-rgc_rep.shape[0]:]):
                raise Exception("Somehow V1's representation does not have the mean luminance "
                                "in the location expected! for image %s!" % fname)

    def test_v1_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        v1(im)
        v1.save_reduced(op.join(tmp_path, 'test_v1_save_load.pt'))
        v1_copy = po.simul.PrimaryVisualCortex.load_reduced(op.join(tmp_path,
                                                                    'test_v1_save_load.pt'))
        if not len(v1.PoolingWindows.windows) == len(v1_copy.PoolingWindows.windows):
            raise Exception("Something went wrong saving and loading, the lists of windows are"
                            " not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(v1.PoolingWindows.windows)):
            if not v1.PoolingWindows.windows[i].allclose(v1_copy.PoolingWindows.windows[i]):
                raise Exception("Something went wrong saving and loading, the windows %d are"
                                " not identical!" % i)

    def test_v1_metamer(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3)

    @pytest.mark.parametrize("frontend", [True, False])
    @pytest.mark.parametrize("steer", [True, False])
    def test_v2(self, frontend, steer):
        x = po.make_basic_stimuli()
        v2 = po.simul.V2(frontend=frontend, steer=steer)
        v2(x)

    @pytest.mark.parametrize("frontend", [True, False])
    @pytest.mark.parametrize("steer", [True, False])
    def test_v2_metamer(self, frontend, steer):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device, requires_grad=True).unsqueeze(0).unsqueeze(0)
        v2 = po.simul.V2(frontend=frontend, steer=steer)
        metamer = po.synth.Metamer(im, v2)
        metamer.synthesize(max_iter=3)


class TestMetamers(object):

    def test_metamer_save_load(self, tmp_path):

        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.save(op.join(tmp_path, 'test_metamer_save_load.pt'))
        met_copy = po.synth.Metamer.load(op.join(tmp_path, "test_metamer_save_load.pt"))
        for k in ['target_image', 'saved_representation', 'saved_image', 'matched_representation',
                  'matched_image', 'target_representation']:
            if not getattr(metamer, k).allclose(getattr(met_copy, k)):
                raise Exception("Something went wrong with saving and loading! %s not the same"
                                % k)

    def test_metamer_save_load_reduced(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.save(op.join(tmp_path, 'test_metamer_save_load_reduced.pt'), True)
        with pytest.raises(Exception):
            met_copy = po.synth.Metamer.load(op.join(tmp_path,
                                                     "test_metamer_save_load_reduced.pt"))
        met_copy = po.synth.Metamer.load(op.join(tmp_path, 'test_metamer_save_load_reduced.pt'),
                                         po.simul.PrimaryVisualCortex.from_state_dict_reduced)
        for k in ['target_image', 'saved_representation', 'saved_image', 'matched_representation',
                  'matched_image', 'target_representation']:
            if not getattr(metamer, k).allclose(getattr(met_copy, k)):
                raise Exception("Something went wrong with saving and loading! %s not the same" % k)

    def test_metamer_save_rep(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=2)

    def test_metamer_save_rep_2(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=True)

    def test_metamer_save_rep_3(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=6, store_progress=3)

    def test_metamer_animate(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3, store_progress=True)
        # this will test several related functions for us:
        # plot_metamer_status, plot_representation_ratio,
        # representation_ratio
        metamer.animate(figsize=(17, 5), plot_representation_ratio=True, ylim='rescale100',
                        framerate=40)

    def test_metamer_nans(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = im / 255
        im = torch.tensor(im, dtype=torch.float32, device=device)
        initial_image = .5*torch.ones_like(im, requires_grad=True, device=device,
                                           dtype=torch.float32)

        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        clamper = po.RangeClamper((0, 1))
        # this gets raised because we try to use saved_image_ticker,
        # which was never initialized, since we're not saving images
        with pytest.raises(IndexError):
            metamer.synthesize(clamper=clamper, learning_rate=10, max_iter=4, loss_thresh=1e-8,
                               initial_image=initial_image)
        # need to re-initialize this for the following run
        initial_image = .5*torch.ones_like(im, requires_grad=True, device=device,
                                           dtype=torch.float32)
        matched_im, _ = metamer.synthesize(clamper=clamper, learning_rate=10, store_progress=True,
                                           max_iter=4, loss_thresh=1e-8,
                                           initial_image=initial_image)
        # this should hit a nan as it runs, leading the second saved
        # image to be all nans, but, because of our way of handling
        # this, matched_image should have no nans
        assert torch.isnan(metamer.saved_image[-1]).all(), "This should be all NaNs!"
        assert not torch.isnan(metamer.matched_image).any(), "There should be no NaNs!"

    def test_metamer_save_progress(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        save_path = op.join(tmp_path, 'test_metamer_save_progress.pt')
        metamer.synthesize(max_iter=3, store_progress=True, save_progress=True,
                           save_path=save_path)
        po.synth.Metamer.load(save_path, po.simul.PrimaryVisualCortex.from_state_dict_reduced)

    def test_metamer_fraction_removed(self):

        X = np.load(op.join(op.join(op.dirname(op.realpath(__file__)), '..', 'examples'), 'metamer_PS_samples.npy'))
        sigma = X.std(axis=1)
        sigma[sigma < .00001] = 1
        normalizationFactor = 1 / sigma
        normalizationFactor = torch.diag(torch.tensor(normalizationFactor, dtype=torch.float32))

        model = po.simul.Texture_Statistics([256, 256], normalizationFactor=normalizationFactor)
        image = plt.imread(op.join(DATA_DIR, 'nuts.pgm')).astype(float) / 255.
        im0 = torch.tensor(image, requires_grad=True, dtype=torch.float32).squeeze().unsqueeze(0).unsqueeze(0)
        c = po.RangeClamper([image.min(), image.max()])
        M = po.synth.Metamer(im0, model)

        matched_image, matched_representation = M.synthesize(max_iter=3, learning_rate=1, seed=1, optimizer='SGD',
                                                             fraction_removed=.1, clamper=c)

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
