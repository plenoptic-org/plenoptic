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
        matched_image, matched_representation = M.synthesize(max_iter=10, learning_rate=1, seed=1)

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
        matched_image, matched_representation = M.synthesize(max_iter=10, learning_rate=1,seed=0)


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

        A, B = po.pol2rect(*po.rect2pol(a, b))

        assert torch.norm(a - A) < 1e-3
        assert torch.norm(b - B) < 1e-3

        a = torch.rand(10, 5, 256, 256)
        b = po.rescale(torch.randn(10, 5, 256, 256), -np.pi / 2, np.pi / 2)

        A, B = po.rect2pol(*po.pol2rect(a, b))

        assert torch.norm(a - A) < 1e-3
        assert torch.norm(b - B) < 1e-3

    def test_rect2pol_dict(self):
        x = po.make_basic_stimuli()
        spc = po.simul.Steerable_Pyramid_Freq(x.shape[-2:], height=5, order=1, is_complex=True)
        y = spc(x)
        energy, state = po.simul.non_linearities.rect2pol_dict(y)

    def test_real_rectangular_to_polar(self):
        x = torch.randn(10, 1, 256, 256)
        po.simul.non_linearities.real_rectangular_to_polar(x)

    def test_local_gain_control(self):
        x = po.make_basic_stimuli()
        spc = po.simul.Steerable_Pyramid_Freq(x.shape[-2:], height=5, order=1, is_complex=False)
        y = spc(x)
        energy, state = po.simul.non_linearities.local_gain_control(y)


def test_find_files(test_files_dir):
    assert op.exists(op.join(test_files_dir, 'buildSCFpyr0.mat'))


class TestPooling(object):

    def test_creation(self):
        windows, theta, ecc = po.simul.create_pooling_windows(.87)

    def test_creation_args(self):
        windows, theta, ecc = po.simul.create_pooling_windows(.87, .2, 30, 1.2, .7, 100, 100)

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
        assert po.simul.pooling.calc_polar_window_width(2) == np.pi
        assert po.simul.pooling.calc_polar_n_windows(2) == np.pi
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

    def test_rgc_metamer(self):
        # literally just testing that it runs
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=10)

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

    def test_v1_metamer(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10)

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
        metamer.synthesize(max_iter=10)


class TestMetamers(object):

    def test_metamer_save_load(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, save_representation=True, save_image=True)
        metamer.save('test.pt')
        met_copy = po.synth.Metamer.load("test.pt")
        for k in ['target_image', 'saved_representation', 'saved_image', 'matched_representation',
                  'matched_image', 'target_representation']:
            if not getattr(metamer, k).allclose(getattr(met_copy, k)):
                raise Exception("Something went wrong with saving and loading! %s not the same" % k)

    def test_metamer_save_rep(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, save_representation=2, save_image=2)

    def test_metamer_save_rep_2(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, save_representation=2, save_image=True)

    def test_metamer_save_rep_3(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, save_representation=3, save_image=True)

    def test_metamer_save_rep_4(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, save_representation=3, save_image=3)


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
