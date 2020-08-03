import pytest
import torch
import requests
import math
import tqdm
import itertools
import tarfile
import os
import imageio
import numpy as np
import pyrtools as pt
import plenoptic as po
import os.path as op
import matplotlib.pyplot as plt
from plenoptic.tools.data import to_numpy, torch_complex_to_numpy


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

def check_pyr_coeffs(coeff_np, coeff_torch, rtol=1e-3, atol=1e-3):
    '''
    function that checks if two sets of pyramid coefficients (one numpy  and one torch) are the same
    We set an absolute and relative tolerance and the following function checks if
    abs(coeff1-coeff2) <= atol + rtol*abs(coeff1)
    Inputs:
    coeff1: numpy pyramid coefficients
    coeff2: torch pyramid coefficients
    Both coeffs must obviously have the same number of scales, orientations etc.
    '''

    for k in coeff_np.keys():
        coeff_np_k = coeff_np[k]
        coeff_torch_k  = coeff_torch[k].squeeze()
        if coeff_torch_k.shape[-1] == 2:
            coeff_torch_k = torch_complex_to_numpy(coeff_torch_k)
        else:
            coeff_torch_k = to_numpy(coeff_torch_k)

        np.testing.assert_allclose(coeff_np_k, coeff_torch_k, rtol=rtol, atol=atol)

def check_band_energies(coeff_1, coeff_2, rtol=1e-4, atol=1e-4):
    '''
    function that checks if the energy in each band of two pyramids are the same.
    We set an absolute and relative tolerance and the function checks for each band if
    abs(coeff_1-coeff_2) <= atol + rtol*abs(coeff_1)
    Args:
    coeff_1: first dictionary of torch tensors corresponding to each band
    coeff_2: second dictionary of torch tensors corresponding to each band
    '''

    for k,v in coeff_1.items():
        band_1 = coeff_1[k].squeeze()
        band_2 = coeff_2[k].squeeze()
        if band_1.shape[-1] == 2:
            band_1 = torch_complex_to_numpy(band_1)
            band_2 = torch_complex_to_numpy(band_2)
        else:
            band_1 = to_numpy(band_1)
            band_2 = to_numpy(band_2)

        np.testing.assert_allclose(np.sum(np.abs(band_1)**2),np.sum(np.abs(band_2)**2), rtol=rtol, atol=atol)


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

    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize('is_complex', [True, False])
    @pytest.mark.parametrize("im_shape", [None, (255, 255), (256, 128), (128, 256), (255, 256),
                                          (256, 255)])
    def test_pyramid(self, height, order, is_complex, im_shape):
        x = po.make_basic_stimuli()
        if im_shape is not None:
            x = x[..., :im_shape[0], :im_shape[1]]
        spc = po.simul.Steerable_Pyramid_Freq(x.shape[-2:], height=height, order=order,
                                              is_complex=is_complex)
        spc(x)

    @pytest.mark.parametrize("height", [3,4,5])
    @pytest.mark.parametrize("order", [1,2,3])
    @pytest.mark.parametrize("is_complex", [False, True])
    def test_not_downsample(self, height, order, is_complex):
        x = plt.imread(op.join(DATA_DIR, 'curie.pgm'))
        x_shape = x.shape
        sp_downsample = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order,
                                                        is_complex = is_complex, downsample = False, fft_normalize=True)
        sp_notdownsample = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order,
                                                            is_complex = is_complex, downsample = True, fft_normalize=True)
        sp_downsample.to(device)
        sp_notdownsample.to(device)

        x_t = torch.tensor(x, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)
        sp_downsample_coeffs = sp_downsample(x_t)
        sp_notdownsample_coeffs = sp_notdownsample(x_t)

        check_band_energies(sp_downsample_coeffs, sp_notdownsample_coeffs)

    @pytest.mark.parametrize("height", [3,4,5])
    @pytest.mark.parametrize("order", [1,2,3])
    @pytest.mark.parametrize("is_complex", [False, True])
    def test_torch_vs_numpy_pyr(self, height, order, is_complex):
        x = plt.imread(op.join(DATA_DIR, 'curie.pgm'))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex)
        torch_sp.to(device)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)

        #Check with non-square image
        x = pt.synthetic_images.ramp((256,128))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex)
        torch_sp.to(device)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)

        #check non-powers-of-2 images
        x = pt.synthetic_images.ramp((200,200))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex)
        torch_sp.to(device)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("fft_normalize", [True, False])
    @pytest.mark.parametrize("downsample", [False, True])
    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize("im_shape", [None, (255, 255), (256, 128), (128, 256), (255, 256),
                                          (256, 255)])
    def test_complete_recon(self, im, is_complex, fft_normalize, downsample, height, order, im_shape):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im = torch.tensor(im, dtype=dtype).unsqueeze(0).unsqueeze(0)
        pyr = po.simul.Steerable_Pyramid_Freq(im.shape[-2:], height, order, is_complex=is_complex, downsample=downsample, fft_normalize = fft_normalize)
        pyr(im)
        recon = pyr.recon_pyr()
        torch.allclose(recon, im)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("fft_normalize", [True, False])
    @pytest.mark.parametrize("downsample", [False, True])
    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize("im_shape", [None, (255, 255), (256, 128), (255, 256)])
    def test_partial_recon(self, im, is_complex, fft_normalize, downsample, height, order, im_shape):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im_tensor = torch.tensor(im, dtype=dtype).unsqueeze(0).unsqueeze(0)
        po_pyr = po.simul.Steerable_Pyramid_Freq(im.shape, height, order, is_complex=is_complex, downsample=downsample, fft_normalize=fft_normalize)
        po_pyr(im_tensor)
        pt_pyr = pt.pyramids.SteerablePyramidFreq(im, height, order, is_complex=is_complex)
        # this is almost certainly over-kill: we're checking every
        # possible combination of reconstructing bands and levels
        recon_levels = []
        for i in range(po_pyr.num_scales):
            recon_levels.extend(list(itertools.combinations(range(po_pyr.num_scales), i)))
        recon_bands = []
        for i in range(po_pyr.num_orientations):
            recon_bands.extend(list(itertools.combinations(range(po_pyr.num_orientations), i)))
        for levels, bands in itertools.product(['all'] + recon_levels, ['all'] + recon_bands):
            po_recon = po.to_numpy(po_pyr.recon_pyr(levels, bands))
            pt_recon = pt_pyr.recon_pyr(levels, bands)
            np.allclose(po_recon, pt_recon)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("fft_normalize", [True, False])
    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize("im_shape", [None, (255, 255), (256, 128), (128, 256), (255, 256),
                                          (256, 255)])
    def test_recon_match_pyrtools(self, im, is_complex, height, order, im_shape):
        # this should fail if and only if test_complete_recon does, but
        # may as well include it just in case
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im_tensor = torch.tensor(im, dtype=dtype).unsqueeze(0).unsqueeze(0)
        po_pyr = po.simul.Steerable_Pyramid_Freq(im.shape, height, order, is_complex=is_complex, fft_normalize=fft_normalize)
        po_pyr(im_tensor)
        pt_pyr = pt.pyramids.SteerablePyramidFreq(im, height, order, is_complex=is_complex)
        po_recon = po.to_numpy(po_pyr.recon_pyr())
        pt_recon = pt_pyr.recon_pyr()
        np.allclose(po_recon, pt_recon)

    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("store_unoriented_bands", [True, False])
    @pytest.mark.parametrize("scales", [[0], [5], [0, 1, 2], [0, 3, 5],
                                        ['residual_highpass', 'residual_lowpass'],
                                        ['residual_highpass', 0, 1, 'residual_lowpass']])
    def test_scales_arg(self, is_complex, store_unoriented_bands, scales):
        img = imageio.imread(op.join(DATA_DIR, 'einstein.pgm'))
        img = torch.tensor(img / 255, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pyr = po.simul.Steerable_Pyramid_Freq(img.shape[-2:], is_complex=is_complex,
                                              store_unoriented_bands=store_unoriented_bands)
        pyr_coeffs = pyr(img).copy()
        if store_unoriented_bands:
            unor = pyr.unoriented_bands.copy()
        reduced_pyr_coeffs = pyr(img, scales).copy()
        for k, v in reduced_pyr_coeffs.items():
            if (v != pyr_coeffs[k]).any():
                raise Exception("Reduced pyr_coeffs should be same as original, but at least key "
                                f"{k} is not")
        if store_unoriented_bands:
            for k, v in pyr.unoriented_bands.items():
                if (v != unor[k]).any():
                    raise Exception("Reduced unoriented_bands should be same as original, but "
                                    f"at least key {k} is not")
        # recon_pyr should always fail
        with pytest.raises(Exception):
            pyr.recon_pyr()
        with pytest.raises(Exception):
            pyr.recon_pyr(scales)


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


class TestPooling(object):

    def test_creation(self):
        ang_windows, ecc_windows = po.simul.pooling.create_pooling_windows(.87, (256, 256))

    def test_creation_args(self):
        ang, ecc = po.simul.pooling.create_pooling_windows(.87, (100, 100), .2, 30, 1.2, .7)
        ang, ecc = po.simul.pooling.create_pooling_windows(.87, (100, 100), .2, 30, 1.2, .5)

    def test_ecc_windows(self):
        windows = po.simul.pooling.log_eccentricity_windows((256, 256), n_windows=4)
        windows = po.simul.pooling.log_eccentricity_windows((256, 256), n_windows=4.5)
        windows = po.simul.pooling.log_eccentricity_windows((256, 256), window_spacing=.5)
        windows = po.simul.pooling.log_eccentricity_windows((256, 256), window_spacing=1)

    def test_angle_windows(self):
        windows = po.simul.pooling.polar_angle_windows(4, (256, 256))
        windows = po.simul.pooling.polar_angle_windows(4, (1000, 1000))
        with pytest.raises(Exception):
            windows = po.simul.pooling.polar_angle_windows(1.5, (256, 256))
        with pytest.raises(Exception):
            windows = po.simul.pooling.polar_angle_windows(1, (256, 256))

    def test_calculations(self):
        # these really shouldn't change, but just in case...
        assert po.simul.pooling.calc_angular_window_spacing(2) == np.pi
        assert po.simul.pooling.calc_angular_n_windows(2) == np.pi
        with pytest.raises(Exception):
            po.simul.pooling.calc_eccentricity_window_spacing()
        assert po.simul.pooling.calc_eccentricity_window_spacing(n_windows=4) == 0.8502993454155389
        assert po.simul.pooling.calc_eccentricity_window_spacing(scaling=.87) == 0.8446653390527211
        assert po.simul.pooling.calc_eccentricity_window_spacing(5, 10, scaling=.87) == 0.8446653390527211
        assert po.simul.pooling.calc_eccentricity_window_spacing(5, 10, n_windows=4) == 0.1732867951399864
        assert po.simul.pooling.calc_eccentricity_n_windows(0.8502993454155389) == 4
        assert po.simul.pooling.calc_eccentricity_n_windows(0.1732867951399864, 5, 10) == 4
        assert po.simul.pooling.calc_scaling(4) == 0.8761474337786708
        assert po.simul.pooling.calc_scaling(4, 5, 10) == 0.17350368946058647
        assert np.isinf(po.simul.pooling.calc_scaling(4, 0))

    @pytest.mark.parametrize('num_scales', [1, 3])
    @pytest.mark.parametrize('transition_region_width', [.5, 1])
    def test_PoolingWindows_cosine(self, num_scales, transition_region_width):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        pw = po.simul.pooling.PoolingWindows(.5, im.shape[2:], num_scales=num_scales,
                                             transition_region_width=transition_region_width,
                                             window_type='cosine',)
        pw = pw.to(device)
        pw(im)
        with pytest.raises(Exception):
            po.simul.PoolingWindows(.2, (64, 64), .5)

    @pytest.mark.parametrize('num_scales', [1, 3])
    def test_PoolingWindows(self, num_scales):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        pw = po.simul.pooling.PoolingWindows(.5, im.shape[2:], num_scales=num_scales,
                                             window_type='gaussian', std_dev=1)
        pw = pw.to(device)
        pw(im)
        # we only support std_dev=1
        with pytest.raises(Exception):
            po.simul.pooling.PoolingWindows(.5, im.shape[2:], num_scales=num_scales,
                                            window_type='gaussian', std_dev=2)
        with pytest.raises(Exception):
            po.simul.pooling.PoolingWindows(.5, im.shape[2:], num_scales=num_scales,
                                            window_type='gaussian', std_dev=.5)

    def test_PoolingWindows_project(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        pw = po.simul.pooling.PoolingWindows(.5, im.shape[2:])
        pw = pw.to(device)
        pooled = pw(im)
        pw.project(pooled)
        pw = po.simul.pooling.PoolingWindows(.5, im.shape[2:], num_scales=3)
        pw = pw.to(device)
        pooled = pw(im)
        pw.project(pooled)

    def test_PoolingWindows_nonsquare(self):
        # test PoolingWindows with weirdly-shaped iamges
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        for sh in [(256, 128), (256, 127), (256, 125), (125, 125), (127, 125)]:
            tmp = im[:sh[0], :sh[1]].unsqueeze(0).unsqueeze(0)
            rgc = po.simul.RetinalGanglionCells(.9, tmp.shape[2:])
            rgc = rgc.to(device)
            rgc(tmp)
            v1 = po.simul.RetinalGanglionCells(.9, tmp.shape[2:])
            v1 = v1.to(device)
            v1(tmp)

    def test_PoolingWindows_plotting(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        pw = po.simul.PoolingWindows(.8, im.shape, num_scales=2)
        pw = pw.to(device)
        pw.plot_window_areas()
        pw.plot_window_widths()
        for i in range(2):
            pw.plot_window_areas('pixels', i)
            pw.plot_window_widths('pixels', i)
        fig = pt.imshow(po.to_numpy(im))
        pw.plot_windows(fig.axes[0])

    def test_PoolingWindows_caching(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device)
        # first time we save, second we load
        pw = po.simul.PoolingWindows(.8, im.shape, num_scales=2, cache_dir=tmp_path)
        pw = po.simul.PoolingWindows(.8, im.shape, num_scales=2, cache_dir=tmp_path)

    def test_PoolingWindows_parallel(self, tmp_path):
        if torch.cuda.device_count() > 1:
            devices = list(range(torch.cuda.device_count()))
            im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
            im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
            pw = po.simul.pooling.PoolingWindows(.5, im.shape[2:])
            pw = pw.parallel(devices)
            pw(im)
            pw = po.simul.pooling.PoolingWindows(.5, im.shape[2:], num_scales=3)
            pw = pw.parallel(devices)
            pw(im)
            pw = po.simul.pooling.PoolingWindows(.5, im.shape[2:], transition_region_width=1)
            pw = pw.parallel(devices)
            pw(im)
            for sh in [(256, 128), (256, 127), (256, 125), (125, 125), (127, 125)]:
                tmp = im[:sh[0], :sh[1]]
                rgc = po.simul.RetinalGanglionCells(.9, tmp.shape[2:])
                rgc = rgc.parallel(devices)
                rgc(tmp)
                v1 = po.simul.RetinalGanglionCells(.9, tmp.shape[2:])
                v1 = v1.parallel(devices)
                v1(tmp)
            pw = po.simul.PoolingWindows(.8, im.shape[2:], num_scales=2)
            pw = pw.parallel(devices)
            pw.plot_window_areas()
            pw.plot_window_widths()
            for i in range(2):
                pw.plot_window_areas('pixels', i)
                pw.plot_window_widths('pixels', i)
            fig = pt.imshow(po.to_numpy(im).squeeze())
            pw.plot_windows(fig.axes[0])
            pw = po.simul.pooling.PoolingWindows(.5, im.shape[2:])
            pw = pw.parallel(devices)
            pooled = pw(im)
            pw.project(pooled)
            pw = po.simul.pooling.PoolingWindows(.5, im.shape[2:], num_scales=3)
            pw = pw.parallel(devices)
            pooled = pw(im)
            pw.project(pooled)

    def test_PoolingWindows_sep(self):
        # test the window and pool function separate of the forward function
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        pw = po.simul.pooling.PoolingWindows(.5, im.shape[2:])
        pw.pool(pw.window(im))

# class TestSpectral(object):
#


class TestVentralStream(object):

    def test_rgc(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:])
        rgc = rgc.to(device)
        rgc(im)
        _ = rgc.plot_window_widths('degrees')
        _ = rgc.plot_window_widths('degrees', jitter=0)
        _ = rgc.plot_window_widths('pixels')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('pixels')
        fig = pt.imshow(po.to_numpy(im).squeeze())
        _ = rgc.plot_windows(fig.axes[0])
        rgc.plot_representation()
        rgc.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(5, 12))
        rgc.plot_representation(ax=axes[1])
        rgc.plot_representation_image(ax=axes[0])

    def test_rgc_2(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:], transition_region_width=1)
        rgc = rgc.to(device)
        rgc(im)
        _ = rgc.plot_window_widths('degrees')
        _ = rgc.plot_window_widths('degrees', jitter=0)
        _ = rgc.plot_window_widths('pixels')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('pixels')
        fig = pt.imshow(po.to_numpy(im).squeeze())
        _ = rgc.plot_windows(fig.axes[0])
        rgc.plot_representation()
        rgc.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(5, 12))
        rgc.plot_representation(ax=axes[1])
        rgc.plot_representation_image(ax=axes[0])

    def test_rgc_metamer(self):
        # literally just testing that it runs
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:])
        rgc = rgc.to(device)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3)
        assert not torch.isnan(metamer.matched_image).any(), "There's a NaN here!"

    def test_rgc_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        # first time we cache the windows...
        rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:], cache_dir=tmp_path)
        rgc = rgc.to(device)
        rgc(im)
        rgc.save_reduced(op.join(tmp_path, 'test_rgc_save_load.pt'))
        rgc_copy = po.simul.RetinalGanglionCells.load_reduced(op.join(tmp_path,
                                                                      'test_rgc_save_load.pt'))
        rgc_copy = rgc_copy.to(device)
        if not len(rgc.PoolingWindows.angle_windows) == len(rgc_copy.PoolingWindows.angle_windows):
            raise Exception("Something went wrong saving and loading, the lists of angle windows"
                            " are not the same length!")
        if not len(rgc.PoolingWindows.ecc_windows) == len(rgc_copy.PoolingWindows.ecc_windows):
            raise Exception("Something went wrong saving and loading, the lists of ecc windows"
                            " are not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(rgc.PoolingWindows.angle_windows)):
            if not rgc.PoolingWindows.angle_windows[i].allclose(rgc_copy.PoolingWindows.angle_windows[i]):
                raise Exception("Something went wrong saving and loading, the angle_windows %d are"
                                " not identical!" % i)
        for i in range(len(rgc.PoolingWindows.ecc_windows)):
            if not rgc.PoolingWindows.ecc_windows[i].allclose(rgc_copy.PoolingWindows.ecc_windows[i]):
                raise Exception("Something went wrong saving and loading, the ecc_windows %d are"
                                " not identical!" % i)
        # ...second time we load them
        rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:], cache_dir=tmp_path)

    def test_rgc_parallel(self):
        if torch.cuda.device_count() > 1:
            devices = list(range(torch.cuda.device_count()))
            im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
            im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
            rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:])
            rgc = rgc.parallel(devices)
            metamer = po.synth.Metamer(im, rgc)
            metamer.synthesize(max_iter=3)
            rgc.plot_representation()
            rgc.plot_representation_image()
            metamer.plot_representation_error()

    def test_frontend(self):
        im = po.make_basic_stimuli()
        frontend = po.simul.Front_End()
        frontend(im)

    def test_frontend_plot(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        frontend = po.simul.Front_End()
        po.tools.display.plot_representation(data=frontend(im), figsize=(11, 5))
        metamer = po.synth.Metamer(im, frontend)
        metamer.synthesize(max_iter=3, store_progress=1)
        metamer.plot_metamer_status(figsize=(35, 5))
        metamer.animate(figsize=(35, 5))

    def test_frontend_PoolingWindows(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        frontend = po.simul.Front_End()
        pw = po.simul.PoolingWindows(.5, (256, 256))
        pw(frontend(im))
        po.tools.display.plot_representation(data=pw(frontend(im)))

    def test_v1(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        v1 = v1.to(device)
        v1(im)
        _ = v1.plot_window_widths('pixels')
        _ = v1.plot_window_areas('pixels')
        for i in range(v1.num_scales):
            _ = v1.plot_window_widths('pixels', i)
            _ = v1.plot_window_areas('pixels', i)
        v1.plot_representation()
        v1.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(27, 12))
        v1.plot_representation(ax=axes[1])
        v1.plot_representation_image(ax=axes[0])

    def test_v1_norm(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        stats = po.simul.non_linearities.generate_norm_stats(v1, DATA_DIR, img_shape=(256, 256))
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:], normalize_dict=stats)
        v1 = v1.to(device)
        v1(im)
        _ = v1.plot_window_widths('pixels')
        _ = v1.plot_window_areas('pixels')
        for i in range(v1.num_scales):
            _ = v1.plot_window_widths('pixels', i)
            _ = v1.plot_window_areas('pixels', i)
        v1.plot_representation()
        v1.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(27, 12))
        v1.plot_representation(ax=axes[1])
        v1.plot_representation_image(ax=axes[0])

    def test_v1_parallel(self):
        if torch.cuda.device_count() > 1:
            devices = list(range(torch.cuda.device_count()))
            im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
            im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
            v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:]).to(device)
            v1 = v1.parallel(devices)
            metamer = po.synth.Metamer(im, v1)
            metamer.synthesize(max_iter=3)
            v1.plot_representation()
            v1.plot_representation_image()
            metamer.plot_representation_error()

    def test_v1_2(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:], transition_region_width=1)
        v1 = v1.to(device)
        v1(im)
        _ = v1.plot_window_widths('pixels')
        _ = v1.plot_window_areas('pixels')
        for i in range(v1.num_scales):
            _ = v1.plot_window_widths('pixels', i)
            _ = v1.plot_window_areas('pixels', i)
        v1.plot_representation()
        v1.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(27, 12))
        v1.plot_representation(ax=axes[1])
        v1.plot_representation_image(ax=axes[0])

    def test_v1_mean_luminance(self):
        for fname in ['nuts', 'einstein']:
            im = plt.imread(op.join(DATA_DIR, fname+'.pgm'))
            im = torch.tensor(im, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
            v1 = v1.to(device)
            v1_rep = v1(im)
            rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:])
            rgc = rgc.to(device)
            rgc_rep = rgc(im)
            if not torch.allclose(rgc.representation, v1.mean_luminance):
                raise Exception("Somehow RGC and V1 mean luminance representations are not the "
                                "same for image %s!" % fname)
            if not torch.allclose(rgc_rep, v1_rep[..., -rgc_rep.shape[-1]:]):
                raise Exception("Somehow V1's representation does not have the mean luminance "
                                "in the location expected! for image %s!" % fname)

    def test_v1_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        # first time we cache the windows...
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:], cache_dir=tmp_path)
        v1 = v1.to(device)
        v1(im)
        v1.save_reduced(op.join(tmp_path, 'test_v1_save_load.pt'))
        v1_copy = po.simul.PrimaryVisualCortex.load_reduced(op.join(tmp_path,
                                                                    'test_v1_save_load.pt'))
        v1_copy = v1_copy.to(device)
        if not len(v1.PoolingWindows.angle_windows) == len(v1_copy.PoolingWindows.angle_windows):
            raise Exception("Something went wrong saving and loading, the lists of angle windows"
                            " are not the same length!")
        if not len(v1.PoolingWindows.ecc_windows) == len(v1_copy.PoolingWindows.ecc_windows):
            raise Exception("Something went wrong saving and loading, the lists of ecc windows"
                            " are not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(v1.PoolingWindows.angle_windows)):
            if not v1.PoolingWindows.angle_windows[i].allclose(v1_copy.PoolingWindows.angle_windows[i]):
                raise Exception("Something went wrong saving and loading, the angle_windows %d are"
                                " not identical!" % i)
        for i in range(len(v1.PoolingWindows.ecc_windows)):
            if not v1.PoolingWindows.ecc_windows[i].allclose(v1_copy.PoolingWindows.ecc_windows[i]):
                raise Exception("Something went wrong saving and loading, the ecc_windows %d are"
                                " not identical!" % i)
        # ...second time we load them
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:], cache_dir=tmp_path)

    def test_v1_metamer(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        v1 = v1.to(device)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3)

    def test_cone_nonlinear(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1_lin = po.simul.PrimaryVisualCortex(1, im.shape[2:], cone_power=1)
        v1 = po.simul.PrimaryVisualCortex(1, im.shape[2:], cone_power=1/3)
        rgc_lin = po.simul.RetinalGanglionCells(1, im.shape[2:], cone_power=1)
        rgc = po.simul.RetinalGanglionCells(1, im.shape[2:], cone_power=1/3)
        for model in [v1, v1_lin, rgc, rgc_lin]:
            model(im)
        # v1 mean luminance and rgc representation, for same cone power
        # and scaling, should be identical
        (v1.representation['mean_luminance'] == rgc.representation).all()
        # v1 mean luminance and rgc representation, for same cone power
        # and scaling, should be identical
        (v1_lin.representation['mean_luminance'] == rgc_lin.representation).all()
        # similarly, the representations should be different if cone
        # power is different
        (v1_lin.representation['mean_luminance'] != v1.representation['mean_luminance']).all()
        # similarly, the representations should be different if cone
        # power is different
        (rgc_lin.representation != rgc.representation).all()

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
        v2 = v2.to(device)
        metamer = po.synth.Metamer(im, v2)
        metamer.synthesize(max_iter=3)


class TestMetamers(object):

    def test_metamer_save_load(self, tmp_path):

        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        v1 = v1.to(device)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.save(op.join(tmp_path, 'test_metamer_save_load.pt'))
        met_copy = po.synth.Metamer.load(op.join(tmp_path, "test_metamer_save_load.pt"),
                                         map_location=device)
        for k in ['target_image', 'saved_representation', 'saved_image', 'matched_representation',
                  'matched_image', 'target_representation']:
            if not getattr(metamer, k).allclose(getattr(met_copy, k)):
                raise Exception("Something went wrong with saving and loading! %s not the same"
                                % k)

    def test_metamer_save_load_reduced(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        v1 = v1.to(device)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.save(op.join(tmp_path, 'test_metamer_save_load_reduced.pt'), True)
        with pytest.raises(Exception):
            met_copy = po.synth.Metamer.load(op.join(tmp_path,
                                                     "test_metamer_save_load_reduced.pt"))
        met_copy = po.synth.Metamer.load(op.join(tmp_path, 'test_metamer_save_load_reduced.pt'),
                                         po.simul.PrimaryVisualCortex.from_state_dict_reduced,
                                         map_location=device)
        for k in ['target_image', 'saved_representation', 'saved_image', 'matched_representation',
                  'matched_image', 'target_representation']:
            if not getattr(metamer, k).allclose(getattr(met_copy, k)):
                raise Exception("Something went wrong with saving and loading! %s not the same" % k)

    def test_metamer_store_rep(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        v1 = v1.to(device)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=2)

    def test_metamer_store_rep_2(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        v1 = v1.to(device)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=True)

    def test_metamer_store_rep_3(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        v1 = v1.to(device)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=6, store_progress=3)

    def test_metamer_store_rep_4(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        v1 = v1.to(device)
        metamer = po.synth.Metamer(im, v1)
        with pytest.raises(Exception):
            metamer.synthesize(max_iter=3, store_progress=False, save_progress=True)

    def test_metamer_plotting_v1(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        v1 = v1.to(device)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=6, store_progress=True)
        metamer.plot_representation_error()
        metamer.model.plot_representation_image(data=metamer.representation_error())
        metamer.plot_metamer_status()
        metamer.plot_metamer_status(iteration=1)

    def test_metamer_plotting_rgc(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:])
        rgc = rgc.to(device)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=6, store_progress=True)
        metamer.plot_representation_error()
        metamer.model.plot_representation_image(data=metamer.representation_error())
        metamer.plot_metamer_status()
        metamer.plot_metamer_status(iteration=1)

    def test_metamer_continue(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:])
        rgc = rgc.to(device)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.synthesize(max_iter=3, initial_image=metamer.matched_image.detach().clone())

    def test_metamer_animate(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:])
        rgc = rgc.to(device)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3, store_progress=True)
        # this will test several related functions for us:
        # plot_metamer_status, plot_representation_error,
        # representation_error
        metamer.animate(figsize=(17, 5), plot_representation_error=True, ylim='rescale100',
                        framerate=40)

    def test_metamer_save_progress(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        v1 = v1.to(device)
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

    def test_metamer_loss_change(self):
        # literally just testing that it runs
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:])
        rgc = rgc.to(device)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=1,
                           loss_change_fraction=.5)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=1,
                           loss_change_fraction=.5, fraction_removed=.1)

    def test_metamer_coarse_to_fine(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape[2:])
        v1 = v1.to(device)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=True)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=True, fraction_removed=.1)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=True, loss_change_fraction=.5)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=True, loss_change_fraction=.5, fraction_removed=.1)

    @pytest.mark.parametrize("clamper", [po.RangeClamper((0, 1)), po.RangeRemapper((0, 1)),
                                         'clamp2', 'clamp4'])
    @pytest.mark.parametrize("clamp_each_iter", [True, False])
    @pytest.mark.parametrize("cone_power", [1, 1/3])
    def test_metamer_clamper(self, clamper, clamp_each_iter, cone_power):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        if type(clamper) == str and clamper == 'clamp2':
            clamper = po.TwoMomentsClamper(im)
        elif type(clamper) == str and clamper == 'clamp4':
            clamper = po.FourMomentsClamper(im)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:], cone_power=cone_power)
        rgc = rgc.to(device)
        metamer = po.synth.Metamer(im, rgc)
        if cone_power == 1/3 and not clamp_each_iter:
            # these will fail because we'll end up outside the 0, 1 range
            with pytest.raises(IndexError):
                metamer.synthesize(max_iter=3, clamper=clamper, clamp_each_iter=clamp_each_iter)
        else:
            metamer.synthesize(max_iter=3, clamper=clamper, clamp_each_iter=clamp_each_iter)

    def test_metamer_no_clamper(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape[2:], cone_power=1)
        rgc = rgc.to(device)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3, clamper=None)


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
