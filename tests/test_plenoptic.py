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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
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

<<<<<<< HEAD
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
        np.testing.assert_allclose(coeff_torch_k, coeff_np_k, rtol=rtol, atol=atol)

def check_band_energies(coeff_1, coeff_2, rtol=1e-4, atol=1e-4):
    '''
    function that checks if the energy in each band of two pyramids are the same.
    We set an absolute and relative tolerance and the function checks for each band if
    abs(coeff_1-coeff_2) <= atol + rtol*abs(coeff_1)
    Args:
    coeff_1: first dictionary of torch tensors corresponding to each band
    coeff_2: second dictionary of torch tensors corresponding to each band
    '''

    for i in range(len(coeff_1.items())):
        k1 = list(coeff_1.keys())[i]
        k2 = list(coeff_2.keys())[i]
        band_1 = coeff_1[k1].squeeze()
        band_2 = coeff_2[k2].squeeze()
        if band_1.shape[-1] == 2:
            band_1 = torch_complex_to_numpy(band_1)
            band_2 = torch_complex_to_numpy(band_2)
        else:
            band_1 = to_numpy(band_1)
            band_2 = to_numpy(band_2)

        np.testing.assert_allclose(np.sum(np.abs(band_1)**2),np.sum(np.abs(band_2)**2), rtol=rtol, atol=atol)

def check_parseval(im ,coeff, rtol=1e-4, atol=0):
    '''
    function that checks if the pyramid is parseval, i.e. energy of coeffs is
    the same as the energy in the original image.
    Args:
    input image: image stimulus as torch.Tensor
    coeff: dictionary of torch tensors corresponding to each band
    '''
    total_band_energy = 0
    im_energy = np.sum(to_numpy(im)**2)
    for k,v in coeff.items():
        band = coeff[k].squeeze()
        if band.shape[-1] == 2:
            band = torch_complex_to_numpy(band)
        else:
            band = to_numpy(band)

        total_band_energy += np.sum(np.abs(band)**2)

    np.testing.assert_allclose(total_band_energy, im_energy, rtol=rtol, atol=atol)

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

class TestNonLinearities(object):

    def test_polar_amplitude_zero(self):
        a = torch.rand(10)*-1
        b = po.rescale(torch.randn(10), -np.pi / 2, np.pi / 2)

    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [1, 2, 3])
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

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("height", ['auto', 1, 2, 3])
    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("downsample", [False, True])
    @pytest.mark.parametrize('is_complex', [True, False])
    @pytest.mark.parametrize("im_shape", [None, (224,224),(256, 128), (128, 256)])
    def test_tight_frame(self, im, height, order, is_complex, downsample, im_shape):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]

        im = im / 255
        im = torch.tensor(im, dtype=dtype).unsqueeze(0).unsqueeze(0)

        pyr = po.simul.Steerable_Pyramid_Freq(im.shape[-2:], height, order, is_complex=is_complex, downsample=downsample, tight_frame = True)
        pyr.forward(im)
        check_parseval(im, pyr.pyr_coeffs)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("height", [3,4,5])
    @pytest.mark.parametrize("order", [1,2,3])
    @pytest.mark.parametrize("is_complex", [False, True])
    @pytest.mark.parametrize("im_shape", [None, (224,224),(256, 128), (128, 256)])
    def test_not_downsample(self, im, height, order, is_complex, im_shape):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]

        im = im / 255
        sp_downsample = po.simul.Steerable_Pyramid_Freq(image_shape = im.shape, height = height, order = order,
                                                        is_complex = is_complex, downsample = False, tight_frame=True)
        sp_notdownsample = po.simul.Steerable_Pyramid_Freq(image_shape = im.shape, height = height, order = order,
                                                            is_complex = is_complex, downsample = True, tight_frame=True)
        sp_downsample.to(device)
        sp_notdownsample.to(device)

        im_t = torch.tensor(im, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)
        sp_downsample.forward(im_t)
        sp_notdownsample.forward(im_t)

        check_band_energies(sp_notdownsample.pyr_coeffs, sp_downsample.pyr_coeffs)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("height", [3,4,5])
    @pytest.mark.parametrize("order", [1,2,3])
    @pytest.mark.parametrize("is_complex", [False, True])
    @pytest.mark.parametrize("im_shape", [None, (224,224),(256, 128), (128, 256)])
    @pytest.mark.parametrize("scales", [[0], [1], [0, 1, 2], [2], [], ['residual_highpass', 'residual_lowpass'],
                                        ['residual_highpass', 0, 1, 'residual_lowpass']])
    def test_pyr_to_tensor(self, im, height, order, is_complex, im_shape, scales, rtol=1e-12, atol=1e-12):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]

        im = im / 255
        sp_notdownsample = po.simul.Steerable_Pyramid_Freq(image_shape = im.shape, height = height, order = order,
                                                                is_complex = is_complex, downsample = False)
        sp_notdownsample.to(device)
        im_t = torch.tensor(im, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)

        pyr_tensor = sp_notdownsample.forward(im_t, scales = scales)
        pyr_coeff_dict = sp_notdownsample.convert_tensor_to_pyr(pyr_tensor)
        for i in range(len(pyr_coeff_dict.keys())):
            k1 = list(pyr_coeff_dict.keys())[i]
            k2 = list(sp_notdownsample.pyr_coeffs.keys())[i]
            np.testing.assert_allclose(to_numpy(pyr_coeff_dict[k1]), to_numpy(sp_notdownsample.pyr_coeffs[k2]), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("height", [3,4,5])
    @pytest.mark.parametrize("order", [1,2,3])
    @pytest.mark.parametrize("is_complex", [False, True])
    def test_torch_vs_numpy_pyr(self, height, order, is_complex):
        x = plt.imread(op.join(DATA_DIR, 'curie.pgm'))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex, tight_frame=False,downsample=True)
        torch_sp.to(device)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)

        #Check with non-square image
        x = pt.synthetic_images.ramp((256,128))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex, tight_frame = False, downsample = True)
        torch_sp.to(device)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)

        #check non-powers-of-2 images
        x = pt.synthetic_images.ramp((200,200))
        x_shape = x.shape
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(x,height=height, order = order, is_complex=is_complex)
        x_t = torch.tensor(x, dtype = dtype).unsqueeze(0).unsqueeze(0).to(device)
        torch_sp = po.simul.Steerable_Pyramid_Freq(image_shape = x.shape, height = height, order = order, is_complex = is_complex, tight_frame = False, downsample = True)
        torch_sp.to(device)
        torch_spc = torch_sp.forward(x_t)
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)


    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("tight_frame", [True, False])
    @pytest.mark.parametrize("downsample", [False, True])
    @pytest.mark.parametrize("height", ['auto', 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("im_shape", [None, (224,224),(256, 128), (128, 256)])
    def test_complete_recon(self, im, is_complex, tight_frame, downsample, height, order, im_shape):
        print(im,is_complex, tight_frame, downsample, height, order, im_shape)

        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im = torch.tensor(im, dtype=dtype).unsqueeze(0).unsqueeze(0)
        pyr = po.simul.Steerable_Pyramid_Freq(im.shape[-2:], height, order, is_complex=is_complex, downsample=downsample, tight_frame = tight_frame)
        pyr.forward(im)
        recon = to_numpy(pyr.recon_pyr())
        np.testing.assert_allclose(recon, im.data.cpu().numpy(), rtol=1e-4, atol=1e-4)


    @pytest.mark.parametrize("im", ['einstein','curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("tight_frame", [True, False])
    @pytest.mark.parametrize("downsample", [False, True])
    @pytest.mark.parametrize("height", ['auto'])
    @pytest.mark.parametrize("order", [3])
    @pytest.mark.parametrize("im_shape", [None, (224,224), (256, 128), (128,256)])
    def test_partial_recon(self, im, is_complex, tight_frame, downsample, height, order, im_shape):
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im_tensor = torch.tensor(im, dtype=dtype).unsqueeze(0).unsqueeze(0)
        po_pyr = po.simul.Steerable_Pyramid_Freq(im.shape, height, order, is_complex=is_complex, downsample=downsample, tight_frame=tight_frame)
        po_pyr.forward(im_tensor)
        pt_pyr = pt.pyramids.SteerablePyramidFreq(im, height, order, is_complex=is_complex)

        recon_levels = [[0], [1,3], [1,3,4]]
        #for i in range(po_pyr.num_scales):
        #    recon_levels.extend(list(itertools.combinations(range(po_pyr.num_scales), i)))
        recon_bands = [[1],[1,3]]
        #for i in range(po_pyr.num_orientations):
        #    recon_bands.extend(list(itertools.combinations(range(po_pyr.num_orientations), i)))
        for levels, bands in itertools.product(['all'] + recon_levels, ['all'] + recon_bands):
            po_recon = po.to_numpy(po_pyr.recon_pyr(levels, bands).squeeze())
            pt_recon = pt_pyr.recon_pyr(levels, bands)
            np.testing.assert_allclose(po_recon, pt_recon,rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("im", ['einstein', 'curie'])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("height", ['auto', 1, 3, 4])
    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("im_shape", [None, (224,224),  (256, 128), (128, 256)])
    def test_recon_match_pyrtools(self, im, is_complex, height, order, im_shape):
        # this should fail if and only if test_complete_recon does, but
        # may as well include it just in case
        im = plt.imread(op.join(DATA_DIR, '%s.pgm' % im))
        if im_shape is not None:
            im = im[:im_shape[0], :im_shape[1]]
        im = im / 255
        im_tensor = torch.tensor(im, dtype=dtype).unsqueeze(0).unsqueeze(0)
        po_pyr = po.simul.Steerable_Pyramid_Freq(im.shape, height, order, is_complex=is_complex, tight_frame=False)
        po_pyr.forward(im_tensor)
        pt_pyr = pt.pyramids.SteerablePyramidFreq(im, height, order, is_complex=is_complex)
        po_recon = po.to_numpy(po_pyr.recon_pyr())
        pt_recon = pt_pyr.recon_pyr()
        np.testing.assert_allclose(po_recon, pt_recon)

    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("downsample", [True, False])
    @pytest.mark.parametrize("scales", [[0], [5], [0, 1, 2], [0, 3, 5],
                                        ['residual_highpass', 'residual_lowpass'],
                                        ['residual_highpass', 0, 1, 'residual_lowpass']])
    def test_scales_arg(self, is_complex, downsample, scales):
        img = imageio.imread(op.join(DATA_DIR, 'einstein.pgm'))
        img = torch.tensor(img / 255, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pyr = po.simul.Steerable_Pyramid_Freq(img.shape[-2:], is_complex=is_complex, downsample=downsample)
        pyr.forward(img)
        pyr_coeffs = pyr.pyr_coeffs.copy()
        pyr.forward(img, scales)
        reduced_pyr_coeffs = pyr.pyr_coeffs.copy()
        for k, v in reduced_pyr_coeffs.items():
            if (v != pyr_coeffs[k]).any():
                raise Exception("Reduced pyr_coeffs should be same as original, but at least key "
                                f"{k} is not")

        # recon_pyr should always fail
        with pytest.raises(Exception):
            pyr.recon_pyr()
        with pytest.raises(Exception):
            pyr.recon_pyr(scales)
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
