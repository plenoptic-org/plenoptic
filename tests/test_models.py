# /usr/bin/env python3
# we do this to enable deterministic behavior on the gpu, see
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility for
# details
from conftest import DEVICE, DATA_DIR
from test_metric import osf_download
import os.path as op
import scipy.io as sio
import pyrtools as pt
from plenoptic.simulate.canonical_computations import gaussian1d, circular_gaussian2d
import plenoptic as po
import torch
import numpy as np
import pytest
import matplotlib.pyplot as plt
from packaging import version
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


@pytest.fixture()
def image_input():
    return torch.rand(1, 1, 100, 100)


@pytest.fixture()
def portilla_simoncelli_matlab_test_vectors():
    return osf_download('portilla_simoncelli_matlab_test_vectors.tar.gz')


@pytest.fixture()
def portilla_simoncelli_test_vectors():
    return osf_download('portilla_simoncelli_test_vectors.tar.gz')


def get_portilla_simoncelli_synthesize_filename(torch_version=None):
    """Helper function to get pathname.

    We can't call fixtures directly (feature removed in pytest 4.0), so we use
    this helper function to get the name, which we use in
    tests/utils.update_ps_synthesis_test_file()

    """
    if torch_version is None:
        # the bit after the + defines the CUDA version used (if any), which
        # doesn't appear to be relevant for this.
        torch_version = torch.__version__.split('+')[0]
    # following https://stackoverflow.com/a/11887885 for how to compare version
    # strings
    if version.parse(torch_version) < version.parse('1.12') or DEVICE.type == 'cuda':
        torch_version = ''
    # going from 1.11 to 1.12 only changes this synthesis output on cpu, not
    # gpu
    else:
        torch_version = '_torch_v1.12.0'
    # synthesis gives differnet outputs on cpu vs gpu, so we have two different
    # versions to test against
    name_template = 'portilla_simoncelli_synthesize{gpu}{torch_version}.npz'
    if DEVICE.type == 'cpu':
        gpu = ''
    elif DEVICE.type == 'cuda':
        gpu = '_gpu'
    return name_template.format(gpu=gpu, torch_version=torch_version)


@pytest.fixture()
def portilla_simoncelli_synthesize(torch_version=None):
    return osf_download(get_portilla_simoncelli_synthesize_filename(torch_version))


@pytest.fixture()
def portilla_simoncelli_scales():
    return osf_download('portilla_simoncelli_scales.npz')


class TestNonLinearities(object):
    def test_rectangular_to_polar_dict(self, basic_stim):
        spc = po.simul.Steerable_Pyramid_Freq(basic_stim.shape[-2:], height=5,
                                              order=1, is_complex=True, tight_frame=True).to(DEVICE)
        y = spc(basic_stim)
        energy, state = po.simul.non_linearities.rectangular_to_polar_dict(y, residuals=True)
        y_hat = po.simul.non_linearities.polar_to_rectangular_dict(energy, state, residuals=True)
        for key in y.keys():
            assert torch.norm(y[key] - y_hat[key]) < 1e-5

    def test_local_gain_control(self):
        x = torch.randn((10, 1, 256, 256), device=DEVICE)
        norm, direction = po.simul.non_linearities.local_gain_control(x)
        x_hat = po.simul.non_linearities.local_gain_release(norm, direction)
        assert torch.norm(x - x_hat) < 1e-4

    def test_local_gain_control_dict(self, basic_stim):
        spr = po.simul.Steerable_Pyramid_Freq(basic_stim.shape[-2:], height=5,
                                              order=1, is_complex=False, tight_frame=True).to(DEVICE)
        y = spr(basic_stim)
        energy, state = po.simul.non_linearities.local_gain_control_dict(y, residuals=True)
        y_hat = po.simul.non_linearities.local_gain_release_dict(energy, state, residuals=True)
        for key in y.keys():
            assert torch.norm(y[key] - y_hat[key]) < 1e-5


class TestLaplacianPyramid(object):

    def test_grad(self, basic_stim):
        lpyr = po.simul.LaplacianPyramid().to(DEVICE)
        y = lpyr.forward(basic_stim)
        assert y[0].requires_grad

    @pytest.mark.parametrize("n_scales", [3, 4, 5, 6])
    def test_synthesis(self, curie_img, n_scales):
        img = curie_img[:, :, 0:253, 0:234]  # Original 256x256 shape is not good for testing padding
        lpyr = po.simul.LaplacianPyramid(n_scales=n_scales).to(DEVICE)
        y = lpyr.forward(img)
        img_recon = lpyr.recon_pyr(y)
        assert torch.allclose(img, img_recon)

    @pytest.mark.parametrize("n_scales", [3, 4, 5, 6])
    def test_match_pyrtools(self, curie_img, n_scales):
        img = curie_img[:, :, 0:253, 0:234]
        lpyr_po = po.simul.LaplacianPyramid(n_scales=n_scales).to(DEVICE)
        y_po = lpyr_po(img)
        lpyr_pt = pt.pyramids.LaplacianPyramid(img.squeeze().cpu(), height=n_scales)
        y_pt = [lpyr_pt.pyr_coeffs[(i, 0)] for i in range(n_scales)]
        assert len(y_po) == len(y_pt)
        for x_po, x_pt in zip(y_po, y_pt):
            x_po = x_po.squeeze().detach().cpu().numpy()
            assert np.abs(x_po - x_pt)[:-2, :-2].max() < 1e-5
            # The pyrtools implementation `pt.upConv performs`` padding after upsampling.
            # Our implementation `po.tools.upsample_convolve`` performs padding before upsampling,
            # and, depending on the parity of the image, sometimes performs additional zero padding
            # after upsampling up to one row/column. This causes inconsistency on the right and
            # bottom edges, so they are exluded in the comparison.


class TestFrontEnd:

    all_models = [
        "frontend.LinearNonlinear",
        "frontend.LuminanceGainControl",
        "frontend.LuminanceContrastGainControl",
        "frontend.OnOff",
    ]

    @pytest.mark.parametrize("model", all_models[:-1], indirect=True)
    def test_output_shape(self, model):
        img = torch.ones(1, 1, 100, 100).to(DEVICE)
        assert model(img).shape == img.shape

    @pytest.mark.parametrize("model", all_models, indirect=True)
    def test_gradient_flow(self, model):
        img = torch.ones(1, 1, 100, 100).to(DEVICE)
        y = model(img)
        assert y.requires_grad

    def test_onoff(self):
        mdl = po.simul.OnOff(7, pretrained=False).to(DEVICE)

    @pytest.mark.parametrize("kernel_size", [7, 31])
    @pytest.mark.parametrize("cache_filt", [False, True])
    def test_pretrained_onoff(self, kernel_size, cache_filt):
        if kernel_size != 31:
            with pytest.raises(AssertionError):
                mdl = po.simul.OnOff(kernel_size, pretrained=True, cache_filt=cache_filt).to(DEVICE)
        else:
            mdl = po.simul.OnOff(kernel_size, pretrained=True, cache_filt=cache_filt).to(DEVICE)

    @pytest.mark.parametrize("model", all_models, indirect=True)
    def test_frontend_display_filters(self, model):
        fig = model.display_filters()
        plt.close(fig)


class TestNaive(object):

    all_models = [
        "naive.Identity",
        "naive.Linear",
        "naive.Gaussian",
        "naive.CenterSurround",
    ]

    @pytest.mark.parametrize("model", all_models, indirect=True)
    def test_gradient_flow(self, model):
        img = torch.ones(1, 1, 100, 100).to(DEVICE).requires_grad_()
        y = model(img)
        assert y.requires_grad

    @pytest.mark.parametrize("mdl", ["naive.Gaussian", "naive.CenterSurround"])
    @pytest.mark.parametrize("cache_filt", [False, True])
    def test_cache_filt(self, cache_filt, mdl):
        img = torch.ones(1, 1, 100, 100).to(DEVICE).requires_grad_()
        if mdl == "naive.Gaussian":
            model = po.simul.Gaussian((31, 31), 1., cache_filt=cache_filt).to(DEVICE)
        elif mdl == "naive.CenterSurround":
            model = po.simul.CenterSurround((31, 31), cache_filt=cache_filt).to(DEVICE)

        y = model(img)  # forward pass should cache filt if True

        if cache_filt:
            assert model._filt is not None
        else:
            assert model._filt is None

    @pytest.mark.parametrize("center_std", [1., torch.tensor([1., 2.])])
    @pytest.mark.parametrize("out_channels", [1, 2, 3])
    @pytest.mark.parametrize("on_center", [True, [True, False]])
    def test_CenterSurround_channels(self, center_std, out_channels, on_center):
        if not isinstance(center_std, float) and len(center_std) != out_channels:
            with pytest.raises(AssertionError):
                model = po.simul.CenterSurround((31, 31), center_std=center_std, out_channels=out_channels)
        else:
            model = po.simul.CenterSurround((31, 31), center_std=center_std, out_channels=out_channels)

    def test_linear(self, basic_stim):
        model = po.simul.Linear().to(DEVICE)
        assert model(basic_stim).requires_grad

    def test_linear_metamer(self, einstein_img):
        model = po.simul.Linear().to(DEVICE)
        M = po.synth.Metamer(einstein_img, model)
        M.synthesize(max_iter=3)


class TestPortillaSimoncelli(object):
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", [3, 5, 7, 9])
    @pytest.mark.parametrize("use_true_correlations", [True, False])
    def test_portilla_simoncelli(
        self,
        n_scales,
        n_orientations,
        spatial_corr_width,
        use_true_correlations,
    ):
        im_shape = (256, 256)
        x = po.tools.make_synthetic_stimuli().to(DEVICE)
        x = x.unsqueeze(0).unsqueeze(0)
        if im_shape is not None:
            x = x[0, 0, : im_shape[0], : im_shape[1]]
        ps = po.simul.PortillaSimoncelli(
            x.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
            use_true_correlations=use_true_correlations,
        ).to(DEVICE)
        ps(x[0, :, :, :])

    # tests for whether output matches the original matlab output.  This implicitly tests that Portilla_simoncelli.forward() returns an object of the correct size.
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", [3, 5, 7, 9])
    @pytest.mark.parametrize("im_shape", [(256, 256)])
    @pytest.mark.parametrize("im", ["curie", "einstein", "metal", "nuts"])
    def test_ps_torch_v_matlab(self, n_scales, n_orientations,
                               spatial_corr_width, im_shape, im,
                               portilla_simoncelli_matlab_test_vectors):

        torch.set_default_dtype(torch.float64)
        x = plt.imread(op.join(DATA_DIR, f"256/{im}.pgm")).copy()
        im0 = torch.Tensor(x).unsqueeze(0).unsqueeze(0).to(DEVICE)
        ps = po.simul.PortillaSimoncelli(
            x.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
            use_true_correlations=False,
        ).to(DEVICE)
        python_vector = po.to_numpy(ps(im0))

        matlab = sio.loadmat(f"{portilla_simoncelli_matlab_test_vectors}/"
                             f"{im}-scales{n_scales}-ori{n_orientations}"
                             f"-spat{spatial_corr_width}.mat")
        matlab_vector = matlab["params_vector"].flatten()

        np.testing.assert_allclose(
            python_vector.squeeze(), matlab_vector.squeeze(), rtol=1e-4, atol=1e-4
        )

    # tests for whether output matches the saved python output.  This implicitly tests that Portilla_simoncelli.forward() returns an object of the correct size.
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", [3, 5, 7, 9])
    @pytest.mark.parametrize("use_true_correlations", [False, True])
    @pytest.mark.parametrize("im", ["curie", "einstein", "metal", "nuts"])
    def test_ps_torch_output(self, n_scales, n_orientations,
                             spatial_corr_width, im, use_true_correlations,
                             portilla_simoncelli_test_vectors):

        torch.set_default_dtype(torch.float64)
        x = plt.imread(op.join(DATA_DIR, f"256/{im}.pgm")).copy() / 255
        im0 = torch.Tensor(x).unsqueeze(0).unsqueeze(0).to(DEVICE)
        ps = po.simul.PortillaSimoncelli(
            x.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
            use_true_correlations=use_true_correlations,
        ).to(DEVICE)
        output = po.to_numpy(ps(im0))

        saved = np.load(f"{portilla_simoncelli_test_vectors}/"
                        f"{im}-scales{n_scales}-ori{n_orientations}-"
                        f"spat{spatial_corr_width}-corr{use_true_correlations}.npy")

        np.testing.assert_allclose(
            output.squeeze(), saved.squeeze(), rtol=1e-5, atol=1e-5
        )

    def test_ps_synthesis(self, portilla_simoncelli_synthesize,
                          run_test=True):
        """Test PS texture metamer synthesis.

        Parameters
        ----------
        portilla_simoncelli_synthesize : str
            Path to the .npz file to test against
        run_test : bool, optional
            If True, we run the test, comparing the current synthesis against
            the saved results. If False, we don't run the test, and return the
            Metamer object instead (used when updating the file to test
            against, as in tests/utils.update_ps_synthesis_test_file)

        """
        # this tests whether the output of metamer synthesis is consistent.
        # this is probably the most likely to fail as our requirements change,
        # because if something in how torch computes its gradients changes,
        # then our outputs will change. for example, in release 1.10
        # (https://github.com/pytorch/pytorch/releases/tag/v1.10.0), they fixed
        # the sub-gradient for torch.a{max,min}, which resulted in our PS
        # synthesis getting worse, somehow. this is just a note to keep an eye
        # on this; you might need to update the output to test against as
        # versions change. you probably only need to store the most recent
        # version, because that's what we test against.
        torch.use_deterministic_algorithms(True)
        torch.set_default_dtype(torch.float64)
        with np.load(portilla_simoncelli_synthesize) as f:
            im = f['im']
            im_init = f['im_init']
            im_synth = f['im_synth']
            rep_synth = f['rep_synth']

        im0 = torch.tensor(im).unsqueeze(0).unsqueeze(0).to(DEVICE)
        model = po.simul.PortillaSimoncelli(im0.shape[-2:],
                                            n_scales=4,
                                            n_orientations=4,
                                            spatial_corr_width=9,
                                            use_true_correlations=True).to(DEVICE)

        po.tools.set_seed(1)
        im_init = torch.tensor(im_init).unsqueeze(0).unsqueeze(0)
        met = po.synth.Metamer(im0, model, initial_image=im_init,
                               loss_function=po.tools.optim.l2_norm,
                               range_penalty_lambda=0)

        coarse_to_fine_kwargs = {'change_scale_criterion': None,
                                 'ctf_iters_to_check': 15}
        # this is the same as the default optimizer, but we explicitly
        # instantiate it anyway, in case we change the defaults at some point
        optim = torch.optim.Adam([met.synthesized_signal], lr=.01,
                                 amsgrad=True)
        output = met.synthesize(max_iter=200, optimizer=optim,
                                coarse_to_fine='together',
                                coarse_to_fine_kwargs=coarse_to_fine_kwargs)

        if run_test:
            np.testing.assert_allclose(
                po.to_numpy(output).squeeze(), im_synth.squeeze(), rtol=1e-4, atol=1e-4,
            )

            np.testing.assert_allclose(
                po.to_numpy(model(output)).squeeze(), rep_synth.squeeze(), rtol=1e-4, atol=1e-4
            )
        else:
            return met


    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", [3, 5, 7, 9])
    @pytest.mark.parametrize("use_true_correlations", [False, True])
    def test_portilla_simoncelli_scales(
        self,
        n_scales,
        n_orientations,
        spatial_corr_width,
        use_true_correlations,
        portilla_simoncelli_scales
    ):
        with np.load(portilla_simoncelli_scales) as f:
            key = f'scale_{n_scales}_ori_{n_orientations}_width_{spatial_corr_width}_corr_{use_true_correlations}'
            saved = f[key]

        model = po.simul.PortillaSimoncelli(
            [256, 256],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
            use_true_correlations=use_true_correlations).to(DEVICE)

        output = model._get_representation_scales()

        for (oo, ss) in zip(output, saved):
            if str(oo) != ss:
                raise ValueError("Scales do not match.")
            
    @pytest.mark.parametrize("im_shape", [(256,256), (128,256)])
    def test_ps_expand(self, im_shape):
        mult = 4
        
        im = po.tools.make_synthetic_stimuli().to(DEVICE)
        im = im[0,0,:im_shape[0], :im_shape[1]]
        out_im = po.simul.PortillaSimoncelli(im_shape).expand(im, mult)
        
        assert out_im.shape == (im_shape[0] * mult, im_shape[1] * mult)


class TestFilters:
    @pytest.mark.parametrize("std", [5., torch.tensor(1., device=DEVICE), -1., 0.])
    @pytest.mark.parametrize("kernel_size", [(31, 31), (3, 2), (7, 7), 5])
    @pytest.mark.parametrize("out_channels", [1, 3, 10])
    def test_circular_gaussian2d_shape(self, std, kernel_size, out_channels):
        if std <= 0.:
            with pytest.raises(AssertionError):
                circular_gaussian2d((7, 7), std)
        else:
            filt = circular_gaussian2d(kernel_size, std, out_channels)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            assert filt.shape == (out_channels, 1, *kernel_size)
            assert filt.sum().isclose(torch.ones(1, device=DEVICE) * out_channels)

    def test_circular_gaussian2d_wrong_std_length(self):
        std = torch.tensor([1., 2.], device=DEVICE)
        out_channels = 3
        with pytest.raises(AssertionError):
            circular_gaussian2d((7, 7), std, out_channels)

    @pytest.mark.parametrize("kernel_size", [5, 11, 20])
    @pytest.mark.parametrize("std", [1., 20., 0.])
    def test_gaussian1d(self, kernel_size, std):
        if std <= 0:
            with pytest.raises(AssertionError):
                gaussian1d(kernel_size, std)
        else:
            filt = gaussian1d(kernel_size, std)
            assert filt.sum().isclose(torch.ones(1))
            assert filt.shape == torch.Size([kernel_size])