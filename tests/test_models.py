#!/usr/bin/env python3
import plenoptic as po
<<<<<<< HEAD
import matplotlib.pyplot as plt
import pytest
import numpy as np
import pyrtools as pt
import scipy.io as sio

from test_plenoptic import DEVICE, DATA_DIR, DTYPE, osf_download

class TestLinear(object):

    def test_linear(self, basic_stim):
        model = po.simul.Linear()
        assert model(basic_stim).requires_grad

    def test_linear_metamer(self, einstein_img):
        model = po.simul.Linear()
        M = po.synth.Metamer(einstein_img, model)
        M.synthesize(max_iter=3, learning_rate=1, seed=1)


class TestLinearNonlinear(object):

    def test_linear_nonlinear(self, basic_stim):
        model = po.simul.Linear_Nonlinear()
        assert model(basic_stim).requires_grad

    def test_linear_nonlinear_metamer(self, einstein_img):
        model = po.simul.Linear_Nonlinear()
        M = po.synth.Metamer(einstein_img, model)
        M.synthesize(max_iter=3, learning_rate=1, seed=0)


class TestLaplacianPyramid(object):

    def test_grad(self, basic_stim):
        L = po.simul.Laplacian_Pyramid()
        y = L.analysis(basic_stim)
        assert y[0].requires_grad


class TestPortillaSimoncelli(object):
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", [3, 5, 7, 9])
    @pytest.mark.parametrize("im_shape", [(256, 256)])
    @pytest.mark.parametrize("use_true_correlations", [True, False])
    def test_portilla_simoncelli(
        self,
        n_scales,
        n_orientations,
        spatial_corr_width,
        im_shape,
        use_true_correlations,
    ):
        x = po.make_basic_stimuli()
        if im_shape is not None:
            x = x[0, 0, : im_shape[0], : im_shape[1]]
        ps = po.simul.PortillaSimoncelli(
            x.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
            use_true_correlations=use_true_correlations,
        )
        ps(x)

    ## tests for whether output matches the original matlab output.  This implicitly tests that Portilla_simoncelli.forward() returns an object of the correct size.
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", [3, 5, 7, 9])
    @pytest.mark.parametrize("im_shape", [(256, 256)])
    @pytest.mark.parametrize("im", ["curie", "einstein", "metal", "nuts"])
    def test_ps_torch_v_matlab(
        self, n_scales, n_orientations, spatial_corr_width, im_shape, im
    ):
        path = osf_download("portilla_simoncelli_matlab_test_vectors.tar.gz")

        torch.set_default_dtype(torch.float64)
        x = plt.imread(op.join(DATA_DIR, f"{im}.pgm")).copy()
        im0 = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
        ps = po.simul.PortillaSimoncelli(
            x.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
            use_true_correlations=False,
        )
        python_vector = ps(im0)

        matlab = sio.loadmat(
            f"{path}/{im}-scales{n_scales}-ori{n_orientations}-spat{spatial_corr_width}.mat"
        )
        matlab_vector = matlab["params_vector"].flatten()

        np.testing.assert_allclose(
            python_vector.squeeze(), matlab_vector.squeeze(), rtol=1e-4, atol=1e-4
        )

    ## tests for whether output matches the saved python output.  This implicitly tests that Portilla_simoncelli.forward() returns an object of the correct size.
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", [3, 5, 7, 9])
    @pytest.mark.parametrize("use_true_correlations", [False, True])
    @pytest.mark.parametrize("im", ["curie", "einstein", "metal", "nuts"])
    def test_ps_torch_output(
        self, n_scales, n_orientations, spatial_corr_width, im, use_true_correlations
    ):
        path = osf_download("portilla_simoncelli_test_vectors.tar.gz")

        print(path)

        torch.set_default_dtype(torch.float64)
        x = plt.imread(op.join(DATA_DIR, f"{im}.pgm")).copy() / 255
        im0 = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
        ps = po.simul.PortillaSimoncelli(
            x.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
            use_true_correlations=use_true_correlations,
        )
        output = ps(im0)

        saved = np.load(
            f"{path}/{im}-scales{n_scales}-ori{n_orientations}-spat{spatial_corr_width}-corr{use_true_correlations}.npy"
        )

        np.testing.assert_allclose(
            output.squeeze(), saved.squeeze(), rtol=1e-5, atol=1e-5
        )

    def test_ps_synthesis(self):
        path = osf_download("portilla_simoncelli_synthesize.npy")

        torch.set_default_dtype(torch.float64)
        with open(path, 'rb') as f:
            im = np.load(f)
            im_init = np.load(f)
            im_synth = np.load(f)
            loss = np.load(f)

        n=256
        im0 = torch.Tensor(im)
        model = po.simul.PortillaSimoncelli(
            [n,n],
            n_scales=4, 
            n_orientations=4, 
            spatial_corr_width=9,
            use_true_correlations=True)

        
        met = po.synth.Metamer(im0, model)

        output=met.synthesize(
            learning_rate=.01,
            seed=1,
            loss_change_thresh=None,
            loss_change_iter=7,
            max_iter=10,
            coarse_to_fine='together',
            optimizer='Adam',
            initial_image = im_init)       

        np.testing.assert_allclose(
            output[0].squeeze().detach().numpy(), im_synth.squeeze(), rtol=1e-2, atol=1e-2
        )

        np.testing.assert_allclose(
            output[1].squeeze().detach().numpy(), loss.squeeze(), rtol=1e-4, atol=1e-4
        )
