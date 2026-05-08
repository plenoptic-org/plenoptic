import os

import einops
import numpy as np
import pytest
import torch

import plenoptic as po
from plenoptic.data import fetch_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if we have a second gpu, can use it for some tests
DEVICE2 = torch.device(1) if torch.cuda.device_count() > 1 else DEVICE
IMG_DIR = fetch_data("test_images.tar.gz")

# we do this to enable deterministic behavior on the gpu, for
# PortillaSimoncelli, see
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility for
# details
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# especially for using LBFGS (e.g., in the PortillaSimoncelli regression tests in
# test_uploaded_files.py), this speeds things up by reducing the number of threads used
# by OpenMP
os.environ["OMP_NUM_THREADS"] = "1"
# torch uses all avail threads which will slow tests
torch.set_num_threads(1)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


class ColorModel(torch.nn.Module):
    """Simple model that takes color image as input and outputs 2d conv."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, 1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture(scope="package")
def curie_img():
    return po.load_images(IMG_DIR / "256" / "curie.pgm").to(DEVICE)


@pytest.fixture(scope="package")
def einstein_img():
    return po.load_images(IMG_DIR / "256" / "einstein.pgm").to(DEVICE)


@pytest.fixture(scope="package")
def einstein_img_small(einstein_img):
    return po.process.center_crop(einstein_img, 64).to(DEVICE)


@pytest.fixture(scope="package")
def color_img():
    img = po.load_images(IMG_DIR / "256" / "color_wheel.jpg", as_gray=False).to(DEVICE)
    return img[..., :256, :256]


@pytest.fixture(scope="package")
def parrot_square():
    img = po.load_images(IMG_DIR / "mixed" / "Parrot.png").to(DEVICE)
    return po.process.center_crop(img, 254)


@pytest.fixture(scope="package")
def parrot_square_double(parrot_square):
    return parrot_square.to(torch.float64)


@pytest.fixture(scope="package")
def einstein_img_double(einstein_img):
    return einstein_img.to(torch.float64)


@pytest.fixture(scope="package")
def basic_stim():
    return po.load_images(IMG_DIR / "256").to(DEVICE)


def get_model(name):
    if name == "SPyr":
        # in order to get a tensor back, need to wrap steerable pyramid so that
        # we can call convert_pyr_to_tensor in the forward call. in order for
        # that to work, downsample must be False
        class spyr(po.process.SteerablePyramidFreq):
            def __init__(self, *args, **kwargs):
                kwargs.pop("downsample", None)
                super().__init__(*args, downsample=False, **kwargs)

            def forward(self, *args, **kwargs):
                coeffs = super().forward(*args, **kwargs)
                pyr_tensor, _ = po.process.SteerablePyramidFreq.convert_pyr_to_tensor(
                    coeffs
                )
                return pyr_tensor

        # setting height=1 and # order=1 limits the size
        return spyr((256, 256), height=1, order=1).to(DEVICE)
    elif name == "LPyr":
        # in order to get a tensor back, need to wrap laplacian pyramid so that
        # we can flatten the output. in practice, not the best way to use this
        class lpyr(po.process.LaplacianPyramid):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, *args, **kwargs):
                coeffs = super().forward(*args, **kwargs)
                return torch.cat([c.flatten(-2) for c in coeffs], -1)

        return lpyr().to(DEVICE)
    elif name == "nlpd":
        return po.metric.nlpd
    elif name == "mse":
        return po.metric.mse
    elif name == "ColorModel":
        model = ColorModel().to(DEVICE)
        po.remove_grad(model)
        model.eval()
        return model

    # naive models
    elif name == "naive.Identity":
        model = po.models.Identity().to(DEVICE)
        model.eval()
        return model
    elif name == "naive.CenterSurround":
        model = po.models.CenterSurround((31, 31)).to(DEVICE)
        model.eval()
        return model
    elif name == "naive.CenterSurround.nograd":
        model = po.models.CenterSurround((31, 31)).to(DEVICE)
        po.remove_grad(model)
        model.eval()
        return model
    elif name == "naive.Gaussian":
        model = po.models.Gaussian((31, 31)).to(DEVICE)
        model.eval()
        return model
    elif name == "naive.Gaussian.nograd":
        model = po.models.Gaussian((31, 31)).to(DEVICE)
        model.eval()
        po.remove_grad(model)
        return model
    elif name == "naive.Linear":
        model = po.models.Linear((31, 31)).to(DEVICE)
        model.eval()
        return model
    elif name == "naive.Linear.nograd":
        model = po.models.Linear((31, 31)).to(DEVICE)
        po.remove_grad(model)
        model.eval()
        return model

    # FrontEnd models:
    elif name == "frontend.LinearNonlinear":
        model = po.models.LinearNonlinear((31, 31)).to(DEVICE)
        model.eval()
        return model
    elif name == "frontend.LinearNonlinear.nograd":
        model = po.models.LinearNonlinear((31, 31)).to(DEVICE)
        po.remove_grad(model)
        model.eval()
        return model
    elif name == "frontend.LuminanceGainControl":
        model = po.models.LuminanceGainControl((31, 31)).to(DEVICE)
        model.eval()
        return model
    elif name == "frontend.LuminanceGainControl.nograd":
        model = po.models.LuminanceGainControl((31, 31)).to(DEVICE)
        po.remove_grad(model)
        model.eval()
        return model
    elif name == "frontend.LuminanceContrastGainControl":
        model = po.models.LuminanceContrastGainControl((31, 31)).to(DEVICE)
        model.eval()
        return model
    elif name == "frontend.LuminanceContrastGainControl.nograd":
        model = po.models.LuminanceContrastGainControl((31, 31)).to(DEVICE)
        po.remove_grad(model)
        model.eval()
        return model
    elif name == "frontend.OnOff":
        model = po.models.OnOff((31, 31), pretrained=True, cache_filt=True).to(DEVICE)
        model.eval()
        return model
    elif name == "frontend.OnOff.nograd":
        model = po.models.OnOff((31, 31), pretrained=True, cache_filt=True).to(DEVICE)
        po.remove_grad(model)
        model.eval()
        return model
    elif name == "frontend.OnOff.nograd.ctf":

        class OnOffCTF(po.models.OnOff):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.scales = [0, 1]

            def forward(self, *args, scales=[], **kwargs):
                rep = super().forward(*args, **kwargs)
                if scales:
                    rep = rep[:, scales]
                return rep

        model = OnOffCTF((31, 31)).to(DEVICE)
        po.remove_grad(model)
        model.eval()
        return model
    elif name == "VideoModel":
        # super simple model that combines across the batch dimension, as a
        # model with a temporal component would do
        class VideoModel(po.models.OnOff):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, *args, **kwargs):
                # this will do on/off on each batch separately
                rep = super().forward(*args, **kwargs)
                return rep.mean(0)

        model = VideoModel((31, 31), pretrained=True, cache_filt=True).to(DEVICE)
        po.remove_grad(model)
        model.eval()
        return model
    elif name == "PortillaSimoncelli":
        return po.models.PortillaSimoncelli((256, 256)).to(DEVICE)
    elif name == "NonModule":

        class NonModule:
            def __init__(self):
                self.name = "nonmodule"

            def __call__(self, x):
                return 1 * x

        return NonModule()
    elif "diff_dims" in name:

        class DimModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.target_dims = int(name.replace("diff_dims-", ""))

            def forward(self, img):
                img = img.flatten()
                while img.ndimension() != self.target_dims:
                    img = img.unsqueeze(0)
                return img

        model = DimModel()
        model.eval()
        return model


@pytest.fixture(scope="package")
def model(request):
    return get_model(request.param)


# this is the same as model() fixture above, in order to get two independent
# fixtures.
@pytest.fixture(scope="package")
def model2(request):
    return get_model(request.param)


def check_loss_saved_synth(
    losses, saved_synth, target_iter, objective_function, store_progress
):
    assert len(saved_synth) == np.ceil(target_iter / store_progress) + 1, (
        "Didn't end up with enough saved synth after first synth!"
    )
    assert len(losses) == target_iter + 1, (
        "Didn't end up with enough losses after first synth!"
    )
    if store_progress is True:
        losses = losses
    elif (target_iter % store_progress) == 0:
        losses = losses[::store_progress]
    else:
        # then we need to add on the loss from the current synth object.
        losses = einops.pack([losses[::store_progress], losses[-1]], "*")[0]
    assert len(losses) == len(saved_synth), "wrong length!"
    for synth_loss, saved in zip(losses.to(DEVICE), saved_synth):
        loss = objective_function(saved.to(DEVICE)).squeeze()
        if not torch.equal(loss, synth_loss):
            raise ValueError("saved_synth and loss are misaligned!")


# this list was created from the api/index.rst page for version 1.4.0
OLD_API = [
    "plenoptic.synthesize.metamer.Metamer",
    "plenoptic.synthesize.metamer.MetamerCTF",
    "plenoptic.synthesize.eigendistortion.Eigendistortion",
    "plenoptic.synthesize.mad_competition.MADCompetition",
    "plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli",
    "plenoptic.simulate.models.frontend.LinearNonlinear",
    "plenoptic.simulate.models.frontend.LuminanceGainControl",
    "plenoptic.simulate.models.frontend.LuminanceContrastGainControl",
    "plenoptic.simulate.models.frontend.OnOff",
    "plenoptic.simulate.models.naive.Identity",
    "plenoptic.simulate.models.naive.Linear",
    "plenoptic.simulate.models.naive.Gaussian",
    "plenoptic.simulate.models.naive.CenterSurround",
    "plenoptic.metric.naive.mse",
    "plenoptic.metric.model_metric.model_metric_factory",
    "plenoptic.metric.perceptual_distance.ssim",
    "plenoptic.metric.perceptual_distance.ms_ssim",
    "plenoptic.metric.perceptual_distance.nlpd",
    "plenoptic.synthesize.metamer.plot_loss",
    "plenoptic.synthesize.metamer.display_metamer",
    "plenoptic.synthesize.metamer.plot_pixel_values",
    "plenoptic.synthesize.metamer.plot_representation_error",
    "plenoptic.synthesize.metamer.plot_synthesis_status",
    "plenoptic.synthesize.metamer.animate",
    "plenoptic.synthesize.mad_competition.display_mad_image",
    "plenoptic.synthesize.mad_competition.display_mad_image_all",
    "plenoptic.synthesize.mad_competition.plot_loss",
    "plenoptic.synthesize.mad_competition.plot_loss_all",
    "plenoptic.synthesize.mad_competition.plot_pixel_values",
    "plenoptic.synthesize.mad_competition.plot_synthesis_status",
    "plenoptic.synthesize.mad_competition.animate",
    "plenoptic.synthesize.eigendistortion.display_eigendistortion",
    "plenoptic.synthesize.eigendistortion.display_eigendistortion_all",
    "plenoptic.metric.perceptual_distance.ssim_map",
    "plenoptic.metric.perceptual_distance.normalized_laplacian_pyramid",
    "plenoptic.simulate.canonical_computations.laplacian_pyramid.LaplacianPyramid",
    "plenoptic.simulate.canonical_computations.steerable_pyramid_freq.SteerablePyramidFreq",
    "plenoptic.simulate.canonical_computations.filters.circular_gaussian2d",
    "plenoptic.tools.signal.rectangular_to_polar",
    "plenoptic.tools.signal.polar_to_rectangular",
    "plenoptic.simulate.canonical_computations.non_linearities.local_gain_control",
    "plenoptic.simulate.canonical_computations.non_linearities.local_gain_release",
    "plenoptic.simulate.canonical_computations.non_linearities.rectangular_to_polar_dict",
    "plenoptic.simulate.canonical_computations.non_linearities.polar_to_rectangular_dict",
    "plenoptic.simulate.canonical_computations.non_linearities.local_gain_control_dict",
    "plenoptic.simulate.canonical_computations.non_linearities.local_gain_release_dict",
    "plenoptic.tools.conv.correlate_downsample",
    "plenoptic.tools.conv.blur_downsample",
    "plenoptic.tools.conv.upsample_convolve",
    "plenoptic.tools.conv.upsample_blur",
    "plenoptic.tools.conv.same_padding",
    "plenoptic.tools.signal.shrink",
    "plenoptic.tools.signal.expand",
    "plenoptic.tools.signal.rescale",
    "plenoptic.tools.signal.add_noise",
    "plenoptic.tools.signal.center_crop",
    "plenoptic.tools.signal.modulate_phase",
    "plenoptic.tools.signal.autocorrelation",
    "plenoptic.tools.stats.variance",
    "plenoptic.tools.stats.skew",
    "plenoptic.tools.stats.kurtosis",
    "plenoptic.tools.load_images",
    "plenoptic.tools.to_numpy",
    "plenoptic.tools.convert_float_to_int",
    "plenoptic.data.einstein",
    "plenoptic.data.curie",
    "plenoptic.data.parrot",
    "plenoptic.data.reptile_skin",
    "plenoptic.data.color_wheel",
    "plenoptic.data.fetch.fetch_data",
    "plenoptic.data.fetch.DOWNLOADABLE_FILES",
    "plenoptic.tools.make_disk",
    "plenoptic.tools.polar_radius",
    "plenoptic.tools.polar_angle",
    "plenoptic.tools.validate.remove_grad",
    "plenoptic.tools.validate.validate_model",
    "plenoptic.tools.validate.validate_input",
    "plenoptic.tools.validate.validate_metric",
    "plenoptic.tools.validate.validate_coarse_to_fine",
    "plenoptic.tools.validate.validate_convert_tensor_dict",
    "plenoptic.tools.display.imshow",
    "plenoptic.tools.display.animshow",
    "plenoptic.tools.display.pyrshow",
    "plenoptic.tools.display.plot_representation",
    "plenoptic.tools.clean_up_axes",
    "plenoptic.tools.display.clean_up_axes",
    "plenoptic.tools.display.clean_stem_plot",
    "plenoptic.tools.display.rescale_ylim",
    "plenoptic.tools.rescale_ylim",
    "plenoptic.tools.display.update_plot",
    "plenoptic.tools.display.update_stem",
    "plenoptic.tools.update_stem",
    "plenoptic.tools.io.examine_saved_synthesis",
    "plenoptic.tools.external.plot_MAD_results",
    "plenoptic.tools.optim.set_seed",
    "plenoptic.tools.optim.mse",
    "plenoptic.tools.optim.l2_norm",
    "plenoptic.tools.optim.relative_sse",
    "plenoptic.tools.optim.portilla_simoncelli_loss_factory",
    "plenoptic.tools.optim.groupwise_relative_l2_norm_factory",
    "plenoptic.tools.regularization.penalize_range",
    "plenoptic.__version__",
]
