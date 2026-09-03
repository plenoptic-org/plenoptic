"""
Test uploaded files.

For the documentation, we have pre-generated some synthesis outputs and uplodaed
them to OSF. Then during docs build, we download them.

These tests check that the outputs haven't changed, notifying us if we need to
update them.
"""

import functools
import itertools
import os
from collections import OrderedDict

import einops
import numpy as np
import pytest
import torch
import torchvision

import plenoptic as po
from conftest import DEVICE, DEVICE2
from plenoptic.tensors import _check_tensor_equality


def compare_eigendistortions(eig, eig_up, rtol=1e-5, atol=1e-7):
    for k in ["_representation_flat", "eigendistortions"]:
        _check_tensor_equality(
            getattr(eig, k),
            getattr(eig_up, k),
            "Test",
            "OSF",
            rtol,
            atol,
            f"{k} has different {{error_type}}! Update the OSF version.",
        )


def compare_metamers(met, met_up, rtol=1e-5, atol=1e-7):
    _check_tensor_equality(
        met.metamer,
        met_up.metamer,
        "Test",
        "OSF",
        rtol,
        atol,
        "metamer has different {error_type}! Update the OSF version.",
    )


def compare_mad(mad, mad_up, rtol=1e-5, atol=1e-7):
    _check_tensor_equality(
        mad.mad_image,
        mad_up.mad_image,
        "Test",
        "OSF",
        rtol,
        atol,
        "mad_image has different {error_type}! Update the OSF version.",
    )


class PortillaSimoncelliRemove(po.models.PortillaSimoncelli):
    r"""Model for measuring a subset of texture statistics reported by
    PortillaSimoncelli

    Parameters
    ----------
    im_shape: int
        the size of the images being processed by the model
    remove_keys: list
        The dictionary keys for the statistics we will "remove".  In practice we set
        them to zero.
        Possible keys: ["pixel_statistics", "auto_correlation_magnitude",
        "skew_reconstructed", "kurtosis_reconstructed",
        "auto_correlation_reconstructed", "std_reconstructed", "magnitude_std",
        "cross_orientation_correlation_magnitude", "cross_scale_correlation_magnitude",
        "cross_scale_correlation_real", "var_highpass_residual"]
    """

    def __init__(
        self,
        im_shape,
        remove_keys,
    ):
        super().__init__(im_shape, n_scales=4, n_orientations=4, spatial_corr_width=7)
        self.remove_keys = remove_keys

    def forward(self, image, scales=None):
        r"""Generate Texture Statistics representation of an image with `remove_keys`
        removed.

        Parameters
        ----------
        image : torch.Tensor
            A tensor containing the image to analyze.
        scales : list, optional
            Which scales to include in the returned representation. If an empty
            list (the default), we include all scales. Otherwise, can contain
            subset of values present in this model's scales attribute.

        Returns
        -------
        representation: torch.Tensor
            3d tensor of shape (batch, channel, stats) containing the measured texture
            stats.

        """
        # create the representation tensor (with all scales)
        stats_vec = super().forward(image)
        # convert to dict so it's easy to zero out the keys we don't care about
        stats_dict = self.convert_to_dict(stats_vec)
        for kk in self.remove_keys:
            # we zero out the stats (instead of removing them) because removing them
            # makes it difficult to keep track of which stats belong to which scale
            # (which is necessary for coarse-to-fine synthesis) -- see discussion above.
            if isinstance(stats_dict[kk], OrderedDict):
                for key, val in stats_dict[kk].items():
                    stats_dict[kk][key] *= 0
            else:
                stats_dict[kk] *= 0
        # then convert back to tensor and remove any scales we don't want
        # (for coarse-to-fine)  -- see discussion above.
        stats_vec = self.convert_to_tensor(stats_dict)
        if scales is not None:
            stats_vec = self.remove_scales(stats_vec, scales)
        return stats_vec


class PortillaSimoncelliMask(po.models.PortillaSimoncelli):
    r"""Extend the PortillaSimoncelli model to operate on masked images.

    Additional Parameters
    ----------
    mask: Tensor
        boolean mask with ``True`` in the part of the image that will be filled in
        during synthesis
    target: Tensor
        image target for synthesis

    """

    def __init__(
        self,
        im_shape,
        n_scales=4,
        n_orientations=4,
        spatial_corr_width=7,
        mask=None,
        target=None,
    ):
        super().__init__(im_shape, n_scales=4, n_orientations=4, spatial_corr_width=7)
        self.mask = mask
        self.target = target

    def forward(self, image, scales=None):
        r"""Generate Texture Statistics representation of an image using the target for
        the masked portion

        Parameters
        ----------
        images : torch.Tensor
            A 4d tensor containing two images to analyze, with shape (2,
            channel, height, width).
        scales : list, optional
            Which scales to include in the returned representation. If an empty
            list (the default), we include all scales. Otherwise, can contain
            subset of values present in this model's scales attribute.

        Returns
        -------
        representation_tensor: torch.Tensor
            3d tensor of shape (batch, channel, stats) containing the measured
            texture statistics.

        """
        if self.mask is not None and self.target is not None:
            image = self.texture_masked_image(image)

        return super().forward(image, scales=scales)

    def texture_masked_image(self, image):
        r"""Fill in part of the image (designated by the mask) with the saved target
        image

        Parameters
        ------------
        image : torch.Tensor
            A tensor containing a single image

        Returns
        -------
        texture_masked_image: torch.Tensor
            An image that is a combination of the input image and the saved target.
            Combination is specified by self.mask

        """
        return self.target * self.mask + image * (~self.mask)


class PortillaSimoncelliMagMeans(po.models.PortillaSimoncelli):
    r"""Include the magnitude means in the PS texture representation.

    Parameters
    ----------
    im_shape: int
        the size of the images being processed by the model

    """

    def __init__(
        self,
        im_shape,
    ):
        super().__init__(im_shape, n_scales=4, n_orientations=4, spatial_corr_width=7)

    def forward(self, image, scales=None):
        r"""Average Texture Statistics representations of two image

        Parameters
        ----------
        image : torch.Tensor
            A 4d tensor (batch, channel, height, width) containing the image(s) to
            analyze.
        scales : list, optional
            Which scales to include in the returned representation. If an empty
            list (the default), we include all scales. Otherwise, can contain
            subset of values present in this model's scales attribute.

        Returns
        -------
        representation_tensor: torch.Tensor
            3d tensor of shape (batch, channel, stats) containing the measured
            texture statistics.

        """
        stats = super().forward(image, scales=scales)
        # this helper function returns a list of tensors containing the steerable
        # pyramid coefficients at each scale
        pyr_coeffs = self._compute_pyr_coeffs(image)[1]
        # only compute the magnitudes for the desired scales
        magnitude_pyr_coeffs = [
            coeff.abs()
            for i, coeff in enumerate(pyr_coeffs)
            if scales is None or i in scales
        ]
        magnitude_means = [mag.mean((-2, -1)) for mag in magnitude_pyr_coeffs]
        return einops.pack([stats, *magnitude_means], "b c *")[0]

    # overwriting this method is necessary for the loss factory to work
    def convert_to_tensor(self, representation_dict: OrderedDict) -> torch.Tensor:
        """Convert dictionary of statistics to a tensor."""
        rep = super().convert_to_tensor(representation_dict)
        return torch.cat(
            [rep, representation_dict["magnitude_means"].flatten(-2, -1)], -1
        )

    # overwriting this method is necessary for the loss factory to work. overwriting
    # this and the following method allows us to use the plot_representation method with
    # the modified model, making examining it easier.
    def convert_to_dict(self, representation_tensor: torch.Tensor) -> OrderedDict:
        """Convert tensor of stats to dictionary."""
        n_mag_means = self.n_scales * self.n_orientations
        rep = super().convert_to_dict(representation_tensor[..., :-n_mag_means])
        mag_means = representation_tensor[..., -n_mag_means:]
        rep["magnitude_means"] = einops.rearrange(
            mag_means,
            "b c (s o) -> b c s o",
            s=self.n_scales,
            o=self.n_orientations,
        )
        return rep

    def _representation_for_plotting(self, rep: OrderedDict) -> OrderedDict:
        r"""Convert the data into a dictionary representation that is more convenient
        for plotting.

        Intended as a helper function for plot_representation.
        """
        mag_means = rep.pop("magnitude_means")
        data = super()._representation_for_plotting(rep)
        data["magnitude_means"] = mag_means.flatten()
        return data


class ColorModel(torch.nn.Module):
    """Simple model that takes color image as input and outputs 2d conv."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, 1)

    def forward(self, x):
        return self.conv(x)


@pytest.mark.order(1)
@pytest.mark.skipif(DEVICE.type == "cpu", reason="Only do this on cuda")
class TestDoctest:
    def test_eigendistortion(self, einstein_img_double):
        po.set_seed(0)
        os.makedirs("uploaded_files", exist_ok=True)
        torch.save(
            torch.random.get_rng_state(),
            "uploaded_files/torch_rng_state_eigendistortion.pt",
        )
        print(np.random.get_state())
        lg = po.models.LuminanceGainControl(
            (31, 31), pad_mode="circular", pretrained=True, cache_filt=True
        )
        po.remove_grad(lg)
        lg = lg.to(DEVICE).to(einstein_img_double.dtype)
        lg.eval()
        eig = po.Eigendistortion(einstein_img_double, lg)
        eig.synthesize(max_iter=1000)
        eig.save("uploaded_files/example_eigendistortion.pt")
        eig_up = po.Eigendistortion(einstein_img_double, lg)
        eig_up.load(
            po.data.fetch_data("example_eigendistortion.pt"),
            tensor_equality_atol=1e-7,
            map_location=DEVICE,
        )
        compare_eigendistortions(eig, eig_up)

    def test_eigendistortion_color(self):
        po.set_seed(0)
        img = po.data.color_wheel().to(torch.float64)
        img = po.process.center_crop(img, 20).to(DEVICE)
        model = ColorModel()
        model.to(img.dtype).to(DEVICE)
        os.makedirs("uploaded_files", exist_ok=True)
        torch.save(
            torch.random.get_rng_state(),
            "uploaded_files/torch_rng_state_eigendistortion.pt",
        )
        print(np.random.get_state())
        po.remove_grad(model)
        model.eval()
        eig = po.Eigendistortion(img, model)
        eig.synthesize(max_iter=500)
        eig.save("uploaded_files/example_eigendistortion_color.pt")
        eig_up = po.Eigendistortion(img, model)
        eig_up.load(
            po.data.fetch_data("example_eigendistortion_color.pt"),
            tensor_equality_atol=1e-7,
            map_location=DEVICE,
        )
        compare_eigendistortions(eig, eig_up)

    @pytest.mark.filterwarnings(
        "ignore:Loss has converged, stopping synthesis:UserWarning"
    )
    def test_example_metamer_gaussian(self, einstein_img_double):
        po.set_seed(0)
        os.makedirs("uploaded_files", exist_ok=True)
        torch.save(
            torch.random.get_rng_state(),
            "uploaded_files/torch_rng_state_metamer_gaussian.pt",
        )
        print(np.random.get_state())
        model = po.models.Gaussian(30).eval()
        po.remove_grad(model)
        model = model.to(DEVICE).to(einstein_img_double.dtype)
        met = po.Metamer(einstein_img_double, model)
        # needed to initialize optimizer for following, see issue #404
        met.setup()
        init_state_dict = met.optimizer.state_dict()
        met.synthesize(110, store_progress=10)
        met.save("uploaded_files/example_metamer_gaussian-cuda.pt")
        met_up = po.Metamer(einstein_img_double, model)
        met_up.load(
            po.data.fetch_data("example_metamer_gaussian-cuda.pt"),
            tensor_equality_atol=1e-7,
            map_location=DEVICE,
        )
        compare_metamers(met, met_up)
        # needed to allow us to move device completely, see issue #404
        met.optimizer.load_state_dict(init_state_dict)
        met.to("cpu")
        met.save("uploaded_files/example_metamer_gaussian.pt")

    @pytest.mark.filterwarnings(
        "ignore:Validating whether model can work with coarse-to-fine:UserWarning"
    )
    def test_example_metamerCTF_ps(self):
        po.set_seed(0)
        os.makedirs("uploaded_files", exist_ok=True)
        torch.save(
            torch.random.get_rng_state(),
            "uploaded_files/torch_rng_state_metamerCTF_ps.pt",
        )
        print(np.random.get_state())
        img = po.data.reptile_skin().to(torch.float64).to(DEVICE)
        model = po.models.PortillaSimoncelli(img.shape[-2:]).to(DEVICE)
        met = po.MetamerCTF(img, model, po.loss.l2_norm)
        # needed to initialize optimizer for following, see issue #404
        met.setup()
        init_state_dict = met.optimizer.state_dict()
        met.synthesize(
            150, change_scale_criterion=None, ctf_iters_to_check=7, store_progress=10
        )
        met.save("uploaded_files/example_metamerCTF_ps-cuda.pt")
        met_up = po.MetamerCTF(img, model, po.loss.l2_norm)
        met_up.load(
            po.data.fetch_data("example_metamerCTF_ps-cuda.pt"),
            tensor_equality_atol=1e-7,
            map_location=DEVICE,
        )
        compare_metamers(met, met_up)
        # needed to allow us to move device completely, see issue #404
        met.optimizer.load_state_dict(init_state_dict)
        met.to("cpu")
        met.save("uploaded_files/example_metamerCTF_ps.pt")

    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_example_mad(self):
        po.set_seed(0)
        os.makedirs("uploaded_files", exist_ok=True)
        torch.save(
            torch.random.get_rng_state(),
            "uploaded_files/torch_rng_state_mad.pt",
        )
        print(np.random.get_state())
        img = po.data.einstein().to(torch.float64).to(DEVICE)

        def ds_ssim(x, y):
            return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")

        mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
        # needed to initialize optimizer for following, see issue #404
        mad.setup(0.04)
        init_state_dict = mad.optimizer.state_dict()
        mad.synthesize(200, store_progress=15)
        mad.save("uploaded_files/example_mad-cuda.pt")
        mad_up = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
        mad_up.load(
            po.data.fetch_data("example_mad-cuda.pt"),
            tensor_equality_atol=1e-7,
            map_location=DEVICE,
        )
        compare_mad(mad, mad_up)
        # needed to allow us to move device completely, see issue #404
        mad.optimizer.load_state_dict(init_state_dict)
        mad.to("cpu")
        mad.save("uploaded_files/example_mad.pt")


@pytest.mark.order(0)
@pytest.mark.skipif(DEVICE.type == "cpu", reason="Only do this on cuda")
@pytest.mark.skipif(
    os.environ.get("RUN_REGRESSION_SYNTH", "") != "1",
    reason="These take a long time, so don't run every time",
)
class TestTutorialNotebooks:
    class TestFeatureExtractor:
        @pytest.mark.parametrize("target_layer", ["layer2", "layer3", "layer4"])
        def test_resnet_macaque_metamer(self, target_layer):
            # torch convolution on cuda is non-deterministic by default
            torch.use_deterministic_algorithms(True)
            po.set_seed(1)
            os.makedirs("uploaded_files", exist_ok=True)
            torch.save(
                torch.random.get_rng_state(),
                f"uploaded_files/torch_rng_state_ResNet50-{target_layer}_macaque_metamer.pt",
            )
            print(np.random.get_state())
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            deepnet = torchvision.models.resnet50(weights=weights)
            deepnet.eval()
            transform = weights.transforms()
            norm = torchvision.transforms.Normalize(transform.mean, transform.std)
            crop = functools.partial(
                po.process.center_crop, output_size=transform.crop_size[0]
            )

            img = po.data.macaque().to(DEVICE).to(torch.float64)
            img = crop(po.process.blur_downsample(img, 2)[..., :-59, :])
            model = po.models.DeepNetFeatures(deepnet, target_layer, norm)
            model.to(torch.float64).to(DEVICE)
            po.remove_grad(model)
            met = po.Metamer(img, model)
            scheduler = torch.optim.lr_scheduler.StepLR
            scheduler_kwargs = {
                "step_size": 5000 if target_layer == "layer4" else 3000,
                "gamma": 0.5,
            }
            lr = 3e-2 if target_layer == "layer4" else 1e-2
            met.setup(
                optimizer_kwargs={"lr": lr, "amsgrad": False},
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
            )
            # by setting stop_iters_to_check=max_iter, we ensure it keeps going through
            # all 12k iterations
            met.synthesize(max_iter=12000, stop_iters_to_check=12000)
            met.save(f"uploaded_files/ResNet50-{target_layer}_macaque_metamer.pt")
            met_up = po.Metamer(img, model)
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    po.data.fetch_data(f"ResNet50-{target_layer}_macaque_metamer.pt"),
                    tensor_equality_atol=1e-6,
                    map_location=DEVICE,
                )
            compare_metamers(met, met_up)

    class TestDemoEigendistortion:
        def test_berardino_onoff(self, parrot_square_double):
            po.set_seed(0)
            os.makedirs("uploaded_files", exist_ok=True)
            torch.save(
                torch.random.get_rng_state(),
                "uploaded_files/torch_rng_state_berardino_onoff.pt",
            )
            print(np.random.get_state())
            model = po.models.OnOff(
                (31, 31),
                pretrained=True,
                cache_filt=True,
                apply_mask=True,
            )
            po.remove_grad(model)
            model = model.to(DEVICE).to(parrot_square_double.dtype)
            model.eval()
            eig = po.Eigendistortion(parrot_square_double, model)
            eig.synthesize(k=3, method="power", max_iter=2000)
            eig.save("uploaded_files/berardino_onoff.pt")
            eig_up = po.Eigendistortion(parrot_square_double, model)
            eig_up.load(
                po.data.fetch_data("berardino_onoff.pt"),
                tensor_equality_atol=1e-7,
                map_location=DEVICE,
            )
            compare_eigendistortions(eig, eig_up)

        def test_berardino_vgg16(self, parrot_square_double):
            po.set_seed(0)
            os.makedirs("uploaded_files", exist_ok=True)
            torch.save(
                torch.random.get_rng_state(),
                "uploaded_files/torch_rng_state_berardino_vgg16.pt",
            )
            print(np.random.get_state())
            weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
            model = torchvision.models.vgg16(weights=weights, progress=False)
            model.eval()
            model = po.models.DeepNetFeatures(model, "features.11")
            po.remove_grad(model)
            # in this case, apply norm outside the model
            transform = weights.transforms()
            norm = torchvision.transforms.Normalize(transform.mean, transform.std)
            img = norm(parrot_square_double.to(DEVICE2).repeat(1, 3, 1, 1))
            model = model.to(DEVICE2).to(img.dtype)
            model.eval()
            with pytest.warns(UserWarning, match="input_tensor range is"):
                eig = po.Eigendistortion(img, model)
            eig.synthesize(k=2, method="power", max_iter=5000)
            eig.save("uploaded_files/berardino_vgg16.pt")
            with pytest.warns(UserWarning, match="input_tensor range is"):
                eig_up = po.Eigendistortion(img, model)
            eig_up.load(
                po.data.fetch_data("berardino_vgg16.pt"),
                tensor_equality_atol=1e-7,
                map_location=DEVICE2,
            )
            compare_eigendistortions(eig, eig_up)

    @pytest.mark.filterwarnings(
        "ignore:Loss has converged, stopping synthesis:UserWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:Validating whether model can work with coarse-to-fine:UserWarning"
    )
    class TestPortillaSimoncelli:
        @pytest.fixture(scope="class")
        @classmethod
        def ps_images(cls):
            img_dir = po.data.fetch_data("portilla_simoncelli_images.tar.gz")
            images = po.load_images(img_dir).to(DEVICE).to(torch.float64)
            filenames = [p.stem for p in sorted(img_dir.iterdir())]
            return images, filenames

        def get_specific_img(self, images, filenames, tgt_filename):
            # the clone is here because torch saving/loading preserves views and
            # so extra info would be saved without it:
            # https://docs.pytorch.org/docs/stable/notes/serialization.html#saving-and-loading-tensors-preserves-views
            return images[filenames.index(tgt_filename)].unsqueeze(0).clone()

        @pytest.fixture(scope="class")
        @classmethod
        def ps_regression(cls):
            return po.data.fetch_data("ps_regression.tar.gz")

        @pytest.mark.parametrize(
            "fig_name",
            [
                "fig4a",
                "fig12a",
                "fig12b",
                "fig12c",
                "fig12d",
                "fig12e",
                "fig12f",
                "fig13a",
                "fig13b",
                "fig13c",
                "fig13d",
                "fig14a",
                "fig14b",
                "fig14c",
                "fig14d",
                "fig14e",
                "fig15a",
                "fig15b",
                "fig15c",
                "fig15d",
                "fig15e",
                "fig15f",
                "fig16a",
                "fig16b",
                "fig16c",
                "fig16d",
                "fig16e",
                "fig16f",
                "fig18a",
                "einstein",
            ],
        )
        def test_ps_basic_synthesis(
            self, ps_images, fig_name, einstein_img_double, ps_regression
        ):
            po.set_seed(0)
            torch.save(
                torch.random.get_rng_state(),
                f"uploaded_files/torch_rng_state_ps_basic_{fig_name}.pt",
            )
            print(np.random.get_state())
            if fig_name.startswith("fig"):
                img = self.get_specific_img(*ps_images, fig_name)
            elif fig_name == "einstein":
                img = einstein_img_double
            # this is a sawtooth grating, with 4 scales the steerable pyramid's
            # residual lowpass is uniform and thus correlation between it and
            # the coarsest scale is all NaNs (i.e., the last scale of
            # auto_correlation_reconstructed is all NaNs)
            n_scales = 3 if fig_name == "fig12b" else 4
            model = po.models.PortillaSimoncelli(img.shape[-2:], n_scales=n_scales)
            model.to(DEVICE)
            loss = po.loss.portilla_simoncelli_loss_factory(model, img)
            met = po.Metamer(img, model, loss_function=loss)
            opt_kwargs = {
                "max_iter": 10,
                "max_eval": 10,
                "history_size": 100,
                "line_search_fn": "strong_wolfe",
                "lr": 1,
            }
            met.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
            # add _lint_ignore so that our linter knows to ignore it
            init_state_dict_lint_ignore = met.optimizer.state_dict()
            n_iters = 150 if fig_name in ["fig12a", "fig12b"] else 100
            met.synthesize(max_iter=n_iters)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save(f"uploaded_files/ps_basic_synthesis_{fig_name}.pt")
            met_up = po.Metamer(img, model, loss_function=loss)
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    ps_regression / f"ps_basic_synthesis_{fig_name}.pt",
                    tensor_equality_atol=1e-7,
                    map_location=DEVICE,
                )
            compare_metamers(met, met_up)

        # make sure we fail if save load with different stats removed
        def test_ps_remove_fail(self, ps_images, tmp_path):
            img = self.get_specific_img(*ps_images, "fig4a")
            model = PortillaSimoncelliRemove(
                img.shape[-2:], remove_keys=["pixel_statistics"]
            )
            model.to(DEVICE).to(torch.float64)
            loss = po.loss.portilla_simoncelli_loss_factory(model, img)
            met = po.Metamer(img, model, loss_function=loss)
            opt_kwargs = {
                "max_iter": 10,
                "max_eval": 10,
                "history_size": 100,
                "line_search_fn": "strong_wolfe",
                "lr": 1,
            }
            met.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
            init_state_dict = met.optimizer.state_dict()
            met.synthesize(max_iter=5)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict)
            met.save(tmp_path / "test_ps_remove_fail.pt")
            model = PortillaSimoncelliRemove(
                img.shape[-2:], remove_keys=["pixel_statistics", "skew_reconstructed"]
            )
            model.to(DEVICE).to(torch.float64)
            met = po.Metamer(img, model, loss_function=loss)
            with pytest.raises(ValueError, match="Saved and initialized model output"):
                met.load(tmp_path / "test_ps_remove_fail.pt", map_location=DEVICE)

        @pytest.mark.parametrize(
            "fn, stats",
            [
                (
                    "fig3a",
                    [
                        "pixel_statistics",
                        "skew_reconstructed",
                        "kurtosis_reconstructed",
                    ],
                ),
                (
                    "fig3b",
                    [
                        "pixel_statistics",
                        "skew_reconstructed",
                        "kurtosis_reconstructed",
                    ],
                ),
                ("fig4a", ["auto_correlation_reconstructed", "std_reconstructed"]),
                ("fig4b", ["auto_correlation_reconstructed", "std_reconstructed"]),
                (
                    "fig6a",
                    [
                        "magnitude_std",
                        "cross_orientation_correlation_magnitude",
                        "cross_scale_correlation_magnitude",
                        "auto_correlation_magnitude",
                    ],
                ),
                (
                    "fig6b",
                    [
                        "magnitude_std",
                        "cross_orientation_correlation_magnitude",
                        "cross_scale_correlation_magnitude",
                        "auto_correlation_magnitude",
                    ],
                ),
                ("fig8a", ["cross_scale_correlation_real"]),
                ("fig8b", ["cross_scale_correlation_real"]),
            ],
        )
        @pytest.mark.parametrize("remove_bool", [True, False])
        def test_ps_remove(self, ps_images, fn, stats, remove_bool, ps_regression):
            po.set_seed(0)
            torch.save(
                torch.random.get_rng_state(),
                f"uploaded_files/torch_rng_state_ps_remove_{fn}_remove-{remove_bool}.pt",
            )
            print(np.random.get_state())
            img = self.get_specific_img(*ps_images, fn).to(DEVICE2)
            if remove_bool:
                model = PortillaSimoncelliRemove(img.shape[-2:], remove_keys=stats)
            else:
                model = po.models.PortillaSimoncelli(img.shape[-2:])
            model.to(DEVICE2).to(torch.float64)
            loss = po.loss.portilla_simoncelli_loss_factory(model, img)
            met = po.Metamer(img, model, loss_function=loss)
            opt_kwargs = {
                "max_iter": 10,
                "max_eval": 10,
                "history_size": 100,
                "line_search_fn": "strong_wolfe",
                "lr": 1,
            }
            met.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
            init_state_dict_lint_ignore = met.optimizer.state_dict()
            met.synthesize(max_iter=100)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save(f"uploaded_files/ps_remove_{fn}_remove-{remove_bool}.pt")
            met_up = po.Metamer(img, model, loss_function=loss)
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    ps_regression / f"ps_remove_{fn}_remove-{remove_bool}.pt",
                    tensor_equality_atol=1e-7,
                    map_location=DEVICE2,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings("ignore:You will need to call setup:UserWarning")
        def test_ps_mask(self, ps_images, ps_regression):
            po.set_seed(0)
            torch.save(
                torch.random.get_rng_state(),
                "uploaded_files/torch_rng_state_ps_mask.pt",
            )
            print(np.random.get_state())
            img = self.get_specific_img(*ps_images, "fig14b").to(DEVICE2)
            mask = torch.zeros_like(img).bool()
            ctr_dim = (img.shape[-2] // 4, img.shape[-1] // 4)
            mask[..., ctr_dim[0] : 3 * ctr_dim[0], ctr_dim[1] : 3 * ctr_dim[1]] = True
            model = PortillaSimoncelliMask(img.shape[-2:], target=img, mask=mask)
            model.to(DEVICE2).to(torch.float64)
            loss = po.loss.portilla_simoncelli_loss_factory(model, img)
            met = po.Metamer(img, model, loss_function=loss)
            opt_kwargs = {
                "max_iter": 10,
                "max_eval": 10,
                "history_size": 100,
                "line_search_fn": "strong_wolfe",
                "lr": 1,
            }
            met.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
            init_state_dict_lint_ignore = met.optimizer.state_dict()
            met.synthesize(max_iter=100)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/ps_mask.pt")
            met_up = po.Metamer(img, model, loss_function=loss)
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    ps_regression / "ps_mask.pt",
                    tensor_equality_atol=1e-7,
                    map_location=DEVICE2,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings("ignore:You will need to call setup:UserWarning")
        @pytest.mark.filterwarnings(
            "ignore:initial_image and image are different sizes:UserWarning"
        )
        @pytest.mark.parametrize(
            "fn",
            [
                ("fig15e", "fig14e"),
                ("fig14b", "fig4a"),
                ("fig15a", "fig15b"),
            ],
        )
        def test_ps_mixture(self, ps_images, fn, ps_regression):
            po.set_seed(0)
            torch.save(
                torch.random.get_rng_state(),
                f"uploaded_files/torch_rng_state_ps_mixture_{'-'.join(fn)}.pt",
            )
            print(np.random.get_state())
            img = torch.cat(
                [
                    self.get_specific_img(*ps_images, fn[0])[..., 128:],
                    self.get_specific_img(*ps_images, fn[1])[..., :128],
                ],
                -1,
            ).to(DEVICE2)
            model = po.models.PortillaSimoncelli(img.shape[-2:])
            model.to(DEVICE2).to(torch.float64)
            loss = po.loss.portilla_simoncelli_loss_factory(model, img)
            met = po.Metamer(img, model, loss_function=loss)
            opt_kwargs = {
                "max_iter": 10,
                "max_eval": 10,
                "history_size": 100,
                "line_search_fn": "strong_wolfe",
                "lr": 1,
            }
            met.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
            init_state_dict_lint_ignore = met.optimizer.state_dict()
            met.synthesize(max_iter=100)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save(f"uploaded_files/ps_mixture_{'-'.join(fn)}.pt")
            met_up = po.Metamer(img, model, loss_function=loss)
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    ps_regression / f"ps_mixture_{'-'.join(fn)}.pt",
                    tensor_equality_atol=1e-7,
                    map_location=DEVICE2,
                )
            compare_metamers(met, met_up)

        @pytest.mark.parametrize("mag_bool", [True, False])
        def test_ps_mag_means(self, ps_images, mag_bool, ps_regression):
            po.set_seed(100)
            torch.save(
                torch.random.get_rng_state(),
                f"uploaded_files/torch_rng_state_ps_mag_means-{mag_bool}.pt",
            )
            print(np.random.get_state())
            img = self.get_specific_img(*ps_images, "fig4a").to(DEVICE2)
            if mag_bool:
                model = PortillaSimoncelliMagMeans(img.shape[-2:])
            else:
                model = po.models.PortillaSimoncelli(
                    img.shape[-2:], spatial_corr_width=7
                )
            model.to(DEVICE2).to(torch.float64)
            loss = po.loss.portilla_simoncelli_loss_factory(model, img)
            met = po.Metamer(img, model, loss_function=loss)
            opt_kwargs = {
                "max_iter": 10,
                "max_eval": 10,
                "history_size": 100,
                "line_search_fn": "strong_wolfe",
                "lr": 1,
            }
            met.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
            init_state_dict_lint_ignore = met.optimizer.state_dict()
            met.synthesize(max_iter=100)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save(f"uploaded_files/ps_mag_means-{mag_bool}.pt")
            met_up = po.Metamer(img, model, loss_function=loss)
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    ps_regression / f"ps_mag_means-{mag_bool}.pt",
                    tensor_equality_atol=1e-7,
                    map_location=DEVICE2,
                )
            compare_metamers(met, met_up)

    # None of these take long to run, so we do it with cpu instead of DEVICE/DEVICE2
    # (which will be GPU for our tests)
    class TestDatasaurus:
        @pytest.fixture(scope="class")
        @classmethod
        def datasaurus(cls):
            data = po.data.fetch_data("datasaurus.tar.gz") / "datasaurus.pt"
            return torch.load(data)[0]

        @pytest.fixture(scope="class")
        @classmethod
        def datasaurus_metamers(cls):
            return po.data.fetch_data("datasaurus_metamers.tar.gz")

        @pytest.fixture(scope="class")
        @classmethod
        def datasaurus_model(cls):
            class DatasaurusModel(torch.nn.Module):
                def __init__(self, n_pts=None, dtype=None):
                    super().__init__()
                    # cache ones to save time
                    if n_pts is not None:
                        self._ones = torch.ones(n_pts, dtype=dtype)
                    else:
                        self._ones = None

                def _prepare_X(self, x):
                    ones = self._ones if self._ones is None else torch.ones_like(x)
                    return torch.stack([ones, x], -1)

                def _compute_linreg(self, x, y):
                    X = self._prepare_X(x)
                    # unsqueezing and squeezing needed because of https://github.com/pytorch/pytorch/issues/158169
                    return torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze()

                def _compute_coeff_determination(self, x, y, solution):
                    X = self._prepare_X(x)
                    pred_y = torch.einsum("x, n x -> n", solution, X)
                    ss_res = (y - pred_y).pow(2).sum()
                    ss_tot = (y - y.mean()).pow(2).sum()
                    return 1 - (ss_res / ss_tot)

                def _vmap_coeff_determination(self, x, solution):
                    f = torch.func.vmap(
                        lambda x, solt: self._compute_coeff_determination(*x, solt)
                    )
                    return f(x, solution).unsqueeze(-1)

                def forward(self, data):
                    if data.ndim == 2:
                        data = data.unsqueeze(0)
                    elif data.ndim != 3:
                        raise ValueError("data must be 2 or 3d!")
                    stats = []
                    stats.append(data.mean(-1))
                    stats.append(data.std(-1))
                    solution = torch.func.vmap(lambda x: self._compute_linreg(*x))(data)
                    stats.append(solution)
                    crosscorr = torch.func.vmap(lambda x: torch.corrcoef(x)[0, 1])(data)
                    stats.append(crosscorr.unsqueeze(-1))
                    stats.append(self._vmap_coeff_determination(data, solution))
                    return torch.cat(stats, -1)

            model = DatasaurusModel(142, torch.float64)
            model.eval()
            return model

        @pytest.mark.filterwarnings(
            "ignore:plenoptic's methods have mostly been tested on 4d:UserWarning"
        )
        @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
        def test_circle(self, datasaurus, datasaurus_model, datasaurus_metamers):
            po.set_seed(0)
            torch.use_deterministic_algorithms(True)

            def circle_penalty(data, target_ctr, target_r):
                target_ctr = torch.as_tensor(target_ctr).unsqueeze(-1)
                R = (data - target_ctr).pow(2).sum(0).sqrt()
                return (R - target_r).pow(2).sum()

            def penalty(x):
                range_penalty = po.regularize.penalize_range(x, (0, 100))
                circle = circle_penalty(x, [50, 50], 35)
                return range_penalty + circle

            met = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0001,
            )
            met.setup(
                initial_image=100 * torch.rand_like(datasaurus),
                optimizer=torch.optim.LBFGS,
            )
            init_state_dict_lint_ignore = met.optimizer.state_dict()

            met.synthesize(50, store_progress=True)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/datasaurus-circle.pt")
            met_up = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0001,
            )
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    datasaurus_metamers / "datasaurus-circle.pt",
                    tensor_equality_atol=1e-7,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings(
            "ignore:plenoptic's methods have mostly been tested on 4d:UserWarning"
        )
        @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
        def test_bullseye(self, datasaurus, datasaurus_model, datasaurus_metamers):
            po.set_seed(0)
            torch.use_deterministic_algorithms(True)

            def circle_penalty(data, target_ctr, target_r):
                target_ctr = torch.as_tensor(target_ctr).unsqueeze(-1)
                R = (data - target_ctr).pow(2).sum(0).sqrt()
                return (R - target_r).pow(2).sum()

            def bullseye_penalty(data, target_ctr, target_rs):
                n_pts = data.shape[-1]
                a = circle_penalty(data[..., n_pts // 2 :], target_ctr, target_rs[0])
                b = circle_penalty(data[..., : n_pts // 2], target_ctr, target_rs[1])
                return a + b

            def penalty(x):
                range_penalty = po.regularize.penalize_range(x, (0, 100))
                bullseye = bullseye_penalty(x, [50, 50], [20, 40])
                return range_penalty + bullseye

            met = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            met.setup(
                initial_image=100 * torch.rand_like(datasaurus),
                optimizer=torch.optim.LBFGS,
            )
            init_state_dict_lint_ignore = met.optimizer.state_dict()

            met.synthesize(50, store_progress=True)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/datasaurus-bullseye.pt")
            met_up = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    datasaurus_metamers / "datasaurus-bullseye.pt",
                    tensor_equality_atol=1e-7,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings(
            "ignore:plenoptic's methods have mostly been tested on 4d:UserWarning"
        )
        @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
        def test_hlines(self, datasaurus, datasaurus_model, datasaurus_metamers):
            po.set_seed(0)
            torch.use_deterministic_algorithms(True)

            def predict_line(data, intercepts, slope):
                return slope * data[0] + intercepts

            def lines_penalty(data, intercepts, slope):
                # intercepts must be shape [n, 1], slope a scalar or same number of
                # elements as intercepts
                errors = []
                n = data.shape[-1] // intercepts.shape[0]
                if hasattr(slope, "__len__") and len(slope) != 1:
                    assert len(slope) == len(intercepts)
                else:
                    slope = len(intercepts) * [slope]
                for i, (inter, sl) in enumerate(zip(intercepts, slope)):
                    if i != len(intercepts) - 1:
                        split = data[..., i * n : (i + 1) * n]
                    else:
                        # extra entries on last one
                        split = data[..., i * n :]
                    pred_y = predict_line(split, inter, sl)
                    errors.append((split[1] - pred_y).pow(2))
                return torch.mean(torch.cat(errors))

            def hlines_penalty(data, y_vals=[10, 30, 50, 70, 90]):
                intercepts = torch.as_tensor(y_vals).unsqueeze(-1)
                return lines_penalty(data, intercepts, 0)

            def penalty(x):
                range_penalty = po.regularize.penalize_range(x, (0, 100))
                hlines = hlines_penalty(x)
                return range_penalty + hlines

            met = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            met.setup(
                initial_image=100 * torch.rand_like(datasaurus),
                optimizer=torch.optim.LBFGS,
            )
            init_state_dict_lint_ignore = met.optimizer.state_dict()

            met.synthesize(50, store_progress=True)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/datasaurus-hlines.pt")
            met_up = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    datasaurus_metamers / "datasaurus-hlines.pt",
                    tensor_equality_atol=1e-7,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings(
            "ignore:plenoptic's methods have mostly been tested on 4d:UserWarning"
        )
        @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
        def test_vlines(self, datasaurus, datasaurus_model, datasaurus_metamers):
            po.set_seed(0)
            torch.use_deterministic_algorithms(True)

            def predict_line(data, intercepts, slope):
                return slope * data[0] + intercepts

            def lines_penalty(data, intercepts, slope):
                # intercepts must be shape [n, 1], slope a scalar or same number of
                # elements as intercepts
                errors = []
                n = data.shape[-1] // intercepts.shape[0]
                if hasattr(slope, "__len__") and len(slope) != 1:
                    assert len(slope) == len(intercepts)
                else:
                    slope = len(intercepts) * [slope]
                for i, (inter, sl) in enumerate(zip(intercepts, slope)):
                    if i != len(intercepts) - 1:
                        split = data[..., i * n : (i + 1) * n]
                    else:
                        # extra entries on last one
                        split = data[..., i * n :]
                    pred_y = predict_line(split, inter, sl)
                    errors.append((split[1] - pred_y).pow(2))
                return torch.mean(torch.cat(errors))

            def vlines_penalty(data, x_vals=[30, 50, 70, 90]):
                intercepts = torch.as_tensor(x_vals).unsqueeze(-1)
                # same as hlines, just swap x and y:
                return lines_penalty(data[[1, 0]], intercepts, 0)

            def penalty(x):
                range_penalty = po.regularize.penalize_range(x, (0, 100))
                vlines = vlines_penalty(x)
                return range_penalty + vlines

            met = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            met.setup(
                initial_image=100 * torch.rand_like(datasaurus),
                optimizer=torch.optim.LBFGS,
            )
            init_state_dict_lint_ignore = met.optimizer.state_dict()

            met.synthesize(50, store_progress=True)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/datasaurus-vlines.pt")
            met_up = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    datasaurus_metamers / "datasaurus-vlines.pt",
                    tensor_equality_atol=1e-7,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings(
            "ignore:plenoptic's methods have mostly been tested on 4d:UserWarning"
        )
        @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
        def test_slantup(self, datasaurus, datasaurus_model, datasaurus_metamers):
            po.set_seed(0)
            torch.use_deterministic_algorithms(True)

            def predict_line(data, intercepts, slope):
                return slope * data[0] + intercepts

            def lines_penalty(data, intercepts, slope):
                # intercepts must be shape [n, 1], slope a scalar or same number of
                # elements as intercepts
                errors = []
                n = data.shape[-1] // intercepts.shape[0]
                if hasattr(slope, "__len__") and len(slope) != 1:
                    assert len(slope) == len(intercepts)
                else:
                    slope = len(intercepts) * [slope]
                for i, (inter, sl) in enumerate(zip(intercepts, slope)):
                    if i != len(intercepts) - 1:
                        split = data[..., i * n : (i + 1) * n]
                    else:
                        # extra entries on last one
                        split = data[..., i * n :]
                    pred_y = predict_line(split, inter, sl)
                    errors.append((split[1] - pred_y).pow(2))
                return torch.mean(torch.cat(errors))

            def slant_penalty(data, slope, intercepts=[-20, -10, 0, 10, 20]):
                slope = torch.as_tensor(slope).unsqueeze(-1)
                intercepts = torch.as_tensor(intercepts).unsqueeze(-1)
                return lines_penalty(data, intercepts, slope)

            def penalty(x):
                range_penalty = po.regularize.penalize_range(x, (0, 100))
                slantup = slant_penalty(x, 1, [-20, -10, 0, 10, 20])
                return range_penalty + slantup

            met = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.001,
            )
            met.setup(
                initial_image=100 * torch.rand_like(datasaurus),
                optimizer=torch.optim.LBFGS,
            )
            init_state_dict_lint_ignore = met.optimizer.state_dict()

            met.synthesize(50, store_progress=True)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/datasaurus-slantup.pt")
            met_up = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.001,
            )
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    datasaurus_metamers / "datasaurus-slantup.pt",
                    tensor_equality_atol=1e-7,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings(
            "ignore:plenoptic's methods have mostly been tested on 4d:UserWarning"
        )
        @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
        def test_slantdown(self, datasaurus, datasaurus_model, datasaurus_metamers):
            po.set_seed(0)
            torch.use_deterministic_algorithms(True)

            def predict_line(data, intercepts, slope):
                return slope * data[0] + intercepts

            def lines_penalty(data, intercepts, slope):
                # intercepts must be shape [n, 1], slope a scalar or same number of
                # elements as intercepts
                errors = []
                n = data.shape[-1] // intercepts.shape[0]
                if hasattr(slope, "__len__") and len(slope) != 1:
                    assert len(slope) == len(intercepts)
                else:
                    slope = len(intercepts) * [slope]
                for i, (inter, sl) in enumerate(zip(intercepts, slope)):
                    if i != len(intercepts) - 1:
                        split = data[..., i * n : (i + 1) * n]
                    else:
                        # extra entries on last one
                        split = data[..., i * n :]
                    pred_y = predict_line(split, inter, sl)
                    errors.append((split[1] - pred_y).pow(2))
                return torch.mean(torch.cat(errors))

            def slant_penalty(data, slope, intercepts=[-20, -10, 0, 10, 20]):
                intercepts = torch.as_tensor(intercepts).unsqueeze(-1)
                slope = torch.as_tensor(slope).unsqueeze(-1)
                return lines_penalty(data, intercepts, slope)

            def penalty(x):
                range_penalty = po.regularize.penalize_range(x, (0, 100))
                slantdown = slant_penalty(x, -0.6, [40, 50, 60, 70, 80])
                return range_penalty + slantdown

            met = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            met.setup(
                initial_image=100 * torch.rand_like(datasaurus),
                optimizer=torch.optim.LBFGS,
            )
            init_state_dict_lint_ignore = met.optimizer.state_dict()

            met.synthesize(50, store_progress=True)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/datasaurus-slantdown.pt")
            met_up = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    datasaurus_metamers / "datasaurus-slantdown.pt",
                    tensor_equality_atol=1e-7,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings(
            "ignore:plenoptic's methods have mostly been tested on 4d:UserWarning"
        )
        @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
        def test_xshape(self, datasaurus, datasaurus_model, datasaurus_metamers):
            po.set_seed(0)
            torch.use_deterministic_algorithms(True)

            def predict_line(data, intercepts, slope):
                return slope * data[0] + intercepts

            def lines_penalty(data, intercepts, slope):
                # intercepts must be shape [n, 1], slope a scalar or same number of
                # elements as intercepts
                errors = []
                n = data.shape[-1] // intercepts.shape[0]
                if hasattr(slope, "__len__") and len(slope) != 1:
                    assert len(slope) == len(intercepts)
                else:
                    slope = len(intercepts) * [slope]
                for i, (inter, sl) in enumerate(zip(intercepts, slope)):
                    if i != len(intercepts) - 1:
                        split = data[..., i * n : (i + 1) * n]
                    else:
                        # extra entries on last one
                        split = data[..., i * n :]
                    pred_y = predict_line(split, inter, sl)
                    errors.append((split[1] - pred_y).pow(2))
                return torch.mean(torch.cat(errors))

            def slant_penalty(data, slope, intercepts):
                intercepts = torch.as_tensor(intercepts).unsqueeze(-1)
                slope = torch.as_tensor(slope).unsqueeze(-1)
                return lines_penalty(data, intercepts, slope)

            def penalty(x):
                range_penalty = po.regularize.penalize_range(x, (0, 100))
                xshape = slant_penalty(x, [1.5, -1.5], [-30, 130])
                return range_penalty + xshape

            met = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            met.setup(
                initial_image=100 * torch.rand_like(datasaurus),
                optimizer=torch.optim.LBFGS,
            )
            init_state_dict_lint_ignore = met.optimizer.state_dict()

            met.synthesize(50, store_progress=True)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/datasaurus-xshape.pt")
            met_up = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    datasaurus_metamers / "datasaurus-xshape.pt",
                    tensor_equality_atol=1e-7,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings(
            "ignore:plenoptic's methods have mostly been tested on 4d:UserWarning"
        )
        @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
        def test_dots(self, datasaurus, datasaurus_model, datasaurus_metamers):
            po.set_seed(0)
            torch.use_deterministic_algorithms(True)

            def dots_penalty(data, target_ctrs, target_r=5):
                target_ctrs = torch.as_tensor(target_ctrs).unsqueeze(-1)
                n = data.shape[-1] // target_ctrs.shape[0]
                errors = []
                for i, ctr in enumerate(target_ctrs):
                    if i != len(target_ctrs) - 1:
                        split = data[..., i * n : (i + 1) * n]
                    else:
                        # extra entries on last one
                        split = data[..., i * n :]
                    rs = (split - ctr).pow(2).sum(0).sqrt()
                    errors.append((rs - target_r).pow(2).mean())
                return torch.stack(errors).mean()

            dot_ctrs = itertools.product(
                [25, datasaurus.mean(-1)[0], 75], [20, datasaurus.mean(-1)[1], 80]
            )
            dot_ctrs = torch.as_tensor(list(dot_ctrs))

            def penalty(x):
                range_penalty = po.regularize.penalize_range(x, (0, 100))
                dots = dots_penalty(x, dot_ctrs)
                return range_penalty + dots

            met = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            met.setup(
                initial_image=100 * torch.rand_like(datasaurus),
                optimizer=torch.optim.LBFGS,
            )
            init_state_dict_lint_ignore = met.optimizer.state_dict()

            met.synthesize(80, store_progress=True)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/datasaurus-dots.pt")
            met_up = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=0.0005,
            )
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    datasaurus_metamers / "datasaurus-dots.pt",
                    tensor_equality_atol=1e-7,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings(
            "ignore:plenoptic's methods have mostly been tested on 4d:UserWarning"
        )
        @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
        def test_away(self, datasaurus, datasaurus_model, datasaurus_metamers):
            po.set_seed(0)
            torch.use_deterministic_algorithms(True)

            def away_penalty(data, target_ctr, std=5):
                target_ctr = torch.as_tensor(target_ctr).unsqueeze(-1)
                r = (data - target_ctr).pow(2).sum(0).sqrt()
                return torch.exp(-r.pow(2) / (2 * std**2)).mean()

            def penalty(x):
                range_penalty = po.regularize.penalize_range(x, (0, 100))
                away = away_penalty(x, [50, 50])
                return range_penalty + away

            met = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=1,
            )
            met.setup(
                initial_image=100 * torch.rand_like(datasaurus),
                optimizer=torch.optim.LBFGS,
            )
            init_state_dict_lint_ignore = met.optimizer.state_dict()

            met.synthesize(50, store_progress=True)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/datasaurus-away.pt")
            met_up = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=1,
            )
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    datasaurus_metamers / "datasaurus-away.pt",
                    tensor_equality_atol=1e-7,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings(
            "ignore:plenoptic's methods have mostly been tested on 4d:UserWarning"
        )
        @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
        def test_star(self, datasaurus, datasaurus_model, datasaurus_metamers):
            po.set_seed(0)
            torch.use_deterministic_algorithms(True)

            def star_penalty(data, target_ctr, target_r, target_theta=-torch.pi / 2):
                target_ctr = torch.as_tensor(target_ctr).unsqueeze(-1)
                # recenter the data and then compute the
                recentered = data - target_ctr
                actual_theta = torch.atan2(*recentered[[1, 0]])
                theta = torch.linspace(
                    -np.pi, np.pi, data.shape[-1], dtype=data.dtype, device=data.device
                )
                r = recentered.pow(2).sum(0).sqrt()

                # modified from https://math.stackexchange.com/a/4293385
                m = 3
                n = 5
                k = torch.as_tensor(1)

                nom = torch.cos((2 * torch.arcsin(k) + torch.pi * m) / (2 * n))
                denom = torch.cos(
                    (
                        2 * torch.arcsin(k * torch.cos(n * (theta + target_theta)))
                        + torch.pi * m
                    )
                    / (2 * n)
                )

                target_r = target_r * nom / denom
                return (r - target_r).pow(2).sum() + (actual_theta - theta).pow(2).sum()

            def range_penalty(x):
                return po.regularize.penalize_range(x, (0, 100))

            def penalty(x):
                star = star_penalty(x, datasaurus.mean(-1), 40)
                return range_penalty(x) + star

            met_star = po.Metamer(
                datasaurus,
                lambda x: x.mean(-1),
                penalty_function=penalty,
            )
            met_star.setup(
                initial_image=100 * torch.rand_like(datasaurus),
                optimizer=torch.optim.LBFGS,
            )
            met_star.synthesize(100, store_progress=True)
            met = po.Metamer(
                datasaurus, datasaurus_model, penalty_function=range_penalty
            )
            met.setup(initial_image=met_star.metamer, optimizer=torch.optim.LBFGS)
            init_state_dict_lint_ignore = met.optimizer.state_dict()
            met.synthesize(50, store_progress=True)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/datasaurus-star.pt")
            met_up = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=1,
            )
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    datasaurus_metamers / "datasaurus-star.pt",
                    tensor_equality_atol=1e-7,
                )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings(
            "ignore:plenoptic's methods have mostly been tested on 4d:UserWarning"
        )
        @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
        def test_oval(self, datasaurus, datasaurus_model, datasaurus_metamers):
            po.set_seed(0)
            torch.use_deterministic_algorithms(True)

            def polygon_penalty(data, target_dist, nbr):
                # break data into "neighborhoods" of nbr points each and tries to make
                # each of their distances match target i.e., form regular polygons of
                # target size
                pts = einops.rearrange(
                    data[..., : nbr * (data.shape[-1] // nbr)],
                    "d (n1 n2) -> n1 n2 d",
                    n2=nbr,
                )
                dist = torch.cdist(pts, pts)
                tril_idx = torch.tril_indices(pts.shape[1], pts.shape[1], -1)
                dist = dist[:, tril_idx[0], tril_idx[1]]
                return (dist - target_dist).pow(2).mean()

            def centroid_penalty(data, target_dist, nbr):
                pts = einops.rearrange(
                    data[..., : nbr * (data.shape[-1] // nbr)],
                    "d (n1 n2) -> n1 n2 d",
                    n2=nbr,
                )
                dist = torch.cdist(pts.mean(1), pts.mean(1))
                tril_idx = torch.tril_indices(pts.shape[0], pts.shape[0], -1)
                dist = dist[tril_idx[0], tril_idx[1]]
                return (dist - target_dist).pow(2).mean()

            nbr = 3

            def penalty(x):
                range_penalty = po.regularize.penalize_range(x, (0, 100))
                # Change these values to whatever you want!
                polygon = polygon_penalty(x, 5, nbr)
                centroid = centroid_penalty(x, 25, nbr)
                return range_penalty + centroid + polygon

            met = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=1,
            )
            met.setup(
                initial_image=100 * torch.rand_like(datasaurus),
                optimizer=torch.optim.LBFGS,
            )
            init_state_dict_lint_ignore = met.optimizer.state_dict()

            met.synthesize(50, store_progress=True)
            # LBFGS's state dict takes a decent amount of memory (it has two keys that
            # are lists of length history_size, where each element is a tensor with the
            # same number of pixels as img), so we reset it for saving purposes -- it's
            # not useful for testing
            met.optimizer.load_state_dict(init_state_dict_lint_ignore)
            met.save("uploaded_files/datasaurus-oval.pt")
            met_up = po.Metamer(
                datasaurus,
                datasaurus_model,
                penalty_function=penalty,
                penalty_lambda=1,
            )
            with pytest.warns(UserWarning, match="You will need to call setup"):
                met_up.load(
                    datasaurus_metamers / "datasaurus-oval.pt",
                    tensor_equality_atol=1e-7,
                )
            compare_metamers(met, met_up)
