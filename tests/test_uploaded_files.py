"""
Test uploaded files.

For the documentation, we have pre-generated some synthesis outputs and uplodaed
them to OSF. Then during docs build, we download them.

These tests check that the outputs haven't changed, notifying us if we need to
update them.
"""

import os
from collections import OrderedDict

import einops
import numpy as np
import pytest
import torch
from torchvision import models
from torchvision.models import feature_extraction

import plenoptic as po
from conftest import DEVICE, DEVICE2
from plenoptic.data.fetch import fetch_data
from plenoptic.tools.data import _check_tensor_equality


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


class PortillaSimoncelliRemove(po.simul.PortillaSimoncelli):
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
        super().__init__(im_shape, n_scales=4, n_orientations=4, spatial_corr_width=9)
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


class PortillaSimoncelliMask(po.simul.PortillaSimoncelli):
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
        spatial_corr_width=9,
        mask=None,
        target=None,
    ):
        super().__init__(im_shape, n_scales=4, n_orientations=4, spatial_corr_width=9)
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


class PortillaSimoncelliMixture(po.simul.PortillaSimoncelli):
    r"""Extend the PortillaSimoncelli model to mix two different images

    Parameters
    ----------
    im_shape: int
        the size of the images being processed by the model

    """

    def __init__(
        self,
        im_shape,
    ):
        super().__init__(im_shape, n_scales=4, n_orientations=4, spatial_corr_width=9)

    def forward(self, images, scales=None):
        r"""Average Texture Statistics representations of two image

        Parameters
        ----------
        images : torch.Tensor
            A 4d tensor containing one or two images to analyze, with shape (i,
            channel, height, width), i in {1,2}.
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
        if images.shape[0] == 2:
            # need the images to be 4d, so we use the "1 element slice"
            stats0 = super().forward(images[:1], scales=scales)
            stats1 = super().forward(images[1:2], scales=scales)
            return (stats0 + stats1) / 2
        else:
            return super().forward(images, scales=scales)


class PortillaSimoncelliMagMeans(po.simul.PortillaSimoncelli):
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

    # overwriting these following two methods allows us to use the plot_representation
    # method with the modified model, making examining it easier.
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


@pytest.mark.skipif(DEVICE.type == "cpu", reason="Only do this on cuda")
class TestDoctest:
    @pytest.mark.xdist_group(name="gpu-0")
    def test_eigendistortion(self, einstein_img_double):
        torch.use_deterministic_algorithms(True)
        po.tools.set_seed(0)
        os.makedirs("uploaded_files", exist_ok=True)
        torch.save(
            torch.random.get_rng_state(),
            "uploaded_files/torch_rng_state_eigendistortion.pt",
        )
        print(np.random.get_state())
        lg = po.simul.LuminanceGainControl(
            (31, 31), pad_mode="circular", pretrained=True, cache_filt=True
        )
        po.tools.remove_grad(lg)
        lg = lg.to(DEVICE).to(einstein_img_double.dtype)
        lg.eval()
        eig = po.synth.Eigendistortion(einstein_img_double, lg)
        eig.synthesize(max_iter=1000)
        eig.save("uploaded_files/example_eigendistortion.pt")
        eig_up = po.synth.Eigendistortion(einstein_img_double, lg)
        eig_up.load(fetch_data("example_eigendistortion.pt"), tensor_equality_atol=1e-7)
        compare_eigendistortions(eig, eig_up)


@pytest.mark.skipif(DEVICE.type == "cpu", reason="Only do this on cuda")
@pytest.mark.skipif(
    os.environ.get("RUN_REGRESSION_SYNTH", "") != "1",
    reason="These take a long time, so don't run every time",
)
class TestTutorialNotebooks:
    class TestDemoEigendistortion:
        @pytest.mark.xdist_group(name="gpu-0")
        def test_berardino_onoff(self, parrot_square_double):
            torch.use_deterministic_algorithms(True)
            po.tools.set_seed(0)
            os.makedirs("uploaded_files", exist_ok=True)
            torch.save(
                torch.random.get_rng_state(),
                "uploaded_files/torch_rng_state_berardino_onoff.pt",
            )
            print(np.random.get_state())
            model = po.simul.OnOff(
                (31, 31),
                pretrained=True,
                cache_filt=True,
                apply_mask=True,
            )
            po.tools.remove_grad(model)
            model = model.to(DEVICE).to(parrot_square_double.dtype)
            model.eval()
            eig = po.synth.Eigendistortion(parrot_square_double, model)
            eig.synthesize(k=3, method="power", max_iter=2000)
            eig.save("uploaded_files/berardino_onoff.pt")
            eig_up = po.synth.Eigendistortion(parrot_square_double, model)
            eig_up.load(fetch_data("berardino_onoff.pt"), tensor_equality_atol=1e-7)
            compare_eigendistortions(eig, eig_up)

        @pytest.mark.xdist_group(name="gpu-1")
        def test_berardino_vgg16(self, parrot_square_double):
            # Create a class that takes the nth layer output of a given model
            class TorchVision(torch.nn.Module):
                def __init__(self, model, return_node: str):
                    super().__init__()
                    self.extractor = feature_extraction.create_feature_extractor(
                        model, return_nodes=[return_node]
                    )
                    self.model = model
                    self.return_node = return_node

                def forward(self, x):
                    return self.extractor(x)[self.return_node]

            def normalize(img_tensor):
                """standardize the image for vgg16"""
                return (img_tensor - img_tensor.mean()) / img_tensor.std()

            torch.use_deterministic_algorithms(True)
            po.tools.set_seed(0)
            os.makedirs("uploaded_files", exist_ok=True)
            torch.save(
                torch.random.get_rng_state(),
                "uploaded_files/torch_rng_state_berardino_vgg16.pt",
            )
            print(np.random.get_state())
            model = models.vgg16(
                weights=models.VGG16_Weights.IMAGENET1K_V1, progress=False
            )
            model = TorchVision(model, "features.11")
            po.tools.remove_grad(model)
            img = parrot_square_double.to(DEVICE2)
            img = normalize(img).repeat(1, 3, 1, 1)
            model = model.to(DEVICE2).to(img.dtype)
            model.eval()
            eig = po.synth.Eigendistortion(img, model)
            eig.synthesize(k=2, method="power", max_iter=5000)
            eig.save("uploaded_files/berardino_vgg16.pt")
            eig_up = po.synth.Eigendistortion(img, model)
            eig_up.load(fetch_data("berardino_vgg16.pt"), tensor_equality_atol=1e-7)
            compare_eigendistortions(eig, eig_up)

    @pytest.mark.filterwarnings(
        "ignore:Loss has converged, stopping synthesis:UserWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:Validating whether model can work with coarse-to-fine:UserWarning"
    )
    class TestPortillaSimoncelli:
        @pytest.fixture(scope="class")
        def ps_images(self):
            img_dir = fetch_data("portilla_simoncelli_images.tar.gz")
            images = po.load_images(img_dir).to(DEVICE).to(torch.float64)
            filenames = [p.stem for p in sorted(img_dir.iterdir())]
            return images, filenames

        def get_specific_img(self, images, filenames, tgt_filename):
            # the clone is here because torch saving/loading preserves views and
            # so extra info would be saved without it:
            # https://docs.pytorch.org/docs/stable/notes/serialization.html#saving-and-loading-tensors-preserves-views
            return images[filenames.index(tgt_filename)].unsqueeze(0).clone()

        @pytest.fixture(scope="class")
        def ps_regression(self):
            return fetch_data("ps_regression.tar.gz")

        @pytest.mark.parametrize(
            "fn",
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
        @pytest.mark.xdist_group(name="gpu-0")
        def test_ps_basic_synthesis(
            self, ps_images, fn, einstein_img_double, ps_regression
        ):
            torch.use_deterministic_algorithms(True)
            po.tools.set_seed(0)
            torch.save(
                torch.random.get_rng_state(),
                f"uploaded_files/torch_rng_state_ps_basic_{fn}.pt",
            )
            print(np.random.get_state())
            if fn.startswith("fig"):
                img = self.get_specific_img(*ps_images, fn)
            elif fn == "einstein":
                img = einstein_img_double
            # this is a sawtooth grating, with 4 scales the steerable pyramid's
            # residual lowpass is uniform and thus correlation between it and
            # the coarsest scale is all NaNs (i.e., the last scale of
            # auto_correlation_reconstructed is all NaNs)
            n_scales = 3 if fn == "fig12b" else 4
            model = po.simul.PortillaSimoncelli(img.shape[-2:], n_scales=n_scales)
            model.to(DEVICE).to(torch.float64)
            met = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            met.setup(
                ((torch.rand_like(img) - 0.5) * 0.1 + img.mean()).clip(min=0, max=1)
            )
            met.synthesize(
                max_iter=350,
                change_scale_criterion=None,
                ctf_iters_to_check=7,
                store_progress=True,
            )
            met.save(f"uploaded_files/ps_basic_synthesis_{fn}.pt")
            met_up = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            met_up.load(
                ps_regression / f"ps_basic_synthesis_{fn}.pt", tensor_equality_atol=1e-7
            )
            compare_metamers(met, met_up)

        # make sure we fail if save load with different stats removed
        def test_ps_remove_fail(self, ps_images, tmp_path):
            img = self.get_specific_img(*ps_images, "fig4a")
            model = PortillaSimoncelliRemove(
                img.shape[-2:], remove_keys=["pixel_statistics"]
            )
            model.to(DEVICE).to(torch.float64)
            im_init = (torch.rand_like(img) - 0.5) * 0.1 + img.mean()
            met = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            met.setup(im_init)
            met.synthesize(
                max_iter=5, change_scale_criterion=None, ctf_iters_to_check=7
            )
            met.save(tmp_path / "test_ps_remove_fail.pt")
            model = PortillaSimoncelliRemove(
                img.shape[-2:], remove_keys=["pixel_statistics", "skew_reconstructed"]
            )
            model.to(DEVICE).to(torch.float64)
            met = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            with pytest.raises(ValueError, match="Saved and initialized model output"):
                met.load(tmp_path / "test_ps_remove_fail.pt")

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
        @pytest.mark.xdist_group(name="gpu-1")
        def test_ps_remove(self, ps_images, fn, stats, remove_bool, ps_regression):
            torch.use_deterministic_algorithms(True)
            po.tools.set_seed(0)
            torch.save(
                torch.random.get_rng_state(),
                f"uploaded_files/torch_rng_state_ps_remove_{fn}_remove-{remove_bool}.pt",
            )
            print(np.random.get_state())
            img = self.get_specific_img(*ps_images, fn).to(DEVICE2)
            if remove_bool:
                model = PortillaSimoncelliRemove(img.shape[-2:], remove_keys=stats)
            else:
                model = po.simul.PortillaSimoncelli(img.shape[-2:])
            model.to(DEVICE2).to(torch.float64)
            met = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            met.setup((torch.rand_like(img) - 0.5) * 0.1 + img.mean())
            met.synthesize(
                max_iter=3000, change_scale_criterion=None, ctf_iters_to_check=7
            )
            met.save(f"uploaded_files/ps_remove_{fn}_remove-{remove_bool}.pt")
            met_up = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            met_up.load(
                ps_regression / f"ps_remove_{fn}_remove-{remove_bool}.pt",
                tensor_equality_atol=1e-7,
            )
            compare_metamers(met, met_up)

        @pytest.mark.filterwarnings("ignore:You will need to call setup:UserWarning")
        @pytest.mark.xdist_group(name="gpu-1")
        def test_ps_mask(self, ps_images, ps_regression):
            torch.use_deterministic_algorithms(True)
            po.tools.set_seed(0)
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
            met = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            met.setup(
                (torch.rand_like(img) - 0.5) * 0.1 + img.mean(),
                optimizer_kwargs={"lr": 0.02, "amsgrad": True},
            )
            met.synthesize(
                max_iter=1000, change_scale_criterion=None, ctf_iters_to_check=7
            )
            met.save("uploaded_files/ps_mask.pt")
            met_up = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            met_up.load(ps_regression / "ps_mask.pt", tensor_equality_atol=1e-7)
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
        @pytest.mark.xdist_group(name="gpu-1")
        def test_ps_mixture(self, ps_images, fn, ps_regression):
            torch.use_deterministic_algorithms(True)
            po.tools.set_seed(0)
            torch.save(
                torch.random.get_rng_state(),
                f"uploaded_files/torch_rng_state_ps_mixture_{'-'.join(fn)}.pt",
            )
            print(np.random.get_state())
            img = torch.cat(
                [
                    self.get_specific_img(*ps_images, fn[0]),
                    self.get_specific_img(*ps_images, fn[1]),
                ]
            ).to(DEVICE2)
            model = PortillaSimoncelliMixture(img.shape[-2:])
            model.to(DEVICE2).to(torch.float64)
            met = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            met.setup(
                (torch.rand_like(img[:1]) - 0.5) * 0.1 + img.mean(),
                optimizer_kwargs={"lr": 0.02, "amsgrad": True},
            )
            met.synthesize(
                max_iter=4000, change_scale_criterion=None, ctf_iters_to_check=7
            )
            met.save(f"uploaded_files/ps_mixture_{'-'.join(fn)}.pt")
            met_up = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            met_up.load(
                ps_regression / f"ps_mixture_{'-'.join(fn)}.pt",
                tensor_equality_atol=1e-7,
            )
            compare_metamers(met, met_up)

        @pytest.mark.parametrize("mag_bool", [True, False])
        @pytest.mark.xdist_group(name="gpu-1")
        def test_ps_mag_means(self, ps_images, mag_bool, ps_regression):
            torch.use_deterministic_algorithms(True)
            po.tools.set_seed(100)
            torch.save(
                torch.random.get_rng_state(),
                f"uploaded_files/torch_rng_state_ps_mag_means-{mag_bool}.pt",
            )
            print(np.random.get_state())
            img = self.get_specific_img(*ps_images, "fig4a").to(DEVICE2)
            if mag_bool:
                model = PortillaSimoncelliMagMeans(img.shape[-2:])
            else:
                model = po.simul.PortillaSimoncelli(
                    img.shape[-2:], spatial_corr_width=7
                )
            model.to(DEVICE2).to(torch.float64)
            met = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            met.setup((torch.rand_like(img) - 0.5) * 0.1 + img.mean())
            met.synthesize(
                max_iter=350,
                change_scale_criterion=None,
                ctf_iters_to_check=7,
                store_progress=True,
            )
            met.save(f"uploaded_files/ps_mag_means-{mag_bool}.pt")
            met_up = po.synth.MetamerCTF(
                img,
                model,
                loss_function=po.tools.optim.l2_norm,
                coarse_to_fine="together",
            )
            met_up.load(
                ps_regression / f"ps_mag_means-{mag_bool}.pt",
                tensor_equality_atol=1e-7,
                map_location=DEVICE2,
            )
            compare_metamers(met, met_up)
