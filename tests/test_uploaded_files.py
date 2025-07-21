"""
Test uploaded files.

For the documentation, we have pre-generated some synthesis outputs and uplodaed
them to OSF. Then during docs build, we download them.

These tests check that the outputs haven't changed, notifying us if we need to
update them.
"""

import os

import numpy as np
import pytest
import torch
from torchvision import models
from torchvision.models import feature_extraction

import plenoptic as po
from conftest import DEVICE, DEVICE2
from plenoptic.data.fetch import fetch_data
from plenoptic.tools.data import _check_tensor_equality


@pytest.mark.skipif(DEVICE.type == "cpu", reason="Only do this on cuda")
class TestDoctest:
    @pytest.mark.xdist_group(name="gpu-0")
    def test_eigendistortion(self, einstein_img_double):
        torch.use_deterministic_algorithms(True)
        po.tools.set_seed(0)
        os.makedirs("uploaded_files", exist_ok=True)
        torch.save(torch.random.get_rng_state(), "uploaded_files/torch_rng_state.pt")
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
        for k in ["image", "_representation_flat", "eigendistortions"]:
            _check_tensor_equality(
                getattr(eig, k),
                getattr(eig_up, k),
                "Test",
                "OSF",
                1e-5,
                1e-7,
                f"{k} has different {{error_type}}! Update the OSF version.",
            )


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
            for k in ["image", "_representation_flat", "eigendistortions"]:
                _check_tensor_equality(
                    getattr(eig, k),
                    getattr(eig_up, k),
                    "Test",
                    "OSF",
                    1e-5,
                    1e-7,
                    f"{k} has different {{error_type}}! Update the OSF version.",
                )

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
            for k in ["image", "_representation_flat", "eigendistortions"]:
                _check_tensor_equality(
                    getattr(eig, k),
                    getattr(eig_up, k),
                    "Test",
                    "OSF",
                    1e-5,
                    1e-7,
                    f"{k} has different {{error_type}}! Update the OSF version.",
                )
