"""
Test uploaded files.

For the documentation, we have pre-generated some synthesis outputs and uplodaed
them to OSF. Then during docs build, we download them.

These tests check that the outputs haven't changed, notifying us if we need to
update them.
"""

import os

import pytest

import plenoptic as po
from conftest import DEVICE
from plenoptic.data.fetch import fetch_data
from plenoptic.tools.data import _check_tensor_equality


@pytest.mark.skipif(DEVICE.type == "cpu", reason="Only do this on cuda")
class TestUploaded:
    def test_eigendistortion(self):
        po.tools.set_seed(0)
        img = po.data.einstein().to(DEVICE)
        lg = po.simul.LuminanceGainControl(
            (31, 31), pad_mode="circular", pretrained=True, cache_filt=True
        )
        po.tools.remove_grad(lg)
        lg = lg.to(DEVICE)
        lg.eval()
        eig = po.synth.Eigendistortion(img, lg)
        eig.synthesize(max_iter=1000)
        os.makedirs("uploaded_files", exist_ok=True)
        eig.save("uploaded_files/example_eigendistortion.pt")
        eig_up = po.synth.Eigendistortion(img, lg)
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
