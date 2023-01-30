import plenoptic as po
import numpy as np
import pytest
import torch
from conftest import DEVICE


class TestSequences(object):

    def test_deviation_from_line_and_brownian_bridge(self):
        """this probabilistic test passes with high probability
        in high dimensions, but for reproducibility we
        set the seed manually."""
        torch.manual_seed(0)
        t = 2**6
        d = 2**14
        sqrt_d = int(np.sqrt(d))
        start = torch.randn(1, d).reshape(1, 1, sqrt_d, sqrt_d)
        stop = torch.randn(1, d).reshape(1, 1, sqrt_d, sqrt_d)
        b = po.tools.sample_brownian_bridge(start, stop,
                                            t, d**.5)
        a, f = po.tools.deviation_from_line(b)
        assert torch.abs(a[t//2] - .5) < 1e-2, f"{a[t//2]}"
        assert torch.abs(f[t//2] - 2**.5/2) < 1e-2, f"{f[t//2]}"


class TestGeodesic(object):

    @pytest.mark.parametrize("init", ["straight", "bridge"])
    @pytest.mark.parametrize("optimizer", [None, "SGD"])
    @pytest.mark.parametrize("n_steps", [5, 10])
    def test_geodesic_texture(self, einstein_img_small, init, optimizer, n_steps):

        model = po.simul.OnOff(kernel_size=(31, 31), pretrained=True)
        po.tools.remove_grad(model)
        sequence = po.tools.translation_sequence(einstein_img_small, n_steps)
        moog = po.synth.Geodesic(sequence[0:1], sequence[-1:],
                                 model, n_steps, init)
        if optimizer == "SGD":
            optimizer = torch.optim.SGD([moog._geodesic], lr=.01)
        moog.synthesize(max_iter=5, optimizer=optimizer)
        po.synth.geodesic.plot_loss(moog)
        po.synth.geodesic.plot_deviation_from_line(moog, natural_video=sequence)
        moog.calculate_jerkiness()


# ADD TESTS FOR failed image shape asserts in init
