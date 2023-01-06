import plenoptic as po
import pytest
import torch
from conftest import DATA_DIR, DEVICE


class TestSequences(object):

    def test_deviation_from_line_and_brownian_bridge(self):
        """this probabilistic test passes with high probability
        in high dimensions, but for reproducibility we
        set the seed manually."""
        torch.manual_seed(0)
        t = 2**6
        d = 2**15
        b = po.tools.sample_brownian_bridge(torch.randn(1, d),
                                      torch.randn(1, d),
                                      t, d**.5)
        a, f = po.tools.deviation_from_line(b)
        assert torch.abs(a[t//2] - .5) < 1e-2, f"{a[t//2]}"
        assert torch.abs(f[t//2] - 2**.5/2) < 1e-2, f"{f[t//2]}"


class TestGeodesic(object):

    @pytest.mark.parametrize("init", ["straight", "bridge"])
    @pytest.mark.parametrize("optimizer", ["Adam", "SGD"])
    @pytest.mark.parametrize("learning_rate", [.001, .01])
    @pytest.mark.parametrize("n_steps", [5, 10])
    def test_geodesic_texture(self, einstein_img_small, init, optimizer, learning_rate, n_steps):

        model = po.simul.OnOff(kernel_size=(31, 31), pretrained=True)
        sequence = po.tools.translation_sequence(einstein_img_small[0], n_steps)
        moog = po.synth.Geodesic(sequence[0:1], sequence[-1:],
                                 model, n_steps, init)
        if optimizer == "SGD":
            optimizer = torch.optim.SGD([moog.x], lr=learning_rate)
        moog.synthesize(max_iter=5, learning_rate=learning_rate,
                                    optimizer=optimizer)
        moog.plot_loss()
        moog.plot_deviation_from_line(video=sequence)
        moog.calculate_jerkiness()


# ADD TESTS FOR failed image shape asserts in init
