import plenoptic as po
import numpy as np
import pytest
import torch
from conftest import DEVICE
from contextlib import nullcontext as does_not_raise


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
        a, f = po.tools.deviation_from_line(b, normalize=True)
        assert torch.abs(a[t//2] - .5) < 1e-2, f"{a[t//2]}"
        assert torch.abs(f[t//2] - 2**.5/2) < 1e-2, f"{f[t//2]}"

    @pytest.mark.parametrize("n_steps", [1, 10])
    @pytest.mark.parametrize("max_norm", [0, 1, 10])
    @pytest.mark.parametrize("multichannel", [False, True])
    def test_brownian_bridge(self, einstein_img, curie_img, n_steps, multichannel, max_norm):
        if multichannel:
            einstein_img = einstein_img.repeat(1, 3, 1, 1)
            curie_img = curie_img.repeat(1, 3, 1, 1)
        bridge = po.tools.sample_brownian_bridge(einstein_img, curie_img, n_steps, max_norm)
        assert bridge.shape == (n_steps+1, *einstein_img.shape[1:]), "sample_brownian_bridge returned a tensor of the wrong shape!"

    @pytest.mark.parametrize("fail", ['batch', 'same_shape', 'n_steps', 'max_norm'])
    def test_brownian_bridge_fail(self, einstein_img, curie_img, fail):
        n_steps = 2
        max_norm = 1
        if fail == 'batch':
            einstein_img = einstein_img.repeat(2, 1, 1, 1)
            curie_img = curie_img.repeat(2, 1, 1, 1)
            expectation = pytest.raises(ValueError, match="input_tensor batch dimension must be 1")
        elif fail == 'same_shape':
            # rand_like preserves DEVICE and dtype
            curie_img = torch.rand_like(curie_img)[..., :128, :128]
            expectation = pytest.raises(ValueError, match="start and stop must be same shape")
        elif fail == 'n_steps':
            n_steps = 0
            expectation = pytest.raises(ValueError, match="n_steps must be positive")
        elif fail == 'max_norm':
            max_norm = -1
            expectation = pytest.raises(ValueError, match="max_norm must be non-negative")
        with expectation:
            po.tools.sample_brownian_bridge(einstein_img, curie_img, n_steps, max_norm)

    @pytest.mark.parametrize("n_steps", [1, 10])
    @pytest.mark.parametrize("multichannel", [False, True])
    def test_straight_line(self, einstein_img, curie_img, n_steps, multichannel):
        if multichannel:
            einstein_img = einstein_img.repeat(1, 3, 1, 1)
            curie_img = curie_img.repeat(1, 3, 1, 1)
        line = po.tools.make_straight_line(einstein_img, curie_img,
                                           n_steps)
        assert line.shape == (n_steps+1, *einstein_img.shape[1:]), "make_straight_line returned a tensor of the wrong shape!"

    @pytest.mark.parametrize("fail", ['batch', 'same_shape', 'n_steps'])
    def test_straight_line_fail(self, einstein_img, curie_img, fail):
        n_steps = 2
        if fail == 'batch':
            einstein_img = einstein_img.repeat(2, 1, 1, 1)
            curie_img = curie_img.repeat(2, 1, 1, 1)
            expectation = pytest.raises(ValueError, match="input_tensor batch dimension must be 1")
        elif fail == 'same_shape':
            # rand_like preserves DEVICE and dtype
            curie_img = torch.rand_like(curie_img)[..., :128, :128]
            expectation = pytest.raises(ValueError, match="start and stop must be same shape")
        elif fail == 'n_steps':
            n_steps = 0
            expectation = pytest.raises(ValueError, match="n_steps must be positive")
        with expectation:
            po.tools.make_straight_line(einstein_img, curie_img, n_steps)

    @pytest.mark.parametrize("n_steps", [0, 1, 10])
    @pytest.mark.parametrize("multichannel", [False, True])
    def test_translation_sequence(self, einstein_img, n_steps, multichannel):
        if n_steps == 0:
            expectation = pytest.raises(ValueError, match="n_steps must be positive")
        else:
            expectation = does_not_raise()
        if multichannel:
            einstein_img = einstein_img.repeat(1, 3, 1, 1)
        with expectation:
            shifted = po.tools.translation_sequence(einstein_img, n_steps)
            assert torch.equal(shifted[0], einstein_img[0]), "somehow first frame changed!"
            assert torch.equal(shifted[1, 0, :, 1], shifted[0, 0, :, 0]), "wrong dimension was translated!"

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


## ADD TESTS FOR:
## - geodesic endpoints don't change, middle does
## - save/load/to
## - in particular: init, synth, change precision, save; init, change precision, load
## - whether it works with multi-channel images -- do the straightness functions work with them?
## - that objective func and calculate jerkiness work with external tensors
## - stop criterion
## - use device
## - continue, amount of saved stuff
