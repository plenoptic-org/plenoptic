from math import pi
import numpy as np
import scipy.ndimage
import plenoptic as po
import pytest
import torch
from numpy.random import randint

from conftest import DEVICE


class TestSignal(object):

    def test_polar_amplitude_zero(self):
        a = torch.rand(10, device=DEVICE) * -1
        b = po.tools.rescale(torch.randn(10, device=DEVICE), -pi / 2, pi / 2)

        with pytest.raises(ValueError) as _:
            _, _ = po.tools.polar_to_rectangular(a, b)

    def test_coordinate_identity_transform_rectangular(self):
        dims = (10, 5, 256, 256)
        x = torch.randn(dims, device=DEVICE)
        y = torch.randn(dims, device=DEVICE)

        z = po.tools.polar_to_rectangular(*po.tools.rectangular_to_polar(torch.complex(x, y)))

        assert torch.norm(x - z.real) < 1e-3
        assert torch.norm(y - z.imag) < 1e-3

    def test_coordinate_identity_transform_polar(self):
        dims = (10, 5, 256, 256)

        # ensure vec len a is non-zero by adding .1 and then re-normalizing
        a = torch.rand(dims, device=DEVICE) + 0.1
        a = a / a.max()
        b = po.tools.rescale(torch.randn(dims, device=DEVICE), -pi / 2, pi / 2)

        A, B = po.tools.rectangular_to_polar(po.tools.polar_to_rectangular(a, b))

        assert torch.norm(a - A) < 1e-3
        assert torch.norm(b - B) < 1e-3

    @pytest.mark.parametrize("n", range(1, 15))
    def test_autocorr(self, n):
        x = po.tools.make_synthetic_stimuli().to(DEVICE)
        x_centered = x - x.mean((2, 3), keepdim=True)
        a = po.tools.autocorr(x_centered, n_shifts=n)

        # import matplotlib.pyplot as plt
        # po.imshow(a, zoom=4)
        # plt.show()

        # autocorr with zero delay is variance
        assert (torch.abs(
                torch.var(x, dim=(2, 3)) - a[..., n//2, n//2])
                < 1e-5).all()

        # autocorr can be computed in signal domain directly with roll
        h = randint(-(n//2), ((n+1)//2))
        assert (torch.abs(
                (x_centered * torch.roll(x_centered, h, dims=2)).sum((2, 3))
                / (x.shape[-2]*x.shape[-1])
                - a[..., n//2+h, n//2])
                < 1e-5).all()

        w = randint(-(n//2), ((n+1)//2))
        assert (torch.abs(
                (x_centered * torch.roll(x_centered, w, dims=3)).sum((2, 3))
                / (x.shape[-2]*x.shape[-1])
                - a[..., n//2, n//2+w])
                < 1e-5).all()


class TestStats(object):

    def test_stats(self):
        torch.manual_seed(0)
        B, D = 32, 512
        x = torch.randn(B, D)
        m = torch.mean(x, dim=1, keepdim=True)
        v = po.tools.variance(x, mean=m, dim=1, keepdim=True)
        assert (torch.abs(v - torch.var(x, dim=1, keepdim=True, unbiased=False)
                          ) < 1e-5).all()
        s = po.tools.skew(x, mean=m, var=v, dim=1)
        k = po.tools.kurtosis(x, mean=m, var=v, dim=1)
        assert torch.abs(k.mean() - 3) < 1e-1

        k = po.tools.kurtosis(torch.rand(B, D), dim=1)
        assert k.mean() < 3

        scale = 2
        exp_samples1 = -scale * torch.log(torch.rand(B, D))
        exp_samples2 = -scale * torch.log(torch.rand(B, D))
        lap_samples = exp_samples1 - exp_samples2
        k = po.tools.kurtosis(lap_samples, dim=1)
        assert k.mean() > 3


class TestDownsampleUpsample(object):

    @pytest.mark.parametrize('odd', [0, 1])
    @pytest.mark.parametrize('size', [9, 10, 11, 12])
    def test_filter(self, odd, size):
        img = torch.zeros([1, 1, 24 + odd, 25], device=DEVICE)
        img[0, 0, 12, 12] = 1
        filt = np.zeros([size, size + 1])
        filt[5, 5] = 1
        filt = scipy.ndimage.gaussian_filter(filt, sigma=1)
        filt = torch.tensor(filt, dtype=torch.float32, device=DEVICE)
        img_down = po.tools.correlate_downsample(img, filt=filt)
        img_up = po.tools.upsample_convolve(img_down, odd=(odd, 1), filt=filt)
        assert np.unravel_index(img_up.cpu().numpy().argmax(), img_up.shape) == (0, 0, 12, 12)

    def test_multichannel(self):
        img = torch.randn([10, 3, 24, 25], device=DEVICE)
        filt = torch.randn([5, 5], device=DEVICE)
        img_down = po.tools.correlate_downsample(img, filt=filt)
        img_up = po.tools.upsample_convolve(img_down, odd=(0, 1), filt=filt)
        assert img_up.shape == img.shape
