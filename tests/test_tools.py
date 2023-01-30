from math import pi
import numpy as np
import scipy.ndimage
import plenoptic as po
import pytest
import torch
from numpy.random import randint
from contextlib import nullcontext as does_not_raise

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
    
    @pytest.mark.parametrize('size_A', [1, 3])
    @pytest.mark.parametrize('size_B', [1, 2, 3])
    def test_add_noise(self, einstein_img, size_A, size_B):
        A = einstein_img.repeat(size_A, 1, 1, 1)
        B = size_B * [4]
        if size_A != size_B and size_A != 1 and size_B != 1:
            with pytest.raises(Exception):
                po.tools.add_noise(A, B)
        else:
            assert po.tools.add_noise(A, B).shape[0] == max(size_A, size_B)


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
        img = torch.zeros([1, 1, 24 + odd, 25], device=DEVICE, dtype=torch.float32)
        img[0, 0, 12, 12] = 1
        filt = np.zeros([size, size + 1])
        filt[5, 5] = 1
        filt = scipy.ndimage.gaussian_filter(filt, sigma=1)
        filt = torch.tensor(filt, dtype=torch.float32, device=DEVICE)
        img_down = po.tools.correlate_downsample(img, filt=filt)
        img_up = po.tools.upsample_convolve(img_down, odd=(odd, 1), filt=filt)
        assert np.unravel_index(img_up.cpu().numpy().argmax(), img_up.shape) == (0, 0, 12, 12)

        img_down = po.tools.blur_downsample(img)
        img_up = po.tools.upsample_blur(img_down, odd=(odd, 1))
        assert np.unravel_index(img_up.cpu().numpy().argmax(), img_up.shape) == (0, 0, 12, 12)

    def test_multichannel(self):
        img = torch.randn([10, 3, 24, 25], device=DEVICE, dtype=torch.float32)
        filt = torch.randn([5, 5], device=DEVICE, dtype=torch.float32)
        img_down = po.tools.correlate_downsample(img, filt=filt)
        img_up = po.tools.upsample_convolve(img_down, odd=(0, 1), filt=filt)
        assert img_up.shape == img.shape

        img_down = po.tools.blur_downsample(img)
        img_up = po.tools.upsample_blur(img_down, odd=(0, 1))
        assert img_up.shape == img.shape


class TestValidate(object):

    # https://docs.pytest.org/en/4.6.x/example/parametrize.html#parametrizing-conditional-raising
    @pytest.mark.parametrize('shape,expectation', [
        ((1, 1, 16, 16), does_not_raise()),
        ((1, 3, 16, 16), does_not_raise()),
        ((2, 1, 16, 16), does_not_raise()),
        ((2, 3, 16, 16), does_not_raise()),
        ((1, 1, 1, 16, 16), pytest.raises(ValueError, match="input_tensor must be torch.Size")),
        ((1, 16, 16), pytest.raises(ValueError, match="input_tensor must be torch.Size")),
        ((16, 16), pytest.raises(ValueError, match="input_tensor must be torch.Size")),
    ])
    def test_input_shape(self, shape, expectation):
        img = torch.rand(*shape)
        with expectation:
            po.tools.validate.validate_input(img)

    def test_input_no_batch(self):
        img = torch.rand(2, 1, 16, 16)
        with pytest.raises(ValueError, match="input_tensor batch dimension must be 1"):
            po.tools.validate.validate_input(img, no_batch=True)

    @pytest.mark.parametrize('minmax,expectation', [
        ('min',pytest.raises(ValueError, match="input_tensor range must lie within")),
        ('max',pytest.raises(ValueError, match="input_tensor range must lie within")),
        ('range',pytest.raises(ValueError, match=r"allowed_range\[0\] must be strictly less")),
    ])
    def test_input_allowed_range(self, minmax, expectation):
        img = torch.rand(1, 1, 16, 16)
        allowed_range = (0, 1)
        if minmax == 'min':
            img -= 1
        elif minmax == 'max':
            img += 1
        elif minmax == 'range':
            allowed_range = (1, 0)
        with expectation:
            po.tools.validate.validate_input(img, allowed_range=allowed_range)

    @pytest.mark.parametrize('model', ['frontend.OnOff'], indirect=True)
    def test_model_learnable(self, model):
        with pytest.raises(ValueError, match="model adds gradient to input"):
            po.tools.validate.validate_model(model)

    def test_model_numpy_comp(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, img):
                return np.fft.fft(img)

        model = TestModel()
        with pytest.raises(ValueError, match="model does not return a torch.Tensor object"):
            po.tools.validate.validate_model(model)

    def test_model_detach(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, img):
                return img.detach()

        model = TestModel()
        with pytest.raises(ValueError, match="model strips gradient from input"):
            po.tools.validate.validate_model(model)

    def test_model_numpy_and_back(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, img):
                return torch.from_numpy(np.fft.fft(img))

        model = TestModel()
        with pytest.raises(ValueError, match="model tries to cast the input into something other"):
            po.tools.validate.validate_model(model)

    def test_model_precision(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, img):
                return img.to(torch.float16)

        model = TestModel()
        with pytest.raises(TypeError, match="model changes precision of input"):
            po.tools.validate.validate_model(model)

    @pytest.mark.parametrize('direction', ['squeeze', 'unsqueeze'])
    def test_model_output_dim(self, direction):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, img):
                if direction == 'squeeze':
                    return img.squeeze()
                elif direction == 'unsqueeze':
                    return img.unsqueeze(0)

        model = TestModel()
        with pytest.raises(ValueError, match="When given a 4d input, model output"):
            po.tools.validate.validate_model(model)

    @pytest.mark.skipif(DEVICE.type == 'cpu', reason="Only makes sense to test on cuda")
    def test_model_device(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, img):
                return img.to('cpu')

        model = TestModel()
        with pytest.raises(RuntimeError, match="model changes device of input"):
            po.tools.validate.validate_model(model)

    @pytest.mark.parametrize("model", ['ColorModel'], indirect=True)
    def test_model_image_shape(self, model):
        img_shape = (1, 3, 16, 16)
        po.tools.validate.validate_model(model, image_shape=img_shape)

    def test_validate_ctf_scales(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, img):
                return img

        model = TestModel()
        with pytest.raises(AttributeError, match="model has no scales attribute"):
            po.tools.validate.validate_coarse_to_fine(model)

    def test_validate_ctf_arg(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scales = [0, 1, 2]
            def forward(self, img):
                return img

        model = TestModel()
        with pytest.raises(TypeError, match="model forward method does not accept scales argument"):
            po.tools.validate.validate_coarse_to_fine(model)

    def test_validate_ctf_shape(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scales = [0, 1, 2]
            def forward(self, img, scales=[]):
                return img

        model = TestModel()
        with pytest.raises(ValueError, match="Output of model forward method doesn't change shape"):
            po.tools.validate.validate_coarse_to_fine(model)

    def test_validate_ctf_pass(self):
        model = po.simul.PortillaSimoncelli((64, 64))
        po.tools.validate.validate_coarse_to_fine(model, image_shape=(1, 1, *model.image_shape))

    def test_validate_metric_inputs(self):
        metric = lambda x: x
        with pytest.raises(TypeError, match="metric should be callable and accept two"):
            po.tools.validate.validate_metric(metric)

    def test_validate_metric_output_shape(self):
        metric = lambda x, y: x-y
        with pytest.raises(ValueError, match="metric should return a scalar value but output"):
            po.tools.validate.validate_metric(metric)

    def test_validate_metric_identical(self):
        metric = lambda x, y : (x+y).mean()
        with pytest.raises(ValueError, match="metric should return <= 5e-7 on two identical"):
            po.tools.validate.validate_metric(metric)

    def test_remove_grad(self):
        # can't use the conftest version, because remove_grad modifies the
        # model in place
        model = po.simul.OnOff((31, 31), pretrained=True, cache_filt=True).to(DEVICE)
        po.tools.remove_grad(model)
        po.tools.validate.validate_model(model)
