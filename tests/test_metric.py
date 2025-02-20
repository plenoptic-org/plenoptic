import os

import numpy as np
import pytest
import scipy.io as sio
import torch

import plenoptic as po
from conftest import DEVICE, IMG_DIR


@pytest.fixture()
def test_files_dir():
    return po.data.fetch_data("plenoptic-test-files.tar.gz")


def test_find_files(test_files_dir):
    assert os.path.exists(os.path.join(test_files_dir, "buildSCFpyr0.mat"))


@pytest.fixture()
def ssim_images():
    return po.data.fetch_data("ssim_images.tar.gz")


@pytest.fixture()
def msssim_images():
    return po.data.fetch_data("msssim_images.tar.gz")


@pytest.fixture()
def ssim_analysis():
    ssim_analysis = po.data.fetch_data("ssim_analysis.mat")
    return sio.loadmat(ssim_analysis, squeeze_me=True)


@pytest.mark.parametrize(
    "paths",
    [
        IMG_DIR / "mixed",
        IMG_DIR / "256" / "einstein.pgm",
        [IMG_DIR / "256" / "einstein.pgm", IMG_DIR / "256" / "curie.pgm"],
    ],
)
@pytest.mark.parametrize("as_gray", [True, False])
def test_load_images(paths, as_gray):
    if paths == IMG_DIR / "mixed":
        # there are images of different sizes in here, which means we should raise
        # an Exception
        with pytest.raises(Exception):
            images = po.tools.data.load_images(paths, as_gray)
    else:
        images = po.tools.data.load_images(paths, as_gray)
        assert images.ndimension() == 4, "load_images did not return a 4d tensor!"


class TestPerceptualMetrics:
    @pytest.mark.parametrize("weighted", [True, False])
    def test_ssim_grad(self, einstein_img, curie_img, weighted):
        curie_img.requires_grad_()
        assert po.metric.ssim(einstein_img, curie_img, weighted=weighted).requires_grad
        curie_img.requires_grad_(False)

    def test_msssim_grad(self, einstein_img, curie_img):
        curie_img.requires_grad_()
        assert po.metric.ms_ssim(einstein_img, curie_img).requires_grad
        curie_img.requires_grad_(False)

    @pytest.mark.parametrize("func_name", ["ssim", "ms-ssim", "nlpd"])
    @pytest.mark.parametrize("size_A", [(), (3,), (1, 1), (6, 3), (6, 1), (6, 4)])
    @pytest.mark.parametrize("size_B", [(), (3,), (1, 1), (6, 3), (3, 1), (1, 4)])
    def test_batch_handling(self, einstein_img, curie_img, func_name, size_A, size_B):
        func = {
            "ssim": po.metric.ssim,
            "ms-ssim": po.metric.ms_ssim,
            "nlpd": po.metric.nlpd,
        }[func_name]
        A = einstein_img[0, 0].repeat(*size_A, 1, 1)
        B = curie_img[0, 0].repeat(*size_B, 1, 1)

        if not len(size_A) == len(size_B) == 2:
            with pytest.raises(
                Exception, match="Input images should have four dimensions"
            ):
                func(A, B)
        else:
            tgt_size = []
            for i in range(len(size_A)):
                if size_A[i] == size_B[i] or size_A[i] == 1 or size_B[i] == 1:
                    tgt_size.append(max(size_A[i], size_B[i]))
                else:
                    tgt_size = None
                    break
            if tgt_size is None:
                with pytest.raises(
                    Exception,
                    match=(
                        "Either img1 and img2 should have the same number of"
                        " elements in each dimension, or one of them should"
                        " be 1"
                    ),
                ):
                    func(A, B)
            elif tgt_size[1] > 1:
                with pytest.warns(
                    Warning, match="computed separately for each channel"
                ):
                    assert func(A, B).shape == tuple(tgt_size)
            else:
                assert func(A, B).shape == tuple(tgt_size)

    @pytest.mark.parametrize("mode", ["many-to-one", "one-to-many"])
    def test_noise_independence(self, einstein_img, mode):
        # this makes sure that we are drawing the noise independently in the
        # two cases here
        if mode == "many-to-one":
            einstein_img = einstein_img.repeat(2, 1, 1, 1)
            noise_lvl = 1
        elif mode == "one-to-many":
            noise_lvl = [1, 1]
        noisy = po.tools.add_noise(einstein_img, noise_lvl)
        assert not torch.equal(*noisy)

    @pytest.mark.parametrize("noise_lvl", [[1], [128], [2, 4], [2, 4, 8], [0]])
    @pytest.mark.parametrize("noise_as_tensor", [True, False])
    def test_add_noise(self, einstein_img, noise_lvl, noise_as_tensor):
        if noise_as_tensor:
            noise_lvl = torch.as_tensor(
                noise_lvl, dtype=torch.float32, device=DEVICE
            ).unsqueeze(1)
        noisy = po.tools.add_noise(einstein_img, noise_lvl).to(DEVICE)
        if not noise_as_tensor:
            # always needs to be a tensor to properly check with allclose
            noise_lvl = torch.as_tensor(
                noise_lvl, dtype=torch.float32, device=DEVICE
            ).unsqueeze(1)
        assert torch.allclose(po.metric.mse(einstein_img, noisy), noise_lvl)

    @pytest.fixture
    def ssim_base_img(self, ssim_images, ssim_analysis):
        return po.load_images(os.path.join(ssim_images, ssim_analysis["base_img"])).to(
            DEVICE
        )

    @pytest.mark.parametrize("weighted", [True, False])
    @pytest.mark.parametrize("other_img", np.arange(1, 11))
    def test_ssim_analysis(
        self, weighted, other_img, ssim_images, ssim_analysis, ssim_base_img
    ):
        mat_type = {True: "weighted", False: "standard"}[weighted]
        other = po.load_images(os.path.join(ssim_images, f"samp{other_img}.tif")).to(
            DEVICE
        )
        # dynamic range is 1 for these images, because po.load_images
        # automatically re-ranges them. They were comptued with
        # dynamic_range=255 in MATLAB, and by correctly setting this value,
        # that should be corrected for
        plen_val = po.metric.ssim(ssim_base_img, other, weighted)
        mat_val = torch.as_tensor(
            ssim_analysis[mat_type][f"samp{other_img}"].astype(np.float32),
            device=DEVICE,
        )
        # float32 precision is ~1e-6 (see `np.finfo(np.float32)`), and the
        # errors increase through multiplication and other operations.
        print(plen_val - mat_val, plen_val, mat_val)
        assert torch.allclose(plen_val, mat_val.view_as(plen_val), atol=1e-5)

    def test_msssim_analysis(self, msssim_images):
        # True values are defined by https://ece.uwaterloo.ca/~z70wang/research/iwssim/msssim.zip
        true_values = torch.as_tensor(
            [1.0000000, 0.9112161, 0.7699084, 0.8785111, 0.9488805],
            device=DEVICE,
        )
        computed_values = torch.zeros_like(true_values)
        base_img = po.load_images(os.path.join(msssim_images, "samp0.tiff")).to(DEVICE)
        for i in range(len(true_values)):
            other_img = po.load_images(os.path.join(msssim_images, f"samp{i}.tiff")).to(
                DEVICE
            )
            computed_values[i] = po.metric.ms_ssim(base_img, other_img)
        assert torch.allclose(true_values, computed_values)

    def test_nlpd_grad(self, einstein_img, curie_img):
        curie_img.requires_grad_()
        assert po.metric.nlpd(einstein_img, curie_img).requires_grad
        curie_img.requires_grad_(False)  # return to previous state for pytest fixtures

    @pytest.mark.parametrize("model", ["frontend.OnOff"], indirect=True)
    def test_model_metric_grad(self, einstein_img, curie_img, model):
        curie_img.requires_grad_()
        assert po.metric.model_metric(einstein_img, curie_img, model).requires_grad
        curie_img.requires_grad_(False)

    def test_ssim_dtype(self, einstein_img, curie_img):
        po.metric.ssim(einstein_img.to(torch.float64), curie_img.to(torch.float64))

    def test_ssim_dtype_exception(self, einstein_img, curie_img):
        with pytest.raises(ValueError, match="must have same dtype"):
            po.metric.ssim(einstein_img.to(torch.float64), curie_img)

    def test_msssim_dtype(self, einstein_img, curie_img):
        po.metric.ms_ssim(einstein_img.to(torch.float64), curie_img.to(torch.float64))

    def test_msssim_dtype_exception(self, einstein_img, curie_img):
        with pytest.raises(ValueError, match="must have same dtype"):
            po.metric.ms_ssim(einstein_img.to(torch.float64), curie_img)
