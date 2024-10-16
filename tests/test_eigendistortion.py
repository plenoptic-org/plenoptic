import os.path as op

import matplotlib.pyplot as plt
import pytest
import torch
from torch import nn

import plenoptic.synthesize.autodiff as autodiff
from conftest import DEVICE, get_model
from plenoptic.simulate import Gaussian, OnOff
from plenoptic.synthesize.eigendistortion import (
    Eigendistortion,
    display_eigendistortion,
)
from plenoptic.tools import remove_grad, set_seed

# to be used for default model instantiation
SMALL_DIM = 20
LARGE_DIM = 100


class TestEigendistortionSynthesis:
    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_method_assertion(self, einstein_img, model):
        einstein_img = einstein_img[..., :SMALL_DIM, :SMALL_DIM]
        ed = Eigendistortion(einstein_img, model)
        with pytest.raises(AssertionError, match="method must be in "):
            ed.synthesize(method="asdfsdfasf")

    @pytest.mark.parametrize(
        "model", ["frontend.OnOff.nograd", "ColorModel"], indirect=True
    )
    def test_method_exact(self, model, einstein_img, color_img):
        # in this case, we're working with grayscale images
        if model.__class__ == OnOff:
            n_chans = 1
            img = einstein_img
        else:
            img = color_img
            n_chans = 3
        img = img[..., :SMALL_DIM, :SMALL_DIM]

        ed = Eigendistortion(img, model)
        # invert matrix explicitly
        ed.synthesize(method="exact")

        assert len(ed.eigenvalues) == n_chans * SMALL_DIM**2
        assert len(ed.eigendistortions) == n_chans * SMALL_DIM**2
        assert len(ed.eigenindex) == n_chans * SMALL_DIM**2

        # test that each eigenvector returned is original img shape
        assert ed.eigendistortions.shape[-3:] == (
            n_chans,
            SMALL_DIM,
            SMALL_DIM,
        )

    @pytest.mark.parametrize(
        "model", ["frontend.OnOff.nograd", "ColorModel"], indirect=True
    )
    def test_method_power(self, model, einstein_img, color_img):
        if model.__class__ == OnOff:
            n_chans = 1
            img = einstein_img
        else:
            img = color_img
            n_chans = 3
        img = img[..., :LARGE_DIM, :LARGE_DIM]
        ed = Eigendistortion(img, model)
        ed.synthesize(method="power", max_iter=3)

        # test it should only return two eigenvectors and values
        assert len(ed.eigenvalues) == 2
        assert len(ed.eigendistortions) == 2
        assert len(ed.eigenindex) == 2

        assert ed.eigendistortions.shape[-3:] == (
            n_chans,
            LARGE_DIM,
            LARGE_DIM,
        )

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_orthog_iter(self, model, einstein_img):
        n, k = 30, 10
        n_chans = 1  # TODO color
        einstein_img = einstein_img[..., :n, :n]
        ed = Eigendistortion(einstein_img, model)
        ed.synthesize(k=k, method="power", max_iter=10)

        assert ed.eigendistortions.shape == (k * 2, n_chans, n, n)
        assert ed.eigenindex.allclose(
            torch.cat((torch.arange(k), torch.arange(n**2 - k, n**2)))
        )
        assert len(ed.eigenvalues) == 2 * k

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_method_randomized_svd(self, model, einstein_img):
        n, k = 30, 10
        n_chans = 1  # TODO color
        einstein_img = einstein_img[..., :n, :n]
        ed = Eigendistortion(einstein_img, model)
        ed.synthesize(k=k, method="randomized_svd")
        assert ed.eigendistortions.shape == (k, n_chans, n, n)
        assert ed.eigenindex.allclose(torch.arange(k))
        assert len(ed.eigenvalues) == k

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_method_accuracy(self, model, einstein_img):
        # test pow and svd against ground-truth jacobian (exact) method
        einstein_img = einstein_img[..., 125 : 125 + 25, 125 : 125 + 25]
        e_jac = Eigendistortion(einstein_img, model)
        e_pow = Eigendistortion(einstein_img, model)
        e_svd = Eigendistortion(einstein_img, model)

        k_pow, k_svd = 1, 75
        e_jac.synthesize(method="exact")
        set_seed(0)
        e_pow.synthesize(k=k_pow, method="power", max_iter=2500)
        e_svd.synthesize(k=k_svd, method="randomized_svd")

        print(
            "synthesized first and last: ",
            e_pow.eigenvalues[0],
            e_pow.eigenvalues[-1],
        )
        print(
            "exact first and last: ",
            e_jac.eigenvalues[0],
            e_jac.eigenvalues[-1],
        )

        assert e_pow.eigenvalues[0].isclose(e_jac.eigenvalues[0], atol=1e-2)
        assert e_pow.eigenvalues[-1].isclose(e_jac.eigenvalues[-1], atol=1e-2)
        assert e_svd.eigenvalues[0].isclose(e_jac.eigenvalues[0], atol=1e-2)

    @pytest.mark.parametrize(
        "model", ["frontend.OnOff.nograd", "ColorModel"], indirect=True
    )
    @pytest.mark.parametrize("method", ["power", "randomized_svd"])
    @pytest.mark.parametrize("k", [2, 3])
    def test_display(self, model, einstein_img, color_img, method, k):
        # in this case, we're working with grayscale images
        img = einstein_img if model.__class__ == OnOff else color_img

        img = img[..., :SMALL_DIM, :SMALL_DIM]
        eigendist = Eigendistortion(img, model)
        eigendist.synthesize(k=k, method=method, max_iter=10)
        display_eigendistortion(eigendist, eigenindex=0)
        display_eigendistortion(eigendist, eigenindex=1)

        if method == "power":
            display_eigendistortion(eigendist, eigenindex=-1)
            display_eigendistortion(eigendist, eigenindex=-2)
        elif method == "randomized_svd":  # svd only has top k not bottom k eigendists
            with pytest.raises(AssertionError):
                display_eigendistortion(eigendist, eigenindex=-1)
        plt.close("all")

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    @pytest.mark.parametrize("fail", [False, "img", "model"])
    @pytest.mark.parametrize("method", ["exact", "power", "randomized_svd"])
    def test_save_load(self, einstein_img, model, fail, method, tmp_path):
        if method in ["exact", "randomized_svd"]:
            img = einstein_img[..., :SMALL_DIM, :SMALL_DIM]
        else:
            img = einstein_img
        ed = Eigendistortion(img, model)
        ed.synthesize(max_iter=4, method=method)
        ed.save(op.join(tmp_path, "test_eigendistortion_save_load.pt"))
        if fail:
            if fail == "img":
                img = torch.rand_like(img)
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized image are different",
                )
            elif fail == "model":
                model = Gaussian(30).to(DEVICE)
                remove_grad(model)
                expectation = pytest.raises(
                    RuntimeError,
                    match=("Attribute representation_flat have different shapes"),
                )
            ed_copy = Eigendistortion(img, model)
            with expectation:
                ed_copy.load(
                    op.join(tmp_path, "test_eigendistortion_save_load.pt"),
                    map_location=DEVICE,
                )
        else:
            ed_copy = Eigendistortion(img, model)
            ed_copy.load(
                op.join(tmp_path, "test_eigendistortion_save_load.pt"),
                map_location=DEVICE,
            )
            for k in ["image", "_representation_flat"]:
                if not getattr(ed, k).allclose(getattr(ed_copy, k), rtol=1e-2):
                    raise ValueError(
                        "Something went wrong with saving and loading! %s not"
                        " the same" % k
                    )
            # check that can resume
            ed_copy.synthesize(max_iter=4, method=method)

    @pytest.mark.parametrize("model", ["Identity", "NonModule"], indirect=True)
    @pytest.mark.parametrize("to_type", ["dtype", "device"])
    def test_to(self, curie_img, model, to_type):
        ed = Eigendistortion(curie_img, model)
        ed.synthesize(max_iter=5, method="power")
        if to_type == "dtype":
            # can't use the power method on a float16 tensor, so we use float64 instead
            # here.
            ed.to(torch.float64)
            assert ed.image.dtype == torch.float64
            assert ed.eigendistortions.dtype == torch.float64
        # can only run this one if we're on a device with CPU and GPU.
        elif to_type == "device" and DEVICE.type != "cpu":
            ed.to("cpu")
        ed.eigendistortions - ed.image
        ed.synthesize(max_iter=5, method="power")

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    @pytest.mark.parametrize("model", ["Identity"], indirect=True)
    def test_map_location(self, curie_img, model, tmp_path):
        curie_img = curie_img.to(DEVICE)
        model.to(DEVICE)
        ed = Eigendistortion(curie_img, model)
        ed.synthesize(max_iter=4, method="power")
        ed.save(op.join(tmp_path, "test_eig_map_location.pt"))
        # calling load with map_location effectively switches everything
        # over to that device
        ed_copy = Eigendistortion(curie_img, model)
        ed_copy.load(op.join(tmp_path, "test_eig_map_location.pt"), map_location="cpu")
        assert ed_copy.eigendistortions.device.type == "cpu"
        assert ed_copy.image.device.type == "cpu"
        ed_copy.synthesize(max_iter=4, method="power")

    @pytest.mark.parametrize("model", ["Identity"], indirect=True)
    def test_change_precision_save_load(self, einstein_img, model, tmp_path):
        # Identity model doesn't change when you call .to() with a dtype
        # (unlike those models that have weights) so we use it here
        ed = Eigendistortion(einstein_img, model)
        ed.synthesize(max_iter=5)
        ed.to(torch.float64)
        assert ed.image.dtype == torch.float64, "dtype incorrect!"
        ed.save(op.join(tmp_path, "test_change_prec_save_load.pt"))
        ed_copy = Eigendistortion(einstein_img.to(torch.float64), model)
        ed_copy.load(op.join(tmp_path, "test_change_prec_save_load.pt"))
        ed_copy.synthesize(max_iter=5)
        assert ed_copy.image.dtype == torch.float64, "dtype incorrect!"


class TestAutodiffFunctions:
    @pytest.fixture(scope="class")
    def state(self, einstein_img):
        """variables to be reused across tests in this class"""

        k = 2  # num vectors with which to compute vjp, jvp, Fv
        einstein_img = einstein_img[
            ..., 100 : 100 + 16, 100 : 100 + 16
        ]  # reduce image size

        # eigendistortion object
        ed = Eigendistortion(einstein_img, get_model("frontend.OnOff.nograd"))

        x, y = ed._image_flat, ed._representation_flat

        x_dim = x.flatten().shape[0]
        y_dim = y.flatten().shape[0]

        return x, y, x_dim, y_dim, k

    def test_jacobian(self, state):
        x, y, x_dim, y_dim, k = state

        jac = autodiff.jacobian(y, x)
        assert jac.shape == (y_dim, x_dim)
        assert jac.requires_grad is False

    @pytest.mark.parametrize("detach", [False, True])
    def test_vec_jac_prod(self, state, detach):
        x, y, x_dim, y_dim, k = state

        U = torch.randn((y_dim, k), device=DEVICE)
        U = U / torch.linalg.vector_norm(U, ord=2, dim=0)

        vjp = autodiff.vector_jacobian_product(y, x, U, detach=detach)
        assert vjp.shape == (x_dim, k)
        assert vjp.requires_grad != detach

    def test_jac_vec_prod(self, state):
        x, y, x_dim, y_dim, k = state

        V = torch.randn((x_dim, k), device=DEVICE)
        V = V / torch.linalg.vector_norm(V, ord=2, dim=0)
        jvp = autodiff.jacobian_vector_product(y, x, V)
        assert jvp.shape == (y_dim, k)
        assert x.requires_grad and y.requires_grad
        assert jvp.requires_grad is False

    def test_fisher_vec_prod(self, state):
        x, y, x_dim, y_dim, k = state

        V, _ = torch.linalg.qr(torch.ones((x_dim, k), device=DEVICE), "reduced")
        U = V.clone()
        Jv = autodiff.jacobian_vector_product(y, x, V)
        Fv = autodiff.vector_jacobian_product(y, x, Jv)

        jac = autodiff.jacobian(y, x)

        Fv2 = jac.T @ jac @ U  # manually compute product to compare accuracy

        assert Fv.shape == (x_dim, k)
        assert Fv2.allclose(Fv, atol=1e-6)

    def test_simple_model_eigenvalues(self):
        """Test if Jacobian is constant in all directions for linear model"""
        singular_value = torch.ones(1, device=DEVICE) * 3.0

        class LM(nn.Module):
            """Simple y = Mx where M=3"""

            def __init__(self):
                super().__init__()
                self.M = nn.Linear(1, 1, bias=False)
                self.M.weight.data = singular_value

            def forward(self, x):
                y = self.M(x)
                return y

        x0 = torch.randn((1, 1, 5, 1), requires_grad=True, device=DEVICE)
        x0 = x0 / torch.linalg.vector_norm(x0, ord=2)
        mdl = LM().to(DEVICE)
        remove_grad(mdl)

        k = 10
        x_dim = x0.numel()
        V = torch.randn((x_dim, k), device=DEVICE)  # random directions
        V = V / torch.linalg.vector_norm(V, ord=2, dim=0)

        e = Eigendistortion(x0, mdl)
        x, y = e._image_flat, e._representation_flat
        Jv = autodiff.jacobian_vector_product(y, x, V)
        Fv = autodiff.vector_jacobian_product(y, x, Jv)
        assert torch.diag(V.T @ Fv).sqrt().allclose(singular_value, rtol=1e-3)
