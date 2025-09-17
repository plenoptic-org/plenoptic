import os.path as op
from contextlib import nullcontext as does_not_raise

import matplotlib.pyplot as plt
import pytest
import torch
from torch import nn

import plenoptic.synthesize.autodiff as autodiff
from conftest import DEVICE, ColorModel, get_model
from plenoptic.metric import mse
from plenoptic.simulate import Gaussian, OnOff
from plenoptic.simulate import LinearNonlinear as LNL
from plenoptic.synthesize import (
    MADCompetition,
    Metamer,
)
from plenoptic.synthesize.eigendistortion import (
    Eigendistortion,
    display_eigendistortion,
    display_eigendistortion_all,
)
from plenoptic.tools import examine_saved_synthesis, remove_grad, set_seed

# to be used for default model instantiation
SMALL_DIM = 20
LARGE_DIM = 100


class TestEigendistortionSynthesis:
    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_method_assertion(self, einstein_img, model):
        einstein_img = einstein_img[..., :SMALL_DIM, :SMALL_DIM]
        ed = Eigendistortion(einstein_img, model)
        with pytest.raises(ValueError, match="method must be in "):
            ed.synthesize(method="asdfsdfasf")

    @pytest.mark.parametrize(
        "model", ["frontend.OnOff.nograd", "ColorModel"], indirect=True
    )
    @pytest.mark.filterwarnings(
        "ignore:Jacobian > 1e6 elements and may cause out-of-memory:UserWarning"
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
    @pytest.mark.filterwarnings(
        "ignore:Randomized SVD complete!:UserWarning",
    )
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
    @pytest.mark.filterwarnings(
        "ignore:Randomized SVD complete!:UserWarning",
    )
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
    @pytest.mark.filterwarnings(
        "ignore:Randomized SVD complete!:UserWarning",
    )
    def test_display(self, model, einstein_img, color_img, method, k):
        img = einstein_img if model.__class__ == OnOff else color_img
        as_rgb = model.__class__ == ColorModel

        img = img[..., :SMALL_DIM, :SMALL_DIM]
        eigendist = Eigendistortion(img, model)
        eigendist.synthesize(k=k, method=method, max_iter=10)
        display_eigendistortion(eigendist, eigenindex=0, as_rgb=as_rgb)
        display_eigendistortion(eigendist, eigenindex=1, as_rgb=as_rgb)

        if method == "power":
            display_eigendistortion(eigendist, eigenindex=-1, as_rgb=as_rgb)
            display_eigendistortion(eigendist, eigenindex=-2, as_rgb=as_rgb)
        elif method == "randomized_svd":  # svd only has top k not bottom k eigendists
            with pytest.raises(ValueError, match="eigenindex must be the index"):
                display_eigendistortion(eigendist, eigenindex=-1)
        plt.close("all")

    @pytest.mark.parametrize(
        "model", ["frontend.OnOff.nograd", "ColorModel"], indirect=True
    )
    @pytest.mark.parametrize("alpha", [1, [1], [1, 10], [1, 10, 10]])
    @pytest.mark.parametrize("eigenindex", [0, [0, -1], [0, -1, 5]])
    @pytest.mark.filterwarnings(
        "ignore:Adding 0.5 to distortion:UserWarning",
    )
    def test_display_all(self, model, einstein_img, color_img, alpha, eigenindex):
        # in this case, we're working with grayscale images
        img = einstein_img if model.__class__ == OnOff else color_img
        as_rgb = model.__class__ == ColorModel

        img = img[..., :SMALL_DIM, :SMALL_DIM]
        eigendist = Eigendistortion(img, model)
        eigendist.synthesize(k=2, method="power", max_iter=10)
        expectation = does_not_raise()
        if isinstance(eigenindex, list):
            if 5 in eigenindex:
                expectation = pytest.raises(
                    ValueError, match="eigenindex must be the index"
                )
            # this will get raised first
            if isinstance(alpha, list) and len(alpha) != len(eigenindex):
                expectation = pytest.raises(ValueError, match="If alpha is a list")
        elif isinstance(alpha, list) and len(alpha) != 1:
            expectation = pytest.raises(ValueError, match="If alpha is a list")
        with expectation:
            display_eigendistortion_all(
                eigendist, eigenindex=eigenindex, alpha=alpha, as_rgb=as_rgb
            )
        plt.close("all")

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("fail", [False, "img", "model"])
    @pytest.mark.parametrize("method", ["exact", "power", "randomized_svd"])
    @pytest.mark.filterwarnings(
        "ignore:Randomized SVD complete!:UserWarning",
    )
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
                    match="Saved and initialized attribute image have different values",
                )
            elif fail == "model":
                model = Gaussian(30).to(DEVICE)
                remove_grad(model)
                model.eval()
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized model output have different values"),
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
                        f"Something went wrong with saving and loading! {k} not"
                        " the same"
                    )
            # check that can resume
            ed_copy.synthesize(max_iter=4, method=method)

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_load_init_fail(self, einstein_img, model, tmp_path):
        eig = Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        eig.save(op.join(tmp_path, "test_eigendistortion_load_init_fail.pt"))
        with pytest.raises(
            ValueError, match="load can only be called with a just-initialized"
        ):
            eig.load(op.join(tmp_path, "test_eigendistortion_load_init_fail.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("fail", [False, "name", "behavior"])
    def test_load_names(self, fail, einstein_img, model, tmp_path):
        # name and behavior same as our LinearNonlinear, but module path is
        # different
        if fail is False:

            class LinearNonlinear(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.model = LNL((31, 31)).to(DEVICE)

                def forward(self, *args, **kwargs):
                    return self.model(*args, **kwargs)

            model2 = LinearNonlinear()
            expectation = does_not_raise()
        # name different but behavior same
        elif fail == "name":

            class LinearNonlinearFAIL(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.model = LNL((31, 31)).to(DEVICE)

                def forward(self, *args, **kwargs):
                    return self.model(*args, **kwargs)

            model2 = LinearNonlinearFAIL()
            expectation = pytest.raises(
                ValueError, match="Saved and initialized model have different names"
            )
        # name same but behavior different
        elif fail == "behavior":

            class LinearNonlinear(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.model = LNL((16, 16)).to(DEVICE)

                def forward(self, *args, **kwargs):
                    return self.model(*args, **kwargs)

            model2 = LinearNonlinear()
            expectation = pytest.raises(
                ValueError,
                match="Saved and initialized model output have different values",
            )
        eig = Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        eig.save(op.join(tmp_path, f"test_eigendistortion_load_names_{fail}.pt"))
        remove_grad(model2)
        model2.eval()
        eig = Eigendistortion(einstein_img, model2)
        with expectation:
            eig.load(op.join(tmp_path, f"test_eigendistortion_load_names_{fail}.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_examine_saved_object(self, einstein_img, model, tmp_path):
        eig = Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        eig.save(op.join(tmp_path, "test_eigendistortion_examine.pt"))
        examine_saved_synthesis(op.join(tmp_path, "test_eigendistortion_examine.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("synth_type", ["met", "mad"])
    def test_load_object_type(self, einstein_img, model, synth_type, tmp_path):
        eig = Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        eig.save(op.join(tmp_path, "test_eigendistortion_load_object_type.pt"))
        if synth_type == "met":
            eig = Metamer(einstein_img, model)
        elif synth_type == "mad":
            eig = MADCompetition(
                einstein_img, mse, mse, "min", metric_tradeoff_lambda=1
            )
        with pytest.raises(
            ValueError, match="Saved object was a.* but initialized object is"
        ):
            eig.load(op.join(tmp_path, "test_eigendistortion_load_object_type.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("model_behav", ["dtype", "shape", "name"])
    def test_load_model_change(self, einstein_img, model, model_behav, tmp_path):
        eig = Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        eig.save(op.join(tmp_path, "test_eigendistortion_load_model_change.pt"))
        if model_behav == "dtype":
            # this actually gets raised in the model validation step (during init), not
            # load.
            expectation = pytest.raises(TypeError, match="model changes precision")
        elif model_behav == "shape":
            expectation = pytest.raises(
                ValueError,
                match="Saved and initialized model output have different shape",
            )
        elif model_behav == "name":
            expectation = pytest.raises(
                ValueError, match="Saved and initialized model have different names"
            )

        class NewModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, x):
                if model_behav == "dtype":
                    return self.model(x).to(torch.float64)
                elif model_behav == "shape":
                    return self.model(x).flatten(-2)
                elif model_behav == "name":
                    return self.model(x)

        model = NewModel()
        model.eval()
        with expectation:
            eig = Eigendistortion(einstein_img, model)
            eig.load(op.join(tmp_path, "test_eigendistortion_load_model_change.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("attribute", ["saved", "init"])
    def test_load_attributes(self, einstein_img, model, attribute, tmp_path):
        eig = Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        if attribute == "saved":
            eig.test = "BAD"
            err_str = "Saved"
        eig.save(op.join(tmp_path, "test_eigendistortion_load_attributes.pt"))
        eig = Eigendistortion(einstein_img, model)
        if attribute == "init":
            eig.test = "BAD"
            err_str = "Initialized"
        with pytest.raises(
            ValueError, match=rf"{err_str} object has 1 attribute\(s\) not present"
        ):
            eig.load(op.join(tmp_path, "test_eigendistortion_load_attributes.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_load_tol(self, einstein_img, model, tmp_path):
        eig = Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=5)
        eig.save(op.join(tmp_path, "test_eigendistortion_load_tol.pt"))
        eig = Eigendistortion(
            einstein_img + 1e-7 * torch.rand_like(einstein_img), model
        )
        with pytest.raises(ValueError, match="Saved and initialized attribute image"):
            eig.load(op.join(tmp_path, "test_eigendistortion_load_tol.pt"))
        eig.load(
            op.join(tmp_path, "test_eigendistortion_load_tol.pt"),
            tensor_equality_atol=1e-7,
        )

    @pytest.mark.parametrize(
        "model", ["naive.Identity", "NonModule", "frontend.OnOff.nograd"], indirect=True
    )
    @pytest.mark.parametrize("to_type", ["dtype", "device"])
    @pytest.mark.filterwarnings("ignore:Unable to call model.to:UserWarning")
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
        # reset so we don't mess up further tests
        if to_type == "dtype":
            ed.to(torch.float32)
        elif to_type == "device" and DEVICE.type != "cpu":
            ed.to(DEVICE)

    # test that we support models with 3d and 4d outputs
    @pytest.mark.parametrize(
        "model",
        ["PortillaSimoncelli", "frontend.LinearNonlinear.nograd"],
        indirect=True,
    )
    def test_model_dimensionality(self, einstein_img, model):
        eig = Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=5, method="power")

    @pytest.mark.parametrize(
        "model",
        [f"diff_dims-{i}" for i in range(1, 6)],
        indirect=True,
    )
    @pytest.mark.parametrize("input_dim", [3, 4, 5])
    @pytest.mark.filterwarnings("ignore:.*mostly been tested on 4d inputs:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:.*mostly been tested on models which:UserWarning"
    )
    def test_dimensionality(self, einstein_img, input_dim, model):
        img = einstein_img.squeeze()[..., :SMALL_DIM, :SMALL_DIM]
        while img.ndimension() < input_dim:
            img = img.unsqueeze(0)
        met = Eigendistortion(img, model)
        met.synthesize(max_iter=5, method="power")

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    @pytest.mark.parametrize("model", ["naive.Identity"], indirect=True)
    def test_map_location(self, curie_img, model, tmp_path):
        ed = Eigendistortion(curie_img, model)
        ed.synthesize(max_iter=4, method="power")
        ed.save(op.join(tmp_path, "test_eig_map_location.pt"))
        # calling load with map_location effectively switches everything
        # over to that device
        model.to("cpu")
        ed_copy = Eigendistortion(curie_img.to("cpu"), model)
        ed_copy.load(op.join(tmp_path, "test_eig_map_location.pt"), map_location="cpu")
        assert ed_copy.eigendistortions.device.type == "cpu"
        assert ed_copy.image.device.type == "cpu"
        ed_copy.synthesize(max_iter=4, method="power")
        # reset model device for other tests
        model.to(DEVICE)

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    @pytest.mark.parametrize("model", ["naive.Identity"], indirect=True)
    def test_to_midsynth(self, curie_img, model):
        ed = Eigendistortion(curie_img, model)
        ed.synthesize(max_iter=4, method="power")
        assert ed.eigendistortions.device.type == "cuda"
        assert ed.image.device.type == "cuda"
        ed.to("cpu")
        ed.synthesize(max_iter=4, method="power")
        assert ed.eigendistortions.device.type == "cpu"
        assert ed.image.device.type == "cpu"
        ed.to("cuda")
        ed.synthesize(max_iter=4, method="power")
        assert ed.eigendistortions.device.type == "cuda"
        assert ed.image.device.type == "cuda"

    @pytest.mark.parametrize("model", ["naive.Identity"], indirect=True)
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
        mdl.eval()

        k = 10
        x_dim = x0.numel()
        V = torch.randn((x_dim, k), device=DEVICE)  # random directions
        V = V / torch.linalg.vector_norm(V, ord=2, dim=0)

        e = Eigendistortion(x0, mdl)
        x, y = e._image_flat, e._representation_flat
        Jv = autodiff.jacobian_vector_product(y, x, V)
        Fv = autodiff.vector_jacobian_product(y, x, Jv)
        assert torch.diag(V.T @ Fv).sqrt().allclose(singular_value, rtol=1e-3)
