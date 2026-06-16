import copy
import re
from contextlib import nullcontext as does_not_raise

import pytest
import torch
from torch import nn

import plenoptic as po
from conftest import DEVICE
from plenoptic._synthesize import autodiff

# to be used for default model instantiation
SMALL_DIM = 20
LARGE_DIM = 100


class TestEigendistortionSynthesis:
    @pytest.fixture(scope="class")
    @classmethod
    def gaussian_einstein_img_eig_saved(cls, einstein_img, tmp_path_factory):
        model = po.models.Gaussian((31, 31)).to(DEVICE)
        po.remove_grad(model)
        model.eval()
        eig = po.Eigendistortion(einstein_img[..., :SMALL_DIM, :SMALL_DIM], model)
        eig.synthesize(max_iter=2)
        save_path = (
            tmp_path_factory.mktemp("data") / "gaussian_einstein_img_eig_saved.pt"
        )
        eig.save(save_path)
        return save_path

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_method_assertion(self, einstein_img, model):
        einstein_img = einstein_img[..., :SMALL_DIM, :SMALL_DIM]
        ed = po.Eigendistortion(einstein_img, model)
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
        if model.__class__ == po.models.OnOff:
            n_chans = 1
            img = einstein_img
        else:
            img = color_img
            n_chans = 3
        img = img[..., :SMALL_DIM, :SMALL_DIM]

        ed = po.Eigendistortion(img, model)
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
        if model.__class__ == po.models.OnOff:
            n_chans = 1
            img = einstein_img
        else:
            img = color_img
            n_chans = 3
        img = img[..., :LARGE_DIM, :LARGE_DIM]
        ed = po.Eigendistortion(img, model)
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
        ed = po.Eigendistortion(einstein_img, model)
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
        ed = po.Eigendistortion(einstein_img, model)
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
        e_jac = po.Eigendistortion(einstein_img, model)
        e_pow = po.Eigendistortion(einstein_img, model)
        e_svd = po.Eigendistortion(einstein_img, model)

        k_pow, k_svd = 1, 75
        e_jac.synthesize(method="exact")
        po.set_seed(0)
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
        ed = po.Eigendistortion(img, model)
        ed.synthesize(max_iter=4, method=method)
        ed.save(tmp_path / "test_eigendistortion_save_load.pt")
        if fail:
            if fail == "img":
                img = torch.rand_like(img)
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized attribute image have different values",
                )
            elif fail == "model":
                model = po.models.Gaussian(30).to(DEVICE)
                po.remove_grad(model)
                model.eval()
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized model output have different values"),
                )
            ed_copy = po.Eigendistortion(img, model)
            with expectation:
                ed_copy.load(
                    tmp_path / "test_eigendistortion_save_load.pt",
                    map_location=DEVICE,
                )
        else:
            ed_copy = po.Eigendistortion(img, model)
            ed_copy.load(
                tmp_path / "test_eigendistortion_save_load.pt",
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
        eig = po.Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        eig.save(tmp_path / "test_eigendistortion_load_init_fail.pt")
        with pytest.raises(
            ValueError, match="load can only be called with a just-initialized"
        ):
            eig.load(tmp_path / "test_eigendistortion_load_init_fail.pt")

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
                    self.model = po.models.LinearNonlinear((31, 31)).to(DEVICE)

                def forward(self, *args, **kwargs):
                    return self.model(*args, **kwargs)

            model2 = LinearNonlinear()
            expectation = does_not_raise()
        # name different but behavior same
        elif fail == "name":

            class LinearNonlinearFAIL(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.model = po.models.LinearNonlinear((31, 31)).to(DEVICE)

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
                    self.model = po.models.LinearNonlinear((16, 16)).to(DEVICE)

                def forward(self, *args, **kwargs):
                    return self.model(*args, **kwargs)

            model2 = LinearNonlinear()
            expectation = pytest.raises(
                ValueError,
                match="Saved and initialized model output have different values",
            )
        eig = po.Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        eig.save(tmp_path / f"test_eigendistortion_load_names_{fail}.pt")
        po.remove_grad(model2)
        model2.eval()
        eig = po.Eigendistortion(einstein_img, model2)
        with expectation:
            eig.load(tmp_path / f"test_eigendistortion_load_names_{fail}.pt")

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_examine_saved_object(self, einstein_img, model, tmp_path):
        eig = po.Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        eig.save(tmp_path / "test_eigendistortion_examine.pt")
        po.io.examine_saved_synthesis(tmp_path / "test_eigendistortion_examine.pt")

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("synth_type", ["met", "mad"])
    def test_load_object_type(self, einstein_img, model, synth_type, tmp_path):
        eig = po.Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        eig.save(tmp_path / "test_eigendistortion_load_object_type.pt")
        if synth_type == "met":
            eig = po.Metamer(einstein_img, model)
        elif synth_type == "mad":
            eig = po.MADCompetition(
                einstein_img,
                po.metric.mse,
                po.metric.mse,
                "min",
                metric_tradeoff_lambda=1,
            )
        with pytest.raises(
            ValueError, match="Saved object was a.* but initialized object is"
        ):
            eig.load(tmp_path / "test_eigendistortion_load_object_type.pt")

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("model_behav", ["dtype", "shape", "name"])
    def test_load_model_change(self, einstein_img, model, model_behav, tmp_path):
        eig = po.Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        eig.save(tmp_path / "test_eigendistortion_load_model_change.pt")
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
            eig = po.Eigendistortion(einstein_img, model)
            eig.load(tmp_path / "test_eigendistortion_load_model_change.pt")

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("attribute", ["saved", "init"])
    def test_load_attributes(self, einstein_img, model, attribute, tmp_path):
        eig = po.Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=4)
        if attribute == "saved":
            eig.test = "BAD"
            err_str = "Saved"
        eig.save(tmp_path / "test_eigendistortion_load_attributes.pt")
        eig = po.Eigendistortion(einstein_img, model)
        if attribute == "init":
            eig.test = "BAD"
            err_str = "Initialized"
        with pytest.raises(
            ValueError, match=rf"{err_str} object has 1 attribute\(s\) not present"
        ):
            eig.load(tmp_path / "test_eigendistortion_load_attributes.pt")

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_load_tol(self, einstein_img, model, tmp_path):
        eig = po.Eigendistortion(einstein_img, model)
        eig.synthesize(max_iter=5)
        eig.save(tmp_path / "test_eigendistortion_load_tol.pt")
        eig = po.Eigendistortion(
            einstein_img * (1 - 1e-7) + 1e-7 * torch.rand_like(einstein_img), model
        )
        with pytest.raises(ValueError, match="Saved and initialized attribute image"):
            eig.load(tmp_path / "test_eigendistortion_load_tol.pt")
        eig.load(
            tmp_path / "test_eigendistortion_load_tol.pt",
            tensor_equality_atol=1e-7,
        )

    @pytest.mark.parametrize(
        "model", ["naive.Identity", "NonModule", "frontend.OnOff.nograd"], indirect=True
    )
    @pytest.mark.parametrize("to_type", ["dtype", "device"])
    @pytest.mark.filterwarnings("ignore:Unable to call model.to:UserWarning")
    def test_to(self, curie_img, model, to_type):
        ed = po.Eigendistortion(curie_img, model)
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
        eig = po.Eigendistortion(einstein_img, model)
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
        met = po.Eigendistortion(img, model)
        met.synthesize(max_iter=5, method="power")

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    @pytest.mark.parametrize("model", ["naive.Identity"], indirect=True)
    def test_map_location(self, curie_img, model, tmp_path):
        ed = po.Eigendistortion(curie_img, model)
        ed.synthesize(max_iter=4, method="power")
        ed.save(tmp_path / "test_eig_map_location.pt")
        # calling load with map_location effectively switches everything
        # over to that device
        model.to("cpu")
        ed_copy = po.Eigendistortion(curie_img.to("cpu"), model)
        ed_copy.load(tmp_path / "test_eig_map_location.pt", map_location="cpu")
        assert ed_copy.eigendistortions.device.type == "cpu"
        assert ed_copy.image.device.type == "cpu"
        ed_copy.synthesize(max_iter=4, method="power")
        # reset model device for other tests
        model.to(DEVICE)

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    @pytest.mark.parametrize("model", ["naive.Identity"], indirect=True)
    def test_to_midsynth(self, curie_img, model):
        ed = po.Eigendistortion(curie_img, model)
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
        ed = po.Eigendistortion(einstein_img, model)
        ed.synthesize(max_iter=5)
        ed.to(torch.float64)
        assert ed.image.dtype == torch.float64, "dtype incorrect!"
        ed.save(tmp_path / "test_change_prec_save_load.pt")
        ed_copy = po.Eigendistortion(einstein_img.to(torch.float64), model)
        ed_copy.load(tmp_path / "test_change_prec_save_load.pt")
        ed_copy.synthesize(max_iter=5)
        assert ed_copy.image.dtype == torch.float64, "dtype incorrect!"

    @pytest.mark.parametrize(
        "model",
        ["naive.Gaussian.nograd"],
        indirect=True,
    )
    def test_load_same_names(
        self, gaussian_einstein_img_eig_saved, einstein_img, model
    ):
        # test that if we have a different object with same name and behavior, there's
        # no problem with loading:
        class Gaussian(po.models.Gaussian):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        model = Gaussian((31, 31)).to(DEVICE)
        po.remove_grad(model)
        model.eval()
        eig = po.Eigendistortion(einstein_img[..., :SMALL_DIM, :SMALL_DIM], model)
        eig.load(gaussian_einstein_img_eig_saved)
        # basically, the following just ensures that the above does what I think it does
        assert eig.model.__class__ != po.models.Gaussian

    @pytest.mark.parametrize(
        "model",
        ["naive.Gaussian.nograd"],
        indirect=True,
    )
    def test_raise_on_checks_diff_name(
        self, gaussian_einstein_img_eig_saved, einstein_img, model
    ):
        # test that if we have a different object with different name and same behavior,
        # need to set raise_on_checks=False to load, and it only raises one warning,
        # about the name

        class GaussianWrong(po.models.Gaussian):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        model = GaussianWrong((31, 31)).to(DEVICE)
        po.remove_grad(model)
        model.eval()
        eig = po.Eigendistortion(einstein_img[..., :SMALL_DIM, :SMALL_DIM], model)
        error_str = "Saved and initialized model have different names"
        with pytest.raises(ValueError, match=error_str):
            eig.load(gaussian_einstein_img_eig_saved)
        eig = po.Eigendistortion(einstein_img[..., :SMALL_DIM, :SMALL_DIM], model)
        with pytest.warns() as record:
            eig.load(gaussian_einstein_img_eig_saved, raise_on_checks=False)
            assert len(record) == 1
            assert record[0].message.args[0].startswith(error_str)

    @pytest.mark.parametrize(
        "model",
        ["naive.Gaussian.nograd"],
        indirect=True,
    )
    def test_raise_on_checks_new_attr(
        self, gaussian_einstein_img_eig_saved, einstein_img, model
    ):
        # proxy for plenoptic changing API of metamer: test that creating a variant of
        # Metamer with a new attribute refuses load regularly, and loads with only that
        # warning if raise_on_checks=False
        class Eigendistortion(po.Eigendistortion):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.new_attribute = True

        eig = Eigendistortion(einstein_img[..., :SMALL_DIM, :SMALL_DIM], model)
        error_str = "Saved object was a plenoptic.Eigendistortion"
        with pytest.raises(ValueError, match=error_str):
            eig.load(gaussian_einstein_img_eig_saved)
        with pytest.warns() as record:
            eig.load(gaussian_einstein_img_eig_saved, raise_on_checks=False)
            assert len(record) == 2
            assert record[0].message.args[0].startswith(error_str)
            assert (
                record[1]
                .message.args[0]
                .startswith("Initialized object has 1 attribute")
            )
        assert eig.new_attribute is True

    @pytest.mark.parametrize(
        "model",
        ["frontend.OnOff.nograd"],
        indirect=True,
    )
    @pytest.mark.parametrize("tensor_problem", ["shape", "value"])
    def test_raise_on_checks_different(
        self, gaussian_einstein_img_eig_saved, einstein_img, model, tensor_problem
    ):
        # users *shouldn't* do the behavior shown here, but just to ensure I know what
        # happens in these cases: we're going to change all the initial arguments and
        # see that the load can happen anyway with raise_on_checks=False, and then
        # assert whether the attribute came from the saved or initialized object. we
        # have to test the dtype, shape, value individually, the others can be tested at
        # once.
        image = einstein_img[..., :SMALL_DIM, :SMALL_DIM]
        if tensor_problem == "shape":
            image = image[..., :16, :16]
        elif tensor_problem == "value":
            image = torch.rand_like(image)

        eig = po.Eigendistortion(image, model)
        with pytest.warns() as record:
            eig.load(gaussian_einstein_img_eig_saved, raise_on_checks=False)
            errors = [
                ("attribute image", tensor_problem),
                ("model output", "shape"),
                ("model", "names"),
            ]
            assert len(record) == len(errors)
            for rec, (attr, err) in zip(record, errors):
                assert rec.message.args[0].startswith(
                    f"Saved and initialized {attr} have different {err}"
                )
        # the non-callables are set to the saved values
        assert torch.equal(eig.image, einstein_img[..., :SMALL_DIM, :SMALL_DIM])
        # the callables are set to the initialized values
        assert eig.model.__class__ == po.models.OnOff

    @pytest.mark.parametrize(
        "model",
        ["naive.Gaussian.nograd"],
        indirect=True,
    )
    def test_raise_on_checks_dtype_error(
        self, gaussian_einstein_img_eig_saved, einstein_img_double, model
    ):
        # dtype and device issues will still raise errors, even if raise_on_checks,
        # because things are weird else: model should only operate on a single
        # dtype/device, so why expect it to operate on different versions in
        # initialization and load? also, there are better ways to fix this issue: the to
        # method / map_location arg.
        model = copy.deepcopy(model).to(torch.float64)
        eig = po.Eigendistortion(
            einstein_img_double[..., :SMALL_DIM, :SMALL_DIM], model
        )
        error_str = "Saved and initialized attribute image have different dtype"
        with pytest.raises(ValueError, match=error_str):
            eig.load(gaussian_einstein_img_eig_saved)
        with pytest.raises(ValueError, match=error_str):
            eig.load(gaussian_einstein_img_eig_saved, raise_on_checks=False)

    @pytest.mark.parametrize(
        "model",
        ["naive.Gaussian.nograd"],
        indirect=True,
    )
    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only do this on cuda")
    def test_raise_on_checks_device_error(
        self, gaussian_einstein_img_eig_saved, einstein_img, model
    ):
        # dtype and device issues will still raise errors, even if raise_on_checks,
        # because things are weird else: model should only operate on a single
        # dtype/device, so why expect it to operate on different versions in
        # initialization and load? also, there are better ways to fix this issue: the to
        # method / map_location arg.
        model = copy.deepcopy(model).to("cpu")
        eig = po.Eigendistortion(
            einstein_img[..., :SMALL_DIM, :SMALL_DIM].to("cpu"), model
        )
        error_str = "Saved and initialized attribute image have different device"
        with pytest.raises(ValueError, match=error_str):
            eig.load(gaussian_einstein_img_eig_saved)
        with pytest.raises(ValueError, match=error_str):
            eig.load(gaussian_einstein_img_eig_saved, raise_on_checks=False)

    @pytest.mark.parametrize(
        "model",
        ["frontend.OnOff.nograd", "naive.Gaussian.nograd"],
        indirect=True,
    )
    @pytest.mark.parametrize("method", ["exact", "power", "randomized_svd"])
    @pytest.mark.filterwarnings(
        "ignore:Randomized SVD complete!:UserWarning",
    )
    def test_repr(self, einstein_img, model, method):
        # don't need the regex match in this case, but keeping it here to be consistent
        # with the other synthesis object test_repr and in case we need to extend it in
        # the future.
        eig = po.Eigendistortion(einstein_img[..., :SMALL_DIM, :SMALL_DIM], model)
        if isinstance(model, po.models.OnOff):
            model_str = (
                r"OnOff\(\n    \(center_surround\): CenterSurround\(\)\n    "
                r"\(luminance\): Gaussian\(\)\n    \(contrast\): Gaussian\(\)"
                r"\n  \)"
            )
        elif isinstance(model, po.models.Gaussian):
            model_str = r"Gaussian\(\)"
        expected_str = (
            rf"Eigendistortion\(\n  image = torch.Size\(\[1, 1, 20, 20\]\)"
            rf" \(torch.float32\),\n  model = {model_str},\n\)"
        )
        assert repr(eig) == str(eig)
        assert re.match(expected_str, repr(eig))
        # synthesize doesn't change repr
        eig.synthesize(method=method, max_iter=2)
        assert repr(eig) == str(eig)
        assert re.match(expected_str, repr(eig))


class TestAutodiffFunctions:
    @pytest.fixture(scope="class")
    @classmethod
    def state(cls, einstein_img_double):
        """variables to be reused across tests in this class"""

        # num vectors with which to compute vjp, jvp, Fv
        k = 2
        # reduce image size
        einstein_img = einstein_img_double[..., 100 : 100 + 16, 100 : 100 + 16]

        model = po.models.OnOff((31, 31), pretrained=True, cache_filt=True)
        po.remove_grad(model)
        model.to(einstein_img.dtype).to(DEVICE)
        model.eval()

        # eigendistortion object
        ed = po.Eigendistortion(einstein_img, model)

        x, y = ed._image_flat, ed._representation_flat

        x_dim = x.flatten().shape[0]
        y_dim = y.flatten().shape[0]

        return x, y, x_dim, y_dim, k

    def test_jacobian(self, state):
        x, y, x_dim, y_dim, k = state

        jac = autodiff._jacobian(y, x)
        assert jac.shape == (y_dim, x_dim)
        assert jac.requires_grad is False

    @pytest.mark.parametrize("detach", [False, True])
    def test_vec_jac_prod(self, state, detach):
        x, y, x_dim, y_dim, k = state

        U = torch.randn((y_dim, k), device=DEVICE)
        U = U / torch.linalg.vector_norm(U, ord=2, dim=0)

        vjp = autodiff._vector_jacobian_product(y, x, U, detach=detach)
        assert vjp.shape == (x_dim, k)
        assert vjp.requires_grad != detach

    def test_jac_vec_prod(self, state):
        x, y, x_dim, y_dim, k = state

        V = torch.randn((x_dim, k), device=DEVICE)
        V = V / torch.linalg.vector_norm(V, ord=2, dim=0)
        jvp = autodiff._jacobian_vector_product(y, x, V)
        assert jvp.shape == (y_dim, k)
        assert x.requires_grad and y.requires_grad
        assert jvp.requires_grad is False

    def test_fisher_vec_prod(self, state):
        x, y, x_dim, y_dim, k = state

        V, _ = torch.linalg.qr(
            torch.ones((x_dim, k), device=DEVICE, dtype=x.dtype), "reduced"
        )
        U = V.clone()
        Jv = autodiff._jacobian_vector_product(y, x, V)
        Fv = autodiff._vector_jacobian_product(y, x, Jv)

        jac = autodiff._jacobian(y, x)

        Fv2 = jac.T @ jac @ U  # manually compute product to compare accuracy

        assert Fv.shape == (x_dim, k)
        assert Fv2.allclose(Fv, atol=1e-5)

    @pytest.mark.filterwarnings("ignore:input_tensor range is:UserWarning")
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
        po.remove_grad(mdl)
        mdl.eval()

        k = 10
        x_dim = x0.numel()
        V = torch.randn((x_dim, k), device=DEVICE)  # random directions
        V = V / torch.linalg.vector_norm(V, ord=2, dim=0)

        e = po.Eigendistortion(x0, mdl)
        x, y = e._image_flat, e._representation_flat
        Jv = autodiff._jacobian_vector_product(y, x, V)
        Fv = autodiff._vector_jacobian_product(y, x, Jv)
        assert torch.diag(V.T @ Fv).sqrt().allclose(singular_value, rtol=1e-3)
