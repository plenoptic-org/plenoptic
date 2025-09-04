import math
import os.path as op
from contextlib import nullcontext as does_not_raise

import pytest
import torch

import plenoptic as po
from conftest import DEVICE, check_loss_saved_synth


# in order for pickling to work with functions, they must be defined at top of
# module: https://stackoverflow.com/a/36995008
def custom_loss(x1, x2):
    return (x1 - x2).sum()


class TestMetamers:
    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("loss_func", ["mse", "l2", "custom"])
    @pytest.mark.parametrize(
        "fail",
        [False, "img", "model", "loss", "range_penalty", "dtype", "allowed_range"],
    )
    @pytest.mark.parametrize("range_penalty", [0.1, 0])
    @pytest.mark.parametrize("allowed_range", [(0, 1), (-1, 1)])
    def test_save_load(
        self,
        einstein_img,
        model,
        loss_func,
        fail,
        range_penalty,
        allowed_range,
        tmp_path,
    ):
        if loss_func == "mse":
            loss = po.tools.optim.mse
        elif loss_func == "l2":
            loss = po.tools.optim.l2_norm
        elif loss_func == "custom":
            loss = custom_loss
        met = po.synth.Metamer(
            einstein_img,
            model,
            loss_function=loss,
            allowed_range=allowed_range,
            range_penalty_lambda=range_penalty,
        )
        met.synthesize(max_iter=4, store_progress=True)
        met.save(op.join(tmp_path, "test_metamer_save_load.pt"))
        if fail:
            if fail == "img":
                einstein_img = torch.rand_like(einstein_img)
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized attribute image have different values",
                )
            elif fail == "model":
                model = po.simul.Gaussian(30).to(DEVICE)
                po.tools.remove_grad(model)
                model.eval()
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized model output have different values"),
                )
            elif fail == "loss":
                loss = po.tools.optim.relative_sse
                expectation = pytest.raises(
                    ValueError,
                    match=(
                        "Saved and initialized loss_function output have different "
                        "values"
                    ),
                )
            elif fail == "allowed_range":
                allowed_range = (0, 5)
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized allowed_range are different"),
                )
            elif fail == "range_penalty":
                range_penalty = 0.5
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized range_penalty_lambda are different"),
                )
            elif fail == "dtype":
                einstein_img = einstein_img.to(torch.float64)
                # need to create new instance of model, because otherwise the
                # version with doubles as weights will persist for other tests
                model = po.simul.LinearNonlinear((31, 31)).to(DEVICE)
                po.tools.remove_grad(model)
                model.to(torch.float64)
                model.eval()
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized attribute image have different dtype",
                )
            met_copy = po.synth.Metamer(
                einstein_img,
                model,
                loss_function=loss,
                allowed_range=allowed_range,
                range_penalty_lambda=range_penalty,
            )
            with expectation:
                met_copy.load(
                    op.join(tmp_path, "test_metamer_save_load.pt"),
                    map_location=DEVICE,
                )
        else:
            met_copy = po.synth.Metamer(
                einstein_img,
                model,
                loss_function=loss,
                range_penalty_lambda=range_penalty,
                allowed_range=allowed_range,
            )
            met_copy.load(
                op.join(tmp_path, "test_metamer_save_load.pt"),
                map_location=DEVICE,
            )
            for k in [
                "image",
                "saved_metamer",
                "metamer",
                "target_representation",
            ]:
                if not getattr(met, k).allclose(getattr(met_copy, k), rtol=1e-2):
                    raise ValueError(
                        f"Something went wrong with saving and loading! {k} not"
                        " the same"
                    )
            # check loss functions correctly saved
            met_loss = met.loss_function(
                met.model(met.metamer), met.target_representation
            )
            met_copy_loss = met_copy.loss_function(
                met.model(met.metamer), met_copy.target_representation
            )
            if not torch.allclose(met_loss, met_copy_loss, rtol=1e-2):
                raise ValueError(
                    "Loss function not properly saved! Before saving was"
                    f" {met_loss}, after loading was {met_copy_loss}"
                )
            # check that can resume
            met_copy.synthesize(
                max_iter=4,
                store_progress=True,
            )

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_setup_initial_image(self, einstein_img, curie_img, model):
        met = po.synth.Metamer(einstein_img, model)
        met.setup(curie_img)
        met.synthesize(5)

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("warn", ["shape", "ndim"])
    @pytest.mark.filterwarnings("ignore:.*mostly been tested on 4d inputs:UserWarning")
    def test_setup_initial_image_warn(self, einstein_img, curie_img, model, warn):
        met = po.synth.Metamer(einstein_img, model)
        if warn == "shape":
            img = curie_img[..., :255, :255]
        elif warn == "ndim":
            img = curie_img[0]
        txt = "initial_image and image are different sizes"
        with pytest.warns(UserWarning, match=txt):
            met.setup(img)

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_setup_fail(self, einstein_img, model):
        met = po.synth.Metamer(einstein_img, model)
        met.setup()
        with pytest.raises(ValueError, match=r"setup\(\) can only be called once"):
            met.setup()

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.filterwarnings("ignore:You will need to call setup:UserWarning")
    def test_synth_then_setup(self, einstein_img, model, tmp_path):
        met = po.synth.Metamer(einstein_img, model)
        met.setup(optimizer=torch.optim.SGD)
        met.synthesize(max_iter=4)
        met.save(op.join(tmp_path, "test_metamer_synth_then_setup.pt"))
        met = po.synth.Metamer(einstein_img, model)
        met.load(op.join(tmp_path, "test_metamer_synth_then_setup.pt"))
        with pytest.raises(ValueError, match="Don't know how to initialize"):
            met.synthesize(5)
        met.setup(optimizer=torch.optim.SGD)
        met.synthesize(5)

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_setup_load_fail(self, einstein_img, model, tmp_path):
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=4)
        met.save(op.join(tmp_path, "test_metamer_setup_load_fail.pt"))
        met = po.synth.Metamer(einstein_img, model)
        met.load(op.join(tmp_path, "test_metamer_setup_load_fail.pt"))
        with pytest.raises(
            ValueError, match="Cannot set initial_image after calling load"
        ):
            met.setup(po.data.curie())

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("fail", ["synth", "setup", "continue"])
    def test_load_init_fail(self, einstein_img, model, fail, tmp_path):
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=4, store_progress=True)
        met.save(op.join(tmp_path, "test_metamer_load_init_fail.pt"))
        if fail != "continue":
            met = po.synth.Metamer(einstein_img, model)
            if fail == "setup":
                met.setup()
            elif fail == "synth":
                met.synthesize(max_iter=4, store_progress=True)
        with pytest.raises(
            ValueError, match="load can only be called with a just-initialized"
        ):
            met.load(op.join(tmp_path, "test_metamer_load_init_fail.pt"))

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
                    self.model = po.simul.LinearNonlinear((31, 31)).to(DEVICE)

                def forward(self, *args, **kwargs):
                    return self.model(*args, **kwargs)

            model2 = LinearNonlinear()
            expectation = does_not_raise()
        # name different but behavior same
        elif fail == "name":

            class LinearNonlinearFAIL(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.model = po.simul.LinearNonlinear((31, 31)).to(DEVICE)

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
                    self.model = po.simul.LinearNonlinear((16, 16)).to(DEVICE)

                def forward(self, *args, **kwargs):
                    return self.model(*args, **kwargs)

            model2 = LinearNonlinear()
            expectation = pytest.raises(
                ValueError,
                match="Saved and initialized model output have different values",
            )
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=4, store_progress=True)
        met.save(op.join(tmp_path, f"test_metamer_load_names_{fail}.pt"))
        po.tools.remove_grad(model2)
        model2.eval()
        met = po.synth.Metamer(einstein_img, model2)
        with expectation:
            met.load(op.join(tmp_path, f"test_metamer_load_names_{fail}.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_examine_saved_object(self, einstein_img, model, tmp_path):
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=4, store_progress=True)
        met.save(op.join(tmp_path, "test_metamer_examine.pt"))
        po.tools.examine_saved_synthesis(op.join(tmp_path, "test_metamer_examine.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("synth_type", ["eig", "mad"])
    def test_load_object_type(self, einstein_img, model, synth_type, tmp_path):
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=4, store_progress=True)
        met.save(op.join(tmp_path, "test_metamer_load_object_type.pt"))
        if synth_type == "eig":
            met = po.synth.Eigendistortion(einstein_img, model)
        elif synth_type == "mad":
            met = po.synth.MADCompetition(
                einstein_img,
                po.metric.mse,
                po.metric.mse,
                "min",
                metric_tradeoff_lambda=1,
            )
        with pytest.raises(
            ValueError, match="Saved object was a.* but initialized object is"
        ):
            met.load(op.join(tmp_path, "test_metamer_load_object_type.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("model_behav", ["dtype", "shape", "name"])
    def test_load_model_change(self, einstein_img, model, model_behav, tmp_path):
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=4, store_progress=True)
        met.save(op.join(tmp_path, "test_metamer_load_model_change.pt"))
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
            met = po.synth.Metamer(einstein_img, model)
            met.load(op.join(tmp_path, "test_metamer_load_model_change.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("loss_behav", ["dtype", "shape", "name"])
    def test_load_loss_change(self, einstein_img, model, loss_behav, tmp_path):
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=4, store_progress=True)
        met.save(op.join(tmp_path, "test_metamer_load_loss_change.pt"))

        def new_loss(x, y):
            if loss_behav == "dtype":
                return po.tools.optim.mse(x, y).to(torch.float64)
            elif loss_behav == "shape":
                return torch.stack(
                    [po.tools.optim.mse(x, y) for _ in range(2)]
                ).unsqueeze(0)
            elif loss_behav == "name":
                return po.tools.optim.mse(x, y)

        met = po.synth.Metamer(einstein_img, model, loss_function=new_loss)
        if loss_behav == "name":
            expectation_str = "Saved and initialized loss_function have different names"
        else:
            expectation_str = (
                "Saved and initialized loss_function output have different"
                f" {loss_behav}"
            )
        with pytest.raises(ValueError, match=expectation_str):
            met.load(op.join(tmp_path, "test_metamer_load_loss_change.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("attribute", ["saved", "init"])
    def test_load_attributes(self, einstein_img, model, attribute, tmp_path):
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=4, store_progress=True)
        if attribute == "saved":
            met.test = "BAD"
            err_str = "Saved"
        met.save(op.join(tmp_path, "test_metamer_load_attributes.pt"))
        met = po.synth.Metamer(einstein_img, model)
        if attribute == "init":
            met.test = "BAD"
            err_str = "Initialized"
        with pytest.raises(
            ValueError, match=rf"{err_str} object has 1 attribute\(s\) not present"
        ):
            met.load(op.join(tmp_path, "test_metamer_load_attributes.pt"))

    @pytest.mark.parametrize(
        "model",
        ["frontend.OnOff.nograd.ctf"],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "optim_opts",
        [
            None,
            "SGD",
            "SGD-args",
            "Adam",
            "Adam-args",
            "Scheduler",
            "Scheduler-args",
            "LBFGS",
            "LBFGS-args",
        ],
    )
    @pytest.mark.parametrize("fail", [True, False])
    @pytest.mark.parametrize("coarse_to_fine", [True, False])
    @pytest.mark.filterwarnings("ignore:You will need to call setup:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:Validating whether model can work with coarse-to-fine:UserWarning"
    )
    def test_load_optimizer(
        self, curie_img, model, coarse_to_fine, optim_opts, fail, tmp_path
    ):
        if coarse_to_fine:
            met = po.synth.MetamerCTF(curie_img, model)
        else:
            met = po.synth.Metamer(curie_img, model)
        scheduler = None
        optimizer = None
        optimizer_kwargs = None
        scheduler_kwargs = None
        check_optimizer = [torch.optim.Adam, {"eps": 1e-8, "lr": 0.01}]
        check_scheduler = None
        if optim_opts is not None:
            if "Scheduler" in optim_opts:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
                check_scheduler = [
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                    {"factor": 0.1},
                ]
                if "args" in optim_opts:
                    scheduler_kwargs = {"factor": 1e-3}
                    check_scheduler[1] = {"factor": 1e-3}
            else:
                if "Adam" in optim_opts:
                    optimizer = torch.optim.Adam
                elif "SGD" in optim_opts:
                    optimizer = torch.optim.SGD
                    check_optimizer[0] = torch.optim.SGD
                    check_optimizer[1] = {"lr": 0.01}
                elif "LBFGS" in optim_opts:
                    optimizer = torch.optim.LBFGS
                    check_optimizer[0] = torch.optim.LBFGS
                    check_optimizer[1] = {"lr": 0.01}
                if "args" in optim_opts:
                    optimizer_kwargs = {"lr": 1}
                    check_optimizer[1] = {"lr": 1}
        met.setup(
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        met.synthesize(max_iter=5)
        met.save(op.join(tmp_path, "test_metamer_optimizer.pt"))
        if coarse_to_fine:
            met = po.synth.MetamerCTF(curie_img, model)
        else:
            met = po.synth.Metamer(curie_img, model)
        met.load(op.join(tmp_path, "test_metamer_optimizer.pt"))
        optimizer_kwargs = None
        scheduler_kwargs = None
        if not fail:
            if optim_opts is not None:
                if "Adam" in optim_opts:
                    optimizer = torch.optim.Adam
                elif "SGD" in optim_opts:
                    optimizer = torch.optim.SGD
                elif "LBFGS" in optim_opts:
                    optimizer = torch.optim.LBFGS
                if "Scheduler" in optim_opts:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            expectation = does_not_raise()
        else:
            expect_str = "User-specified optimizer must have same type"
            if optim_opts is None:
                optimizer = torch.optim.SGD
            else:
                if optim_opts == "Adam":
                    optimizer = torch.optim.SGD
                elif optim_opts == "Adam-args":
                    optimizer = torch.optim.Adam
                    optimizer_kwargs = {"lr": 1}
                    expect_str = (
                        "When initializing optimizer after load, optimizer_kwargs"
                    )
                elif optim_opts == "SGD":
                    optimizer = None
                    expect_str = "Don't know how to initialize saved optimizer"
                elif optim_opts == "SGD-args":
                    optimizer = torch.optim.SGD
                    optimizer_kwargs = {"lr": 1}
                    expect_str = (
                        "When initializing optimizer after load, optimizer_kwargs"
                    )
                elif optim_opts == "LBFGS":
                    optimizer = None
                    expect_str = "Don't know how to initialize saved optimizer"
                elif optim_opts == "LBFGS-args":
                    optimizer = torch.optim.LBFGS
                    optimizer_kwargs = {"lr": 1}
                    expect_str = (
                        "When initializing optimizer after load, optimizer_kwargs"
                    )
                elif optim_opts == "Scheduler":
                    scheduler = torch.optim.lr_scheduler.ConstantLR
                    expect_str = "User-specified scheduler must have same type"
                elif optim_opts == "Scheduler-args":
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
                    scheduler_kwargs = {"factor": 1e-3}
                    expect_str = (
                        "When initializing scheduler after load, scheduler_kwargs"
                    )
            expectation = pytest.raises(ValueError, match=expect_str)
        # these fail during setup
        with expectation:
            met.setup(
                optimizer=optimizer,
                scheduler=scheduler,
                optimizer_kwargs=optimizer_kwargs,
                scheduler_kwargs=scheduler_kwargs,
            )
            met.synthesize(max_iter=5)
            if not isinstance(met.optimizer, check_optimizer[0]):
                raise ValueError("Didn't properly set optimizer!")
            state_dict = met.optimizer.state_dict()["param_groups"][0]
            for k, v in check_optimizer[1].items():
                if state_dict[k] != v:
                    raise ValueError(
                        "Didn't properly set optimizer kwargs! "
                        f"Expected {v} but got {state_dict[k]}!"
                    )
            if check_scheduler is not None:
                if not isinstance(met.scheduler, check_scheduler[0]):
                    raise ValueError("Didn't properly set scheduler!")
                state_dict = met.scheduler.state_dict()
                for k, v in check_scheduler[1].items():
                    if met.scheduler.state_dict()[k] != v:
                        raise ValueError("Didn't properly set scheduler kwargs!")
            elif met.scheduler is not None:
                raise ValueError("Didn't set scheduler to None!")

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_load_tol(self, einstein_img, model, tmp_path):
        met = po.synth.Metamer(einstein_img, model, allowed_range=(-1, 2))
        met.synthesize(5)
        met.save(op.join(tmp_path, "test_metamer_load_tol.pt"))
        met = po.synth.Metamer(
            einstein_img + 1e-7 * torch.rand_like(einstein_img),
            model,
            allowed_range=(-1, 2),
        )
        with pytest.raises(ValueError, match="Saved and initialized attribute image"):
            met.load(op.join(tmp_path, "test_metamer_load_tol.pt"))
        met.load(
            op.join(tmp_path, "test_metamer_load_tol.pt"), tensor_equality_atol=1e-7
        )

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("load", [True, False])
    @pytest.mark.filterwarnings("ignore:You will need to call setup:UserWarning")
    def test_resume_synthesis(self, einstein_img, curie_img, model, load, tmp_path):
        met = po.synth.Metamer(einstein_img, model)
        # Adam has some stochasticity in its initialization(?), so this test doesn't
        # quite work with it (it does if you do po.tools.set_seed(2) at the top of the
        # function)
        met.setup(curie_img, optimizer=torch.optim.SGD)
        met.synthesize(10)
        met_copy = po.synth.Metamer(einstein_img, model)
        met_copy.setup(curie_img, optimizer=torch.optim.SGD)
        met_copy.synthesize(5)
        if load:
            met_copy.save(op.join(tmp_path, "test_metamer_resume_synthesis.pt"))
            met_copy = po.synth.Metamer(einstein_img, model)
            met_copy.load(op.join(tmp_path, "test_metamer_resume_synthesis.pt"))
            met_copy.setup(optimizer=torch.optim.SGD)
            met_copy.synthesize(5)
        else:
            met_copy.synthesize(5)
        if not torch.allclose(met.metamer, met_copy.metamer):
            raise ValueError("Resuming synthesis different than just continuing!")

    # test that we support models with 3d and 4d outputs
    @pytest.mark.parametrize(
        "model",
        ["PortillaSimoncelli", "frontend.LinearNonlinear.nograd"],
        indirect=True,
    )
    def test_model_dimensionality_real(self, einstein_img, model):
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(5)

    @pytest.mark.parametrize(
        "model",
        [f"diff_dims-{i}" for i in range(1, 6)],
        indirect=True,
    )
    @pytest.mark.parametrize("input_dim", [2, 3, 4, 5])
    @pytest.mark.filterwarnings("ignore:.*mostly been tested on 4d inputs:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:.*mostly been tested on models which:UserWarning"
    )
    def test_dimensionality(self, einstein_img, input_dim, model):
        img = einstein_img.squeeze()
        while img.ndimension() < input_dim:
            img = img.unsqueeze(0)
        met = po.synth.Metamer(img, model)
        met.synthesize(5)

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("store_progress", [True, 2, 3])
    def test_store_rep(self, einstein_img, model, store_progress):
        metamer = po.synth.Metamer(einstein_img, model)
        max_iter = 3
        if store_progress == 3:
            max_iter = 6
        metamer.synthesize(max_iter=max_iter, store_progress=store_progress)
        check_loss_saved_synth(
            metamer.losses,
            metamer.saved_metamer,
            max_iter,
            metamer.objective_function,
            metamer.store_progress,
        )
        metamer.synthesize(max_iter=max_iter, store_progress=store_progress)
        check_loss_saved_synth(
            metamer.losses,
            metamer.saved_metamer,
            2 * max_iter,
            metamer.objective_function,
            metamer.store_progress,
        )

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_save_metamer_empty(self, einstein_img, model):
        metamer = po.synth.Metamer(einstein_img, model)
        torch.equal(metamer.saved_metamer, torch.empty(0))
        metamer.synthesize(max_iter=3)
        torch.equal(metamer.saved_metamer, metamer.metamer.to("cpu"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_metamer_empty_loss(self, einstein_img, model):
        met = po.synth.Metamer(einstein_img, model)
        assert met.objective_function().numel() == 0
        torch.equal(met.losses, torch.empty(0))
        met.setup()
        assert isinstance(met.objective_function(), torch.Tensor)
        assert met.losses.numel() > 0
        met.synthesize(max_iter=2)
        assert isinstance(met.objective_function(), torch.Tensor)
        assert met.losses.numel() > 0

    @pytest.mark.parametrize("iteration", [None, 0, -2, -3, 2, 1, 6, -7])
    @pytest.mark.parametrize("store_progress", [True, False, 2])
    @pytest.mark.parametrize("store_progress_behavior", ["floor", "ceiling", "round"])
    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_metamer_get_progress(
        self, einstein_img, model, iteration, store_progress, store_progress_behavior
    ):
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=5, store_progress=store_progress)
        if store_progress_behavior == "floor":
            func = math.floor
        elif store_progress_behavior == "ceiling":
            func = math.ceil
        else:
            func = round
        expected_dict = {}
        if iteration is None:
            expected_dict["losses"] = met.losses[-1]
            expected_dict.update(
                {
                    "iteration": 5,
                    "gradient_norm": None,
                    "pixel_change_norm": None,
                }
            )
            if store_progress:
                expected_dict.update(
                    {
                        "saved_metamer": met.metamer,
                        "store_progress_iteration": 5,
                    }
                )
        elif iteration not in [6, -7]:
            expected_dict["losses"] = met.losses[iteration]
            if iteration < 0:
                # add one to account for loss and these attributes being off by one
                expected_dict.update(
                    {
                        "iteration": 6 + iteration,
                        "gradient_norm": met.gradient_norm[iteration + 1],
                        "pixel_change_norm": met.pixel_change_norm[iteration + 1],
                    }
                )
                if store_progress:
                    iter = func((6 + iteration) / store_progress)
                    expected_dict.update(
                        {
                            "saved_metamer": met.saved_metamer[iter],
                            "store_progress_iteration": iter * store_progress,
                        }
                    )
            else:
                expected_dict.update(
                    {
                        "iteration": iteration,
                        "gradient_norm": met.gradient_norm[iteration],
                        "pixel_change_norm": met.pixel_change_norm[iteration],
                    }
                )
                if store_progress:
                    iter = func(iteration / store_progress)
                    expected_dict.update(
                        {
                            "saved_metamer": met.saved_metamer[iter],
                            "store_progress_iteration": iter * store_progress,
                        }
                    )
        if iteration in [6, -7]:
            expectation = pytest.raises(IndexError, match=".*out of bounds with.*")
        elif store_progress == 2 and iteration in [1, -3]:
            expectation = pytest.warns(
                UserWarning, match="loss iteration and iteration"
            )
        else:
            expectation = does_not_raise()
        with expectation:
            progress = met.get_progress(iteration, store_progress_behavior)
            assert progress.keys() == expected_dict.keys()
            for k, v in progress.items():
                if isinstance(v, torch.Tensor):
                    assert torch.equal(v, expected_dict[k]), f"{k} not as expected!"
                else:
                    assert v == expected_dict[k], f"{k} not as expected!"

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_continue(self, einstein_img, model):
        metamer = po.synth.Metamer(einstein_img, model)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.synthesize(max_iter=3, store_progress=True)

    @pytest.mark.parametrize("model", ["SPyr"], indirect=True)
    @pytest.mark.parametrize("coarse_to_fine", ["separate", "together"])
    @pytest.mark.filterwarnings(
        "ignore:Validating whether model can work with coarse-to-fine:UserWarning"
    )
    @pytest.mark.filterwarnings("ignore:Loss has converged:UserWarning")
    def test_coarse_to_fine(self, einstein_img, model, coarse_to_fine, tmp_path):
        metamer = po.synth.MetamerCTF(
            einstein_img, model, coarse_to_fine=coarse_to_fine
        )
        metamer.synthesize(
            max_iter=5,
            stop_iters_to_check=1,
            change_scale_criterion=10,
            ctf_iters_to_check=1,
        )
        assert len(metamer.scales_finished) > 0, "Didn't actually switch scales!"

        metamer.save(op.join(tmp_path, "test_metamer_ctf.pt"))
        metamer_copy = po.synth.MetamerCTF(
            einstein_img, model, coarse_to_fine=coarse_to_fine
        )
        metamer_copy.load(op.join(tmp_path, "test_metamer_ctf.pt"), map_location=DEVICE)
        # check the ctf-related attributes all saved correctly
        for k in [
            "coarse_to_fine",
            "scales",
            "scales_loss",
            "scales_timing",
            "scales_finished",
        ]:
            if not getattr(metamer, k) == (getattr(metamer_copy, k)):
                raise ValueError(
                    f"Something went wrong with saving and loading! {k} not the same"
                )
        # check we can resume
        metamer.synthesize(
            max_iter=5,
            stop_iters_to_check=1,
            change_scale_criterion=10,
            ctf_iters_to_check=1,
        )

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd.ctf"], indirect=True)
    @pytest.mark.parametrize(
        "optimizer",
        [
            "SGD",
            "SGD-args",
            "Adam",
            "Adam-args",
            None,
            "Scheduler-args",
            "Scheduler",
            "LBFGS",
            "LBFGS-args",
        ],
    )
    @pytest.mark.parametrize("coarse_to_fine", [True, False])
    @pytest.mark.filterwarnings(
        "ignore:Validating whether model can work with coarse-to-fine:UserWarning"
    )
    def test_optimizer(self, curie_img, model, coarse_to_fine, optimizer):
        if coarse_to_fine:
            met = po.synth.MetamerCTF(curie_img, model)
        else:
            met = po.synth.Metamer(curie_img, model)
        optimizer = None
        scheduler = None
        optimizer_kwargs = None
        scheduler_kwargs = None
        check_optimizer = [torch.optim.Adam, {"eps": 1e-8, "lr": 0.01}]
        check_scheduler = None
        if optimizer == "Adam":
            optimizer = torch.optim.Adam
        elif optimizer == "Adam-args":
            optimizer = torch.optim.Adam
            optimizer_kwargs = {"eps": 1e-5}
            check_optimizer[1]["eps"] = 1e-5
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD
            check_optimizer = [torch.optim.SGD, {"lr": 0.01}]
        elif optimizer == "SGD-args":
            optimizer = torch.optim.SGD
            optimizer_kwargs = {"lr": 1}
            check_optimizer = [torch.optim.SGD, {"lr": 1}]
        elif optimizer == "LBFGS":
            optimizer = torch.optim.LBFGS
            check_optimizer = [torch.optim.LBFGS, {"lr": 0.01}]
        elif optimizer == "LBFGS-args":
            optimizer = torch.optim.LBFGS
            optimizer_kwargs = {"lr": 1, "history_size": 10}
            check_optimizer = [torch.optim.LBFGS, {"lr": 1, "history_size": 10}]
        elif optimizer == "Scheduler":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            check_scheduler = [
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                {"factor": 0.1},
            ]
        elif optimizer == "Scheduler-args":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            scheduler_kwargs = {"factor": 1e-3}
            check_scheduler = [
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                {"factor": 1e-3},
            ]
        met.setup(
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        met.synthesize(max_iter=5)
        if not isinstance(met.optimizer, check_optimizer[0]):
            raise ValueError("Didn't properly set optimizer!")
        state_dict = met.optimizer.state_dict()["param_groups"][0]
        for k, v in check_optimizer[1].items():
            if state_dict[k] != v:
                raise ValueError(
                    "Didn't properly set optimizer kwargs! "
                    f"Expected {v} but got {state_dict[k]}!"
                )
        if check_scheduler is not None:
            if not isinstance(met.scheduler, check_scheduler[0]):
                raise ValueError("Didn't properly set scheduler!")
            state_dict = met.scheduler.state_dict()
            for k, v in check_scheduler[1].items():
                if met.scheduler.state_dict()[k] != v:
                    raise ValueError("Didn't properly set scheduler kwargs!")
        elif met.scheduler is not None:
            raise ValueError("Didn't set scheduler to None!")

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    @pytest.mark.parametrize("model", ["naive.Identity"], indirect=True)
    def test_map_location(self, curie_img, model, tmp_path):
        met = po.synth.Metamer(curie_img, model)
        met.synthesize(max_iter=4, store_progress=True)
        met.save(op.join(tmp_path, "test_metamer_map_location.pt"))
        # calling load with map_location effectively switches everything
        # over to that device
        model.to("cpu")
        met_copy = po.synth.Metamer(curie_img.to("cpu"), model)
        met_copy.load(
            op.join(tmp_path, "test_metamer_map_location.pt"),
            map_location="cpu",
        )
        assert met_copy.metamer.device.type == "cpu"
        assert met_copy.image.device.type == "cpu"
        met_copy.synthesize(max_iter=4, store_progress=True)
        # reset model device for other tests
        model.to(DEVICE)

    @pytest.mark.parametrize(
        "model", ["naive.Identity", "NonModule", "frontend.OnOff.nograd"], indirect=True
    )
    @pytest.mark.parametrize("to_type", ["dtype", "device"])
    @pytest.mark.filterwarnings("ignore:Unable to call model.to:UserWarning")
    def test_to(self, curie_img, model, to_type):
        met = po.synth.Metamer(curie_img, model)
        met.synthesize(max_iter=5)
        if to_type == "dtype":
            met.to(torch.float16)
            assert met.image.dtype == torch.float16
            assert met.metamer.dtype == torch.float16
        # can only run this one if we're on a device with CPU and GPU.
        elif to_type == "device" and DEVICE.type != "cpu":
            met.to("cpu")
        met.metamer - met.image
        met.synthesize(max_iter=5)
        # reset so we don't mess up further tests
        if to_type == "dtype":
            met.to(torch.float32)
        elif to_type == "device" and DEVICE.type != "cpu":
            met.to(DEVICE)

    # this determines whether we mix across channels or treat them separately,
    # both of which are supported
    @pytest.mark.parametrize("model", ["ColorModel", "naive.Identity"], indirect=True)
    def test_multichannel(self, model, color_img):
        met = po.synth.Metamer(color_img, model)
        met.synthesize(max_iter=5)
        assert met.metamer.shape == color_img.shape, (
            "Metamer image should have the same shape as input!"
        )

    # this determines whether we mix across batches (e.g., a video model) or
    # treat them separately, both of which are supported
    @pytest.mark.parametrize("model", ["VideoModel", "naive.Identity"], indirect=True)
    def test_multibatch(self, model, einstein_img, curie_img):
        img = torch.cat([curie_img, einstein_img], dim=0)
        met = po.synth.Metamer(img, model)
        met.synthesize(max_iter=5)
        assert met.metamer.shape == img.shape, (
            "Metamer image should have the same shape as input!"
        )

    # we assume that the target representation has no gradient attached, so
    # doublecheck that (validate_model should ensure this)
    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_rep_no_grad(self, model, einstein_img):
        met = po.synth.Metamer(einstein_img, model)
        assert met.target_representation.grad is None, (
            "Target representation has a gradient attached, how?"
        )
        met.synthesize(max_iter=5)
        assert met.target_representation.grad is None, (
            "Target representation has a gradient attached, how?"
        )

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_nan_loss(self, model, einstein_img):
        # clone to prevent NaN from showing up in other tests
        img = einstein_img.clone()
        met = po.synth.Metamer(img, model)
        met.synthesize(max_iter=5)
        met.target_representation[..., 0, 0] = torch.nan
        with pytest.raises(ValueError, match="Found a NaN in loss during optimization"):
            met.synthesize(max_iter=1)

    @pytest.mark.parametrize("model", ["naive.Identity"], indirect=True)
    def test_change_precision_save_load(self, model, einstein_img, tmp_path):
        # Identity model doesn't change when you call .to() with a dtype
        # (unlike those models that have weights) so we use it here
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=5)
        met.to(torch.float64)
        assert met.metamer.dtype == torch.float64, "dtype incorrect!"
        met.save(op.join(tmp_path, "test_metamer_change_prec_save_load.pt"))
        met_copy = po.synth.Metamer(einstein_img.to(torch.float64), model)
        met_copy.load(op.join(tmp_path, "test_metamer_change_prec_save_load.pt"))
        met_copy.synthesize(max_iter=5)
        assert met_copy.metamer.dtype == torch.float64, "dtype incorrect!"

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    @pytest.mark.filterwarnings("ignore:Loss has converged:UserWarning")
    def test_stop_criterion(self, einstein_img, model):
        # checking that this hits the criterion and stops early, so set seed
        # for reproducibility
        po.tools.set_seed(0)
        met = po.synth.Metamer(einstein_img, model)
        # takes different numbers of iter to converge on GPU and CPU
        met.synthesize(max_iter=35, stop_criterion=1e-5, stop_iters_to_check=5)
        assert abs(met.losses[-5] - met.losses[-1]) < 1e-5, (
            "Didn't stop when hit criterion!"
        )
        assert abs(met.losses[-6] - met.losses[-2]) > 1e-5, (
            "Stopped after hit criterion!"
        )
