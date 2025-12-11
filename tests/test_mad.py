import inspect
import math
import os.path as op
from contextlib import nullcontext as does_not_raise

import pytest
import torch

import plenoptic as po
from conftest import DEVICE, check_loss_saved_synth


def rgb_mse(*args):
    return po.metric.mse(*args).mean()


def rgb_l2_norm(*args):
    return po.tools.optim.l2_norm(*args).mean()


# MAD requires metrics are *dis*-similarity metrics, so that they
# return 0 if two images are identical (SSIM normally returns 1)
def dis_ssim(*args):
    return (1 - po.metric.ssim(*args)).mean()


def custom_penalty(x1):
    return po.tools.regularization.penalize_range(x1, allowed_range=(0.2, 0.8))


def custom_penalty2(x1):
    return po.tools.regularization.penalize_range(x1, allowed_range=(0.3, 0.7))


class ModuleMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mdl = po.simul.Gaussian((31, 31)).to(DEVICE)

    def forward(self, x, y):
        return (self.mdl(x) - self.mdl(y)).abs().mean()


class NonModuleMetric:
    def __init__(self):
        self.name = "nonmodule"

    def __call__(self, x, y):
        return (x - y).abs().sum()


class TestMAD:
    @pytest.mark.parametrize("target", ["min", "max"])
    @pytest.mark.parametrize("model_order", ["mse-ssim", "ssim-mse"])
    @pytest.mark.parametrize("store_progress", [False, True, 2])
    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_basic(self, curie_img, target, model_order, store_progress):
        if model_order == "mse-ssim":
            model = po.metric.mse
            model2 = dis_ssim
        elif model_order == "ssim-mse":
            model = dis_ssim
            model2 = po.metric.mse
        mad = po.synth.MADCompetition(
            curie_img, model, model2, target, metric_tradeoff_lambda=1
        )
        mad.synthesize(max_iter=5, store_progress=store_progress)
        if store_progress:
            mad.synthesize(max_iter=5, store_progress=store_progress)

    @pytest.mark.parametrize(
        "fail",
        [
            False,
            "img",
            "metric1",
            "metric2",
            "target",
            "tradeoff",
            "penalty",
            "penalty_lambda",
        ],
    )
    @pytest.mark.parametrize("penalty_lambda", [0.1, 0])
    @pytest.mark.parametrize("rgb", [False, True])
    @pytest.mark.parametrize("penalty_function", ["range", "custom"])
    @pytest.mark.filterwarnings(
        "ignore:SSIM was designed for grayscale images:UserWarning"
    )
    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_save_load(
        self, curie_img, fail, penalty_lambda, rgb, penalty_function, tmp_path
    ):
        # this works with either rgb or grayscale images
        metric = rgb_mse
        if rgb:
            curie_img = curie_img.repeat(1, 3, 1, 1)
            metric2 = rgb_l2_norm
        else:
            metric2 = dis_ssim
        target = "min"
        tradeoff = 1
        if penalty_function == "range":
            penalty = po.tools.regularization.penalize_range
        elif penalty_function == "custom":
            penalty = custom_penalty
        mad = po.synth.MADCompetition(
            curie_img,
            metric,
            metric2,
            target,
            metric_tradeoff_lambda=tradeoff,
            penalty_lambda=penalty_lambda,
            penalty_function=penalty,
        )
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, "test_mad_save_load.pt"))
        if fail:
            if fail == "img":
                curie_img = torch.rand_like(curie_img)
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized attribute image have different values",
                )
            elif fail == "metric1":
                # this works with either rgb or grayscale images (though note
                # that SSIM just operates on each RGB channel independently,
                # which is probably not the right thing to do)
                metric = dis_ssim
                expectation = pytest.raises(
                    ValueError,
                    match=(
                        "Saved and initialized optimized_metric output have different"
                        " values"
                    ),
                )
            elif fail == "metric2":
                # this works with either rgb or grayscale images
                metric2 = rgb_mse
                expectation = pytest.raises(
                    ValueError,
                    match=(
                        "Saved and initialized reference_metric output have different"
                        " values"
                    ),
                )
            elif fail == "target":
                target = "max"
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized minmax are different",
                )
            elif fail == "tradeoff":
                tradeoff = 10
                expectation = pytest.raises(
                    ValueError,
                    match=(
                        "Saved and initialized metric_tradeoff_lambda are different"
                    ),
                )
            elif fail == "penalty":
                penalty = custom_penalty2
                expectation = pytest.raises(
                    ValueError,
                    match=(
                        "Saved and initialized penalty_function output have different"
                        " values"
                    ),
                )
            elif fail == "penalty_lambda":
                penalty_lambda = 0.5
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized penalty_lambda are different"),
                )
            mad_copy = po.synth.MADCompetition(
                curie_img,
                metric,
                metric2,
                target,
                metric_tradeoff_lambda=tradeoff,
                penalty_lambda=penalty_lambda,
                penalty_function=penalty,
            )
            with expectation:
                mad_copy.load(
                    op.join(tmp_path, "test_mad_save_load.pt"),
                    map_location=DEVICE,
                )
        else:
            mad_copy = po.synth.MADCompetition(
                curie_img,
                metric,
                metric2,
                target,
                metric_tradeoff_lambda=tradeoff,
                penalty_lambda=penalty_lambda,
                penalty_function=penalty,
            )
            mad_copy.load(
                op.join(tmp_path, "test_mad_save_load.pt"), map_location=DEVICE
            )
            # check that can resume
            mad_copy.synthesize(max_iter=5, store_progress=True)
        if rgb:
            # since this is a fixture, get this back to a grayscale image
            curie_img = curie_img.mean(1, True)

    def test_setup_initial_noise(self, einstein_img):
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.setup(0.5)
        mad.synthesize(5)

    def test_setup_fail(self, einstein_img):
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.setup()
        with pytest.raises(ValueError, match=r"setup\(\) can only be called once"):
            mad.setup()

    def test_setup_load_fail(self, einstein_img, tmp_path):
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.synthesize(max_iter=4)
        mad.save(op.join(tmp_path, "test_mad_setup_load_fail.pt"))
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.load(op.join(tmp_path, "test_mad_setup_load_fail.pt"))
        with pytest.raises(
            ValueError, match="Cannot set initial_noise after calling load"
        ):
            mad.setup(0.5)

    @pytest.mark.filterwarnings("ignore:You will need to call setup:UserWarning")
    def test_synth_then_setup(self, einstein_img, tmp_path):
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.setup(optimizer=torch.optim.SGD)
        mad.synthesize(max_iter=4)
        mad.save(op.join(tmp_path, "test_mad_synth_then_setup.pt"))
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.load(op.join(tmp_path, "test_mad_synth_then_setup.pt"))
        with pytest.raises(ValueError, match="Don't know how to initialize"):
            mad.synthesize(5)
        mad.setup(optimizer=torch.optim.SGD)
        mad.synthesize(5)

    def test_load_init_fail(self, einstein_img, tmp_path):
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, "test_mad_load_init_fail.pt"))
        with pytest.raises(
            ValueError, match="load can only be called with a just-initialized"
        ):
            mad.load(op.join(tmp_path, "test_mad_load_init_fail.pt"))

    @pytest.mark.parametrize("fail", [False, "name", "behavior"])
    def test_load_names(self, fail, einstein_img, tmp_path):
        # name and behavior same, but module path is different
        if fail is False:

            def mse(x, y):
                return po.tools.optim.mse(x, y)

            metric2 = mse
            expectation = does_not_raise()
        # name different but behavior same
        elif fail == "name":

            def bad_metric(x, y):
                return po.tools.optim.mse(x, y)

            metric2 = bad_metric
            expectation = pytest.raises(
                ValueError,
                match="Saved and initialized optimized_metric have different names",
            )
        # name same but behavior different
        elif fail == "behavior":

            def mse(x, y):
                return po.tools.optim.l2_norm(x, y)

            metric2 = mse
            expectation = pytest.raises(
                ValueError,
                match=(
                    "Saved and initialized optimized_metric output have different"
                    " values"
                ),
            )
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, f"test_mad_load_names_{fail}.pt"))
        mad = po.synth.MADCompetition(
            einstein_img,
            metric2,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        with expectation:
            mad.load(op.join(tmp_path, f"test_mad_load_names_{fail}.pt"))

    def test_examine_saved_object(self, einstein_img, tmp_path):
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, "test_mad_examine.pt"))
        po.tools.examine_saved_synthesis(op.join(tmp_path, "test_mad_examine.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("synth_type", ["eig", "met"])
    def test_load_object_type(self, einstein_img, model, synth_type, tmp_path):
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, "test_mad_load_object_type.pt"))
        if synth_type == "eig":
            mad = po.synth.Eigendistortion(einstein_img, model)
        elif synth_type == "met":
            mad = po.synth.Metamer(einstein_img, model)
        with pytest.raises(
            ValueError, match="Saved object was a.* but initialized object is"
        ):
            mad.load(op.join(tmp_path, "test_mad_load_object_type.pt"))

    @pytest.mark.parametrize("metric_behav", ["dtype", "shape", "name"])
    @pytest.mark.parametrize("metric", ["optimized", "reference"])
    def test_load_metric_change(self, einstein_img, metric, metric_behav, tmp_path):
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, "test_mad_load_metric_change.pt"))

        def new_metric(x, y):
            if metric_behav == "dtype":
                return po.tools.optim.mse(x, y).to(torch.float64)
            elif metric_behav == "shape":
                return torch.stack(
                    [po.tools.optim.mse(x, y) for _ in range(2)]
                ).unsqueeze(0)
            elif metric_behav == "name":
                if metric == "optimized":
                    return po.tools.optim.mse(x, y)
                elif metric == "reference":
                    return po.tools.optim.l2_norm(x, y)

        if metric_behav == "name":
            expectation_str = (
                f"Saved and initialized {metric}_metric have different names"
            )
        elif metric_behav == "shape":
            # this gets raised during the metric validation step
            expectation_str = "metric should return a scalar value but output had shape"
        else:
            expectation_str = (
                f"Saved and initialized {metric}_metric output have different"
                f" {metric_behav}"
            )
        with pytest.raises(ValueError, match=expectation_str):
            if metric == "optimized":
                mad = po.synth.MADCompetition(
                    einstein_img,
                    new_metric,
                    po.tools.optim.l2_norm,
                    "min",
                    metric_tradeoff_lambda=1,
                )
            elif metric == "reference":
                mad = po.synth.MADCompetition(
                    einstein_img,
                    po.metric.mse,
                    new_metric,
                    "min",
                    metric_tradeoff_lambda=1,
                )
            mad.load(op.join(tmp_path, "test_mad_load_metric_change.pt"))

    @pytest.mark.parametrize("penalty_behav", ["dtype", "shape", "name"])
    def test_load_penalty_change(self, einstein_img, penalty_behav, tmp_path):
        def base_penalty(x):
            return po.tools.regularization.penalize_range(x)

        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
            penalty_lambda=0.1,
            penalty_function=base_penalty,
        )
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, "test_mad_load_penalty_change.pt"))

        def new_penalty(x):
            penalty = base_penalty(x)
            if penalty_behav == "dtype":
                return penalty.to(torch.float64)
            if penalty_behav == "shape":
                return torch.stack([penalty, penalty])
            return penalty

        if penalty_behav == "name":
            expectation = "Saved and initialized penalty_function have different names"
        else:
            expectation = (
                "Saved and initialized penalty_function output have different"
                f" {penalty_behav}"
            )
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
            penalty_lambda=0.1,
            penalty_function=new_penalty,
        )
        with pytest.raises(ValueError, match=expectation):
            mad.load(op.join(tmp_path, "test_mad_load_penalty_change.pt"))

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("attribute", ["saved", "init"])
    def test_load_attributes(self, einstein_img, model, attribute, tmp_path):
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.synthesize(max_iter=4, store_progress=True)
        if attribute == "saved":
            mad.test = "BAD"
            err_str = "Saved"
        mad.save(op.join(tmp_path, "test_mad_load_attributes.pt"))
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        if attribute == "init":
            mad.test = "BAD"
            err_str = "Initialized"
        with pytest.raises(
            ValueError, match=rf"{err_str} object has 1 attribute\(s\) not present"
        ):
            mad.load(op.join(tmp_path, "test_mad_load_attributes.pt"))

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
    @pytest.mark.filterwarnings("ignore:You will need to call setup:UserWarning")
    def test_load_optimizer(self, curie_img, optim_opts, fail, tmp_path):
        mad = po.synth.MADCompetition(
            curie_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
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
        mad.setup(
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        mad.synthesize(max_iter=5)
        mad.save(op.join(tmp_path, "test_mad_optimizer.pt"))
        mad = po.synth.MADCompetition(
            curie_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.load(op.join(tmp_path, "test_mad_optimizer.pt"))
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
            mad.setup(
                optimizer=optimizer,
                scheduler=scheduler,
                optimizer_kwargs=optimizer_kwargs,
                scheduler_kwargs=scheduler_kwargs,
            )
            mad.synthesize(max_iter=5)
            if not isinstance(mad.optimizer, check_optimizer[0]):
                raise ValueError("Didn't properly set optimizer!")
            state_dict = mad.optimizer.state_dict()["param_groups"][0]
            for k, v in check_optimizer[1].items():
                if state_dict[k] != v:
                    raise ValueError(
                        "Didn't properly set optimizer kwargs! "
                        f"Expected {v} but got {state_dict[k]}!"
                    )
            if check_scheduler is not None:
                if not isinstance(mad.scheduler, check_scheduler[0]):
                    raise ValueError("Didn't properly set scheduler!")
                state_dict = mad.scheduler.state_dict()
                for k, v in check_scheduler[1].items():
                    if mad.scheduler.state_dict()[k] != v:
                        raise ValueError("Didn't properly set scheduler kwargs!")
            elif mad.scheduler is not None:
                raise ValueError("Didn't set scheduler to None!")

    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_load_tol(self, einstein_img, tmp_path):
        mad = po.synth.MADCompetition(
            einstein_img,
            po.metric.mse,
            lambda *args: 1 - po.metric.ssim(*args),
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.synthesize(5)
        mad.save(op.join(tmp_path, "test_mad_load_tol.pt"))
        mad = po.synth.MADCompetition(
            (1 - 1e-7) * einstein_img + 1e-7 * torch.rand_like(einstein_img),
            po.metric.mse,
            lambda *args: 1 - po.metric.ssim(*args),
            "min",
            metric_tradeoff_lambda=1,
        )
        with pytest.raises(ValueError, match="Saved and initialized attribute image"):
            mad.load(op.join(tmp_path, "test_mad_load_tol.pt"))
        mad.load(op.join(tmp_path, "test_mad_load_tol.pt"), tensor_equality_atol=1e-7)

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
    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_optimizer(self, curie_img, optimizer):
        mad = po.synth.MADCompetition(
            curie_img,
            po.metric.mse,
            lambda *args: 1 - po.metric.ssim(*args),
            "min",
            metric_tradeoff_lambda=1,
        )
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
            check_optimizer[1] = {"eps": 1e-5}
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
        mad.setup(
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        mad.synthesize(max_iter=5)
        if not isinstance(mad.optimizer, check_optimizer[0]):
            raise ValueError("Didn't properly set optimizer!")
        state_dict = mad.optimizer.state_dict()["param_groups"][0]
        for k, v in check_optimizer[1].items():
            if state_dict[k] != v:
                raise ValueError(
                    "Didn't properly set optimizer kwargs! "
                    f"Expected {v} but got {state_dict[k]}!"
                )
        if check_scheduler is not None:
            if not isinstance(mad.scheduler, check_scheduler[0]):
                raise ValueError("Didn't properly set scheduler!")
            state_dict = mad.scheduler.state_dict()
            for k, v in check_scheduler[1].items():
                if mad.scheduler.state_dict()[k] != v:
                    raise ValueError("Didn't properly set scheduler kwargs!")
        elif mad.scheduler is not None:
            raise ValueError("Didn't set scheduler to None!")

    @pytest.mark.parametrize("metric", [po.metric.mse, ModuleMetric, NonModuleMetric])
    @pytest.mark.parametrize("to_type", ["dtype", "device"])
    def test_to(self, curie_img, metric, to_type):
        # if metric is not the po.metric.mse function above, initialize it here,
        # otherwise we can get a weird state-dependence
        if not inspect.isfunction(metric):
            metric = metric()
        mad = po.synth.MADCompetition(
            curie_img, metric, po.tools.optim.l2_norm, "min", metric_tradeoff_lambda=1
        )
        mad.synthesize(max_iter=5)
        if to_type == "dtype":
            mad.to(torch.float64)
            assert mad.initial_image.dtype == torch.float64
            assert mad.image.dtype == torch.float64
            assert mad.mad_image.dtype == torch.float64
        # can only run this one if we're on a device with CPU and GPU.
        elif to_type == "device" and DEVICE.type != "cpu":
            mad.to("cpu")
        # initial_image doesn't get used anywhere after init, so check it like
        # this
        mad.initial_image - mad.image
        mad.mad_image - mad.image
        mad.synthesize(max_iter=5)

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    def test_map_location(self, curie_img, tmp_path):
        mad = po.synth.MADCompetition(
            curie_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, "test_mad_map_location.pt"))
        mad_copy = po.synth.MADCompetition(
            curie_img.to("cpu"),
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        assert mad_copy.image.device.type == "cpu"
        mad_copy.load(op.join(tmp_path, "test_mad_map_location.pt"), map_location="cpu")
        assert mad_copy.mad_image.device.type == "cpu"
        mad_copy.synthesize(max_iter=4, store_progress=True)

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    def test_to_midsynth(self, curie_img):
        mad = po.synth.MADCompetition(
            curie_img,
            po.metric.mse,
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.synthesize(max_iter=4, store_progress=2)
        assert mad.image.device.type == "cuda"
        assert mad.mad_image.device.type == "cuda"
        mad.to("cpu")
        mad.synthesize(max_iter=4, store_progress=2)
        assert mad.image.device.type == "cpu"
        assert mad.mad_image.device.type == "cpu"
        mad.to("cuda")
        mad.synthesize(max_iter=4, store_progress=2)
        assert mad.image.device.type == "cuda"
        assert mad.mad_image.device.type == "cuda"

    # MAD can accept multiple images on the batch dimension, but the metrics
    # must return a single number. This means, effectively, that we can do
    # synthesis for e.g., video metrics, but that we cannot synthesize several
    # images in parallel
    def test_batch_synthesis(self, curie_img, einstein_img):
        img = torch.cat([curie_img, einstein_img], dim=0)
        mad = po.synth.MADCompetition(
            img,
            lambda *args: po.metric.mse(*args).mean(),
            po.tools.optim.l2_norm,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad.synthesize(max_iter=10)
        assert mad.mad_image.shape == img.shape, (
            "MAD image should have the same shape as input!"
        )

    @pytest.mark.parametrize("input_dim", [2, 3, 4, 5])
    @pytest.mark.filterwarnings("ignore:.*mostly been tested on 4d inputs:UserWarning")
    def test_dimensionality(self, einstein_img, input_dim):
        img = einstein_img.squeeze()
        while img.ndimension() < input_dim:
            img = img.unsqueeze(0)
        mad = po.synth.MADCompetition(img, rgb_mse, rgb_l2_norm, "min", 1)
        mad.synthesize(5)

    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    @pytest.mark.parametrize("store_progress", [True, 2, 3])
    def test_store_rep(self, einstein_img, store_progress):
        mad = po.synth.MADCompetition(
            einstein_img, po.metric.mse, dis_ssim, "min", metric_tradeoff_lambda=1
        )
        max_iter = 3
        if store_progress == 3:
            max_iter = 6
        mad.synthesize(max_iter=max_iter, store_progress=store_progress)
        check_loss_saved_synth(
            mad.losses,
            mad.saved_mad_image,
            max_iter,
            mad.objective_function,
            store_progress,
        )
        mad.synthesize(max_iter=max_iter, store_progress=store_progress)
        check_loss_saved_synth(
            mad.losses,
            mad.saved_mad_image,
            2 * max_iter,
            mad.objective_function,
            store_progress,
        )

    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_save_mad_empty(self, einstein_img):
        mad = po.synth.MADCompetition(
            einstein_img, po.metric.mse, dis_ssim, "min", metric_tradeoff_lambda=1
        )
        torch.equal(mad.saved_mad_image, torch.empty(0))
        mad.synthesize(max_iter=3)
        torch.equal(mad.saved_mad_image, mad.mad_image.to("cpu"))

    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_mad_empty_loss(self, einstein_img):
        mad = po.synth.MADCompetition(
            einstein_img, po.metric.mse, dis_ssim, "min", metric_tradeoff_lambda=1
        )
        assert mad.objective_function().numel() == 0
        torch.equal(mad.losses, torch.empty(0))
        mad.setup()
        assert isinstance(mad.objective_function(), torch.Tensor)
        assert mad.losses.numel() > 0
        mad.synthesize(max_iter=2)
        assert isinstance(mad.objective_function(), torch.Tensor)
        assert mad.losses.numel() > 0

    @pytest.mark.parametrize("iteration", [None, 0, -2, -3, 2, 1, 6, -7])
    @pytest.mark.parametrize("store_progress", [True, False, 2])
    @pytest.mark.parametrize("iteration_selection", ["floor", "ceiling", "round"])
    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_mad_get_progress(
        self, einstein_img, iteration, store_progress, iteration_selection
    ):
        mad = po.synth.MADCompetition(
            einstein_img, po.metric.mse, dis_ssim, "min", metric_tradeoff_lambda=1
        )
        mad.synthesize(max_iter=5, store_progress=store_progress)
        if iteration_selection == "floor":
            func = math.floor
        elif iteration_selection == "ceiling":
            func = math.ceil
        else:
            func = round
        expected_dict = {}
        if iteration is None:
            expected_dict["losses"] = mad.losses[-1]
            expected_dict.update(
                {
                    "iteration": 5,
                    "gradient_norm": None,
                    "pixel_change_norm": None,
                    "reference_metric_loss": mad.reference_metric_loss[-1],
                    "optimized_metric_loss": mad.optimized_metric_loss[-1],
                }
            )
            if store_progress:
                expected_dict.update(
                    {
                        "saved_mad_image": mad.mad_image.to("cpu"),
                        "store_progress_iteration": 5,
                    }
                )
        elif iteration not in [6, -7]:
            expected_dict["losses"] = mad.losses[iteration]
            if iteration < 0:
                # add one to account for loss and these attributes being off by one
                expected_dict.update(
                    {
                        "iteration": 6 + iteration,
                        "gradient_norm": mad.gradient_norm[iteration + 1],
                        "pixel_change_norm": mad.pixel_change_norm[iteration + 1],
                        "reference_metric_loss": mad.reference_metric_loss[iteration],
                        "optimized_metric_loss": mad.optimized_metric_loss[iteration],
                    }
                )
                if store_progress:
                    iter = func((6 + iteration) / store_progress)
                    expected_dict.update(
                        {
                            "saved_mad_image": mad.saved_mad_image[iter],
                            "store_progress_iteration": iter * store_progress,
                        }
                    )
            else:
                expected_dict.update(
                    {
                        "iteration": iteration,
                        "gradient_norm": mad.gradient_norm[iteration],
                        "pixel_change_norm": mad.pixel_change_norm[iteration],
                        "reference_metric_loss": mad.reference_metric_loss[iteration],
                        "optimized_metric_loss": mad.optimized_metric_loss[iteration],
                    }
                )
                if store_progress:
                    iter = func(iteration / store_progress)
                    expected_dict.update(
                        {
                            "saved_mad_image": mad.saved_mad_image[iter],
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
            progress = mad.get_progress(iteration, iteration_selection)
            assert progress.keys() == expected_dict.keys()
            for k, v in progress.items():
                if isinstance(v, torch.Tensor):
                    assert torch.equal(v, expected_dict[k]), f"{k} not as expected!"
                else:
                    assert v == expected_dict[k], f"{k} not as expected!"

    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_continue(self, einstein_img):
        mad = po.synth.MADCompetition(
            einstein_img, po.metric.mse, dis_ssim, "min", metric_tradeoff_lambda=1
        )
        mad.synthesize(max_iter=3, store_progress=True)
        mad.synthesize(max_iter=3, store_progress=True)

    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_nan_loss(self, einstein_img):
        # clone to prevent NaN from showing up in other tests
        img = einstein_img.clone()
        mad = po.synth.MADCompetition(
            img, po.metric.mse, dis_ssim, "min", metric_tradeoff_lambda=1
        )
        mad.synthesize(max_iter=5)
        mad.image[..., 0, 0] = torch.nan
        with pytest.raises(ValueError, match="Found a NaN in loss during optimization"):
            mad.synthesize(max_iter=1)

    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_change_precision_save_load(self, einstein_img, tmp_path):
        # Identity model doesn't change when you call .to() with a dtype
        # (unlike those models that have weights) so we use it here
        mad = po.synth.MADCompetition(
            einstein_img, po.metric.mse, dis_ssim, "min", metric_tradeoff_lambda=1
        )
        mad.synthesize(max_iter=5)
        mad.to(torch.float64)
        assert mad.mad_image.dtype == torch.float64, "dtype incorrect!"
        mad.save(op.join(tmp_path, "test_change_prec_save_load.pt"))
        mad_copy = po.synth.MADCompetition(
            einstein_img.to(torch.float64),
            po.metric.mse,
            dis_ssim,
            "min",
            metric_tradeoff_lambda=1,
        )
        mad_copy.load(op.join(tmp_path, "test_change_prec_save_load.pt"))
        mad_copy.synthesize(max_iter=5)
        assert mad_copy.mad_image.dtype == torch.float64, "dtype incorrect!"

    @pytest.mark.filterwarnings("ignore:Loss has converged:UserWarning")
    @pytest.mark.filterwarnings("ignore:Image range falls outside:UserWarning")
    def test_stop_criterion(self, einstein_img):
        # checking that this hits the criterion and stops early, so set seed
        # for reproducibility
        po.tools.set_seed(0)
        mad = po.synth.MADCompetition(
            einstein_img, po.metric.mse, dis_ssim, "min", metric_tradeoff_lambda=0.1
        )
        # losses[-1] corresponds to the *current* loss (of mad.mad_image), not the loss
        # from the most recent iteration. so losses[-2] is the loss of the last
        # synthesis iteration.
        mad.synthesize(max_iter=15, stop_criterion=1e-3, stop_iters_to_check=5)
        assert abs(mad.losses[-6] - mad.losses[-2]) < 1e-3, (
            "Didn't stop when hit criterion!"
        )
        assert abs(mad.losses[-7] - mad.losses[-3]) > 1e-3, (
            "Stopped after hit criterion!"
        )

    def test_warn_out_of_range_input(self, einstein_img):
        img = einstein_img + 1
        with pytest.warns(UserWarning, match="outside the tested range \\(0, 1\\)"):
            mad = po.synth.MADCompetition(
                img,
                po.metric.mse,
                po.tools.optim.l2_norm,
                "min",
                metric_tradeoff_lambda=1,
            )
