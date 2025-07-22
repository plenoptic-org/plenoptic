"""
Run MAD Competition.

Classes to perform the synthesis of Maximum Differentiation Competition.
"""

import contextlib
import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyrtools.tools.display import make_figure as pt_make_figure
from torch import Tensor
from tqdm.auto import tqdm

from ..tools import data, display, optim, regularization
from ..tools.convergence import loss_convergence
from ..tools.validate import validate_input, validate_metric
from .synthesis import OptimizedSynthesis


class MADCompetition(OptimizedSynthesis):
    r"""
    Synthesize a single maximally-differentiating image for two metrics.

    Following the basic idea in [1]_, this class synthesizes a
    maximally-differentiating image for two given metrics, based on a given
    image. We start by adding noise to this image and then iteratively
    adjusting its pixels so as to either minimize or maximize
    ``optimized_metric`` while holding the value of ``reference_metric`` constant.

    MADCompetiton accepts two metrics as its input. These should be callables
    that take two images and return a single number, and that number should be
    0 if and only if the two images are identical (thus, the larger the number,
    the more different the two images).

    Note that a full set of MAD Competition images consists of two pairs: a maximal and
    a minimal image for each metric. A single instantiation of ``MADCompetition`` will
    generate one of these four images.

    Parameters
    ----------
    image
        A tensor, this is the image we use as the reference point.
    optimized_metric
        The metric whose value you wish to minimize or maximize, which takes
        two tensors and returns a scalar.
    reference_metric
        The metric whose value you wish to keep fixed, which takes two tensors
        and returns a scalar.
    minmax
        Whether you wish to minimize or maximize ``optimized_metric``.
    metric_tradeoff_lambda
        Lambda to multiply by ``reference_metric`` loss and add to
        ``optimized_metric`` loss. If ``None``, we pick a value so the two
        initial losses are approximately equal in magnitude.
    penalty_function
        A penalty function to help constrain the synthesized
        image by penalizing specific image properties.
    penalty_lambda
        Strength of the regularizer. Must be non-negative.

    References
    ----------
    .. [1] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
           competition: A methodology for comparing computational models of
           perceptual discriminability. Journal of Vision, 8(12), 1–13.
           https://dx.doi.org/10.1167/8.12.8
    """

    def __init__(
        self,
        image: Tensor,
        optimized_metric: torch.nn.Module | Callable[[Tensor, Tensor], Tensor],
        reference_metric: torch.nn.Module | Callable[[Tensor, Tensor], Tensor],
        minmax: Literal["min", "max"],
        metric_tradeoff_lambda: float | None = None,
        penalty_function: Callable[[Tensor], Tensor] = regularization.penalize_range,
        penalty_lambda: float = 0.1,
    ):
        super().__init__(
          penalty_function=penalty_function,
          penalty_lambda=penalty_lambda
        )
        validate_input(image, allowed_range=(0, 1))
        validate_metric(
            optimized_metric,
            image_shape=image.shape,
            image_dtype=image.dtype,
            device=image.device,
        )
        validate_metric(
            reference_metric,
            image_shape=image.shape,
            image_dtype=image.dtype,
            device=image.device,
        )
        self._optimized_metric = optimized_metric
        self._reference_metric = reference_metric
        self._image = image.detach()
        self._image_shape = image.shape
        self._scheduler = None
        self._scheduler_step_arg = False
        self._optimized_metric_loss = []
        self._reference_metric_loss = []
        if minmax not in ["min", "max"]:
            raise ValueError(
                "synthesis_target must be one of {'min', 'max'}, but got "
                f"value {minmax} instead!"
            )
        self._mad_image = None
        self._initial_image = None
        self._reference_metric_target = None
        # If no metric_tradeoff_lambda is specified, pick one that gets them to
        # approximately the same magnitude
        if metric_tradeoff_lambda is None:
            other_image = torch.rand_like(image)
            optim_loss = optimized_metric(image, other_image)
            loss_ratio = optim_loss / reference_metric(image, other_image)
            metric_tradeoff_lambda = torch.pow(
                torch.as_tensor(10), torch.round(torch.log10(loss_ratio))
            ).item()
            warnings.warn(
                "Since metric_tradeoff_lamda was None, automatically set"
                f" to {metric_tradeoff_lambda} to roughly balance metrics."
            )
        self._metric_tradeoff_lambda = metric_tradeoff_lambda
        self._minmax = minmax
        self._store_progress = None
        self._saved_mad_image = []

    def setup(
        self,
        initial_noise: float | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        optimizer_kwargs: dict | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scheduler_kwargs: dict | None = None,
    ):
        """
        Initialize the MAD image, optimizer, and scheduler.

        Can only be called once. If ``load()`` has been called, ``initial_noise`` must
        be None.

        Parameters
        ----------
        initial_noise
            :attr:`mad_image` is initialized to ``self.image + initial_noise *
            torch.randn_like(self.image)``, so this gives the standard deviation of the
            Gaussian noise. If ``None``, we use a value of 0.1.
        optimizer
            The un-initialized optimizer object to use. If ``None``, we use Adam.
        optimizer_kwargs
            The keyword arguments to pass to the optimizer on initialization. If
            ``None``, we use ``{"lr": .01}`` and, if optimizer is ``None``,
            ``{"amsgrad": True}``.
        scheduler
            The un-initialized learning rate scheduler object to use. If ``None``, we
            don't use one.
        scheduler_kwargs
            The keyword arguments to pass to the scheduler on initialization.

        Raises
        ------
        ValueError
            If you try to set ``initial_noise`` after calling :func:`load`.
        ValueError
            If ``setup`` is called more than once or after :func:`synthesize`.

        Examples
        --------
        Set initial noise:

        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> mad = po.synth.MADCompetition(
        ...     img,
        ...     lambda x, y: 1 - po.metric.ssim(x, y),
        ...     po.metric.mse,
        ...     "min",
        ...     metric_tradeoff_lambda=0.1,
        ... )
        >>> mad.setup(1)
        >>> mad.synthesize(10)

        Set optimizer:

        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> mad = po.synth.MADCompetition(
        ...     img,
        ...     lambda x, y: 1 - po.metric.ssim(x, y),
        ...     po.metric.mse,
        ...     "min",
        ...     metric_tradeoff_lambda=0.1,
        ... )
        >>> mad.setup(optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.01})
        >>> mad.synthesize(10)

        Use with save/load. Only the optimizer object is necessary, its kwargs and the
        initial noise are handled by load.

        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> mad = po.synth.MADCompetition(
        ...     img,
        ...     lambda x, y: 1 - po.metric.ssim(x, y),
        ...     po.metric.mse,
        ...     "min",
        ...     metric_tradeoff_lambda=0.1,
        ... )
        >>> mad.setup(1, optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.01})
        >>> mad.synthesize(10)
        >>> mad.save("mad_setup.pt")
        >>> mad = po.synth.MADCompetition(
        ...     img,
        ...     lambda x, y: 1 - po.metric.ssim(x, y),
        ...     po.metric.mse,
        ...     "min",
        ...     metric_tradeoff_lambda=0.1,
        ... )
        >>> mad.load("mad_setup.pt")
        >>> mad.setup(optimizer=torch.optim.SGD)
        >>> mad.synthesize(10)
        """
        if self._mad_image is None:
            if initial_noise is None:
                initial_noise = 0.1
            mad_image = self.image + initial_noise * torch.randn_like(self.image)
            mad_image = mad_image.clamp(0, 1)
            self._initial_image = mad_image.clone()
            mad_image.requires_grad_()
            self._mad_image = mad_image
            self._reference_metric_target = self.reference_metric(
                self.image, self.mad_image
            ).item()
            self._reference_metric_loss.append(self._reference_metric_target)
            self._optimized_metric_loss.append(
                self.optimized_metric(self.image, self.mad_image).item()
            )
        else:
            if self._loaded:
                if initial_noise is not None:
                    raise ValueError("Cannot set initial_noise after calling load()!")
            else:
                raise ValueError(
                    "setup() can only be called once and must be called"
                    " before synthesize()!"
                )

        # initialize the optimizer
        self._initialize_optimizer(optimizer, self.mad_image, optimizer_kwargs)
        # and scheduler
        self._initialize_scheduler(scheduler, self.optimizer, scheduler_kwargs)
        # reset _loaded, if everything ran successfully
        self._loaded = False

    def synthesize(
        self,
        max_iter: int = 100,
        store_progress: bool | int = False,
        stop_criterion: float = 1e-4,
        stop_iters_to_check: int = 50,
    ):
        r"""
        Synthesize a MAD image.

        Update the pixels of :attr:`initial_image` to maximize or minimize
        (depending on the value of ``minmax``) the value of
        ``optimized_metric(image, mad_image)`` while keeping the value of
        ``reference_metric(image, mad_image)`` constant.

        We run this until either we reach ``max_iter`` or the change over the
        past ``stop_iters_to_check`` iterations is less than
        ``stop_criterion``, whichever comes first.

        Parameters
        ----------
        max_iter
            The maximum number of iterations to run before we end synthesis
            (unless we hit the stop criterion).
        store_progress
            Whether we should store the MAD image in progress during synthesis. If
            ``False``, we don't save anything. If True, we save every iteration. If an
            int, we save every ``store_progress`` iterations (note then that ``0`` is
            the same as ``False`` and ``1`` the same as ``True``).
        stop_criterion
            If the loss over the past ``stop_iters_to_check`` has changed
            less than ``stop_criterion``, we terminate synthesis.
        stop_iters_to_check
            How many iterations back to check in order to see if the
            loss has stopped decreasing (for ``stop_criterion``).

        Raises
        ------
        ValueError
            If we find a NaN during optimization.
        """
        # if setup hasn't been called manually, call it now.
        if self._mad_image is None or isinstance(self._scheduler, tuple):
            self.setup()
        self._current_loss = None

        # get ready to store progress
        self.store_progress = store_progress

        pbar = tqdm(range(max_iter))

        for _ in pbar:
            # update saved_* attrs. len(_losses) gives the total number of
            # iterations and will be correct across calls to `synthesize`
            self._store(len(self._losses))

            loss = self._optimizer_step(pbar)

            if not np.isfinite(loss):
                raise ValueError("Found a NaN in loss during optimization.")

            if self._check_convergence(stop_criterion, stop_iters_to_check):
                warnings.warn("Loss has converged, stopping synthesis")
                break

        # compute current loss, no need to compute gradient
        with torch.no_grad():
            self._current_loss = self.objective_function().item()

        pbar.close()

    def objective_function(
        self,
        mad_image: Tensor | None = None,
        image: Tensor | None = None,
    ) -> Tensor:
        r"""
        Compute the MADCompetition synthesis loss.

        This computes:

        .. math::

            t L_1(x, \hat{x}) &+ \lambda_1 [L_2(x, x+\epsilon) - L_2(x, \hat{x})]^2 \\
                              &+ \lambda_2 \mathcal{B}(\hat{x})

        where :math:`t` is 1 if :attr:`minmax` is ``'min'`` and -1 if it's ``'max'``,
        :math:`L_1` is :attr:`optimized_metric`, :math:`L_2` is
        :attr:`reference_metric`, :math:`x` is :attr:`image`, :math:`\hat{x}` is
        :attr:`mad_image`, :math:`\epsilon` is the initial noise, :math:`\mathcal{B}` is
        the penalty function, :math:`\lambda_1` is :attr:`metric_tradeoff_lambda`
        and :math:`\lambda_2` is :attr:`penalty_lambda`.

        Parameters
        ----------
        mad_image
            Proposed ``mad_image``, :math:`\hat{x}` in the above equation. If
            ``None``, use ``self.mad_image``.
        image
            Proposed ``image``, :math:`x` in the above equation. If
            ``None``, use ``self.image``.

        Returns
        -------
        loss
            1-element tensor containing the loss on this step.
        """
        if image is None:
            image = self.image
        if mad_image is None:
            mad_image = self.mad_image
            # if this is empty, then self.mad_image hasn't been initialized
            if mad_image.numel() == 0:
                return torch.empty(0)
        synth_target = {"min": 1, "max": -1}[self.minmax]
        synthesis_loss = self.optimized_metric(image, mad_image)
        fixed_loss = (
            self._reference_metric_target - self.reference_metric(image, mad_image)
        ).pow(2)
        penalty = self.penalty_function(mad_image)
        return (
            synth_target * synthesis_loss
            + self.metric_tradeoff_lambda * fixed_loss
            + self.penalty_lambda * penalty
        )

    def get_progress(
        self,
        iteration: int,
        iteration_selection: Literal["floor", "ceiling", "round"] = "round",
    ) -> dict:
        """
        Return dictionary summarizing synthesis progress at ``iteration``.

        This returns a dictionary containing info from :attr:`losses`,
        :attr:`pixel_change_norm`, :attr:`gradient_norm`, and
        :attr:`saved_mad_image` corresponding to ``iteration``. If synthesis was
        run with ``store_progress=False`` (and so we did not cache anything in
        :attr:`saved_mad_image`), then that key will be missing. If synthesis was
        run with ``store_progress>1``, we will grab the corresponding tensor
        from :attr:`saved_mad_image`, with behavior determined by
        ``iteration_selection``.

        The returned dictionary will additionally contain the keys:

        - ``"iteration"``: the (0-indexed positive) synthesis iteration that the
          values for :attr:`losses`, :attr:`pixel_change_norm`, and
          :attr:`gradient_norm` come from.

        - If ``self.store_progress``, ``"store_progress_iteration"``: the (0-indexed
          positive) synthesis iteration that the value for :attr:`saved_mad_image` comes
          from.

        Note that for the most recent iteration (``iteration=-1`` or ``iteration=None``
        or ``iteration==len(self.losses)-1``), we do not have values for
        :attr:`pixel_change_norm` or :attr:`gradient_norm`, since in this case we are
        showing the loss and value for the current MAD image.

        Parameters
        ----------
        iteration
            Synthesis iteration to summarize. If ``None``, grab the most recent.
            Negative values are allowed.
        iteration_selection

            How to select the relevant iteration from :attr:`saved_mad_image`
            when the request iteration wasn't stored.

            When synthesis was run with ``store_progress=n`` (where ``n>1``),
            MAD images are only saved every ``n`` iterations. If you request an
            iteration where a MAD image wasn't saved, this determines which available
            iteration is used instead:

            * ``"floor"``: use the closest saved iteration **before** the
              requested one.

            * ``"ceiling"``: use the closest saved iteration **after** the
              requested one.

            * ``"round"``: use the closest saved iteration.

        Returns
        -------
        progress_info
            Dictionary summarizing synthesis progress.

        Raises
        ------
        IndexError
            If ``iteration`` takes an illegal value.

        Warns
        -----
        UserWarning
            If the iteration used for ``saved_mad_image`` is not the same as the
            argument ``iteration`` (because e.g., you set ``iteration=3`` but
            ``self.store_progress=2``).
        """
        return super().get_progress(
            iteration,
            iteration_selection,
            ["reference_metric_loss", "optimized_metric_loss"],
            store_progress_attributes=["saved_mad_image"],
        )

    def _optimizer_step(self, pbar: tqdm) -> Tensor:
        r"""
        Compute and propagate gradients, then step optimizer to update mad_image.

        Parameters
        ----------
        pbar
            A tqdm progress-bar, which we update with a postfix
            describing the current loss, gradient norm, and learning
            rate (it already tells us which iteration and the time
            elapsed).

        Returns
        -------
        loss
            1-element tensor containing the loss on this step.
        """  # numpydoc ignore=ES01
        last_iter_mad_image = self.mad_image.clone()
        loss = self.optimizer.step(self._closure)
        self._losses.append(loss)
        grad_norm = torch.linalg.vector_norm(self.mad_image.grad.data, ord=2, dim=None)
        self._gradient_norm.append(grad_norm.item())

        fm = self.reference_metric(self.image, self.mad_image)
        self._reference_metric_loss.append(fm.item())
        sm = self.optimized_metric(self.image, self.mad_image)
        self._optimized_metric_loss.append(sm.item())

        # optionally step the scheduler, passing loss if needed
        if self.scheduler is not None:
            if self._scheduler_step_arg:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

        pixel_change_norm = torch.linalg.vector_norm(
            self.mad_image - last_iter_mad_image, ord=2, dim=None
        )
        self._pixel_change_norm.append(pixel_change_norm.item())

        # add extra info here if you want it to show up in progress bar
        pbar.set_postfix(
            OrderedDict(
                loss=f"{loss:.04e}",
                learning_rate=self.optimizer.param_groups[0]["lr"],
                gradient_norm=f"{grad_norm.item():.04e}",
                pixel_change_norm=f"{pixel_change_norm.item():.04e}",
                reference_metric=f"{fm.item():.04e}",
                optimized_metric=f"{sm.item():.04e}",
            )
        )
        return loss

    def _check_convergence(
        self, stop_criterion: float, stop_iters_to_check: int
    ) -> bool:
        r"""
        Check whether the loss has stabilized and, if so, return True.

        Uses :func:`loss_convergence`.

        Parameters
        ----------
        stop_criterion
            If the loss over the past ``stop_iters_to_check`` has changed
            less than ``stop_criterion``, we terminate synthesis.
        stop_iters_to_check
            How many iterations back to check in order to see if the
            loss has stopped decreasing (for ``stop_criterion``).

        Returns
        -------
        loss_stabilized
            Whether the loss has stabilized or not.
        """
        return loss_convergence(self, stop_criterion, stop_iters_to_check)

    def _store(self, i: int) -> bool:
        """
        Store mad_image and model response, if appropriate.

        If it's the right iteration, we update :attr:`saved_mad_image`.

        Parameters
        ----------
        i
            The current iteration.

        Returns
        -------
        stored
            True if we stored this iteration, False if not.
        """
        if self.store_progress and (i % self.store_progress == 0):
            # want these to always be on cpu, to reduce memory use for GPUs
            self._saved_mad_image.append(self.mad_image.clone().to("cpu"))
            stored = True
        else:
            stored = False
        return stored

    def save(self, file_path: str):
        r"""
        Save all relevant variables in .pt file.

        Note that if ``store_progress`` is True, this will probably be very
        large.

        See :func:`load` docstring for an example of use.

        Parameters
        ----------
        file_path
            The path to save the MADCompetition object to.
        """
        save_io_attrs = [
            ("_optimized_metric", ("_image", "_mad_image")),
            ("_reference_metric", ("_image", "_mad_image")),
            ("penalty_function", ("_image",)),
        ]
        save_state_dict_attrs = ["_optimizer", "_scheduler"]
        super().save(file_path, save_io_attrs, save_state_dict_attrs)

    def to(self, *args: Any, **kwargs: Any):
        r"""
        Move and/or casts the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)

        .. function:: to(dtype, non_blocking=False)

        .. function:: to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See :meth:`torch.nn.Module.to` for examples.

        .. note::
            This method modifies the module in-place.

        Parameters
        ----------
        device : torch.device
            The desired device of the parameters and buffers in this module.
        dtype : torch.dtype
            The desired floating point type of the floating point parameters and
            buffers in this module.
        tensor : torch.Tensor
            Tensor whose dtype and device are the desired dtype and device for
            all parameters and buffers in this module.
        """  # numpydoc ignore=PR01,PR02
        attrs = ["_initial_image", "_image", "_mad_image", "_saved_mad_image"]
        super().to(*args, attrs=attrs, **kwargs)
        # if the metrics are Modules, then we should pass them as well. If
        # they're functions then nothing needs to be done.
        with contextlib.suppress(AttributeError):
            self.reference_metric.to(*args, **kwargs)
        with contextlib.suppress(AttributeError):
            self.optimized_metric.to(*args, **kwargs)

    def load(
        self,
        file_path: str,
        map_location: str | None = None,
        tensor_equality_atol: float = 1e-8,
        tensor_equality_rtol: float = 1e-5,
        **pickle_load_args: Any,
    ):
        r"""
        Load all relevant stuff from a .pt file.

        This must be called by a ``MADCompetition`` object initialized just like the
        saved object.

        Note this operates in place and so doesn't return anything.

        .. versionchanged:: 1.2
           load behavior changed in a backwards-incompatible manner in order to
           compatible with breaking changes in torch 2.6.

        Parameters
        ----------
        file_path
            The path to load the synthesis object from.
        map_location
            Argument to pass to ``torch.load`` as ``map_location``. If you save
            stuff that was being run on a GPU and are loading onto a
            CPU, you'll need this to make sure everything lines up
            properly. This should be structured like the str you would
            pass to :class:`torch.device`.
        tensor_equality_atol
            Absolute tolerance to use when checking for tensor equality during load,
            passed to :func:`torch.allclose`. It may be necessary to increase if you are
            saving and loading on two machines with torch built by different cuda
            versions. Be careful when changing this! See
            :class:`torch.finfo<torch.torch.finfo>` for more details about floating
            point precision of different data types (especially, ``eps``); if you have
            to increase this by more than 1 or 2 decades, then you are probably not
            dealing with a numerical issue.
        tensor_equality_rtol
            Relative tolerance to use when checking for tensor equality during load,
            passed to :func:`torch.allclose`. It may be necessary to increase if you are
            saving and loading on two machines with torch built by different cuda
            versions. Be careful when changing this! See
            :class:`torch.finfo<torch.torch.finfo>` for more details about floating
            point precision of different data types (especially, ``eps``); if you have
            to increase this by more than 1 or 2 decades, then you are probably not
            dealing with a numerical issue.
        **pickle_load_args
            Any additional kwargs will be added to ``pickle_module.load`` via
            :func:`torch.load`, see that function's docstring for details.

        Raises
        ------
        ValueError
            If :func:`setup` or :func:`synthesize` has been called before this call
            to ``load``.
        ValueError
            If the object saved at ``file_path`` is not a ``MADCompetition`` object.
        ValueError
            If the saved and loading ``MADCompetition`` objects have a different value
            for any of :attr:`image`, :attr:`penalty_lambda`,
            :attr:`metric_tradeoff_lambda`, or :attr:`minimax`.
        ValueError
            If the behavior of :attr:`optimized_metric` or :attr:`reference_metric` is
            different between the saved and loading objects.

        Warns
        -----
        UserWarning
            If :func:`setup` will need to be called after ``load``, to finish
            initializing :attr:`optimizer` or :attr:`scheduler`.

        See Also
        --------
        :func:`~plenoptic.tools.io.examine_saved_synthesis`
            Examine metadata from saved object: pytorch and plenoptic versions, name of
            the synthesis object, shapes of tensors, etc.

        Examples
        --------
        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> def ds_ssim(x, y):
        ...     return 1 - po.metric.ssim(x, y)
        >>> mad = po.synth.MADCompetition(
        ...     img, po.metric.mse, ds_ssim, "min", metric_tradeoff_lambda=10
        ... )
        >>> mad.synthesize(max_iter=5, store_progress=True)
        >>> mad.save("mad.pt")
        >>> mad_copy = po.synth.MADCompetition(
        ...     img, po.metric.mse, ds_ssim, "min", metric_tradeoff_lambda=10
        ... )
        >>> mad_copy.load("mad.pt")
        """
        check_attributes = [
            "_image",
            "_metric_tradeoff_lambda",
            "_penalty_lambda",
            "_minmax",
        ]
        check_io_attrs = [
            ("_optimized_metric", ("_image", "_mad_image")),
            ("_reference_metric", ("_image", "_mad_image")),
            ("penalty_function", ("_image",)),
        ]
        super().load(
            file_path,
            "losses",
            map_location=map_location,
            check_attributes=check_attributes,
            check_io_attributes=check_io_attrs,
            state_dict_attributes=["_optimizer", "_scheduler"],
            tensor_equality_atol=tensor_equality_atol,
            tensor_equality_rtol=tensor_equality_rtol,
            **pickle_load_args,
        )
        # make this require a grad again
        self.mad_image.requires_grad_()
        # these are always supposed to be on cpu, but may get copied over to
        # gpu on load (which can cause problems when resuming synthesis), so
        # fix that.
        if len(self._saved_mad_image) and self._saved_mad_image[0].device.type != "cpu":
            self._saved_mad_image = [mad.to("cpu") for mad in self._saved_mad_image]

    @property
    def mad_image(self) -> Tensor:
        """Maximally-differentiating image, the parameter we are optimizing."""
        # numpydoc ignore=RT01,ES01
        if self._mad_image is None:
            return torch.empty(0)
        return self._mad_image

    @property
    def optimized_metric(self) -> torch.nn.Module | Callable[[Tensor, Tensor], Tensor]:
        """The metric whose value we are minimizing or maximizing."""
        # numpydoc ignore=RT01,ES01
        return self._optimized_metric

    @property
    def reference_metric(self) -> torch.nn.Module | Callable[[Tensor, Tensor], Tensor]:
        """The metric whose value we are keeping constant."""
        # numpydoc ignore=RT01,ES01
        return self._reference_metric

    @property
    def image(self) -> Tensor:
        """The reference image for this MAD Competition."""
        # numpydoc ignore=RT01,ES01
        return self._image

    @property
    def initial_image(self) -> Tensor:
        """
        Initial image for MAD Competition.

        This is the image whose distance to ``image``, the reference, we are
        maximizing/minimizing for ``optimized_metric``, while keeping constant for
        ``reference_metric``.
        """
        # numpydoc ignore=RT01
        return self._initial_image

    @property
    def reference_metric_loss(self) -> Tensor:
        """
        :attr:`reference_metric` loss over iterations.

        That is, the value of ``reference_metric(image, mad_image)``. Ideally, this is
        equal to ``reference_metric(image, initial_image)``.

        This tensor always lives on the CPU, regardless of the device of the
        ``MADCompetition`` object.
        """
        # numpydoc ignore=RT01
        return torch.as_tensor(self._reference_metric_loss)

    @property
    def optimized_metric_loss(self) -> Tensor:
        """
        :attr:`optimized_metric` loss over iterations.

        That is, the value of ``optimized_metric(image, mad_image)``. Ideally, this is
        either very different from ``optimized_metric(image, initial_image)``.

        This tensor always lives on the CPU, regardless of the device of the
        ``MADCompetition`` object.
        """
        # numpydoc ignore=RT01
        return torch.as_tensor(self._optimized_metric_loss)

    @property
    def metric_tradeoff_lambda(self) -> float:
        """Tradeoff between the two metrics in synthesis loss."""
        # numpydoc ignore=RT01,ES01
        return self._metric_tradeoff_lambda

    @property
    def minmax(self) -> Literal["min", "max"]:
        """Whether we are minimizing or maximizing :attr:`optimized_metric`."""
        # numpydoc ignore=RT01,ES01
        return self._minmax

    @property
    def saved_mad_image(self) -> Tensor:
        """
        :attr:`mad_image`, cached over time for later examination.

        How often the MAD image is cached is determined by the ``store_progress``
        argument to the :func:`synthesize` function.

        The last entry will always be the current :attr:`mad_image`.

        If ``store_progress==1``, then this corresponds directly to :attr:`losses`:
        ``losses[i]`` is the error for ``saved_mad_image[i]``

        This tensor always lives on the CPU, regardless of the device of the
        ``MADCompetition`` object.
        """  # numpydoc ignore=RT01
        if self._mad_image is None:
            return torch.empty(0)
        else:
            # for memory purposes, always on CPU
            return torch.stack([*self._saved_mad_image, self.mad_image.to("cpu")])


def plot_loss(
    mad: MADCompetition,
    iteration: int | None = None,
    axes: list[mpl.axes.Axes] | mpl.axes.Axes | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Plot metric losses.

    Plots ``mad.optimized_metric_loss`` and ``mad.reference_metric_loss`` on two
    separate axes, over all iterations. Also plots a red dot at ``iteration``,
    to highlight the loss there. If ``iteration=None``, then the dot will be at
    the final iteration.

    Parameters
    ----------
    mad
        MADCompetition object whose loss we want to plot.
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed.
    axes
        Pre-existing axes for plot. If a list of axes, must be the two axes to
        use for this plot. If a single axis, we'll split it in half
        horizontally. If ``None``, we call :func:`matplotlib.pyplot.gca()`.
    **kwargs
        Passed to :func:`matplotlib.pyplot.semilogy`.

    Returns
    -------
    axes :
        The matplotlib axes containing the plot.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.

    Notes
    -----
    We plot ``abs(mad.losses)`` because if we're maximizing the synthesis
    metric, we minimized its negative. By plotting the absolute value, we get
    them all on the same scale.
    """
    # this warning is not relevant for this plotting function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="loss iteration and iteration for")
        progress = mad.get_progress(iteration)

    if axes is None:
        axes = plt.gca()
    if not hasattr(axes, "__iter__"):
        axes = display.clean_up_axes(
            axes, False, ["top", "right", "bottom", "left"], ["x", "y"]
        )
        gs = axes.get_subplotspec().subgridspec(1, 2)
        fig = axes.figure
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    losses = [mad.reference_metric_loss, mad.optimized_metric_loss]
    names = ["reference_metric_loss", "optimized_metric_loss"]
    for ax, loss, name in zip(axes, losses, names):
        ax.plot(loss, **kwargs)
        ax.scatter(progress["iteration"], progress[name], c="r")
        ax.set(xlabel="Synthesis iteration", ylabel=name.capitalize().replace("_", " "))
    return ax


def display_mad_image(
    mad: MADCompetition,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    zoom: float | None = None,
    iteration: int | None = None,
    ax: mpl.axes.Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Display MAD image.

    We use :func:`~plenoptic.tools.display.imshow` to display the synthesized image and
    attempt to automatically find the most reasonable zoom value. You can override this
    value using the zoom arg, but remember that :func:`~plenoptic.tools.display.imshow`
    is opinionated about the size of the resulting image and will throw an Exception if
    the axis created is not big enough for the selected zoom.

    Parameters
    ----------
    mad
        MADCompetition object whose MAD image we want to display.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we assume
        image is RGB(A) and show all channels.
    zoom
        How much to zoom in / enlarge the synthesized image, the ratio
        of display pixels to image pixels. If ``None``, we
        attempt to find the best value ourselves.
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``mad.store_progress>1`` (that is, the MAD image was not cached on every
        iteration), then we show the cached MAD image from the nearest iteration.
    ax
        Pre-existing axes for plot. If ``None``, we call :func:`matplotlib.pyplot.gca`.
    title
        Title to add to axis. If ``None``, we use ``"MAD Image [iteration={iter}]"``,
        where ``iter`` gives the iteration corresponding to the displayed image.
    **kwargs
        Passed to :func:`~plenoptic.tools.display.imshow`.

    Returns
    -------
    ax :
        The matplotlib axes containing the plot.

    Raises
    ------
    ValueError
        If ``batch_idx`` is not an int.
    IndexError
        If ``iteration`` takes an illegal value.

    Warns
    -----
    UserWarning
        If the iteration used for ``saved_mad_image`` is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``mad.store_progress=2``).
    """
    progress = mad.get_progress(iteration)
    try:
        image = progress["saved_mad_image"]
        iter = progress["store_progress_iteration"]
    except KeyError:
        if iteration is not None:
            raise IndexError("When mad.store_progress=False, iteration must be None!")
        image = mad.mad_image
        # losses will always have one extra value, the current loss.
        iter = len(mad.losses) - 1

    if batch_idx is None:
        raise ValueError("batch_idx must be an integer!")
    # we're only plotting one image here, so if the user wants multiple
    # channels, they must be RGB
    as_rgb = bool(channel_idx is None and image.shape[1] > 1)
    if ax is None:
        ax = plt.gca()
    if title is None:
        title = f"MAD Image [iteration={iter}]"
    display.imshow(
        image,
        ax=ax,
        title=title,
        zoom=zoom,
        batch_idx=batch_idx,
        channel_idx=channel_idx,
        as_rgb=as_rgb,
        **kwargs,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax


def plot_pixel_values(
    mad: MADCompetition,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    iteration: int | None = None,
    ylim: tuple[float] | Literal[False] = False,
    ax: mpl.axes.Axes | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    r"""
    Plot histogram of pixel values of reference and MAD images.

    As a way to check the distributions of pixel intensities and see
    if there's any values outside the allowed range.

    Parameters
    ----------
    mad
        MADCompetition object with the images whose pixel values we want to compare.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) images).
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``mad.store_progress>1`` (that is, the MAD image was not cached on every
        iteration), then we use the cached MAD image from the nearest iteration.
    ylim
        If tuple, the ylimit to set for this axis. If ``False``, we leave
        it untouched.
    ax
        Pre-existing axes for plot. If ``None``, we call
        :func:`matplotlib.pyplot.gca()`.
    **kwargs
        Passed to :func:`matplotlib.pyplot.hist`.

    Returns
    -------
    ax :
        Creates axes.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.

    Warns
    -----
    UserWarning
        If the iteration used for ``saved_mad_image`` is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``mad.store_progress=2``).
    """

    def _freedman_diaconis_bins(a: np.ndarray) -> int:
        """
        Calculate number of hist bins using Freedman-Diaconis rule.

        Copied from seaborn.

        Parameters
        ----------
        a
            The array to histogram.

        Returns
        -------
        n_bins
            Number of bins to use for histogram.
        """
        # From https://stats.stackexchange.com/questions/798/
        a = np.asarray(a)
        iqr = np.diff(np.percentile(a, [0.25, 0.75]))[0]
        if len(a) < 2:
            return 1
        h = 2 * iqr / (len(a) ** (1 / 3))
        # fall back to sqrt(a) bins if iqr is 0
        if h == 0:
            return int(np.sqrt(a.size))
        else:
            return int(np.ceil((a.max() - a.min()) / h))

    kwargs.setdefault("alpha", 0.4)
    progress = mad.get_progress(iteration)
    try:
        mad_image = progress["saved_mad_image"]
        iter = progress["store_progress_iteration"]
    except KeyError:
        if iteration is not None:
            raise IndexError("When mad.store_progress=False, iteration must be None!")
        mad_image = mad.mad_image
        # losses will always have one extra value, the current loss.
        iter = len(mad.losses) - 1
    image = mad.image[batch_idx]
    if channel_idx is not None:
        image = image[channel_idx]
        mad_image = mad_image[channel_idx]
    image = data.to_numpy(image).flatten()
    mad_image = data.to_numpy(mad_image).flatten()

    if ax is None:
        ax = plt.gca()
    ax.hist(
        mad_image,
        bins=min(_freedman_diaconis_bins(image), 50),
        label=f"MAD image [iteration={iter}]",
        **kwargs,
    )
    ax.hist(
        image,
        bins=min(_freedman_diaconis_bins(image), 50),
        label="Reference image",
        **kwargs,
    )
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title("Histogram of pixel values")
    return ax


def _check_included_plots(to_check: list[str] | dict[str, int], to_check_name: str):
    """
    Check whether the user wanted us to create plots that we can't.

    Helper function for :func:`plot_synthesis_status` and :func:`animate`.

    Raises a ``ValueError`` if ``to_check`` contains any values that are not allowed.

    Parameters
    ----------
    to_check
        The variable to check. We ensure that it doesn't contain any extra (not
        allowed) values. If a list, we check its contents. If a dict, we check
        its keys.
    to_check_name
        Name of the ``to_check`` variable, used in the error message.

    Raises
    ------
    ValueError
        If ``to_check`` takes an illegal value.
    """
    allowed_vals = [
        "display_mad_image",
        "plot_loss",
        "plot_pixel_values",
        "misc",
    ]
    try:
        vals = to_check.keys()
    except AttributeError:
        vals = to_check
    not_allowed = [v for v in vals if v not in allowed_vals]
    if not_allowed:
        raise ValueError(
            f"{to_check_name} contained value(s) {not_allowed}! "
            f"Only {allowed_vals} are permissible!"
        )


def _setup_synthesis_fig(
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float] | None = None,
    included_plots: list[str] = [
        "display_mad_image",
        "plot_loss",
        "plot_pixel_values",
    ],
    display_mad_image_width: float = 1,
    plot_loss_width: float = 2,
    plot_pixel_values_width: float = 1,
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes], dict[str, int]]:
    """
    Set up figure for :func:`plot_synthesis_status`.

    Creates figure with enough axes for the all the plots you want. Will
    also create index in ``axes_idx`` for them if you haven't done so already.

    If ``fig=None``, all axes will be on the same row and have the same width. If
    you want them to be on different rows, will need to initialize ``fig`` yourself
    and pass that in. For changing width, change the corresponding ``*_width`` arg,
    which gives width relative to other axes. So if you want the axis for the
    loss plot to be three times as wide as the others, set ``loss_width=3``.

    Parameters
    ----------
    fig
        The figure to plot on or ``None``. If ``None``, we create a new figure.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows for more
        fine-grained control of the resulting figure. Probably only helpful if fig is
        also defined. Possible keys: ``"loss"``, ``"pixel_values"``, ``"misc"``. Values
        should all be ints. If you tell this function to create a plot that doesn't have
        a corresponding key, we find the lowest int that is not already in the dict, so
        if you have axes that you want unchanged, place their idx in ``"misc"``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have relative width=1 correspond to 5.
    included_plots
        Which plots to include. Must be some subset of ``'display_mad_image',
        'plot_loss', 'plot_pixel_values'``.
    display_mad_image_width
        Relative width of the axis for the synthesized image.
    plot_loss_width
        Relative width of the axis for loss plot.
    plot_pixel_values_width
        Relative width of the axis for image pixel intensities histograms.

    Returns
    -------
    fig
        The figure to plot on.
    axes
        List or array of axes contained in fig.
    axes_idx
        Dictionary identifying the idx for each plot type.
    """
    n_subplots = 0
    axes_idx = axes_idx.copy()
    width_ratios = []
    if "display_mad_image" in included_plots:
        n_subplots += 1
        width_ratios.append(display_mad_image_width)
        if "display_mad_image" not in axes_idx:
            axes_idx["display_mad_image"] = data._find_min_int(axes_idx.values())
    if "plot_loss" in included_plots:
        n_subplots += 1
        width_ratios.append(plot_loss_width)
        if "plot_loss" not in axes_idx:
            axes_idx["plot_loss"] = data._find_min_int(axes_idx.values())
    if "plot_pixel_values" in included_plots:
        n_subplots += 1
        width_ratios.append(plot_pixel_values_width)
        if "plot_pixel_values" not in axes_idx:
            axes_idx["plot_pixel_values"] = data._find_min_int(axes_idx.values())
    if fig is None:
        width_ratios = np.array(width_ratios)
        if figsize is None:
            # we want (5, 5) for each subplot, with a bit of room between
            # each subplot
            figsize = ((width_ratios * 5).sum() + width_ratios.sum() - 1, 5)
        width_ratios = width_ratios / width_ratios.sum()
        fig, axes = plt.subplots(
            1,
            n_subplots,
            figsize=figsize,
            gridspec_kw={"width_ratios": width_ratios},
        )
        if n_subplots == 1:
            axes = [axes]
    else:
        axes = fig.axes
    # make sure misc contains all the empty axes
    misc_axes = axes_idx.get("misc", [])
    if not hasattr(misc_axes, "__iter__"):
        misc_axes = [misc_axes]
    all_axes = []
    for i in axes_idx.values():
        # so if it's a list of ints
        if hasattr(i, "__iter__"):
            all_axes.extend(i)
        else:
            all_axes.append(i)
    misc_axes += [i for i, _ in enumerate(fig.axes) if i not in all_axes]
    axes_idx["misc"] = misc_axes
    return fig, axes, axes_idx


def plot_synthesis_status(
    mad: MADCompetition,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    iteration: int | None = None,
    vrange: tuple[float] | str = "indep1",
    zoom: float | None = None,
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float] | None = None,
    included_plots: list[str] = [
        "display_mad_image",
        "plot_loss",
        "plot_pixel_values",
    ],
    width_ratios: dict[str, float] = {},
) -> tuple[mpl.figure.Figure, dict[str, int]]:
    r"""
    Make a plot showing synthesis status.

    We create several subplots to analyze this. The plots to include are
    specified by including their name in the ``included_plots`` list. All plots
    can be created separately using the method with the same name.

    Parameters
    ----------
    mad
        MADCompetition object whose status we want to plot.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) image).
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``mad.store_progress>1`` (that is, the MAD image was not cached on every
        iteration), then we use the cached MAD image from the nearest iteration.
    vrange
        The vrange option to pass to ``display_mad_image()``. See
        docstring of :func:`~plenoptic.tools.display.imshow` for possible values.
    zoom
        How much to zoom in / enlarge the synthesized image, the ratio
        of display pixels to image pixels. If ``None``, we
        attempt to find the best value ourselves.
    fig
        If ``None``, we create a new figure. otherwise we assume this is
        an empty figure that has the appropriate size and number of
        subplots.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'mad_image',
        'loss', 'pixel_values', 'misc'``. Values should all be ints. If you
        tell this function to create a plot that doesn't have a corresponding
        key, we find the lowest int that is not already in the dict, so if you
        have axes that you want unchanged, place their idx in ``'misc'``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have each axis be of size ``(5, 5)``.
    included_plots
        Which plots to include. Must be some subset of ``'display_mad_image',
        'plot_loss', 'plot_pixel_values'``.
    width_ratios
        If ``width_ratios`` is an empty dictionary, ``plot_loss`` will have
        double the width of the other plots. To change that, specify their
        relative widths using the keys: ['display_mad_image', 'plot_loss',
        'plot_pixel_values'] and floats specifying their relative width.

    Returns
    -------
    fig
        The figure containing this plot.
    axes_idx
        Dictionary giving index of each plot.

    Raises
    ------
    ValueError
        If ``mad.mad_image`` object is not 3d or 4d.
    ValueError
        If the ``iteration is not None`` and the given ``mad`` object was run
        with ``store_progress=False``.

    Warns
    -----
    UserWarning
        If the iteration used for ``saved_mad_image`` is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``mad.store_progress=2``).
    """
    if iteration is not None and not mad.store_progress:
        raise ValueError(
            "synthesis() was run with store_progress=False, "
            "cannot specify which iteration to plot (only"
            " last one, with iteration=None)"
        )
    if mad.mad_image.ndim not in [3, 4]:
        raise ValueError(
            "plot_synthesis_status() expects 3 or 4d data;"
            "unexpected behavior will result otherwise!"
        )
    _check_included_plots(included_plots, "included_plots")
    _check_included_plots(width_ratios, "width_ratios")
    _check_included_plots(axes_idx, "axes_idx")
    width_ratios = {f"{k}_width": v for k, v in width_ratios.items()}
    fig, axes, axes_idx = _setup_synthesis_fig(
        fig, axes_idx, figsize, included_plots, **width_ratios
    )

    if "display_mad_image" in included_plots:
        display_mad_image(
            mad,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=iteration,
            ax=axes[axes_idx["display_mad_image"]],
            zoom=zoom,
            vrange=vrange,
        )
    if "plot_loss" in included_plots:
        plot_loss(mad, iteration=iteration, axes=axes[axes_idx["plot_loss"]])
        # this function creates a single axis for loss, which plot_loss then
        # split into two. this makes sure the right two axes are present in the
        # dict
        all_axes = []
        for i in axes_idx.values():
            # so if it's a list of ints
            if hasattr(i, "__iter__"):
                all_axes.extend(i)
            else:
                all_axes.append(i)
        new_axes = [i for i, _ in enumerate(fig.axes) if i not in all_axes]
        axes_idx["plot_loss"] = new_axes
    if "plot_pixel_values" in included_plots:
        plot_pixel_values(
            mad,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=iteration,
            ax=axes[axes_idx["plot_pixel_values"]],
        )
    return fig, axes_idx


def animate(
    mad: MADCompetition,
    framerate: int = 10,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    zoom: float | None = None,
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float] | None = None,
    included_plots: list[str] = [
        "display_mad_image",
        "plot_loss",
        "plot_pixel_values",
    ],
    width_ratios: dict[str, float] = {},
) -> mpl.animation.FuncAnimation:
    r"""
    Animate synthesis progress.

    This is essentially the figure produced by
    ``mad.plot_synthesis_status`` animated over time, for each stored
    iteration.

    This functions returns a matplotlib FuncAnimation object. See our
    documentation (e.g., :ref:`quickstart-nb`) for examples on how to view it in
    a Jupyter notebook. In order to save, use ``anim.save(filename)``. In either
    case, this can take a while and you'll need the appropriate writer installed
    and on your path, e.g., ffmpeg, imagemagick, etc). See
    :doc:`matplotlib documentation <matplotlib:api/animation_api>` for more details.

    Parameters
    ----------
    mad
        MADCompetition object whose synthesis we want to animate.
    framerate
        How many frames a second to display.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) image).
    zoom
        How much to zoom in / enlarge the synthesized image, the ratio
        of display pixels to image pixels. If ``None``, we
        attempt to find the best value ourselves.
    fig
        If ``None``, we create a new figure. otherwise we assume this is
        an empty figure that has the appropriate size and number of
        subplots.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'mad_image',
        'loss', 'pixel_values', 'misc'``. Values should all be ints. If you
        tell this function to create a plot that doesn't have a corresponding
        key, we find the lowest int that is not already in the dict, so if you
        have axes that you want unchanged, place their idx in ``'misc'``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have each axis be of size ``(5, 5)``.
    included_plots
        Which plots to include. Must be some subset of ``'display_mad_image',
        'plot_loss', 'plot_pixel_values'``.
    width_ratios
        If ``width_ratios`` is an empty dictionary, ``plot_loss`` will have
        double the width of the other plots. To change that, specify their
        relative widths using the keys: ['display_mad_image', 'plot_loss',
        'plot_pixel_values'] and floats specifying their relative width.

    Returns
    -------
    anim
        The animation object. In order to view, must convert to HTML
        or save.

    Raises
    ------
    ValueError
        If the given ``mad`` object was run with ``store_progress=False``.
    ValueError
        If ``mad.mage_image`` object is not 3d or 4d.
    ValueError
        If we do not know how to interpret the value of ``ylim``.

    Notes
    -----
    Unless specified, we use the ffmpeg backend, which requires that you have
    ffmpeg installed and on your path (https://ffmpeg.org/download.html).
    To use a different, use the matplotlib rcParams:
    ``matplotlib.rcParams['animation.writer'] = writer``, see
    `matplotlib documentation
    <https://matplotlib.org/stable/api/animation_api.html#writer-classes>`_ for more
    details.
    """
    if not mad.store_progress:
        raise ValueError(
            "synthesize() was run with store_progress=False, cannot animate!"
        )
    if mad.mad_image.ndim not in [3, 4]:
        raise ValueError(
            "animate() expects 3 or 4d data; unexpected behavior will result otherwise!"
        )
    _check_included_plots(included_plots, "included_plots")
    _check_included_plots(width_ratios, "width_ratios")
    _check_included_plots(axes_idx, "axes_idx")
    # we run plot_synthesis_status to initialize the figure if either fig is
    # None or if there are no titles on any axes, which we assume means that
    # it's an empty figure
    if fig is None or not any([ax.get_title() for ax in fig.axes]):
        fig, axes_idx = plot_synthesis_status(
            mad=mad,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=0,
            figsize=figsize,
            zoom=zoom,
            fig=fig,
            included_plots=included_plots,
            axes_idx=axes_idx,
            width_ratios=width_ratios,
        )
    # grab the artist for the second plot (we don't need to do this for the
    # MAD image plot, because we use the update_plot function for that)
    if "plot_loss" in included_plots:
        scat = [fig.axes[i].collections[0] for i in axes_idx["plot_loss"]]
    if "display_mad_image" in included_plots:
        fig.axes[axes_idx["display_mad_image"]].set_title("MAD Image")

    def movie_plot(i: int) -> list[mpl.artist.Artist]:
        """
        Matplotlib function for animation.

        Update plots for frame ``i``.

        Parameters
        ----------
        i
            The frame to plot.

        Returns
        -------
        artists
            The updated matplotlib artists.
        """
        # this warning is not relevant for animate
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="loss iteration and iteration for"
            )
            artists = []
            if "display_mad_image" in included_plots:
                artists.extend(
                    display.update_plot(
                        fig.axes[axes_idx["display_mad_image"]],
                        data=mad.saved_mad_image[i],
                        batch_idx=batch_idx,
                    )
                )
            if "plot_pixel_values" in included_plots:
                # this is the dumbest way to do this, but it's simple --
                # clearing the axes can cause problems if the user has, for
                # example, changed the tick locator or formatter. not sure how
                # to handle this best right now
                fig.axes[axes_idx["plot_pixel_values"]].clear()
                plot_pixel_values(
                    mad,
                    batch_idx=batch_idx,
                    channel_idx=channel_idx,
                    iteration=min(i * mad.store_progress, len(mad.losses) - 1),
                    ax=fig.axes[axes_idx["plot_pixel_values"]],
                )
            if "plot_loss" in included_plots:
                # loss always contains values from every iteration, but everything
                # else will be subsampled.
                x_val = mad._convert_iteration(i, False)
                scat[0].set_offsets((x_val, mad.reference_metric_loss[x_val]))
                scat[1].set_offsets((x_val, mad.optimized_metric_loss[x_val]))
                artists.extend(scat)
            # as long as blitting is True, need to return a sequence of artists
            return artists

    # don't need an init_func, since we handle initialization ourselves
    anim = mpl.animation.FuncAnimation(
        fig,
        movie_plot,
        frames=len(mad.saved_mad_image),
        blit=True,
        interval=1000.0 / framerate,
        repeat=False,
    )
    plt.close(fig)
    return anim


def display_mad_image_all(
    mad_metric1_min: MADCompetition,
    mad_metric2_min: MADCompetition,
    mad_metric1_max: MADCompetition,
    mad_metric2_max: MADCompetition,
    metric1_name: str | None = None,
    metric2_name: str | None = None,
    zoom: int | float = 1,
    **kwargs: Any,
) -> mpl.figure.Figure:
    """
    Display all MAD Competition images.

    To generate a full set of MAD Competition images, you need four instances:
    one for minimizing and maximizing each metric. This helper function creates
    a figure to display the full set of images.

    In addition to the four MAD Competition images, this also plots the initial
    image from ``mad_metric1_min``, for comparison.

    Parameters
    ----------
    mad_metric1_min
        ``MADCompetition`` object that minimized the first metric.
    mad_metric2_min
        ``MADCompetition`` object that minimized the second metric.
    mad_metric1_max
        ``MADCompetition`` object that maximized the first metric.
    mad_metric2_max
        ``MADCompetition`` object that maximized the second metric.
    metric1_name
        Name of the first metric. If ``None``, we use the name of the
        ``optimized_metric`` function from ``mad_metric1_min``.
    metric2_name
        Name of the second metric. If ``None``, we use the name of the
        ``optimized_metric`` function from ``mad_metric2_min``.
    zoom
        Ratio of display pixels to image pixels. See
        :func:`~plenoptic.tools.display.imshow` for details.
    **kwargs
        Passed to :func:`~plenoptic.tools.display.imshow`.

    Returns
    -------
    fig
        Figure containing the images.

    Raises
    ------
    ValueError
        If the four ``MADCompetition`` instances do not have the same ``image``
        attribute.
    """
    # this is a bit of a hack right now, because they don't all have same
    # initial image
    if not torch.allclose(mad_metric1_min.image, mad_metric2_min.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if not torch.allclose(mad_metric1_min.image, mad_metric1_max.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if not torch.allclose(mad_metric1_min.image, mad_metric2_max.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if metric1_name is None:
        metric1_name = mad_metric1_min.optimized_metric.__name__
    if metric2_name is None:
        metric2_name = mad_metric2_min.optimized_metric.__name__
    fig = pt_make_figure(3, 2, [zoom * i for i in mad_metric1_min.image.shape[-2:]])
    mads = [mad_metric1_min, mad_metric1_max, mad_metric2_min, mad_metric2_max]
    titles = [
        f"Minimize {metric1_name}",
        f"Maximize {metric1_name}",
        f"Minimize {metric2_name}",
        f"Maximize {metric2_name}",
    ]
    # we're only plotting one image here, so if the user wants multiple
    # channels, they must be RGB
    if kwargs.get("channel_idx") is None and mad_metric1_min.initial_image.shape[1] > 1:
        as_rgb = True
    else:
        as_rgb = False
    display.imshow(
        mad_metric1_min.image,
        ax=fig.axes[0],
        title="Reference image",
        zoom=zoom,
        as_rgb=as_rgb,
        **kwargs,
    )
    display.imshow(
        mad_metric1_min.initial_image,
        ax=fig.axes[1],
        title="Initial (noisy) image",
        zoom=zoom,
        as_rgb=as_rgb,
        **kwargs,
    )
    for ax, mad, title in zip(fig.axes[2:], mads, titles):
        display_mad_image(mad, zoom=zoom, ax=ax, title=title, **kwargs)
    return fig


def plot_loss_all(
    mad_metric1_min: MADCompetition,
    mad_metric2_min: MADCompetition,
    mad_metric1_max: MADCompetition,
    mad_metric2_max: MADCompetition,
    metric1_name: str | None = None,
    metric2_name: str | None = None,
    metric1_kwargs: dict = {"c": "C0"},
    metric2_kwargs: dict = {"c": "C1"},
    min_kwargs: dict = {"linestyle": "--"},
    max_kwargs: dict = {"linestyle": "-"},
    figsize: tuple[int, int] = (10, 5),
) -> mpl.figure.Figure:
    """
    Plot loss for full set of MAD Competiton instances.

    To generate a full set of MAD Competition images, you need four instances:
    one for minimizing and maximizing each metric. This helper function creates
    a two-axis figure to display the loss for this full set.

    Parameters
    ----------
    mad_metric1_min
        ``MADCompetition`` object that minimized the first metric.
    mad_metric2_min
        ``MADCompetition`` object that minimized the second metric.
    mad_metric1_max
        ``MADCompetition`` object that maximized the first metric.
    mad_metric2_max
        ``MADCompetition`` object that maximized the second metric.
    metric1_name
        Name of the first metric. If ``None``, we use the name of the
        ``optimized_metric`` function from ``mad_metric1_min``.
    metric2_name
        Name of the second metric. If ``None``, we use the name of the
        ``optimized_metric`` function from ``mad_metric2_min``.
    metric1_kwargs
        Dictionary of arguments to pass to :func:`matplotlib.pyplot.plot` to identify
        synthesis instance where the first metric was being optimized.
    metric2_kwargs
        Dictionary of arguments to pass to :func:`matplotlib.pyplot.plot` to identify
        synthesis instance where the second metric was being optimized.
    min_kwargs
        Dictionary of arguments to pass to :func:`matplotlib.pyplot.plot` to identify
        synthesis instance where ``optimized_metric`` was being minimized.
    max_kwargs
        Dictionary of arguments to pass to :func:`matplotlib.pyplot.plot` to identify
        synthesis instance where ``optimized_metric`` was being maximized.
    figsize
        Size of the figure we create.

    Returns
    -------
    fig
        Figure containing the plot.

    Raises
    ------
    ValueError
        If the four ``MADCompetition`` instances do not have the same ``image``
        attribute.
    """
    if not torch.allclose(mad_metric1_min.image, mad_metric2_min.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if not torch.allclose(mad_metric1_min.image, mad_metric1_max.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if not torch.allclose(mad_metric1_min.image, mad_metric2_max.image):
        raise ValueError("All four instances of MADCompetition must have same image!")
    if metric1_name is None:
        metric1_name = mad_metric1_min.optimized_metric.__name__
    if metric2_name is None:
        metric2_name = mad_metric2_min.optimized_metric.__name__
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_loss(
        mad_metric1_min,
        axes=axes,
        label=f"Minimize {metric1_name}",
        **metric1_kwargs,
        **min_kwargs,
    )
    plot_loss(
        mad_metric1_max,
        axes=axes,
        label=f"Maximize {metric1_name}",
        **metric1_kwargs,
        **max_kwargs,
    )
    # we pass the axes backwards here because the fixed and synthesis metrics are
    # the opposite as they are in the instances above.
    plot_loss(
        mad_metric2_min,
        axes=axes[::-1],
        label=f"Minimize {metric2_name}",
        **metric2_kwargs,
        **min_kwargs,
    )
    plot_loss(
        mad_metric2_max,
        axes=axes[::-1],
        label=f"Maximize {metric2_name}",
        **metric2_kwargs,
        **max_kwargs,
    )
    axes[0].set(ylabel="Loss", title=metric2_name)
    axes[1].set(ylabel="Loss", title=metric1_name)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.1, 0.5))
    return fig
