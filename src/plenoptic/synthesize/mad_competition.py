"""
Maximum Differentiation Competition.

Maximum Differentiation Competition synthesizes images which maximally distinguish
between a pair of metrics. Generally speaking, they are synthesized in pairs (two images
that one metric considers identical and the other considers as different as possible) or
groups of four (a pair of such pairs, one for each of the two metrics). They emphasize
the features that distinguish metrics, highlighting the features that one metric
considers important that the other is invariant to.
"""

import contextlib
import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from .. import optim
from ..convergence import _loss_convergence
from ..validate import validate_input, validate_metric
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
    range_penalty_lambda
        Lambda to multiply by range penalty and add to loss.
    allowed_range
        Range (inclusive) of allowed pixel values. Any values outside this
        range will be penalized.

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
        range_penalty_lambda: float = 0.1,
        allowed_range: tuple[float, float] = (0, 1),
    ):
        super().__init__(range_penalty_lambda, allowed_range)
        validate_input(image, allowed_range=allowed_range)
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
            mad_image = mad_image.clamp(*self.allowed_range)
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

        We run this until either we reach ``max_iter`` or the loss changes less than
        ``stop_criterion`` over the past ``stop_iters_to_check`` iterations,
        whichever comes first.

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
        the quadratic bound penalty, :math:`\lambda_1` is :attr:`metric_tradeoff_lambda`
        and :math:`\lambda_2` is :attr:`range_penalty_lambda`.

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
        range_penalty = optim.penalize_range(mad_image, self.allowed_range)
        return (
            synth_target * synthesis_loss
            + self.metric_tradeoff_lambda * fixed_loss
            + self.range_penalty_lambda * range_penalty
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

        Uses :func:`~plenoptic.tools.convergence._loss_convergence`.

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
        return _loss_convergence(self, stop_criterion, stop_iters_to_check)

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
        ]
        save_state_dict_attrs = ["_optimizer", "_scheduler"]
        super().save(file_path, save_io_attrs, save_state_dict_attrs)

    def to(self, *args: Any, **kwargs: Any):
        r"""
        Move and/or casts the parameters and buffers.

        This can be called as

        .. code:: python

            to(device=None, dtype=None, non_blocking=False)

        .. code:: python

            to(dtype, non_blocking=False)

        .. code:: python

            to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired ``dtype``. In addition, this method will
        only cast the floating point parameters and buffers to ``dtype``
        (if given). The integral parameters and buffers will be moved
        ``device``, if that is given, but with dtypes unchanged. When
        `on_blocking`` is set, it tries to convert/move asynchronously
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
            for any of :attr:`image`, :attr:`range_penalty_lambda`,
            :attr:`allowed_range`, :attr:`metric_tradeoff_lambda`, or :attr:`minmax`.
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
            "_range_penalty_lambda",
            "_allowed_range",
            "_minmax",
        ]
        check_io_attrs = [
            ("_optimized_metric", ("_image", "_mad_image")),
            ("_reference_metric", ("_image", "_mad_image")),
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
