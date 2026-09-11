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

from .. import regularize
from ..convergence import _loss_convergence
from ..validate import validate_input, validate_metric, validate_penalty
from .synthesis import _OptimizedSynthesis

__all__ = [
    "MADCompetition",
]


def __dir__() -> list[str]:
    return __all__


class MADCompetition(_OptimizedSynthesis):
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
        A function applied to the metamer during optimization, that returns
        a scalar penalty to be minimized. By penalizing certain properties of
        the image, like pixels values outside an allowed range, we can constrain
        those image properties. See :ref:`how-to-penalty` in the
        documentation for details and examples.
    penalty_lambda
        Weight of the penalty term. Must be non-negative.

    References
    ----------
    .. [1] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
           competition: A methodology for comparing computational models of
           perceptual discriminability. Journal of Vision, 8(12), 1–13.
           https://dx.doi.org/10.1167/8.12.8

    Examples
    --------
    Synthesize and visualize MAD Competition between SSIM and MSE. Note that MAD
    Competition requires two distance metrics (where a value of 0 means the inputs are
    identical), and thus we must pass ``1 - ssim`` as our metric, since
    :func:`~plenoptic.metric.ssim` is a similarity metric, and thus 0 corresponds to
    "completely different" and 1 "identical".

    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import matplotlib.pyplot as plt
      >>> img = po.data.einstein()
      >>> def ds_ssim(x, y):
      ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
      >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
      >>> mad.synthesize(200)
      >>> fig, axes = plt.subplots(1, 3, figsize=(16, 4), width_ratios=[1, 1, 2])
      >>> po.plot.imshow(img, ax=axes[0], title="Target image")
      <Figure size ... with 3 Axes>
      >>> axes[0].xaxis.set_visible(False)
      >>> axes[0].yaxis.set_visible(False)
      >>> po.plot.synthesis_status(mad, fig=fig, axes_idx={"misc": 0})
      <Figure size ...>
      >>> fig.subplots_adjust(wspace=0.3)

    If ``metric_tradeoff_lambda`` is not specified, we will attempt to select a
    reasonable value based on the values of the two metrics comparing ``image``
    and some random image. You can use this as a starting point, but we recommend
    adjusting it.

    .. plot::
      :context: reset

      >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max")
      >>> mad.metric_tradeoff_lambda
      10.0

    Set ``minmax`` to determine whether to minimize or maximize ``optimized_metric``.
    Notice that its value, plotted in the rightmost subplot below, decreases, as opposed
    to the increase seen above.

    .. plot::
      :context: reset

      >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "min", 1e6)
      >>> mad.synthesize(200)
      >>> fig, axes = plt.subplots(1, 3, figsize=(16, 4), width_ratios=[1, 1, 2])
      >>> po.plot.imshow(img, ax=axes[0], title="Target image")
      <Figure size ... with 3 Axes>
      >>> axes[0].xaxis.set_visible(False)
      >>> axes[0].yaxis.set_visible(False)
      >>> po.plot.synthesis_status(mad, fig=fig, axes_idx={"misc": 0})
      <Figure size ...>
      >>> fig.subplots_adjust(wspace=0.3)
    """

    def __init__(
        self,
        image: Tensor,
        optimized_metric: torch.nn.Module | Callable[[Tensor, Tensor], Tensor],
        reference_metric: torch.nn.Module | Callable[[Tensor, Tensor], Tensor],
        minmax: Literal["min", "max"],
        metric_tradeoff_lambda: float | None = None,
        penalty_function: Callable[[Tensor], Tensor] = regularize.penalize_range,
        penalty_lambda: float = 0.1,
    ):
        super().__init__(
            penalty_function=penalty_function, penalty_lambda=penalty_lambda
        )
        validate_input(image)
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
        validate_penalty(penalty_function, image.shape, image.dtype, image.device)
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
        self._current_ref_metric = None
        self._current_opt_metric = None

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
        >>> mad = po.MADCompetition(
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
        >>> mad = po.MADCompetition(
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
        >>> mad = po.MADCompetition(
        ...     img,
        ...     lambda x, y: 1 - po.metric.ssim(x, y),
        ...     po.metric.mse,
        ...     "min",
        ...     metric_tradeoff_lambda=0.1,
        ... )
        >>> mad.setup(1, optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.01})
        >>> mad.synthesize(10)
        >>> mad.save("mad_setup.pt")
        >>> mad = po.MADCompetition(
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
            self._initial_image = mad_image.clone()
            mad_image.requires_grad_()
            self._mad_image = mad_image
            self._reference_metric_target = self.reference_metric(
                self.image, self.mad_image
            ).item()
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

        See Also
        --------
        :func:`~plenoptic.plot.synthesis_status`
            Create a plot summarizing synthesis status at a given iteration.
        :func:`~plenoptic.plot.synthesis_animate`
            Create a video of the metamer changing over the course of
            synthesis.

        Examples
        --------
        >>> import plenoptic as po
        >>> po.set_seed(0)
        >>> img = po.data.einstein()
        >>> def ds_ssim(x, y):
        ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max")
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> mad.synthesize(5)
        >>> mad.losses
        tensor([-0.0142, -0.1498, -0.2685, -0.3714, -0.4603, -0.5369])

        Synthesize MAD image, using ``store_progress`` so we can examine progress later.
        (This also enables us to create a video of the MAD image changing over the
        course of synthesis, see :func:`~plenoptic.plot.synthesis_animate`.)

        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max")
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> mad.synthesize(5, store_progress=2)
        >>> mad.saved_mad_image.shape
        torch.Size([4, 1, 1, 256, 256])
        >>> # see loss, etc on the 4th iteration
        >>> progress = mad.get_progress(4)
        >>> progress.keys()
        dict_keys(['losses', ..., 'saved_mad_image', 'store_progress_iteration'])
        >>> progress["losses"]
        tensor(-0.4511)

        Adjust ``stop_criterion`` and ``stop_iters_to_check`` to change how convergence
        is determined. In this case, we stop early by making ``stop_criterion`` fairly
        large. In practice, you're more likely to make ``stop_criterion`` smaller to let
        synthesis run for longer.

        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max")
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> mad.synthesize(12, stop_criterion=0.1, stop_iters_to_check=2)
        >>> len(mad.losses)
        6
        """
        # if setup hasn't been called manually, call it now.
        if self._mad_image is None or isinstance(self._scheduler, tuple):
            self.setup()
        self._current_loss = None
        self._current_penalty = None
        self._current_ref_metric = None
        self._current_opt_metric = None

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
            sm, fm, penalty = self._objective_function()
            self._current_penalty = penalty.item()
            self._current_ref_metric = fm.item()
            self._current_opt_metric = sm.item()

        pbar.close()

    def _objective_function(
        self,
        mad_image: Tensor | None = None,
        image: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute objective function components.

        This calls :attr:`optimized_metric`, :attr:`reference_metric`, and
        :attr:`penalty_function` and returns their output, without combining them. It is
        not meant to be called directly, but is used by both :func:`_closure` and
        (public) :func:`objective_function`.

        Parameters
        ----------
        mad_image
            Proposed ``mad_image``. If ``None``, use ``self.mad_image``.
        image
            Proposed ``image``. If ``None``, use ``self.image``.

        Returns
        -------
        sm
            1-element tensor containing optimized_metric(image, mad_image).
        fm
            1-element tensor containing reference_metric(image, mad_image).
        penalty
            1-element tensor containing the penalty on this step.
        """
        if image is None:
            image = self.image
        if mad_image is None:
            mad_image = self.mad_image
            # if this is empty, then self.mad_image hasn't been initialized
            if mad_image.numel() == 0:
                return torch.empty(0), torch.empty(0), torch.empty(0)
        sm = self.optimized_metric(image, mad_image)
        fm = self.reference_metric(image, mad_image)
        penalty = self.penalty_function(mad_image)
        return sm, fm, penalty

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

        If :meth:`setup` or :meth:`synthesize` has not been called to initialize the MAD
        image, then this will return an empty tensor.

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

        Examples
        --------
        >>> import plenoptic as po
        >>> po.set_seed(0)
        >>> img = po.data.einstein()
        >>> def ds_ssim(x, y):
        ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max")

        Before :meth:`setup` or :meth:`synthesize` is called, this returns an
        empty tensor because the MAD image attribute hasn't been initialized:

        >>> mad.objective_function()
        tensor([])
        >>> mad.synthesize(5, store_progress=True)

        When called without any arguments, this returns the current loss:

        >>> mad.objective_function()
        tensor([[-0.5369]], grad_fn=<AddBackward0>)
        >>> mad.losses[-1]
        tensor(-0.5369)

        Can be called with a different image. (Note that, because we called
        :meth:`synthesize` with ``store_progress=True``, we cached the MAD image
        over the course of synthesis):

        >>> mad.objective_function(mad.saved_mad_image[0])
        tensor([[-0.0142]], grad_fn=<AddBackward0>)
        >>> mad.losses[0]
        tensor(-0.0142)

        The objective function computes a weighted sum of three components:

        - :attr:`optimized_metric_loss`, which is either being maximized or minimized,
          depending on the value of :attr:`minmax` set at initialization.

        - :attr:`reference_metric_loss`, which should be held to the same value as that
          between :attr:`image` and :attr:`initial_image`, its initial value.

        - :attr:`penalties`, the output of :attr:`penalty_function`, which will be
          minimized.

        >>> opt_loss = mad.optimized_metric_loss[-1]
        >>> ref_loss = mad.reference_metric_loss[-1]
        >>> init_ref_loss = mad.reference_metric_loss[0]
        >>> penalty = mad.penalties[-1]
        >>> opt_loss, ref_loss, penalty
        (tensor(0.7615), tensor(0.0187), tensor(2.2383))
        >>> # We maximize opt_loss by minizimizing its negative
        >>> opt_comp = {"min": 1, "max": -1}[mad.minmax] * opt_loss
        >>> ref_comp = mad.metric_tradeoff_lambda * (ref_loss - init_ref_loss).pow(2)
        >>> penalty_comp = mad.penalty_lambda * penalty
        >>> opt_comp + ref_comp + penalty_comp
        tensor(-0.5369)
        >>> mad.objective_function()
        tensor([[-0.5369]], grad_fn=<AddBackward0>)
        """
        if self._reference_metric_target is None:
            return torch.empty(0)
        sm, fm, penalty = self._objective_function(mad_image, image)
        synth_target = {"min": 1, "max": -1}[self.minmax]
        fixed_loss = (self._reference_metric_target - fm).pow(2)
        return (
            synth_target * sm
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
        :attr:`pixel_change_norm`, :attr:`gradient_norm`, :attr:`penalties`, and
        :attr:`saved_mad_image` corresponding to ``iteration``. If synthesis was
        run with ``store_progress=False`` (and so we did not cache anything in
        :attr:`saved_mad_image`), then that key will be missing. If synthesis was
        run with ``store_progress>1``, we will grab the corresponding tensor
        from :attr:`saved_mad_image`, with behavior determined by
        ``iteration_selection``.

        The returned dictionary will additionally contain the keys:

        - ``"iteration"``: the (0-indexed positive) synthesis iteration that the
          values for :attr:`losses`, :attr:`pixel_change_norm`, :attr:`penalties`
          and :attr:`gradient_norm` come from.

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

        See Also
        --------
        :func:`~plenoptic.plot.synthesis_status`
            Create a plot summarizing synthesis status at a given iteration.
        :func:`~plenoptic.plot.synthesis_animate`
            Create a video of the MAD image changing over the course of
            synthesis.

        Examples
        --------
        >>> import plenoptic as po
        >>> po.set_seed(0)
        >>> img = po.data.einstein()
        >>> def ds_ssim(x, y):
        ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max")
        >>> mad.synthesize(5)

        Get values from the first iteration:

        >>> mad.get_progress(0)
        {'losses': tensor(-0.0142),
        'iteration': 0,
        'penalties': tensor(5.8395),
        'pixel_change_norm': tensor(2.5577),
        'gradient_norm': tensor(0.4831),
        'reference_metric_loss': tensor(0.0100),
        'optimized_metric_loss': tensor(0.5982)}

        Get values from last iteration of synthesis:

        >>> mad.get_progress(-2)
        {'losses': tensor(-0.4603),
        'iteration': 4,
        'penalties': tensor(2.7274),
        'pixel_change_norm': tensor(2.4336),
        'gradient_norm': tensor(0.3309),
        'reference_metric_loss': tensor(0.0167),
        'optimized_metric_loss': tensor(0.7335)}

        Get current values:

        >>> mad.get_progress(-1)
        {'losses': tensor(-0.5369),
        'iteration': 5,
        'penalties': tensor(2.2383),
        'pixel_change_norm': None,
        'gradient_norm': None,
        'reference_metric_loss': tensor(0.0187),
        'optimized_metric_loss': tensor(0.7615)}

        When synthesis is run with ``store_progress=True``, this function also
        returns the MAD image from the corresponding iteration:

        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max")
        >>> mad.synthesize(5, store_progress=True)
        >>> mad.get_progress(-1)
        {'losses': tensor(-0.5298),
        'iteration': 5,
        'penalties': tensor(2.3197),
        'pixel_change_norm': None,
        'gradient_norm': None,
        'reference_metric_loss': tensor(0.0187),
        'optimized_metric_loss': tensor(0.7626),
        'saved_mad_image': tensor([[[[ 0.0554, ...]]]], grad_fn=<SelectBackward0>),
        'store_progress_iteration': 5}
        >>> torch.equal(
        ...     mad.saved_mad_image[-1], mad.get_progress(-1)["saved_mad_image"]
        ... )
        True

        When synthesis is run with ``store_progress>1``, this function returns the
        metamer from the closest iteration:

        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max")
        >>> mad.synthesize(5, store_progress=2)
        >>> mad.get_progress(-3)
        {'losses': tensor(-0.3482),
        'iteration': 3,
        'penalties': tensor(3.5309),
        'pixel_change_norm': tensor(2.4719),
        'gradient_norm': tensor(0.3763),
        'reference_metric_loss': tensor(0.0147),
        'optimized_metric_loss': tensor(0.7016),
        'saved_mad_image': tensor([[[[ 7.9802e-02, ...]]]], grad_fn=<SelectBackward0>),
        'store_progress_iteration': 4}

        When we cannot grab the saved metamer corresponding to the requested
        iteration, ``iteration_selection`` controls how we determine "closest":

        >>> mad.get_progress(-3, iteration_selection="floor")
        {'losses': tensor(-0.3482),
        'iteration': 3,
        'penalties': tensor(3.5309),
        'pixel_change_norm': tensor(2.4719),
        'gradient_norm': tensor(0.3763),
        'reference_metric_loss': tensor(0.0147),
        'optimized_metric_loss': tensor(0.7016),
        'saved_mad_image': tensor([[[[ 5.9717e-02, ...]]]], grad_fn=<SelectBackward0>),
        'store_progress_iteration': 2}
        """
        return super().get_progress(
            iteration,
            iteration_selection,
            ["reference_metric_loss", "optimized_metric_loss"],
            store_progress_attributes=["saved_mad_image"],
        )

    def _closure(self) -> float:
        r"""
        Calculate the gradient, before the optimization step.

        This enables optimization algorithms that perform several evaluations
        of the gradient before taking a step (e.g., second order methods like
        LBFGS or methods with line searches).

        Additionally, this is where ``loss`` is calculated, ``loss.backward()`` is
        called, and ``self._penalties``, ``self._reference_metric_loss``, and
        ``self._optimized_metric_loss`` are updated (but not ``self._losses``!
        that happens in ``_optimizer_step``).

        Returns
        -------
        loss
            Loss of the current objective function.
        """
        self.optimizer.zero_grad()
        sm, fm, penalty = self._objective_function()
        synth_target = {"min": 1, "max": -1}[self.minmax]
        fixed_loss = (self._reference_metric_target - fm).pow(2)
        loss = (
            synth_target * sm
            + self.metric_tradeoff_lambda * fixed_loss
            + self.penalty_lambda * penalty
        )
        loss.backward(retain_graph=False)
        self._reference_metric_tmp.append(fm.item())
        self._optimized_metric_tmp.append(sm.item())
        self._penalty_tmp.append(penalty.item())
        return loss.item()

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
        # For some reason, the loss actually returned by optimizer.step above for
        # optimizers that call closure multiple time (like LBFGS) is the one that
        # corresponds to the *first* call, not the last. Therefore, to make penalty
        # match it, we keep track of the penalty on each call to closure and then ...
        self._penalty_tmp = []
        self._reference_metric_tmp = []
        self._optimized_metric_tmp = []
        loss = self.optimizer.step(self._closure)
        self._losses.append(loss)
        # ... grab the first one. This also allows the stored penalty to line up with
        # penalty_function(saved_mad_image). (and same for metrics)
        self._penalties.append(self._penalty_tmp[0])
        self._reference_metric_loss.append(self._reference_metric_tmp[0])
        self._optimized_metric_loss.append(self._optimized_metric_tmp[0])

        grad_norm = torch.linalg.vector_norm(
            self.mad_image.grad.data, ord=2, dim=None
        ).item()
        self._gradient_norm.append(grad_norm)

        # optionally step the scheduler, passing loss if needed
        if self.scheduler is not None:
            if self._scheduler_step_arg:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

        pixel_change_norm = torch.linalg.vector_norm(
            self.mad_image - last_iter_mad_image, ord=2, dim=None
        ).item()
        self._pixel_change_norm.append(pixel_change_norm)

        # add extra info here if you want it to show up in progress bar
        pbar.set_postfix(
            OrderedDict(
                loss=f"{loss:.04e}",
                learning_rate=self.optimizer.param_groups[0]["lr"],
                penalty=f"{self._penalties[-1]:.04e}",
                gradient_norm=f"{grad_norm:.04e}",
                pixel_change_norm=f"{pixel_change_norm:.04e}",
                reference_metric=f"{self._reference_metric_loss[-1]:.04e}",
                optimized_metric=f"{self._optimized_metric_loss[-1]:.04e}",
            )
        )
        return loss

    def _check_convergence(
        self, stop_criterion: float, stop_iters_to_check: int
    ) -> bool:
        r"""
        Check whether the loss has stabilized and, if so, return True.

        Uses :func:`~plenoptic.convergence._loss_convergence`.

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

        See Also
        --------
        load
            Method to load in saved ``MADCompetition`` objects.

        Examples
        --------
        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> def ds_ssim(x, y):
        ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max")
        >>> mad.synthesize(5, store_progress=True)
        >>> mad.save("mad.pt")
        """
        save_io_attrs = [
            ("_optimized_metric", ("_image", "_mad_image")),
            ("_reference_metric", ("_image", "_mad_image")),
            ("_penalty_function", ("_image",)),
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

        Examples
        --------
        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> def ds_ssim(x, y):
        ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max")
        >>> mad.image.dtype
        torch.float32
        >>> mad.optimized_metric(mad.image, torch.rand_like(mad.image)).dtype
        torch.float32
        >>> mad.to(torch.float64)
        >>> mad.image.dtype
        torch.float64
        >>> mad.optimized_metric(mad.image, torch.rand_like(mad.image)).dtype
        torch.float64
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
        raise_on_checks: bool = True,
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

        .. versionchanged:: 2.0.0
           Adds ``raise_on_checks`` argument.

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
        raise_on_checks
            During load, we perform several checks to ensure that the saved object was
            initialized in the same way as the loading object. This is to ensure that
            the model, image, etc. are all the same and avoid unpleasant surprises. If
            ``True``, we raise a ``ValueError`` if any of these checks fail. If
            ``False``, we instead raise a ``LoadWarning``. The intended use here is if
            you're loading something that was saved with an older version of plenoptic
            and you're sure that you're doing everything correctly. Note that different
            devices or dtypes will always result in a ``ValueError``. See
            :ref:`raise-on-checks` on the "Reproducibility and Compatibility" page of
            the documentation for more info. Additionally, note that, if the
            ``MADCompetition`` object itself has changed, we cannot ensure that methods
            are the same -- proceed at your own risk.
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
            :attr:`metric_tradeoff_lambda`, or :attr:`minmax`.
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
        :func:`~plenoptic.io.examine_saved_synthesis`
            Examine metadata from saved object: pytorch and plenoptic versions, name of
            the synthesis object, shapes of tensors, etc.

        Examples
        --------
        In order to load a saved ``MADCompetition`` object, we must first initialize
        one using the same arguments. (We use float64 / "double" precision rather than
        torch's default float32 because it increases reproducibility, see the
        :ref:`Reproducibility <reproduce>` page of our documentations for more details.)
        Here, we load in a cached example:

        >>> import plenoptic as po
        >>> img = po.data.einstein().to(torch.float64)
        >>> def ds_ssim(x, y):
        ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
        >>> print(mad.mad_image)
        tensor([])
        >>> mad.load(po.data.fetch_data("example_mad.pt"))
        >>> print(mad.mad_image)
        tensor([[[[0.0230, ...]]]], dtype=torch.float64, requires_grad=True)

        If the saved ``MADCompetition`` object lived on a CUDA device and you do not
        have CUDA on the loading machine, use ``map_location`` to change device:

        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
        >>> mad.image.device
        device(type='cpu')
        >>> mad.load(po.data.fetch_data("example_mad-cuda.pt"))
        Traceback (most recent call last):
        RuntimeError: Attempting to deserialize object on a CUDA device but
        torch.cuda.is_available() is False...
        >>> mad.load(
        ...     po.data.fetch_data("example_mad-cuda.pt"),
        ...     map_location="cpu",
        ... )
        >>> print(mad.mad_image)
        tensor([[[[0.0230, ...]]]], dtype=torch.float64, requires_grad=True)

        If the loading ``MADCompetition`` object was not initialized with same values
        as the saved object, an error will be raised:

        >>> rand_img = torch.rand_like(img)
        >>> mad = po.MADCompetition(rand_img, ds_ssim, po.metric.mse, "max", 1e6)
        >>> mad.load(po.data.fetch_data("example_mad.pt"))
        Traceback (most recent call last):
        ValueError: Saved and initialized attribute image have different values...

        If the loading ``MADCompetition`` object has a different data type than the
        saved object, an error will be raised:

        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
        >>> mad.to(torch.float32)
        >>> mad.load(po.data.fetch_data("example_mad.pt"))
        Traceback (most recent call last):
        ValueError: Saved and initialized attribute image have different dtype...
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
            ("_penalty_function", ("_image",)),
        ]
        super().load(
            file_path,
            "_mad_image",
            map_location=map_location,
            check_attributes=check_attributes,
            check_io_attributes=check_io_attrs,
            state_dict_attributes=["_optimizer", "_scheduler"],
            raise_on_checks=raise_on_checks,
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

    def __repr__(self) -> str:
        # numpydoc ignore=GL08
        return super()._repr_format(
            [
                "image",
                "optimized_metric",
                "reference_metric",
                "minmax",
                "metric_tradeoff_lambda",
                "penalty_function",
                "penalty_lambda",
            ]
        )

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
        current_ref = self._current_ref_metric
        # this will happen if we haven't run synthesize() yet or got
        # interrupted
        if current_ref is None:
            if self.mad_image.numel() == 0:
                # this will happen if setup() has not been called and so we can't
                # compute penalty because synthesis hasn't been initialized.
                return torch.empty(0)
            else:
                # compute current penalty, no need to compute gradient
                with torch.no_grad():
                    current_ref = self.reference_metric(self.image, self.mad_image)
                    current_ref = current_ref.item()
        return torch.as_tensor(
            [*self._reference_metric_loss, current_ref],
            dtype=self.image.dtype,
        )

    @property
    def optimized_metric_loss(self) -> Tensor:
        """
        :attr:`optimized_metric` loss over iterations.

        That is, the value of ``optimized_metric(image, mad_image)``. Ideally, this is
        very different from ``optimized_metric(image, initial_image)``.

        This tensor always lives on the CPU, regardless of the device of the
        ``MADCompetition`` object.
        """
        # numpydoc ignore=RT01
        current_opt = self._current_opt_metric
        # this will happen if we haven't run synthesize() yet or got
        # interrupted
        if current_opt is None:
            if self.mad_image.numel() == 0:
                # this will happen if setup() has not been called and so we can't
                # compute penalty because synthesis hasn't been initialized.
                return torch.empty(0)
            else:
                # compute current penalty, no need to compute gradient
                with torch.no_grad():
                    current_opt = self.optimized_metric(self.image, self.mad_image)
                    current_opt = current_opt.item()
        return torch.as_tensor(
            [*self._optimized_metric_loss, current_opt],
            dtype=self.image.dtype,
        )

    @property
    def metric_tradeoff_lambda(self) -> float:
        """Tradeoff between the two metrics in synthesis loss."""
        # numpydoc ignore=RT01,ES01
        return self._metric_tradeoff_lambda

    @property
    def minmax(self) -> str:
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

        Examples
        --------
        If synthesize is called without ``store_progress``, then this attribute
        just contains the MAD image, though the number of dimensions is different:

        >>> import plenoptic as po
        >>> po.set_seed(0)
        >>> img = po.data.einstein()
        >>> def ds_ssim(x, y):
        ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
        >>> mad.saved_mad_image
        tensor([])
        >>> mad.synthesize(5)
        >>> mad.saved_mad_image
        tensor([[[[[ 0.1316, ...]]]]], grad_fn=<StackBackward0>)
        >>> mad.mad_image
        tensor([[[[ 0.1316, ...]]]], requires_grad=True)
        >>> mad.saved_mad_image.shape
        torch.Size([1, 1, 1, 256, 256])
        >>> mad.mad_image.shape
        torch.Size([1, 1, 256, 256])

        If synthesize is called with ``store_progress=1``, then this attribute
        contains the metamer at each iteration, and ``losses[i]`` contains the error
        for ``saved_mad_image[i]``.

        >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
        >>> mad.synthesize(5, store_progress=True)
        >>> mad.saved_mad_image.shape
        torch.Size([6, 1, 1, 256, 256])
        >>> mad.objective_function(mad.saved_mad_image[2])
        tensor([[-0.1495]], grad_fn=<AddBackward0>)
        >>> mad.losses[2]
        tensor(-0.1495)

        (In the above example, ``saved_mad_image`` has 6 elements because it includes
        the MAD image at the start of each of the 5 synthesis iterations, plus the
        current one.)
        """  # numpydoc ignore=RT01
        if self._mad_image is None:
            return torch.empty(0)
        else:
            # for memory purposes, always on CPU
            return torch.stack([*self._saved_mad_image, self.mad_image.to("cpu")])

    @property
    def penalties(self) -> torch.Tensor:
        """
        Penalty function output over iterations.

        Will have ``length=num_iter+1``, where ``num_iter`` is the number of
        iterations of synthesis run so far.

        This tensor always lives on the CPU.
        """  # numpydoc ignore=RT01
        return super().penalties(self.mad_image)
