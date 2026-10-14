"""
Model metamers.

Model metamers are images whose pixel values differ but whose model outputs are
identical. They allow researchers to better understand the information which have no
effect on a model's output, also known as their invariances.
"""  # numpydoc ignore=EX01

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
from ..validate import (
    validate_input,
    validate_penalty,
)
from .synthesis import _OptimizedSynthesis

__all__ = [
    "GenericSynthesis",
]


def __dir__() -> list[str]:
    return __all__


class GenericSynthesis(_OptimizedSynthesis):
    r"""
    Synthesize metamers for image-computable differentiable models.

    Following the basic idea in [1]_, this class creates a metamer for a given model on
    a given image. We iteratively adjust the pixel values so as to match the
    representation of the :attr:`metamer` and :attr:`image`.

    Parameters
    ----------
    loss_function
        The callable whose output we will minimize.
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
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model
       based on Joint Statistics of Complex Wavelet Coefficients. Int'l
       Journal of Computer Vision. 40(1):49-71, October, 2000.
       https://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
       https://www.cns.nyu.edu/~lcv/texture/

    Examples
    --------
    Synthesize and visualize a metamer for a simple model.

    .. attention::
       For all of these examples, synthesis has not been run to completion. These
       examples just demonstrate basic usage.

    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import matplotlib.pyplot as plt
      >>> img = po.data.einstein()
      >>> model = po.models.Gaussian(30).eval()
      >>> po.remove_grad(model)
      >>> met = po.Metamer(img, model)
      >>> met.synthesize(110)
      >>> # Define a helper function for this, since we'll use it in the other examples
      >>> def example_figure(met):
      ...     fig, axes = plt.subplots(1, 4, figsize=(22, 5))
      ...     po.plot.imshow(img, ax=axes[0], title="Target image")
      ...     axes[0].xaxis.set_visible(False)
      ...     axes[0].yaxis.set_visible(False)
      ...     po.plot.synthesis_status(
      ...         met,
      ...         fig=fig,
      ...         axes_idx={"misc": 0},
      ...     )
      ...     return fig
      >>> example_figure(met)
      <Figure size ...>

    Change the loss function. A loss function must accept two tensors and return a
    scalar. Note that metamer and synthesis efficiency change when using a different
    loss function.

    .. plot::
      :context: close-figs

      >>> po.loss.l2_norm(img, 2 * img)
      tensor(123.5844)
      >>> met = po.Metamer(img, model, loss_function=po.loss.l2_norm)
      >>> met.synthesize(50)
      >>> example_figure(met)
      <Figure size ...>

    Change the penalty function. A penalty function must accept one tensor and return a
    scalar. See :ref:`how-to-penalty` for more information about penalty functions. Note
    the different ranges in the title of the subplots showing the metamers in the
    following two examples.

    .. plot::
      :context: close-figs

      >>> # Penalty encourages metamer pixel values to lie between 0.3 and 0.7
      >>> def penalty(x):
      ...     return po.regularize.penalize_range(x, [0.3, 0.7])
      >>> penalty(img)
      tensor(213.6589)
      >>> met = po.Metamer(
      ...     img, model, loss_function=po.loss.l2_norm, penalty_function=penalty
      ... )
      >>> met.synthesize(50)
      >>> example_figure(met)
      <Figure size ...>

    Change the contribution of the penalty to the objective function.
    :attr:`penalty_lambda` must be a non-negative scalar.

    .. plot::
      :context: close-figs

      >>> met = po.Metamer(
      ...     img,
      ...     model,
      ...     loss_function=po.loss.l2_norm,
      ...     penalty_function=penalty,
      ...     penalty_lambda=0.2,
      ... )
      >>> met.synthesize(50)
      >>> example_figure(met)
      <Figure size ...>
    """

    def __init__(
        self,
        loss_function: Callable[[Tensor], Tensor],
        penalty_function: Callable[[Tensor], Tensor] = regularize.penalize_range,
        penalty_lambda: float = 0.1,
    ):
        super().__init__(
            penalty_function=penalty_function, penalty_lambda=penalty_lambda
        )
        self._loss_function = loss_function
        self._scheduler = None
        self._scheduler_step_arg = False
        self._saved_metamer = []
        self._store_progress = None
        self._synthesized_image = None
        # WHAT DO ABOUT CACHED IMAGE?
        self._image = None

    def setup(
        self,
        initial_image: Tensor | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        optimizer_kwargs: dict | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scheduler_kwargs: dict | None = None,
    ):
        """
        Initialize the metamer, optimizer, and scheduler.

        Can only be called once. If ``load()`` has been called, ``initial_image`` must
        be ``None``.

        Parameters
        ----------
        initial_image
            The tensor we use to initialize the metamer. If ``None``, we initialize with
            random noise uniformly-distributed in [0,1].
        optimizer
            The un-initialized optimizer object to use. If ``None``, we use
            :class:`torch.optim.Adam`.
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
            If you try to set ``initial_image`` after calling :func:`load`.
        ValueError
            If ``setup`` is called more than once or after :func:`synthesize`.
        ValueError
            If you try to set ``optimizer_kwargs`` after calling :func:`load`.
        TypeError
            If the loaded object had a non-Adam optimizer, but the ``optimizer`` arg
            is not specified.
        ValueError
            If the loaded object had an optimizer, and the ``optimizer`` arg is
            a different type.
        ValueError
            If you try to set ``scheduler_kwargs`` after calling :func:`load`.
        TypeError
            If the loaded object had a scheduler, but the ``scheduler`` arg is not
            specified.
        ValueError
            If the loaded object had a scheduler, but the ``scheduler`` arg is
            a different type.

        Warns
        -----
        UserWarning
            If ``initial_image`` is a different shape than ``self.image``.

        Examples
        --------
        Set initial image:

        >>> import plenoptic as po
        >>> import torch
        >>> img = po.data.einstein()
        >>> model = po.models.Gaussian(30).eval()
        >>> po.remove_grad(model)
        >>> met = po.Metamer(img, model)
        >>> met.setup(po.data.curie())

        Set optimizer:

        >>> met = po.Metamer(img, model)
        >>> met.setup(optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.01})

        Set optimizer and scheduler:

        >>> met = po.Metamer(img, model)
        >>> met.setup(
        ...     optimizer=torch.optim.SGD,
        ...     optimizer_kwargs={"lr": 0.01},
        ...     scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        ... )

        Use with save/load. We only pass the optimizer/scheduler objects when calling
        setup after load, their kwargs and the initial image are handled during the
        load.

        >>> met = po.Metamer(img, model)
        >>> met.setup(
        ...     po.data.curie(),
        ...     optimizer=torch.optim.SGD,
        ...     optimizer_kwargs={"lr": 0.01},
        ...     scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        ... )
        >>> met.synthesize(5)
        >>> met.save("metamer_setup.pt")
        >>> met = po.Metamer(img, model)
        >>> met.load("metamer_setup.pt")
        >>> met.setup(
        ...     optimizer=torch.optim.SGD,
        ...     scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        ... )
        """
        if self._synthesized_image is None:
            validate_input(initial_image)
            synth_image = initial_image.clone().detach()
            synth_image.requires_grad_()
            self._synthesized_image = synth_image
        else:
            if self._loaded:
                if initial_image is not None:
                    raise ValueError("Cannot set initial_image after calling load()!")
            else:
                raise ValueError(
                    "setup() can only be called once and must be called"
                    " before synthesize()!"
                )

        self._image = synth_image.clone().detach()
        validate_input(synth_image)
        validate_penalty(
            self._penalty_function,
            synth_image.shape,
            synth_image.dtype,
            synth_image.device,
        )
        # initialize the optimizer
        self._initialize_optimizer(optimizer, self.synthesized_image, optimizer_kwargs)
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
        Synthesize a metamer.

        Update the pixels of :attr:`metamer` until its representation matches that of
        :attr:`image`.

        We run this until either we reach ``max_iter`` or the loss changes less than
        ``stop_criterion`` over the past ``stop_iters_to_check`` iterations,
        whichever comes first.

        Parameters
        ----------
        max_iter
            The maximum number of iterations to run before we end synthesis
            (unless we hit the stop criterion).
        store_progress
            Whether we should store the metamer image in progress during
            synthesis. If ``False``, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress`` iterations
            (note then that 0 is the same as False and 1 the same as True). This is
            primarily useful for using
            :func:`~plenoptic.plot.synthesis_animate` to create a video of the
            course of synthesis.
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
        RuntimeError
            If :func:`setup` hasn't been called.

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
        >>> model = po.models.Gaussian(30).eval()
        >>> po.remove_grad(model)
        >>> met = po.Metamer(img, model)
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> met.synthesize(5)
        >>> met.losses
        tensor([0.0194, 0.0198, 0.0179, 0.0160, 0.0145, 0.0132])

        Set ``store_progress`` in order to examine the metamer-in-progress over the
        course of synthesis. (This also enables us to create a video of the metamer
        changing over the course of synthesis, see
        :func:`~plenoptic.plot.synthesis_animate`.)

        >>> met = po.Metamer(img, model)
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> met.synthesize(5, store_progress=2)
        >>> met.saved_metamer.shape
        torch.Size([4, 1, 1, 256, 256])
        >>> # see loss, etc on the 4th iteration
        >>> progress = met.get_progress(4)
        >>> progress.keys()
        dict_keys(['losses', ..., 'saved_metamer', 'store_progress_iteration'])
        >>> progress["losses"]
        tensor(0.0139)

        Adjust ``stop_criterion`` and ``stop_iters_to_check`` to change how convergence
        is determined. In this case, we stop early by making ``stop_criterion`` fairly
        large. In practice, you're more likely to make ``stop_criterion`` smaller to let
        synthesis run for longer.

        >>> met = po.Metamer(img, model)
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> met.synthesize(12, stop_criterion=0.001, stop_iters_to_check=2)
        >>> len(met.losses)
        9
        """
        # if setup hasn't been called manually, call it now.
        if self._synthesized_image is None or isinstance(self._scheduler, tuple):
            raise RuntimeError("Must call setup before synthesize!")
        self._current_loss = None
        self._current_penalty = None

        # get ready to store progress
        self.store_progress = store_progress

        pbar = tqdm(range(max_iter))

        for i in pbar:
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
            self._current_penalty = self.penalty_function(self.synthesized_image).item()

        pbar.close()

    def _objective_function(
        self,
        synthesized_image: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute objective function components.

        This calls :attr:`loss_function` and :attr:`penalty_function` and returns their
        output, without combining them. It is not meant to be called directly, but is
        used by both :func:`_closure` and (public) :func:`objective_function`.

        Parameters
        ----------
        synthesized_image
            Current ``synthesized_image``. If ``None``, we use
            ``self.synthesized_image``.

        Returns
        -------
        loss
            1-element tensor containing the metamer loss on this step
            (without penalty).
        penalty
            1-element tensor containing the penalty on this step.
        """
        if synthesized_image is None:
            synthesized_image = self.synthesized_image
            # if this is empty, then self.synthesized_image hasn't been initialized
            if synthesized_image.numel() == 0:
                return torch.empty(0), torch.empty(0)
        loss = self.loss_function(synthesized_image)
        penalty = self.penalty_function(synthesized_image)
        return loss, penalty

    def objective_function(
        self,
        synthesized_image: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the metamer synthesis loss.

        This calls :attr:`loss_function` on ``self.model(metamer, **analyze_kwargs)``
        and ``target_representation`` and then adds :attr:`penalty_lambda` times
        :attr:`penalty_function` on ``metamer``.

        Its output over time is stored in :attr:`losses`.

        Parameters
        ----------
        synthesized_image
            Current ``synthesized_image``. If ``None``, we use
            ``self.synthesized_image``.

        Returns
        -------
        loss
            1-element tensor containing the loss on this step.

        Examples
        --------
        >>> import plenoptic as po
        >>> po.set_seed(0)
        >>> img = po.data.einstein()
        >>> model = po.models.Gaussian(30).eval()
        >>> po.remove_grad(model)
        >>> met = po.Metamer(img, model)

        Before :meth:`setup` or :meth:`synthesize` is called, this returns an
        empty tensor because the metamer attribute hasn't been initialized:

        >>> met.objective_function()
        tensor([])
        >>> met.synthesize(5, store_progress=True)

        When called without any arguments, this returns the current loss:

        >>> met.objective_function()
        tensor(0.0132, grad_fn=<AddBackward0>)
        >>> met.losses[-1]
        tensor(0.0132)

        Can be called with a different image. In the following example, we compute the
        objective function on the initial metamer, which thus matches the first stored
        loss value (note that, because we called :meth:`synthesize` with
        ``store_progress=True``, we cached the metamer over the course of synthesis).

        >>> met.objective_function(met.saved_metamer[0])
        tensor(0.0194, grad_fn=<AddBackward0>)
        >>> met.losses[0]
        tensor(0.0194)

        This method differs from the :attr:`loss_function` attribute because of its
        inclusion of the penalty. In the following block, the pixels of
        ``rand_img`` all lie within [0, 1], and so the outputs of
        :attr:`objective_function` and :attr:`loss_function` are the same:

        >>> rand_img = torch.rand_like(img)
        >>> rand_img.min(), rand_img.max()
        (tensor(7.9870e-06), tensor(1.0000))
        >>> met.objective_function(rand_img)
        tensor(0.0190)
        >>> met.loss_function(model(img), model(rand_img))
        tensor(0.0190)

        In this block, the image's lie outside [0, 1], and so the outputs of
        :attr:`objective_function` and :attr:`loss_function` are different:

        >>> rand_img *= 2
        >>> rand_img.min(), rand_img.max()
        (tensor(0.0001), tensor(2.0000))
        >>> met.objective_function(rand_img)
        tensor(1100.9663)
        >>> loss = met.loss_function(model(img), model(rand_img))
        >>> loss
        tensor(0.3133)

        To compute the output of the objective function, we take the output of
        :attr:`loss_function` and add the output of :attr:`penalty_function` times
        :attr:`penalty_lambda`:

        >>> penalty = met.penalty_function(rand_img)
        >>> penalty
        tensor(11006.5293)
        >>> loss + met.penalty_lambda * penalty
        tensor(1100.9663)
        """
        loss, penalty = self._objective_function(synthesized_image)
        return loss + self.penalty_lambda * penalty

    def get_progress(
        self,
        iteration: int | None,
        iteration_selection: Literal["floor", "ceiling", "round"] = "round",
    ) -> dict:
        r"""
        Return dictionary summarizing synthesis progress at ``iteration``.

        This returns a dictionary containing info from :attr:`losses`,
        :attr:`pixel_change_norm`, :attr:`gradient_norm`, :attr:`penalties`, and
        :attr:`saved_metamer` corresponding to ``iteration``. If synthesis was
        run with ``store_progress=False`` (and so we did not cache anything in
        :attr:`saved_metamer`), then that key will be missing. If synthesis was
        run with ``store_progress>1``, we will grab the corresponding tensor
        from :attr:`saved_metamer`, with behavior determined by
        ``iteration_selection``.

        The returned dictionary will additionally contain the keys:

        - ``"iteration"``: the (0-indexed positive) synthesis iteration that the
          values for :attr:`losses`, :attr:`pixel_change_norm`, :attr:`penalties`,
          and :attr:`gradient_norm` come from.

        - If ``self.store_progress``, ``"store_progress_iteration"``: the (0-indexed
          positive) synthesis iteration that the value for :attr:`saved_metamer` comes
          from.

        Note that for the most recent iteration (``iteration=-1`` or ``iteration=None``
        or ``iteration==len(self.losses)-1``), we do not have values for
        :attr:`pixel_change_norm` or :attr:`gradient_norm`, since in this case we are
        showing the loss and value for the current metamer.

        Parameters
        ----------
        iteration
            Synthesis iteration to summarize. If ``None``, grab the most recent.
            Negative values are allowed.
        iteration_selection

            How to select the relevant iteration from :attr:`saved_metamer`
            when the request iteration wasn't stored.

            When synthesis was run with ``store_progress=n`` (where ``n>1``),
            metamers are only saved every ``n`` iterations. If you request an
            iteration where a metamer wasn't saved, this determines which available
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
            If the iteration used for ``saved_metamer`` is not the same as the argument
            ``iteration`` (because e.g., you set ``iteration=3`` but
            ``self.store_progress=2``).

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
        >>> import torch
        >>> po.set_seed(0)
        >>> img = po.data.einstein()
        >>> model = po.models.Gaussian(30).eval()
        >>> po.remove_grad(model)
        >>> met = po.Metamer(img, model)
        >>> met.synthesize(5)

        Get values from the first iteration:

        >>> met.get_progress(0)
        {'losses': tensor(0.0194),
        'iteration': 0,
        'penalties': tensor(0.),
        'pixel_change_norm': tensor(2.5326),
        'gradient_norm': tensor(0.0010)}

        Get current values:

        >>> met.get_progress(-1)
        {'losses': tensor(0.0132),
        'iteration': 5,
        'penalties': tensor(0.0174),
        'pixel_change_norm': None,
        'gradient_norm': None}

        Get values from last iteration of synthesis:

        >>> met.get_progress(-2)
        {'losses': tensor(0.0145),
        'iteration': 4,
        'penalties': tensor(0.0180),
        'pixel_change_norm': tensor(2.2698),
        'gradient_norm': tensor(0.0268)}

        When synthesis is run with ``store_progress=True``, this function also
        returns the metamer from the corresponding iteration:

        >>> met = po.Metamer(img, model)
        >>> met.synthesize(5, store_progress=True)
        >>> met.get_progress(-1)
        {'losses': tensor(0.0124),
        'iteration': 5,
        'penalties': tensor(0.0168),
        'pixel_change_norm': None,
        'gradient_norm': None,
        'saved_metamer': tensor([[[[0.4554, ...]]]], grad_fn=<SelectBackward0>),
        'store_progress_iteration': 5}
        >>> torch.equal(met.saved_metamer[-1], met.get_progress(-1)["saved_metamer"])
        True

        When synthesis is run with ``store_progress>1``, this function returns the
        metamer from the closest iteration. This might not be the specified iteration,
        since we only cached the metamer every ``store_progress`` iterations (2, in
        the following example):

        >>> met = po.Metamer(img, model)
        >>> met.synthesize(5, store_progress=2)
        >>> met.get_progress(-3)
        {'losses': tensor(0.0152),
        'iteration': 3,
        'penalties': tensor(0.0182),
        'pixel_change_norm': tensor(2.3592),
        'gradient_norm': tensor(0.0269),
        'saved_metamer': tensor([[[[0.8532, ...]]]], grad_fn=<SelectBackward0>),
        'store_progress_iteration': 4}

        When we cannot grab the saved metamer corresponding to the requested
        iteration, ``iteration_selection`` controls how we determine "closest":

        >>> met.get_progress(-3, iteration_selection="floor")
        {'losses': tensor(0.0152),
        'iteration': 3,
        'penalties': tensor(0.0182),
        'pixel_change_norm': tensor(2.3592),
        'gradient_norm': tensor(0.0269),
        'saved_metamer': tensor([[[[ 0.8730, ...]]]], grad_fn=<SelectBackward0>),
        'store_progress_iteration': 2}
        """
        return super().get_progress(
            iteration,
            iteration_selection,
            store_progress_attributes=["saved_synthesized_image"],
        )

    def _closure(self) -> float:
        r"""
        Calculate the gradient, before the optimization step.

        This enables optimization algorithms that perform several evaluations
        of the gradient before taking a step (ie. second order methods like
        LBFGS).

        Additionally, this is where ``loss`` is calculated, ``loss.backward()`` is
        called, and ``self._penalties`` is updated (but not ``self._losses``!
        that happens in ``_optimizer_step``).

        Returns
        -------
        loss
            Loss of the current objective function.
        """
        self.optimizer.zero_grad()
        loss, penalty = self._objective_function()
        loss = loss + self.penalty_lambda * penalty
        loss.backward(retain_graph=False)
        self._penalty_tmp.append(penalty.item())
        return loss.item()

    def _optimizer_step(self, pbar: tqdm) -> Tensor:
        r"""
        Compute and propagate gradients, then step the optimizer to update metamer.

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
        """  # numpydoc ignore=ES01,EX01
        last_iter_synth = self.synthesized_image.clone()
        # For some reason, the loss actually returned by optimizer.step above for
        # optimizers that call closure multiple time (like LBFGS) is the one that
        # corresponds to the *first* call, not the last. Therefore, to make penalty
        # match it, we keep track of the penalty on each call to closure and then ...
        self._penalty_tmp = []
        loss = self.optimizer.step(self._closure)
        self._losses.append(loss)
        # ... grab the first one. This also allows the stored penalty to line up with
        # penalty_function(saved_synthesized_image)
        self._penalties.append(self._penalty_tmp[0])

        grad_norm = torch.linalg.vector_norm(
            self.synthesized_image.grad.data, ord=2, dim=None
        ).item()
        self._gradient_norm.append(grad_norm)

        # optionally step the scheduler, passing loss if needed
        if self.scheduler is not None:
            if self._scheduler_step_arg:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

        pixel_change_norm = torch.linalg.vector_norm(
            self.synthesized_image - last_iter_synth, ord=2, dim=None
        ).item()
        self._pixel_change_norm.append(pixel_change_norm)
        # add extra info here if you want it to show up in progress bar
        pbar.set_postfix(
            OrderedDict(
                loss=f"{self._losses[-1]:.04e}",
                learning_rate=self.optimizer.param_groups[0]["lr"],
                penalty=f"{self._penalties[-1]:.04e}",
                gradient_norm=f"{grad_norm:.04e}",
                pixel_change_norm=f"{pixel_change_norm:.04e}",
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
        """  # numpydoc ignore=EX01
        return _loss_convergence(self, stop_criterion, stop_iters_to_check)

    def _store(self, i: int) -> bool:
        """
        Store metamer, if appropriate.

        If it's the right iteration, we update :attr:`saved_metamer`.

        Parameters
        ----------
        i
            The current iteration.

        Returns
        -------
        stored
            True if we stored this iteration, False if not.
        """  # numpydoc ignore=EX01
        if self.store_progress and (i % self.store_progress == 0):
            # want these to always be on cpu, to reduce memory use for GPUs
            self._saved_synthesized_image.append(
                self.synthesized_image.clone().to("cpu")
            )
            stored = True
        else:
            stored = False
        return stored

    def save(self, file_path: str):
        r"""
        Save all relevant variables in .pt file.

        Note that if ``store_progress`` is True, this will probably be very
        large.

        Parameters
        ----------
        file_path :
            The path to save the metamer object to.

        See Also
        --------
        load
            Method to load in saved ``Metamer`` objects.

        Examples
        --------
        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> model = po.models.Gaussian(30).eval()
        >>> po.remove_grad(model)
        >>> met = po.Metamer(img, model)
        >>> met.synthesize(max_iter=5, store_progress=True)
        >>> met.save("metamers.pt")
        """
        raise NotImplementedError(
            "Unsure how to handle this! don't have cached tensors"
        )
        save_io_attrs = [
            (
                "_loss_function",
                ("_target_representation", "2 * _target_representation"),
            ),
            ("_penalty_function", ("_image",)),
        ]
        save_state_dict_attrs = ["_optimizer", "_scheduler"]
        super().save(file_path, save_io_attrs, save_state_dict_attrs)

    def to(self, *args: Any, **kwargs: Any):
        r"""
        Move and/or cast the parameters and buffers.

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
        >>> import torch
        >>> img = po.data.einstein()
        >>> model = po.models.Gaussian(30).eval()
        >>> po.remove_grad(model)
        >>> met = po.Metamer(img, model)
        >>> met.image.dtype
        torch.float32
        >>> met.model(met.image).dtype
        torch.float32
        >>> met.to(torch.float64)
        >>> met.image.dtype
        torch.float64
        >>> met.model(met.image).dtype
        torch.float64
        """  # numpydoc ignore=PR01,PR02
        attrs = ["_synthesized_image", "_saved_synthesized_image"]
        super().to(*args, attrs=attrs, **kwargs)
        # try to call .to() on model. this should work, but it might fail if e.g., this
        # a custom model that doesn't inherit torch.nn.Module
        try:
            self._loss_function = self._loss_function.to(*args, **kwargs)
        except AttributeError:
            warnings.warn("Unable to call loss_function.to(), so we leave it as is.")

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

        This should be called by an initialized ``Metamer`` object -- we will
        ensure that ``image``, ``target_representation`` (and thus
        ``model``), and ``loss_function`` are all identical.

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
            Argument to pass to :func:`torch.load` as ``map_location``. If you
            save stuff that was being run on a GPU and are loading onto a
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
            the documentation for more info. Additionally, note that, if the ``Metamer``
            object itself has changed, we cannot ensure that methods are the same --
            proceed at your own risk.
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
            If the object saved at ``file_path`` is not a ``Metamer`` object.
        ValueError
            If the saved and loading ``Metamer`` objects have a different value
            for any of :attr:`image` or :attr:`penalty_lambda`,
        ValueError
            If the behavior of :attr:`loss_function` or :attr:`model` is different
            between the saved and loading objects.

        Warns
        -----
        UserWarning
            If :func:`setup` will need to be called after load, to finish initializing
            :attr:`optimizer` or :attr:`scheduler`.

        See Also
        --------
        :func:`~plenoptic.io.examine_saved_synthesis`
            Examine metadata from saved object: pytorch and plenoptic versions, name of
            the synthesis object, shapes of tensors, etc.

        Examples
        --------
        In order to load a saved ``Metamer`` object, we must first initialize
        one using the same arguments. (We use float64 / "double" precision rather than
        torch's default float32 because it increases reproducibility, see the
        :ref:`Reproducibility <reproduce>` page of our documentations for more details.)
        Here, we load in a cached example:

        >>> import plenoptic as po
        >>> import torch
        >>> img = po.data.einstein().to(torch.float64)
        >>> model = po.models.Gaussian(30).eval().to(torch.float64)
        >>> po.remove_grad(model)
        >>> met = po.Metamer(img, model)
        >>> met.metamer
        tensor([])
        >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
        >>> met.metamer
        tensor([[[[0.0692, ...]]]], dtype=torch.float64, requires_grad=True)

        If the saved ``Metamer`` object lived on a CUDA device and you do not have
        CUDA on the loading machine, use ``map_location`` to change device:

        >>> met = po.Metamer(img, model)
        >>> met.image.device
        device(type='cpu')
        >>> met.load(po.data.fetch_data("example_metamer_gaussian-cuda.pt"))
        Traceback (most recent call last):
        RuntimeError: Attempting to deserialize object on a CUDA device but
        torch.cuda.is_available() is False...
        >>> met.load(
        ...     po.data.fetch_data("example_metamer_gaussian-cuda.pt"),
        ...     map_location="cpu",
        ... )
        >>> met.metamer
        tensor([[[[0.0692, ...]]]], dtype=torch.float64, requires_grad=True)

        If the loading ``Metamer`` object was not initialized with same values
        as the saved object, an error will be raised:

        >>> met = po.Metamer(torch.rand_like(img), model)
        >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
        Traceback (most recent call last):
        ValueError: Saved and initialized attribute image have different values...

        If the loading ``Metamer`` object has a different data type than the saved
        object, an error will be raised:

        >>> met = po.Metamer(img, model)
        >>> met.to(torch.float32)
        >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
        Traceback (most recent call last):
        ValueError: Saved and initialized attribute image have different dtype...

        If the name of the model, loss function, or penalty function has changed (even
        if their behavior is identical), an error will be raised:

        >>> class WrongModel(torch.nn.Module):
        ...     def forward(self, x):
        ...         return model(x)
        >>> model.to(torch.float64)
        Gaussian()
        >>> wrong_model = WrongModel().eval()
        >>> po.remove_grad(wrong_model)
        >>> met = po.Metamer(img, wrong_model)
        >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
        Traceback (most recent call last):
        ValueError: Saved and initialized model have different names...

        If you wish to proceed anyway, you can set ``raise_on_checks=False`` to turn the
        errors into warnings. Do so at your own risk and read the resulting warning
        messages in order to ensure the **only** differences are those you expect.
        See :ref:`raise-on-checks` on the "Reproducibility and Compatibility" page of
        the documentation for more info.

        >>> met = po.Metamer(img, wrong_model)
        >>> met.load(
        ...     po.data.fetch_data("example_metamer_gaussian.pt"), raise_on_checks=False
        ... )
        """
        raise NotImplementedError(
            "Unsure how to handle this! don't have cached tensors"
        )
        check_attributes = [
            "_penalty_lambda",
        ]
        check_io_attrs = [
            ("_loss_function", ("_image")),
            ("_penalty_function", ("_image",)),
        ]
        super().load(
            file_path,
            "_synthesized_image",
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
        self.synthesized_image.requires_grad_()
        # these are always supposed to be on cpu, but may get copied over to
        # gpu on load (which can cause problems when resuming synthesis), so
        # fix that.
        if (
            len(self._saved_synthesized_image)
            and self._saved_synthesized_image[0].device.type != "cpu"
        ):
            self._saved_synthesized_image = [
                mad.to("cpu") for mad in self._saved_synthesized_image
            ]

    def __repr__(self) -> str:
        # numpydoc ignore=GL08
        return super()._repr_format(
            ["loss_function", "penalty_function", "penalty_lambda"]
        )

    @property
    def loss_function(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Callable which specifies how close metamer representation is to target."""
        # numpydoc ignore=RT01,ES01
        return self._loss_function

    @property
    def synthesized_image(self) -> torch.Tensor:
        """Model metamer, the parameter we are optimizing."""
        # numpydoc ignore=RT01,ES01,EX01
        if self._synthesized_image is None:
            return torch.empty(0)
        return self._synthesized_image

    @property
    def saved_synthesized_image(self) -> torch.Tensor:
        """
        :attr:`synthesized_image`, cached over time for later examination.

        How often the synthesized_image is cached is determined by the
        ``store_progress`` argument to the :func:`synthesize` function.

        The last entry will always be the current :attr:`synthesized_image`.

        If ``store_progress==1``, then this corresponds directly to :attr:`losses`:
        ``losses[i]`` is the error for ``saved_synthesized_image[i]``

        This tensor always lives on the CPU, regardless of the device of the
        ``Synthesized_Image`` object.

        Examples
        --------
        If synthesize is called without ``store_progress``, then this attribute
        just contains the synthesized_image, though there will be an additional
        dimension:

        >>> import plenoptic as po
        >>> po.set_seed(0)
        >>> img = po.data.einstein()
        >>> model = po.models.Gaussian(30).eval()
        >>> po.remove_grad(model)
        >>> met = po.Synthesized_Image(img, model)
        >>> met.saved_synthesized_image
        tensor([])
        >>> met.synthesize(5)
        >>> met.saved_synthesized_image
        tensor([[[[[ 0.0098, ...]]]]], grad_fn=<StackBackward0>)
        >>> met.synthesized_image
        tensor([[[[ 0.0098, ...]]]], requires_grad=True)
        >>> met.saved_synthesized_image.shape
        torch.Size([1, 1, 1, 256, 256])
        >>> met.synthesized_image.shape
        torch.Size([1, 1, 256, 256])

        If synthesize is called with ``store_progress=1``, then this attribute
        contains the synthesized_image at each iteration, and ``losses[i]`` contains
        the error for ``saved_synthesized_image[i]``.

        >>> met = po.Synthesized_Image(img, model)
        >>> met.synthesize(5, store_progress=True)
        >>> met.saved_synthesized_image.shape
        torch.Size([6, 1, 1, 256, 256])
        >>> met.objective_function(met.saved_synthesized_image[2])
        tensor(0.0169, grad_fn=<AddBackward0>)
        >>> met.losses[2]
        tensor(0.0169)

        (In the above example, ``saved_synthesized_image`` has 6 elements because it
        includes the synthesized_image at the start of each of the 5 synthesis
        iterations, plus the current one.)
        """  # numpydoc ignore=RT01,EX01
        if self._synthesized_image is None:
            return torch.empty(0)
        else:
            # for memory purposes, always on CPU
            return torch.stack(
                [*self._saved_synthesized_image, self.synthesized_image.to("cpu")]
            )

    @property
    def penalties(self) -> torch.Tensor:
        """
        Penalty function output over iterations.

        Will have ``length=num_iter+1``, where ``num_iter`` is the number of
        iterations of synthesis run so far.

        This tensor always lives on the CPU.
        """  # numpydoc ignore=RT01
        return super().penalties(self.synthesized_image)

    @property
    def image(self) -> torch.Tensor:
        """Target image of metamer optimization."""
        # numpydoc ignore=RT01,ES01,EX01
        return self._image
