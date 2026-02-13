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

from .. import optim
from ..convergence import _coarse_to_fine_enough, _loss_convergence
from ..model_components import signal
from ..validate import validate_coarse_to_fine, validate_input, validate_model
from .synthesis import OptimizedSynthesis


class Metamer(OptimizedSynthesis):
    r"""
    Synthesize metamers for image-computable differentiable models.

    Following the basic idea in [1]_, this class creates a metamer for a given model on
    a given image. We iteratively adjust the pixel values so as to match the
    representation of the :attr:`metamer` and :attr:`image`.

    Parameters
    ----------
    image
        A tensor, this is the image whose representation we wish to
        match.
    model
        A visual model.
    loss_function
        The loss function to use to compare the representations of the models
        in order to determine their loss.
    range_penalty_lambda
        Strength of the regularizer that enforces the allowed_range. Must be
        non-negative.
    allowed_range
        Range (inclusive) of allowed pixel values. Any values outside this
        range will be penalized.

    References
    ----------
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model
       based on Joint Statistics of Complex Wavelet Coefficients. Int'l
       Journal of Computer Vision. 40(1):49-71, October, 2000.
       https://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
       https://www.cns.nyu.edu/~lcv/texture/

    Examples
    --------
    Synthesize and visualize a metamer for a simple model:

    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import matplotlib.pyplot as plt
      >>> img = po.data.einstein()
      >>> model = po.simul.Gaussian(30).eval()
      >>> po.tools.remove_grad(model)
      >>> met = po.synth.Metamer(img, model)
      >>> met.synthesize(110)
      >>> fig, axes = plt.subplots(1, 4, figsize=(16, 4))
      >>> po.imshow(img, ax=axes[0], title="Target image")
      <Figure size ... with 4 Axes>
      >>> axes[0].xaxis.set_visible(False)
      >>> axes[0].yaxis.set_visible(False)
      >>> po.synth.metamer.plot_synthesis_status(met, fig=fig, axes_idx={"misc": 0})[0]
      <Figure size ...>
    """

    loss_function: Callable[[Tensor, Tensor], Tensor]
    """Callable which specifies how close metamer representation is to target."""

    def __init__(
        self,
        image: Tensor,
        model: torch.nn.Module,
        loss_function: Callable[[Tensor, Tensor], Tensor] = optim.mse,
        range_penalty_lambda: float = 0.1,
        allowed_range: tuple[float, float] = (0, 1),
    ):
        super().__init__(range_penalty_lambda, allowed_range)
        validate_input(image, allowed_range=allowed_range)
        validate_model(
            model,
            image_shape=image.shape,
            image_dtype=image.dtype,
            device=image.device,
        )
        self._model = model
        self._image = image
        self._image_shape = image.shape
        self._target_representation = self.model(self.image)
        self._scheduler = None
        self._scheduler_step_arg = False
        self.loss_function = loss_function
        self._saved_metamer = []
        self._store_progress = None
        self._metamer = None

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
            uniformly-distributed random noise lying within ``self.allowed_range``.
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
        >>> img = po.data.einstein()
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> met = po.synth.Metamer(img, model)
        >>> met.setup(po.data.curie())

        Set optimizer:

        >>> met = po.synth.Metamer(img, model)
        >>> met.setup(optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.01})

        Set optimizer and scheduler:

        >>> met = po.synth.Metamer(img, model)
        >>> met.setup(
        ...     optimizer=torch.optim.SGD,
        ...     optimizer_kwargs={"lr": 0.01},
        ...     scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        ... )

        Use with save/load. We only pass the optimizer/scheduler objects when calling
        setup after load, their kwargs and the initial image are handled during the
        load.

        >>> met = po.synth.Metamer(img, model)
        >>> met.setup(
        ...     po.data.curie(),
        ...     optimizer=torch.optim.SGD,
        ...     optimizer_kwargs={"lr": 0.01},
        ...     scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        ... )
        >>> met.synthesize(5)
        >>> met.save("metamer_setup.pt")
        >>> met = po.synth.Metamer(img, model)
        >>> met.load("metamer_setup.pt")
        >>> met.setup(
        ...     optimizer=torch.optim.SGD,
        ...     scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        ... )
        """
        if self._metamer is None:
            if initial_image is None:
                metamer = torch.rand_like(self.image)
                # rescale metamer to lie within the interval
                # self.allowed_range
                metamer = signal.rescale(metamer, *self.allowed_range)
            else:
                validate_input(initial_image, allowed_range=self.allowed_range)
                if initial_image.size() != self.image.size():
                    warnings.warn(
                        "initial_image and image are different sizes! This "
                        "has not been tested as much, open an issue if you have "
                        "any problems! https://github.com/plenoptic-org/plenoptic/"
                        "issues/new?template=bug_report.md"
                    )
                metamer = initial_image.clone().detach()
                metamer = metamer.to(dtype=self.image.dtype, device=self.image.device)
            metamer.requires_grad_()
            self._metamer = metamer
        else:
            if self._loaded:
                if initial_image is not None:
                    raise ValueError("Cannot set initial_image after calling load()!")
            else:
                raise ValueError(
                    "setup() can only be called once and must be called"
                    " before synthesize()!"
                )

        # initialize the optimizer
        self._initialize_optimizer(optimizer, self.metamer, optimizer_kwargs)
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
            :func:`~plenoptic.synthesize.metamer.animate` to create a video of the
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

        See Also
        --------
        :func:`~plenoptic.synthesize.metamer.plot_synthesis_status`
            Create a plot summarizing synthesis status at a given iteration.
        :func:`~plenoptic.synthesize.metamer.animate`
            Create a video of the metamer changing over the course of
            synthesis.

        Examples
        --------
        >>> import plenoptic as po
        >>> po.tools.set_seed(0)
        >>> img = po.data.einstein()
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> met = po.synth.Metamer(img, model)
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> met.synthesize(5)
        >>> met.losses
        tensor([0.0194, 0.0198, 0.0179, 0.0160, 0.0145, 0.0132])

        Synthesize a metamer, using ``store_progress`` so we can examine progress
        later. (This also enables us to create a video of the metamer changing over
        the course of synthesis, see
        :func:`~plenoptic.synthesize.metamer.animate`.)

        >>> met = po.synth.Metamer(img, model)
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

        >>> met = po.synth.Metamer(img, model)
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> met.synthesize(12, stop_criterion=0.001, stop_iters_to_check=2)
        >>> len(met.losses)
        9
        """
        # if setup hasn't been called manually, call it now.
        if self._metamer is None or isinstance(self._scheduler, tuple):
            self.setup()
        self._current_loss = None

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

        pbar.close()

    def objective_function(
        self,
        metamer: Tensor | None = None,
        target_representation: Tensor | None = None,
        **analyze_kwargs: Any,
    ) -> Tensor:
        """
        Compute the metamer synthesis loss.

        This calls self.loss_function on
        ``self.model(metamer, **analyze_kwargs)`` and
        ``target_representation`` and then adds the weighted range penalty
        on ``metamer``.

        Its output over time is stored in :attr:`losses`.

        Parameters
        ----------
        metamer
            Current ``metamer``. If ``None``, we use ``self.metamer``.
        target_representation
            Model response to ``image``. If ``None``, we use
            ``self.target_representation``.
        **analyze_kwargs
            Additional kwargs to pass to ``self.model(metamer)``.

        Returns
        -------
        loss
            1-element tensor containing the loss on this step.

        Examples
        --------
        >>> import plenoptic as po
        >>> po.tools.set_seed(0)
        >>> img = po.data.einstein()
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> met = po.synth.Metamer(img, model)

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

        Can be called with a different image. (Note that, because we called
        :meth:`synthesize` with ``store_progress=True``, we cached the metamer
        over the course of synthesis):

        >>> met.objective_function(met.saved_metamer[0])
        tensor(0.0194, grad_fn=<AddBackward0>)
        >>> met.losses[0]
        tensor(0.0194)

        This method differs from the :attr:`loss_function` attribute because of its
        inclusion of the penalty. In the following block, the pixels of
        ``rand_img`` all lie within $[0, 1]$, and so the outputs of
        :attr:`objective_function` and :attr:`loss_function` are the same:

        >>> rand_img = torch.rand_like(img)
        >>> rand_img.min(), rand_img.max()
        (tensor(7.9870e-06), tensor(1.0000))
        >>> met.objective_function(rand_img)
        tensor(0.0190)
        >>> met.loss_function(model(img), model(rand_img))
        tensor(0.0190)

        In this block, the image's lie outside $[0, 1]$, and so the outputs of
        :attr:`objective_function` and :attr:`loss_function` are different:

        >>> rand_img *= 2
        >>> rand_img.min(), rand_img.max()
        (tensor(0.0001), tensor(2.0000))
        >>> met.objective_function(rand_img)
        tensor(1100.9663)
        >>> met.loss_function(model(img), model(rand_img))
        tensor(0.3133)
        """
        if metamer is None:
            metamer = self.metamer
            # if this is empty, then self.metamer hasn't been initialized
            if metamer.numel() == 0:
                return torch.empty(0)
        if target_representation is None:
            target_representation = self.target_representation
        metamer_representation = self.model(metamer, **analyze_kwargs)
        loss = self.loss_function(metamer_representation, target_representation)
        range_penalty = optim.penalize_range(metamer, self.allowed_range)
        return loss + self.range_penalty_lambda * range_penalty

    def get_progress(
        self,
        iteration: int | None,
        iteration_selection: Literal["floor", "ceiling", "round"] = "round",
    ) -> dict:
        r"""
        Return dictionary summarizing synthesis progress at ``iteration``.

        This returns a dictionary containing info from :attr:`losses`,
        :attr:`pixel_change_norm`, :attr:`gradient_norm`, and
        :attr:`saved_metamer` corresponding to ``iteration``. If synthesis was
        run with ``store_progress=False`` (and so we did not cache anything in
        :attr:`saved_metamer`), then that key will be missing. If synthesis was
        run with ``store_progress>1``, we will grab the corresponding tensor
        from :attr:`saved_metamer`, with behavior determined by
        ``iteration_selection``.

        The returned dictionary will additionally contain the keys:

        - ``"iteration"``: the (0-indexed positive) synthesis iteration that the
          values for :attr:`losses`, :attr:`pixel_change_norm`, and
          :attr:`gradient_norm` come from.

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
        :func:`~plenoptic.synthesize.metamer.plot_synthesis_status`
            Create a plot summarizing synthesis status at a given iteration.
        :func:`~plenoptic.synthesize.metamer.animate`
            Create a video of the metamer changing over the course of
            synthesis.

        Examples
        --------
        >>> import plenoptic as po
        >>> po.tools.set_seed(0)
        >>> img = po.data.einstein()
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> met = po.synth.Metamer(img, model)
        >>> met.synthesize(5)

        Get values from the first iteration:

        >>> met.get_progress(0)
        {'losses': tensor(0.0194),
        'iteration': 0,
        'pixel_change_norm': tensor(2.5326),
        'gradient_norm': tensor(0.0010)}

        Get values from most last iteration of synthesis:

        >>> print(met.get_progress(-2))
        {'losses': tensor(0.0145),
        'iteration': 4,
        'pixel_change_norm': tensor(2.2698),
        'gradient_norm': tensor(0.0268)}

        Get current values:

        >>> print(met.get_progress(-1))
        {'losses': tensor(0.0132),
        'iteration': 5,
        'pixel_change_norm': None,
        'gradient_norm': None}

        When synthesis is run with ``store_progress=True``, this function also
        returns the metamer from the corresponding iteration:

        >>> met = po.synth.Metamer(img, model)
        >>> met.synthesize(5, store_progress=True)
        >>> print(met.get_progress(-1))
        {'losses': tensor(0.0124),
        'iteration': 5,
        'pixel_change_norm': None,
        'gradient_norm': None,
        'saved_metamer': tensor([[[[0.4554, ...]]]], grad_fn=<SelectBackward0>),
        'store_progress_iteration': 5}
        >>> torch.equal(met.saved_metamer[-1], met.get_progress(-1)["saved_metamer"])
        True

        When synthesis is run with ``store_progress>1``, this function returns the
        metamer from the closest iteration:

        >>> met = po.synth.Metamer(img, model)
        >>> met.synthesize(5, store_progress=2)
        >>> print(met.get_progress(-3))
        {'losses': tensor(0.0152),
        'iteration': 3,
        'pixel_change_norm': tensor(2.3592),
        'gradient_norm': tensor(0.0269),
        'saved_metamer': tensor([[[[0.8532, ...]]]], grad_fn=<SelectBackward0>),
        'store_progress_iteration': 4}

        When we cannot grab the saved metamer corresponding to the requested
        iteration, ``iteration_selection`` controls how we determine "closest":

        >>> print(met.get_progress(-3, iteration_selection="floor"))
        {'losses': tensor(0.0152),
        'iteration': 3,
        'pixel_change_norm': tensor(2.3592),
        'gradient_norm': tensor(0.0269),
        'saved_metamer': tensor([[[[ 0.8730, ...]]]], grad_fn=<SelectBackward0>),
        'store_progress_iteration': 2}
        """
        return super().get_progress(
            iteration,
            iteration_selection,
            store_progress_attributes=["saved_metamer"],
        )

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
        last_iter_metamer = self.metamer.clone()
        loss = self.optimizer.step(self._closure)
        self._losses.append(loss)

        grad_norm = torch.linalg.vector_norm(self.metamer.grad.data, ord=2, dim=None)
        self._gradient_norm.append(grad_norm.item())

        # optionally step the scheduler, passing loss if needed
        if self.scheduler is not None:
            if self._scheduler_step_arg:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

        pixel_change_norm = torch.linalg.vector_norm(
            self.metamer - last_iter_metamer, ord=2, dim=None
        )
        self._pixel_change_norm.append(pixel_change_norm.item())
        # add extra info here if you want it to show up in progress bar
        pbar.set_postfix(
            OrderedDict(
                loss=f"{loss:.04e}",
                learning_rate=self.optimizer.param_groups[0]["lr"],
                gradient_norm=f"{grad_norm.item():.04e}",
                pixel_change_norm=f"{pixel_change_norm.item():.04e}",
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
            self._saved_metamer.append(self.metamer.clone().to("cpu"))
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
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> met = po.synth.Metamer(img, model)
        >>> met.synthesize(max_iter=5, store_progress=True)
        >>> met.save("metamers.pt")
        """
        save_io_attrs = [
            ("loss_function", ("_target_representation", "2 * _target_representation")),
            ("_model", ("_image",)),
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
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> met = po.synth.Metamer(img, model)
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
        attrs = ["_image", "_target_representation", "_metamer", "_saved_metamer"]
        super().to(*args, attrs=attrs, **kwargs)
        # try to call .to() on model. this should work, but it might fail if e.g., this
        # a custom model that doesn't inherit torch.nn.Module
        try:
            self._model = self._model.to(*args, **kwargs)
        except AttributeError:
            warnings.warn("Unable to call model.to(), so we leave it as is.")

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

        This must be called by a ``Metamer`` object initialized just like the saved
        object.

        Note this operates in place and so doesn't return anything.

        .. versionchanged:: 1.2
           load behavior changed in a backwards-incompatible manner in order to
           compatible with breaking changes in torch 2.6.

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
            for any of :attr:`image`, :attr:`range_penalty_lambda`,
            or :attr:`allowed_range`.
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
        :func:`~plenoptic.tools.io.examine_saved_synthesis`
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
        >>> img = po.data.einstein().to(torch.float64)
        >>> model = po.simul.Gaussian(30).eval().to(torch.float64)
        >>> po.tools.remove_grad(model)
        >>> met = po.synth.Metamer(img, model)
        >>> print(met.metamer)
        tensor([])
        >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
        >>> print(met.metamer)
        tensor([[[[0.0692, ...]]]], dtype=torch.float64, requires_grad=True)

        If the saved ``Metamer`` object lived on a CUDA device and you do not have
        CUDA on the loading machine, use ``map_location`` to change device:

        >>> met = po.synth.Metamer(img, model)
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
        >>> print(met.metamer)
        tensor([[[[0.0692, ...]]]], dtype=torch.float64, requires_grad=True)

        If the loading ``Metamer`` object was not initialized with same values
        as the saved object, an error will be raised:

        >>> met = po.synth.Metamer(torch.rand_like(img), model)
        >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
        Traceback (most recent call last):
        ValueError: Saved and initialized attribute image have different values...

        If the loading ``Metamer`` object has a different data type than the saved
        object, an error will be raised:

        >>> met = po.synth.Metamer(img, model)
        >>> met.to(torch.float32)
        >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
        Traceback (most recent call last):
        ValueError: Saved and initialized attribute image have different dtype...
        """
        self._load(
            file_path,
            map_location,
            tensor_equality_atol=tensor_equality_atol,
            tensor_equality_rtol=tensor_equality_rtol,
            **pickle_load_args,
        )

    def _load(
        self,
        file_path: str,
        map_location: str | None = None,
        additional_check_attributes: list[str] = [],
        additional_check_io_attributes: list[str] = [],
        tensor_equality_atol: float = 1e-8,
        tensor_equality_rtol: float = 1e-5,
        **pickle_load_args: Any,
    ):
        r"""
        Load from a file.

        This is a helper function for loading.

        Users interact with ``load`` (without the underscore), this is to allow
        subclasses to specify additional attributes or loss functions to check.

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
        additional_check_attributes
            Any additional attributes to check for equality. Intended for use by any
            subclasses, to add other attributes set at initialization.
        additional_check_io_attributes
            Any additional attributes whose input/output behavior we should check.
            Intended for use by any subclasses.
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
        """  # numpydoc ignore=EX01
        check_attributes = [
            "_image",
            "_range_penalty_lambda",
            "_allowed_range",
        ]
        check_attributes += additional_check_attributes
        check_io_attrs = [
            ("loss_function", ("_target_representation", "2 * _target_representation")),
            ("_model", ("_image",)),
        ]
        check_io_attrs += additional_check_io_attributes
        super().load(
            file_path,
            "_metamer",
            map_location=map_location,
            check_attributes=check_attributes,
            check_io_attributes=check_io_attrs,
            state_dict_attributes=["_optimizer", "_scheduler"],
            tensor_equality_atol=tensor_equality_atol,
            tensor_equality_rtol=tensor_equality_rtol,
            **pickle_load_args,
        )
        # make this require a grad again
        self.metamer.requires_grad_()
        # these are always supposed to be on cpu, but may get copied over to
        # gpu on load (which can cause problems when resuming synthesis), so
        # fix that.
        if len(self._saved_metamer) and self._saved_metamer[0].device.type != "cpu":
            self._saved_metamer = [met.to("cpu") for met in self._saved_metamer]

    @property
    def model(self) -> torch.nn.Module:
        """The model for which the metamer is synthesized."""
        # numpydoc ignore=RT01,ES01,EX01
        return self._model

    @property
    def image(self) -> torch.Tensor:
        """Target image of metamer optimization."""
        # numpydoc ignore=RT01,ES01,EX01
        return self._image

    @property
    def target_representation(self) -> torch.Tensor:
        """
        :attr:`model` representation of :attr:`image`.

        The goal of synthesis is for ``model(metamer)`` to match this value.

        Examples
        --------
        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> met = po.synth.Metamer(img, model)
        >>> torch.equal(model(img), met.target_representation)
        True
        """  # numpydoc ignore=RT01
        return self._target_representation

    @property
    def metamer(self) -> torch.Tensor:
        """Model metamer, the parameter we are optimizing."""
        # numpydoc ignore=RT01,ES01,EX01
        if self._metamer is None:
            return torch.empty(0)
        return self._metamer

    @property
    def saved_metamer(self) -> torch.Tensor:
        """
        :attr:`metamer`, cached over time for later examination.

        How often the metamer is cached is determined by the ``store_progress`` argument
        to the :func:`synthesize` function.

        The last entry will always be the current :attr:`metamer`.

        If ``store_progress==1``, then this corresponds directly to :attr:`losses`:
        ``losses[i]`` is the error for ``saved_metamer[i]``

        This tensor always lives on the CPU, regardless of the device of the ``Metamer``
        object.

        Examples
        --------
        If synthesize is called without ``store_progress``, then this attribute
        just contains the metamer, though the number of dimensions is different:

        >>> import plenoptic as po
        >>> po.tools.set_seed(0)
        >>> img = po.data.einstein()
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> met = po.synth.Metamer(img, model)
        >>> met.saved_metamer
        tensor([])
        >>> met.synthesize(5)
        >>> met.saved_metamer
        tensor([[[[[ 0.0098, ...]]]]], grad_fn=<StackBackward0>)
        >>> met.metamer
        tensor([[[[ 0.0098, ...]]]], requires_grad=True)
        >>> met.saved_metamer.shape
        torch.Size([1, 1, 1, 256, 256])
        >>> met.metamer.shape
        torch.Size([1, 1, 256, 256])

        If synthesize is called with ``store_progress=1``, then this attribute
        contains the metamer at each iteration, and ``losses[i]`` contains the error
        for ``saved_metamer[i]``.

        >>> met = po.synth.Metamer(img, model)
        >>> met.synthesize(5, store_progress=True)
        >>> met.saved_metamer.shape
        torch.Size([6, 1, 1, 256, 256])
        >>> met.objective_function(met.saved_metamer[2])
        tensor(0.0169, grad_fn=<AddBackward0>)
        >>> met.losses[2]
        tensor(0.0169)

        (In the above example, ``saved_metamer`` has 6 elements because it includes the
        metamer at the start of each of the 5 synthesis iterations, plus the current
        one.)
        """  # numpydoc ignore=RT01,EX01
        if self._metamer is None:
            return torch.empty(0)
        else:
            # for memory purposes, always on CPU
            return torch.stack([*self._saved_metamer, self.metamer.to("cpu")])


class MetamerCTF(Metamer):
    """
    Synthesize model metamers with coarse-to-fine synthesis.

    This is a special case of ``Metamer``, which uses the coarse-to-fine
    synthesis procedure described in [1]_: we start by updating metamer with
    respect to only a subset of the model's representation (generally, that
    which corresponds to the lowest spatial frequencies), and changing which
    subset we consider over the course of synthesis. This is similar to
    optimizing with a blurred version of the objective function and gradually
    adding in finer details. It improves synthesis performance for some models.

    Parameters
    ----------
    image
        A tensor, this is the image whose representation we wish to
        match.
    model
        A visual model.
    loss_function
        The loss function to use to compare the representations of the models
        in order to determine their loss.
    range_penalty_lambda
        Strength of the regularizer that enforces the allowed_range. Must be
        non-negative.
    allowed_range
        Range (inclusive) of allowed pixel values. Any values outside this
        range will be penalized.
    coarse_to_fine
        - ``"together"``: start with the coarsest scale, then gradually
          add each finer scale.
        - ``"separate"``: compute the gradient with respect to each
          scale separately (ignoring the others), then with respect
          to all of them at the end.

        (see :ref:`Metamer tutorial <metamer-nb>` for more details).

    References
    ----------
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model
       based on Joint Statistics of Complex Wavelet Coefficients. Int'l
       Journal of Computer Vision. 40(1):49-71, October, 2000.
       https://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
       https://www.cns.nyu.edu/~lcv/texture/

    Examples
    --------
    Synthesize and visualize a metamer using coarse-to-fine synthesis:

    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import matplotlib.pyplot as plt
      >>> import torch
      >>> img = po.data.reptile_skin()
      >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
      >>> # to work with MetamerCTF, models must have a scales attribute
      >>> model.scales
      ['pixel_statistics', 'residual_lowpass', 3, 2, 1, 0, 'residual_highpass']
      >>> met = po.synth.MetamerCTF(img, model, loss_function=po.tools.optim.l2_norm)
      >>> # initialize with an image that has a comparable mean and standard deviation
      >>> init_img = (torch.rand_like(img) - 0.5) * 0.1 + img.mean()
      >>> met.setup(init_img)
      >>> met.synthesize(150, change_scale_criterion=None, ctf_iters_to_check=7)
      >>> fig, axes = plt.subplots(1, 4, figsize=(25, 4), width_ratios=[1, 1, 1, 3])
      >>> po.imshow(img, ax=axes[0], title="Target image")
      <Figure size ... with 4 Axes>
      >>> axes[0].xaxis.set_visible(False)
      >>> axes[0].yaxis.set_visible(False)
      >>> po.synth.metamer.plot_synthesis_status(met, fig=fig, axes_idx={"misc": 0})[0]
      <Figure size ...>

    Not all models work with ``MetamerCTF``:

    >>> import plenoptic as po
    >>> img = po.data.einstein()
    >>> model = po.simul.Gaussian(30).eval()
    >>> po.tools.remove_grad(model)
    >>> met = po.synth.MetamerCTF(img, model)
    Traceback (most recent call last):
    AttributeError: model has no scales attribute ...
    """

    def __init__(
        self,
        image: Tensor,
        model: torch.nn.Module,
        loss_function: Callable[[Tensor, Tensor], Tensor] = optim.mse,
        range_penalty_lambda: float = 0.1,
        allowed_range: tuple[float, float] = (0, 1),
        coarse_to_fine: Literal["together", "separate"] = "together",
    ):
        super().__init__(
            image,
            model,
            loss_function,
            range_penalty_lambda,
            allowed_range,
        )
        self._init_ctf(coarse_to_fine)

    def _init_ctf(self, coarse_to_fine: Literal["together", "separate"]):
        """
        Initialize stuff related to coarse-to-fine.

        - Validates value of ``coarse_to_fine``

        - Validates ``self.model`` for coarse-to-fine synthesis (calls
          :func:`validate_coarse_to_fine`).

        - Initializes attributes for coarse-to-fine synthesis.

        Parameters
        ----------
        coarse_to_fine
            Which mode of coarse-to-fine to use, see initial docstring for details.

        Raises
        ------
        ValueError
            If ``coarse_to_fine`` takes an illegal value.
        """  # numpydoc ignore=EX01
        # this will hold the reduced representation of the target image.
        if coarse_to_fine not in ["separate", "together"]:
            raise ValueError(
                f"Don't know how to handle value {coarse_to_fine}!"
                " Must be one of: 'separate', 'together'"
            )
        self._ctf_target_representation = None
        validate_coarse_to_fine(
            self.model, image_shape=self.image.shape, device=self.image.device
        )
        # if self.scales is not None, we're continuing a previous version
        # and want to continue. this list comprehension creates a new
        # object, so we don't modify model.scales
        self._scales = [i for i in self.model.scales[:-1]]
        if coarse_to_fine == "separate":
            self._scales += [self.model.scales[-1]]
        self._scales += ["all"]
        self._scales_timing = dict((k, []) for k in self.scales)
        self._scales_timing[self.scales[0]].append(0)
        self._scales_loss = []
        self._scales_finished = []
        self._coarse_to_fine = coarse_to_fine
        self._initial_lr = None

    def _initialize_optimizer(
        self,
        optimizer: torch.optim.Optimizer | None,
        synth_attr: torch.Tensor,
        optimizer_kwargs: dict | None = None,
    ):
        """
        Initialize optimizer.

        Calls ``super._initialize_optimizer()``, passing all arguments through, and also
        caches the initial learning rate (``self._initial_lr``), which we use when
        switching scales.

        Parameters
        ----------
        optimizer
            The (un-initialized) optimizer object to use. If ``None``, we use
            :class:`torch.optim.Adam`.
        synth_attr
            The tensor we will optimize.
        optimizer_kwargs
            The keyword arguments to pass to the optimizer on initialization. If
            ``None``, we use ``{"lr": .01}`` and, if optimizer is ``None``,
            ``{"amsgrad": True}``.
        """  # numpydoc ignore=EX01
        super()._initialize_optimizer(optimizer, synth_attr, optimizer_kwargs)
        # save the initial learning rate so we can reset it when we change scales
        self._initial_lr = [pg["lr"] for pg in self.optimizer.param_groups]

    def synthesize(
        self,
        max_iter: int = 100,
        store_progress: bool | int = False,
        stop_criterion: float = 1e-4,
        stop_iters_to_check: int = 50,
        change_scale_criterion: float | None = 1e-2,
        ctf_iters_to_check: int = 50,
    ):
        r"""
        Synthesize a metamer.

        Update the pixels of ``metamer`` until its representation matches
        that of ``image``.

        We run this until either we reach ``max_iter`` or the change over the
        past ``stop_iters_to_check`` iterations is less than
        ``stop_criterion``, whichever comes first.

        Parameters
        ----------
        max_iter
            The maximum number of iterations to run before we end synthesis
            (unless we hit the stop criterion).
        store_progress
            Whether we should store the metamer image in progress on every
            iteration. If ``False``, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress`` iterations
            (note then that 0 is the same as False and 1 the same as True). This is
            primarily useful for using
            :func:`~plenoptic.synthesize.metamer.animate` to create a video of the
            course of synthesis.
        stop_criterion
            If the loss over the past ``stop_iters_to_check`` has changed
            less than ``stop_criterion``, we terminate synthesis.
        stop_iters_to_check
            How many iterations back to check in order to see if the
            loss has stopped decreasing (for ``stop_criterion``).
        change_scale_criterion
            Scale-specific analogue of ``change_scale_criterion``: we consider
            a given scale finished (and move onto the next) if the loss has
            changed less than this in the past ``ctf_iters_to_check``
            iterations. If ``None``, we'll change scales as soon as we've spent
            ``ctf_iters_to_check`` on a given scale.
        ctf_iters_to_check
            Scale-specific analogue of ``stop_iters_to_check``: how many
            iterations back in order to check in order to see if we should
            switch scales.

        Raises
        ------
        ValueError
            If ``stop_criterion >= change_scale_criterion`` -- behavior is strange
            otherwise.
        ValueError
            If we find a NaN during optimization.

        See Also
        --------
        :func:`~plenoptic.synthesize.metamer.plot_synthesis_status`
            Create a plot summarizing synthesis status at a given iteration.
        :func:`~plenoptic.synthesize.metamer.animate`
            Create a video of the metamer changing over the course of
            synthesis.

        Examples
        --------
        >>> import plenoptic as po
        >>> po.tools.set_seed(0)
        >>> img = po.data.reptile_skin()
        >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
        >>> met = po.synth.MetamerCTF(img, model)
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> met.synthesize(5)
        >>> met.losses
        tensor([0.0821, ..., 0.0805])

        You can examine scales_timing attribute to see when MetamerCTF started and
        stopped optimizing each scale:

        >>> met.scales_timing
        {'pixel_statistics': [0],
         'residual_lowpass': [],
         3: [],
         2: [],
         1: [],
         0: [],
         'all': []}

        Synthesize a metamer, using ``store_progress`` so we can examine progress
        later. (This also enables us to create a video of the metamer changing over
        the course of synthesis, see
        :func:`~plenoptic.synthesize.metamer.animate`.)

        >>> met = po.synth.MetamerCTF(img, model)
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> met.synthesize(5, store_progress=2)
        >>> met.saved_metamer.shape
        torch.Size([4, 1, 1, 256, 256])
        >>> # see loss, etc on the 4th iteration
        >>> progress = met.get_progress(4)
        >>> progress.keys()
        dict_keys(['losses', ..., 'saved_metamer', 'store_progress_iteration'])
        >>> progress["losses"]
        tensor(0.0850)

        Set ``change_scale_criterion`` and ``ctf_iters_to_check`` to change
        scale-switching behavior.

        >>> met = po.synth.MetamerCTF(img, model)
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> met.synthesize(5, change_scale_criterion=None, ctf_iters_to_check=2)
        >>> met.losses
        tensor([0.0863, ..., 0.0569])
        >>> met.scales_timing
        {'pixel_statistics': [0, 1],
         'residual_lowpass': [2, 3],
         3: [4],
         2: [],
         1: [],
         0: [],
         'all': []}

        Adjust ``stop_criterion`` and ``stop_iters_to_check`` to change how convergence
        is determined. In this case, we stop early by making ``stop_criterion`` fairly
        large. In practice, you're more likely to make ``stop_criterion`` smaller to let
        synthesis run for longer.

        >>> met = po.synth.MetamerCTF(img, model)
        >>> # this isn't enough to run synthesis to completion, just an example
        >>> met.synthesize(10, stop_criterion=0.001, stop_iters_to_check=2)
        """
        if (change_scale_criterion is not None) and (
            stop_criterion >= change_scale_criterion
        ):
            raise ValueError(
                "stop_criterion must be strictly less than "
                "change_scale_criterion, or things get weird!"
            )

        # if setup hasn't been called manually, call it now.
        if self._metamer is None or isinstance(self._scheduler, tuple):
            self.setup()
        self._current_loss = None

        # get ready to store progress
        self.store_progress = store_progress

        pbar = tqdm(range(max_iter))

        for i in pbar:
            # update saved_* attrs. len(_losses) gives the total number of
            # iterations and will be correct across calls to `synthesize`
            self._store(len(self._losses))

            loss = self._optimizer_step(
                pbar, change_scale_criterion, ctf_iters_to_check
            )

            if not np.isfinite(loss):
                raise ValueError("Found a NaN in loss during optimization.")

            if self._check_convergence(
                i, stop_criterion, stop_iters_to_check, ctf_iters_to_check
            ):
                warnings.warn("Loss has converged, stopping synthesis")
                break

        # compute current loss, no need to compute gradient
        with torch.no_grad():
            self._current_loss = self.objective_function().item()

        pbar.close()

    def _optimizer_step(
        self,
        pbar: tqdm,
        change_scale_criterion: float,
        ctf_iters_to_check: int,
    ) -> Tensor:
        r"""
        Compute and propagate gradients, then step the optimizer to update metamer.

        Parameters
        ----------
        pbar
            A tqdm progress-bar, which we update with a postfix
            describing the current loss, gradient norm, and learning
            rate (it already tells us which iteration and the time
            elapsed).
        change_scale_criterion
            How many iterations back to check to see if the loss has stopped
            decreasing and we should thus move to the next scale in
            coarse-to-fine optimization.
        ctf_iters_to_check
            Minimum number of iterations coarse-to-fine must run at each scale.

        Returns
        -------
        loss
            1-element tensor containing the loss on this step.
        """  # numpydoc ignore=ES01,EX01
        last_iter_metamer = self.metamer.clone()

        # Check if conditions hold for switching scales:
        # - Check if loss has decreased below the change_scale_criterion and
        # - if we've been optimizing this scale for the required number of iterations
        # - The first check here is because the last scale will be 'all', and
        #   we never remove it

        if (
            len(self.scales) > 1
            and len(self.scales_loss) >= ctf_iters_to_check
            and (
                change_scale_criterion is None
                or abs(self.scales_loss[-1] - self.scales_loss[-ctf_iters_to_check])
                < change_scale_criterion
            )
            and (
                len(self._losses) - self.scales_timing[self.scales[0]][0]
                >= ctf_iters_to_check
            )
        ):
            self._scales_timing[self.scales[0]].append(len(self._losses) - 1)
            self._scales_finished.append(self._scales.pop(0))

            # Only append if scales list is still non-empty after the pop
            if self.scales:
                self._scales_timing[self.scales[0]].append(len(self._losses))

            # Reset optimizer's learning rate
            for pg, lr in zip(self.optimizer.param_groups, self._initial_lr):
                pg["lr"] = lr

            # Reset ctf target representation for the next update
            self._ctf_target_representation = None

        # the loss returned by objective_function is from *before* updating the metamer,
        # so to compute the equivalent for display purposes, we need to call this before
        # calling step()
        if self.scales[0] != "all":
            with torch.no_grad():
                overall_loss = self.objective_function(None, None).item()

        loss = self.optimizer.step(self._closure)
        if self.scales[0] == "all":
            # then the loss computed above includes all scales
            overall_loss = loss

        self._scales_loss.append(loss)
        self._losses.append(overall_loss)

        grad_norm = torch.linalg.vector_norm(self.metamer.grad.data, ord=2, dim=None)
        self._gradient_norm.append(grad_norm.item())

        # optionally step the scheduler, passing loss if needed
        if self.scheduler is not None:
            if self._scheduler_step_arg:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

        pixel_change_norm = torch.linalg.vector_norm(
            self.metamer - last_iter_metamer, ord=2, dim=None
        )
        self._pixel_change_norm.append(pixel_change_norm.item())
        # add extra info here if you want it to show up in progress bar
        pbar.set_postfix(
            OrderedDict(
                loss=f"{overall_loss:.04e}",
                learning_rate=self.optimizer.param_groups[0]["lr"],
                gradient_norm=f"{grad_norm.item():.04e}",
                pixel_change_norm=f"{pixel_change_norm.item():.04e}",
                current_scale=self.scales[0],
                current_scale_loss=f"{loss:.04e}",
            )
        )
        return overall_loss

    def _closure(self) -> Tensor:
        r"""
        Calculate the gradient, before the optimization step.

        This enables optimization algorithms that perform several evaluations
        of the gradient before taking a step (ie. second order methods like
        LBFGS).

        Additionally, this is where:

        - ``metamer_representation`` is calculated, and thus any modifications
          to the model's forward call (e.g., specifying ``scale`` kwarg for
          coarse-to-fine) should happen.

        - ``loss`` is calculated and ``loss.backward()`` is called.

        Returns
        -------
        loss
            Loss of the current objective function.
        """  # numpydoc ignore=EX01
        self.optimizer.zero_grad()
        analyze_kwargs = {}
        # if we've reached 'all', we use the full model
        if self.scales[0] != "all":
            analyze_kwargs["scales"] = [self.scales[0]]
            # if 'together', then we also want all the coarser
            # scales
            if self.coarse_to_fine == "together":
                analyze_kwargs["scales"] += self.scales_finished
        # if analyze_kwargs is empty, we can just compare
        # metamer_representation against our cached target_representation
        if analyze_kwargs:
            if self._ctf_target_representation is None:
                target_rep = self.model(self.image, **analyze_kwargs)
                self._ctf_target_representation = target_rep
            else:
                target_rep = self._ctf_target_representation
        else:
            target_rep = None

        loss = self.objective_function(self.metamer, target_rep, **analyze_kwargs)
        loss.backward(retain_graph=False)

        return loss.item()

    def _check_convergence(
        self,
        i: int,
        stop_criterion: float,
        stop_iters_to_check: int,
        ctf_iters_to_check: int,
    ) -> bool:
        r"""
        Check whether the loss has stabilized and whether we've synthesized all scales.

        We check whether:

        - We have been synthesizing for ``stop_iters_to_check`` iterations,
          i.e. ``len(synth.losses) > stop_iters_to_check``.

        - Loss has decreased by less than ``stop_criterion`` over the past
          ``stop_iters_to_check`` iterations.

        - We have finished synthesizing each individual scale, i.e. ``synth.scales[0] ==
          "all"``.

        - We have been synthesizing all scales for more than ``ctf_iters_to_check``
          iterations, i.e. ``i - synth.scales_timing["all"][0]) > ctf_iters_to_check``.

        If all conditions are met, we return ``True``. Else, we return ``False``.

        Parameters
        ----------
        i
            The current iteration (0-indexed).
        stop_criterion
            If the loss over the past ``stop_iters_to_check`` has changed
            less than ``stop_criterion``, we terminate synthesis.
        stop_iters_to_check
            How many iterations back to check in order to see if the
            loss has stopped decreasing (for ``stop_criterion``).
        ctf_iters_to_check
            Minimum number of iterations coarse-to-fine must run at each scale.

        Returns
        -------
        loss_stabilized
            Whether the loss has stabilized and we've synthesized all scales.
        """  # noqa: E501
        # numpydoc ignore=EX01
        loss_conv = _loss_convergence(self, stop_criterion, stop_iters_to_check)
        return loss_conv and _coarse_to_fine_enough(self, i, ctf_iters_to_check)

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
        >>> img = po.data.reptile_skin()
        >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
        >>> met = po.synth.MetamerCTF(img, model)
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
        super().to(*args, **kwargs)
        # if synthesize has been called at least once and we have not finished moving
        # through all scales, _ctf_target_representation will be a Tensor which get
        # passed to objective_function at some point. thus, need to make sure it's also
        # updated.
        if self._ctf_target_representation is not None:
            self._ctf_target_representation = self._ctf_target_representation.to(
                *args, **kwargs
            )

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

        This should be called by an initialized ``Metamer`` object -- we will
        ensure that ``image``, ``target_representation`` (and thus
        ``model``), and ``loss_function`` are all identical.

        Note this operates in place and so doesn't return anything.

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
            If the object saved at ``file_path`` is not a ``MetamerCTF`` object.
        ValueError
            If the saved and loading ``MetamerCTF`` objects have a different value
            for any of :attr:`image`, :attr:`range_penalty_lambda`,
            :attr:`allowed_range`, or :attr:`coarse_to_fine`.
        ValueError
            If the behavior of :attr:`loss_function` or :attr:`model` is different
            between the saved and loading objects.

        Warns
        -----
        UserWarning
            If :func:`setup` will need to be called after load, to finish initializing
            :attr:`optimizer` or :attr:`scheduler`.

        Examples
        --------
        In order to load a saved ``MetamerCTF`` object, we must first initialize one
        using the same arguments. (We use float64 / "double" precision rather than
        torch's default float32 because it increases reproducibility, see the
        :ref:`Reproducibility <reproduce>` page of our documentations for more details.)
        Here, we load in a cached example:

        >>> import plenoptic as po
        >>> img = po.data.reptile_skin().to(torch.float64)
        >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
        >>> met = po.synth.MetamerCTF(img, model, po.tools.optim.l2_norm)
        >>> print(met.metamer)
        tensor([])
        >>> met.load(po.data.fetch_data("example_metamerCTF_ps.pt"))
        >>> print(met.metamer)
        tensor([[[[0.3016, ...]]]], dtype=torch.float64, requires_grad=True)

        If the saved ``MetamerCTF`` object lived on a CUDA device and you do not have
        CUDA on the loading machine, use ``map_location`` to change device:

        >>> met = po.synth.MetamerCTF(img, model, po.tools.optim.l2_norm)
        >>> met.image.device
        device(type='cpu')
        >>> met.load(po.data.fetch_data("example_metamerCTF_ps-cuda.pt"))
        Traceback (most recent call last):
        RuntimeError: Attempting to deserialize object on a CUDA device but
        torch.cuda.is_available() is False...
        >>> met.load(
        ...     po.data.fetch_data("example_metamerCTF_ps-cuda.pt"), map_location="cpu"
        ... )
        >>> print(met.metamer)
        tensor([[[[0.3016, ...]]]], dtype=torch.float64, requires_grad=True)

        Loading and saving must both be done with ``MetamerCTF``:

        >>> met = po.synth.Metamer(img, model)
        >>> met.load(po.data.fetch_data("example_metamerCTF_ps.pt"))
        Traceback (most recent call last):
        ValueError: Saved object was a plenoptic.synthesize.metamer.MetamerCTF...

        If the loading ``MetamerCTF`` object was not initialized with same values
        as the saved object, an error will be raised:

        >>> met = po.synth.MetamerCTF(
        ...     torch.rand_like(img), model, po.tools.optim.l2_norm
        ... )
        >>> met.load(po.data.fetch_data("example_metamerCTF_ps.pt"))
        Traceback (most recent call last):
        ValueError: Saved and initialized attribute image have different values...

        If the loading ``MetamerCTF`` object has a different data type than the saved
        object, an error will be raised:

        >>> met = po.synth.MetamerCTF(img, model, po.tools.optim.l2_norm)
        >>> met.to(torch.float32)
        >>> met.load(po.data.fetch_data("example_metamerCTF_ps.pt"))
        Traceback (most recent call last):
        ValueError: Saved and initialized attribute image have different dtype...
        """
        super()._load(
            file_path,
            map_location,
            ["_coarse_to_fine"],
            tensor_equality_atol=tensor_equality_atol,
            tensor_equality_rtol=tensor_equality_rtol,
            **pickle_load_args,
        )

    @property
    def coarse_to_fine(self) -> str:
        """How we scales are handled, see :class:`MetamerCTF` for details."""
        # numpydoc ignore=RT01,ES01,EX01
        return self._coarse_to_fine

    @property
    def scales(self) -> tuple:
        """Model scales that we've yet to optimize, modified during optimization."""
        # numpydoc ignore=RT01,ES01,EX01
        return tuple(self._scales)

    @property
    def scales_loss(self) -> tuple:
        """Scale-specific loss at each iteration."""
        # numpydoc ignore=RT01,ES01,EX01
        return tuple(self._scales_loss)

    @property
    def scales_timing(self) -> dict:
        """
        Information about when each scale was started and stopped.

        Keys are the values found in :attr:`scales`, and values are lists specifying
        the iteration where we started and stopped optimizing this scale, which are
        modified during optimization.
        """  # numpydoc ignore=RT01,EX01
        return self._scales_timing

    @property
    def scales_finished(self) -> tuple:
        """Model scales that we've finished optimizing, modified during optimization."""
        # numpydoc ignore=RT01,ES01,EX01
        return tuple(self._scales_finished)
