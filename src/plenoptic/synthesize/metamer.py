"""
Model metamers.

Classes to perform the synthesis of model metamers.
"""

import re
import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from ..tools import data, display, optim, signal
from ..tools.convergence import coarse_to_fine_enough, loss_convergence
from ..tools.validate import validate_coarse_to_fine, validate_input, validate_model
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
    """

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
        >>> met.synthesize(10)

        Set optimizer:

        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> met = po.synth.Metamer(img, model)
        >>> met.setup(optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.01})
        >>> met.synthesize(10)

        Use with save/load. Only the optimizer object is necessary, its kwargs and the
        initial image are handled by load.

        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> met = po.synth.Metamer(img, model)
        >>> met.setup(
        ...     po.data.curie(),
        ...     optimizer=torch.optim.SGD,
        ...     optimizer_kwargs={"lr": 0.01},
        ... )
        >>> met.synthesize(10)
        >>> met.save("metamer_setup.pt")
        >>> met = po.synth.Metamer(img, model)
        >>> met.load("metamer_setup.pt")
        >>> met.setup(optimizer=torch.optim.SGD)
        >>> met.synthesize(10)
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

        We run this until either we reach ``max_iter`` or the change over the
        past ``stop_iters_to_check`` iterations is less than
        ``stop_criterion``, whichever comes first.

        Parameters
        ----------
        max_iter
            The maximum number of iterations to run before we end synthesis
            (unless we hit the stop criterion).
        store_progress
            Whether we should store the metamer image in progress during
            synthesis. If ``False``, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress`` iterations
            (note then that 0 is the same as False and 1 the same as True).
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
        """
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
        """  # numpydoc ignore=ES01
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
        """
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

        See :func:`load` docstring for an example of use.

        Parameters
        ----------
        file_path :
            The path to save the metamer object to.
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
        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> metamer = po.synth.Metamer(img, model)
        >>> metamer.synthesize(max_iter=5, store_progress=True)
        >>> metamer.save("metamers.pt")
        >>> metamer_copy = po.synth.Metamer(img, model)
        >>> metamer_copy.load("metamers.pt")
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
        """
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
        # numpydoc ignore=RT01,ES01
        return self._model

    @property
    def image(self) -> torch.Tensor:
        """Target image of metamer optimization."""
        # numpydoc ignore=RT01,ES01
        return self._image

    @property
    def target_representation(self) -> torch.Tensor:
        """
        :attr:`model` representation of :attr:`image`, i.e. ``model(image)``.

        The goal of synthesis is for ``model(metamer)`` to match this value.
        """  # numpydoc ignore=RT01
        return self._target_representation

    @property
    def metamer(self) -> torch.Tensor:
        """Model metamer, the parameter we are optimizing."""
        # numpydoc ignore=RT01,ES01
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
        """  # numpydoc ignore=RT01
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
        (see :ref:`Metamer tutorial <metamer-nb>`_ for more details).

    References
    ----------
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model
       based on Joint Statistics of Complex Wavelet Coefficients. Int'l
       Journal of Computer Vision. 40(1):49-71, October, 2000.
       https://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
       https://www.cns.nyu.edu/~lcv/texture/
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
        """
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
        """
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
            (note then that 0 is the same as False and 1 the same as True).
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
        """  # numpydoc ignore=ES01
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
        """
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
        loss_conv = loss_convergence(self, stop_criterion, stop_iters_to_check)
        return loss_conv and coarse_to_fine_enough(self, i, ctf_iters_to_check)

    def to(self, *args: Any, **kwargs: Any):
        r"""
        Move and/or cast the parameters and buffers.

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
        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
        >>> metamer = po.synth.MetamerCTF(img, model)
        >>> metamer.synthesize(max_iter=5, store_progress=True)
        >>> metamer.save("metamers_ctf.pt")
        >>> metamer_copy = po.synth.MetamerCTF(img, model)
        >>> metamer_copy.load("metamers_ctf.pt")
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
        # numpydoc ignore=RT01,ES01
        return self._coarse_to_fine

    @property
    def scales(self) -> tuple:
        """Model scales that we've yet to optimize, modified during optimization."""
        # numpydoc ignore=RT01,ES01
        return tuple(self._scales)

    @property
    def scales_loss(self) -> tuple:
        """Scale-specific loss at each iteration."""
        # numpydoc ignore=RT01,ES01
        return tuple(self._scales_loss)

    @property
    def scales_timing(self) -> dict:
        """
        Information about when each scale was started and stopped.

        Keys are the values found in :attr:`scales`, and values are lists specifying
        the iteration where we started and stopped optimizing this scale, which are
        modified during optimization.
        """  # numpydoc ignore=RT01
        return self._scales_timing

    @property
    def scales_finished(self) -> tuple:
        """Model scales that we've finished optimizing, modified during optimization."""
        # numpydoc ignore=RT01,ES01
        return tuple(self._scales_finished)


def plot_loss(
    metamer: Metamer,
    iteration: int | None = None,
    ax: mpl.axes.Axes | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Plot synthesis loss with log-scaled y axis.

    Plots ``metamer.losses`` over all iterations. Also plots a red dot at
    ``iteration``, to highlight the loss there. If ``iteration=None``, then the
    dot will be at the final iteration.

    Parameters
    ----------
    metamer
        Metamer object whose loss we want to plot.
    iteration
        Which iteration to display. If ``None``,  we show
        the most recent one. Negative values are also allowed.
    ax
        Pre-existing axes for plot. If ``None``, we call
        :func:`matplotlib.pyplot.gca()`.
    **kwargs
        Passed to :func:`matplotlib.pyplot.semilogy`.

    Returns
    -------
    ax
        The matplotlib axes containing the plot.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.
    """
    # this warning is not relevant for this plotting function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="loss iteration and iteration for")
        progress = metamer.get_progress(iteration)

    if ax is None:
        ax = plt.gca()
    ax.semilogy(metamer.losses, **kwargs)
    ax.scatter(progress["iteration"], progress["losses"], c="r")
    ax.set(xlabel="Synthesis iteration", ylabel="Loss")
    return ax


def display_metamer(
    metamer: Metamer,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    zoom: float | None = None,
    iteration: int | None = None,
    ax: mpl.axes.Axes | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Display metamer.

    We use :func:`~plenoptic.tools.display.imshow` to display the metamer and attempt to
    automatically find the most reasonable zoom value. You can override this
    value using the zoom arg, but remember that :func:`~plenoptic.tools.display.imshow`
    is opinionated about the size of the resulting image and will throw an
    exception if the axis created is not big enough for the selected zoom.

    Parameters
    ----------
    metamer
        Metamer object whose synthesized metamer we want to display.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we assume
        image is RGB(A) and show all channels.
    zoom
        How much to zoom in / enlarge the metamer, the ratio of display pixels
        to image pixels. If ``None``, we attempt to find the best
        value ourselves.
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``metamer.store_progress>1`` (that is, the metamer was not cached on every
        iteration), then we show the cached metamer from the nearest iteration.
    ax
        Pre-existing axes for plot. If ``None``, we call :func:`matplotlib.pyplot.gca`.
    **kwargs
        Passed to :func:`~plenoptic.tools.display.imshow`.

    Returns
    -------
    ax
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
        If the iteration for the displayed metamer is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``metamer.store_progress=2``).
    """
    progress = metamer.get_progress(iteration)
    try:
        image = progress["saved_metamer"]
        iter = progress["store_progress_iteration"]
    except KeyError:
        if iteration is not None:
            raise IndexError(
                "When metamer.store_progress=False, iteration must be None!"
            )
        image = metamer.metamer
        # losses will always have one extra value, the current loss.
        iter = len(metamer.losses) - 1

    if not isinstance(batch_idx, int):
        raise ValueError("batch_idx must be an integer!")
    # we're only plotting one image here, so if the user wants multiple
    # channels, they must be RGB
    as_rgb = bool(channel_idx is None and image.shape[1] > 1)
    if ax is None:
        ax = plt.gca()
    display.imshow(
        image,
        ax=ax,
        title=f"Metamer [iteration={iter}]",
        zoom=zoom,
        batch_idx=batch_idx,
        channel_idx=channel_idx,
        as_rgb=as_rgb,
        **kwargs,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax


def _representation_error(
    metamer: Metamer,
    iteration: int | None = None,
    iteration_selection: Literal["floor", "ceiling", "round"] = "round",
    **kwargs: Any,
) -> Tensor:
    r"""
    Get the representation error.

    This is ``metamer.model(metamer) - target_representation)``. If
    ``iteration`` is not ``None``, we use
    ``metamer.model(saved_metamer[iteration])`` instead.

    Parameters
    ----------
    metamer
        Metamer object whose representation error we want to compute.
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``metamer.store_progress>1`` (that is, the metamer was not cached on every
        iteration), then we show the cached metamer from the nearest iteration.
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

    **kwargs
        Passed to ``metamer.model.forward``.

    Returns
    -------
    representation_error
        The representation error at the specified iteration, for displaying.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.

    Warns
    -----
    UserWarning
        If the iteration for the used metamer is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``metamer.store_progress=2``).
    """
    if iteration is not None:
        progress = metamer.get_progress(iteration)
        image = progress["saved_metamer"].to(metamer.target_representation.device)
        metamer_rep = metamer.model(image, **kwargs)
    else:
        metamer_rep = metamer.model(metamer.metamer, **kwargs)
    return metamer_rep - metamer.target_representation


def plot_representation_error(
    metamer: Metamer,
    batch_idx: int = 0,
    iteration: int | None = None,
    ylim: tuple[float, float] | None | Literal[False] = None,
    ax: mpl.axes.Axes | None = None,
    as_rgb: bool = False,
    **kwargs: Any,
) -> list[mpl.axes.Axes]:
    r"""
    Plot distance ratio showing how close we are to convergence.

    We plot ``_representation_error(metamer, iteration)``. For more details, see
    :func:`plenoptic.tools.display.plot_representation`.

    Parameters
    ----------
    metamer
        Metamer object whose synthesized metamer we want to display.
    batch_idx
        Which index to take from the batch dimension.
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``metamer.store_progress>1`` (that is, the metamer was not cached on every
        iteration), then we show the cached metamer from the nearest iteration.
    ylim
        If ``ylim`` is ``None``, we sets the axes' y-limits to be ``(-y_max,
        y_max)``, where ``y_max=np.abs(data).max()``. If it's ``False``, we do
        nothing. If a tuple, we use that range.
    ax
        Pre-existing axes for plot. If ``None``, we call :func:`matplotlib.pyplot.gca`.
    as_rgb
        The representation can be image-like with multiple channels, and we
        have no way to determine whether it should be represented as an RGB
        image or not, so the user must set this flag to tell us. It will be
        ignored if the response doesn't look image-like or if the model has its
        own ``plot_representation_error()`` method. Else, it will be passed to
        :func:`~plenoptic.tools.display.imshow`, see that methods docstring for details.
    **kwargs
        Passed to ``metamer.model.forward``.

    Returns
    -------
    axes :
        List of created axes.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.

    Warns
    -----
    UserWarning
        If the iteration for the metamer used to compute the error is not the same as
        the argument ``iteration`` (because e.g., you set ``iteration=3`` but
        ``metamer.store_progress=2``).
    """
    representation_error = _representation_error(
        metamer=metamer, iteration=iteration, **kwargs
    )
    if ax is None:
        ax = plt.gca()
    return display.plot_representation(
        metamer.model,
        representation_error,
        ax,
        title="Representation error",
        ylim=ylim,
        batch_idx=batch_idx,
        as_rgb=as_rgb,
    )


def plot_pixel_values(
    metamer: Metamer,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    iteration: int | None = None,
    ylim: tuple[float, float] | Literal[False] = False,
    ax: mpl.axes.Axes | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    r"""
    Plot histogram of pixel values of target image and its metamer.

    As a way to check the distributions of pixel intensities and see
    if there's any values outside the allowed range

    Parameters
    ----------
    metamer
        Metamer object with the images whose pixel values we want to compare.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) images).
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``metamer.store_progress>1`` (that is, the metamer was not cached on every
        iteration), then we show the cached metamer from the nearest iteration.
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
    ax
        Created axes.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.

    Warns
    -----
    UserWarning
        If the iteration used for ``saved_metamer`` is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``metamer.store_progress=2``).
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
    progress = metamer.get_progress(iteration)
    try:
        met = progress["saved_metamer"]
        iter = progress["store_progress_iteration"]
    except KeyError:
        if iteration is not None:
            raise IndexError(
                "When metamer.store_progress=False, iteration must be None!"
            )
        met = metamer.metamer
        # losses will always have one extra value, the current loss.
        iter = len(metamer.losses) - 1
    image = metamer.image[batch_idx]
    if channel_idx is not None:
        image = image[channel_idx]
        met = met[channel_idx]
    image = data.to_numpy(image).flatten()
    met = data.to_numpy(met).flatten()

    if ax is None:
        ax = plt.gca()
    ax.hist(
        met,
        bins=min(_freedman_diaconis_bins(image), 50),
        label=f"Metamer [iteration={iter}]",
        **kwargs,
    )
    ax.hist(
        image,
        bins=min(_freedman_diaconis_bins(image), 50),
        label="Target image",
        **kwargs,
    )
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title("Histogram of pixel values")
    return ax


def _check_included_plots(to_check: list[str] | dict[str, float], to_check_name: str):
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
        "display_metamer",
        "plot_loss",
        "plot_representation_error",
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
    figsize: tuple[float, float] | None = None,
    included_plots: list[str] = [
        "display_metamer",
        "plot_loss",
        "plot_representation_error",
    ],
    display_metamer_width: float = 1,
    plot_loss_width: float = 1,
    plot_representation_error_width: float = 1,
    plot_pixel_values_width: float = 1,
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes], dict[str, int]]:
    """
    Set up figure for :func:`plot_synthesis_status`.

    Creates figure with enough axes for the all the plots you want. Will
    also create index in ``axes_idx`` for them if you haven't done so already.

    If ``fig=None``, all axes will be on the same row and have the same width.
    If you want them to be on different rows, will need to initialize ``fig``
    yourself and pass that in. For changing width, change the corresponding
    ``*_width`` arg, which gives width relative to other axes. So if you want
    the axis for the ``representation_error`` plot to be twice as wide as the
    others, set ``representation_error_width=2``.

    Parameters
    ----------
    fig
        The figure to plot on or ``None``. If ``None``, we create a new figure.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows for more
        fine-grained control of the resulting figure. Probably only helpful if fig is
        also defined. Possible keys: ``"loss"``, ``"representation_error"``,
        ``"pixel_values"``, ``"misc"``. Values should all be ints. If you tell this
        function to create a plot that doesn't have a corresponding key, we find the
        lowest int that is not already in the dict, so if you have
        axes that you want unchanged, place their idx in ``"misc"``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have relative width=1 correspond to 5.
    included_plots
        Which plots to include. Must be some subset of ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values'``.
    display_metamer_width
        Relative width of the axis for the synthesized metamer.
    plot_loss_width
        Relative width of the axis for loss plot.
    plot_representation_error_width
        Relative width of the axis for representation error plot.
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
    if "display_metamer" in included_plots:
        n_subplots += 1
        width_ratios.append(display_metamer_width)
        if "display_metamer" not in axes_idx:
            axes_idx["display_metamer"] = data._find_min_int(axes_idx.values())
    if "plot_loss" in included_plots:
        n_subplots += 1
        width_ratios.append(plot_loss_width)
        if "plot_loss" not in axes_idx:
            axes_idx["plot_loss"] = data._find_min_int(axes_idx.values())
    if "plot_representation_error" in included_plots:
        n_subplots += 1
        width_ratios.append(plot_representation_error_width)
        if "plot_representation_error" not in axes_idx:
            axes_idx["plot_representation_error"] = data._find_min_int(
                axes_idx.values()
            )
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
    metamer: Metamer,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    iteration: int | None = None,
    ylim: tuple[float, float] | None | Literal[False] = None,
    vrange: tuple[float, float] | str = "indep1",
    zoom: float | None = None,
    plot_representation_error_as_rgb: bool = False,
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float, float] | None = None,
    included_plots: list[str] = [
        "display_metamer",
        "plot_loss",
        "plot_representation_error",
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
    metamer
        Metamer object whose status we want to plot.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) image).
    iteration
        Which iteration to display. If ``None``, we show the most recent one.
        Negative values are also allowed. If ``iteration!=None`` and
        ``metamer.store_progress>1`` (that is, the metamer was not cached on every
        iteration), then we show the cached metamer from the nearest iteration.
    ylim
        The ylimit to use for the representation_error plot. We pass
        this value directly to ``plot_representation_error``.
    vrange
        The vrange option to pass to :func:`display_metamer()`. See
        docstring of :func:`~plenoptic.tools.display.imshow` for possible values.
    zoom
        How much to zoom in / enlarge the metamer, the ratio
        of display pixels to image pixels. If ``None``, we
        attempt to find the best value ourselves.
    plot_representation_error_as_rgb
        The representation can be image-like with multiple channels, and we
        have no way to determine whether it should be represented as an RGB
        image or not, so the user must set this flag to tell us. It will be
        ignored if the response doesn't look image-like or if the
        model has its own plot_representation_error() method. Else, it will
        be passed to :func:`~plenoptic.tools.display.imshow`, see that method's
        docstring for details.
    fig
        If ``None``, we create a new figure. otherwise we assume this is
        an empty figure that has the appropriate size and number of
        subplots.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values',
        'misc'``. Values should all be ints. If you tell this function to
        create a plot that doesn't have a corresponding key, we find the lowest
        int that is not already in the dict, so if you have axes that you want
        unchanged, place their idx in ``'misc'``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have each axis be of size ``(5, 5)``.
    included_plots
        Which plots to include. Must be some subset of ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values'``.
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
        If ``metamer.metamer`` object is not 3d or 4d.
    ValueError
        If the ``iteration is not None`` and the given ``metamer`` object was run
        with ``store_progress=False``.

    Warns
    -----
    UserWarning
        If the iteration used for ``saved_metamer`` is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``metamer.store_progress=2``).
    """
    if iteration is not None and not metamer.store_progress:
        raise ValueError(
            "synthesis() was run with store_progress=False, "
            "cannot specify which iteration to plot (only"
            " last one, with iteration=None)"
        )
    if metamer.metamer.ndim not in [3, 4]:
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

    def check_iterables(i: int, vals: list | tuple) -> bool:
        """
        Determine whether i is in vals.

        Works with an iterable of iterables and iterable of non-iterables.

        Parameters
        ----------
        i
            The value we're looking for.
        vals
            The iterable it might be in.

        Returns
        -------
        contained
            Whether i is in vals.
        """
        for j in vals:
            try:
                # then it's an iterable
                if i in j:
                    return True
            except TypeError:
                # then it's not an iterable
                if i == j:
                    return True

    if "display_metamer" in included_plots:
        display_metamer(
            metamer,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=iteration,
            ax=axes[axes_idx["display_metamer"]],
            zoom=zoom,
            vrange=vrange,
        )
    if "plot_loss" in included_plots:
        plot_loss(metamer, iteration=iteration, ax=axes[axes_idx["plot_loss"]])
    if "plot_representation_error" in included_plots:
        plot_representation_error(
            metamer,
            batch_idx=batch_idx,
            iteration=iteration,
            ax=axes[axes_idx["plot_representation_error"]],
            ylim=ylim,
            as_rgb=plot_representation_error_as_rgb,
        )
        # this can add a bunch of axes, so this will try and figure
        # them out
        new_axes = [
            i
            for i, _ in enumerate(fig.axes)
            if not check_iterables(i, axes_idx.values())
        ] + [axes_idx["plot_representation_error"]]
        axes_idx["plot_representation_error"] = new_axes
    if "plot_pixel_values" in included_plots:
        plot_pixel_values(
            metamer,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=iteration,
            ax=axes[axes_idx["plot_pixel_values"]],
        )
    return fig, axes_idx


def animate(
    metamer: Metamer,
    framerate: int = 10,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    ylim: str | None | tuple[float, float] | Literal[False] = None,
    vrange: tuple[float, float] | str = (0, 1),
    zoom: float | None = None,
    plot_representation_error_as_rgb: bool = False,
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float, float] | None = None,
    included_plots: list[str] = [
        "display_metamer",
        "plot_loss",
        "plot_representation_error",
    ],
    width_ratios: dict[str, float] = {},
) -> mpl.animation.FuncAnimation:
    r"""
    Animate synthesis progress.

    This is essentially the figure produced by
    ``metamer.plot_synthesis_status`` animated over time, for each stored
    iteration.

    This functions returns a matplotlib FuncAnimation object. See our
    documentation (e.g., :ref:`quickstart-nb`) for examples on how to view it in
    a Jupyter notebook. In order to save, use ``anim.save(filename)``. In either
    case, this can take a while and you'll need the appropriate writer installed
    and on your path, e.g., ffmpeg, imagemagick, etc). See
    :doc:`matplotlib documentation <matplotlib:api/animation_api>` for more details.

    Parameters
    ----------
    metamer
        Metamer object whose synthesis we want to animate.
    framerate
        How many frames a second to display.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) image).
    ylim
        The y-limits of the representation_error plot:

        * If a tuple, then this is the ylim of all plots

        * If ``None``, then all plots have the same limits, all
          symmetric about 0 with a limit of
          ``np.abs(representation_error).max()`` (for the initial
          representation_error).

        * If ``False``, don't modify limits.

        * If a string, must be ``"rescale"`` or of the form ``"rescaleN"``,
          where N can be any integer. If ``"rescaleN"``, we rescale the
          limits every N frames (we rescale as if ``ylim=None``). If
          ``"rescale"``, then we do this 10 times over the course of the
          animation.

    vrange
        The vrange option to pass to :func:`display_metamer()`. See
        docstring of :func:`~plenoptic.tools.display.imshow` for possible values.
    zoom
        How much to zoom in / enlarge the metamer, the ratio
        of display pixels to image pixels. If ``None``, we
        attempt to find the best value ourselves.
    plot_representation_error_as_rgb
        The representation can be image-like with multiple channels, and we
        have no way to determine whether it should be represented as an RGB
        image or not, so the user must set this flag to tell us. It will be
        ignored if the representation doesn't look image-like or if the
        model has its own ``plot_representation_error()`` method. Else, it will
        be passed to :func:`~plenoptic.tools.display.imshow`, see that method's
        docstring for details.
    fig
        If ``None``, create the figure from scratch. Else, should be an empty
        figure with enough axes (the expected use here is have same-size
        movies with different plots).
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values',
        'misc'``. Values should all be ints. If you tell this function to
        create a plot that doesn't have a corresponding key, we find the lowest
        int that is not already in the dict, so if you have axes that you want
        unchanged, place their idx in ``'misc'``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have each axis be of size ``(5, 5)``.
    included_plots :
        Which plots to include. Must be some subset of ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values'``.
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
        If the given ``metamer`` object was run with ``store_progress=False``.
    ValueError
        If ``metamer.metamer`` object is not 3d or 4d.
    ValueError
        If we do not know how to interpret the value of ``ylim``.

    Notes
    -----
    Unless specified, we use the ffmpeg backend, which requires that you have
    ffmpeg installed and on your path (https://ffmpeg.org/download.html). To use
    a different, use the matplotlib rcParams:
    ``matplotlib.rcParams['animation.writer'] = writer``, see `matplotlib
    documentation
    <https://matplotlib.org/stable/api/animation_api.html#writer-classes>`_ for
    more details.
    """
    if not metamer.store_progress:
        raise ValueError(
            "synthesize() was run with store_progress=False, cannot animate!"
        )
    if metamer.metamer.ndim not in [3, 4]:
        raise ValueError(
            "animate() expects 3 or 4d data; unexpected behavior will result otherwise!"
        )
    _check_included_plots(included_plots, "included_plots")
    _check_included_plots(width_ratios, "width_ratios")
    _check_included_plots(axes_idx, "axes_idx")
    if metamer.target_representation.ndimension() == 4:
        # we have to do this here so that we set the
        # ylim_rescale_interval such that we never rescale ylim
        # (rescaling ylim messes up an image axis)
        ylim = False
    try:
        if ylim.startswith("rescale"):
            try:
                ylim_rescale_interval = int(ylim.replace("rescale", ""))
            except ValueError:
                # then there's nothing we can convert to an int there
                ylim_rescale_interval = int((metamer.saved_metamer.shape[0] - 1) // 10)
                if ylim_rescale_interval == 0:
                    ylim_rescale_interval = int(metamer.saved_metamer.shape[0] - 1)
            ylim = None
        else:
            raise ValueError(f"Don't know how to handle ylim {ylim}!")
    except AttributeError:
        # this way we'll never rescale
        ylim_rescale_interval = len(metamer.saved_metamer) + 1
    # we run plot_synthesis_status to initialize the figure if either fig is
    # None or if there are no titles on any axes, which we assume means that
    # it's an empty figure
    if fig is None or not any([ax.get_title() for ax in fig.axes]):
        fig, axes_idx = plot_synthesis_status(
            metamer=metamer,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=0,
            figsize=figsize,
            ylim=ylim,
            vrange=vrange,
            zoom=zoom,
            fig=fig,
            axes_idx=axes_idx,
            included_plots=included_plots,
            plot_representation_error_as_rgb=plot_representation_error_as_rgb,
            width_ratios=width_ratios,
        )
    # grab the artist for the second plot (we don't need to do this for the
    # metamer or representation plot, because we use the update_plot
    # function for that)
    if "plot_loss" in included_plots:
        scat = fig.axes[axes_idx["plot_loss"]].collections[0]
    # can have multiple plots
    if "plot_representation_error" in included_plots:
        try:
            rep_error_axes = [
                fig.axes[i] for i in axes_idx["plot_representation_error"]
            ]
        except TypeError:
            # in this case, axes_idx['plot_representation_error'] is not iterable and
            # so is a single value
            rep_error_axes = [fig.axes[axes_idx["plot_representation_error"]]]
    else:
        rep_error_axes = []
    if "display_metamer" in included_plots:
        fig.axes[axes_idx["display_metamer"]].set_title("Metamer")

    if metamer.target_representation.ndimension() == 4:
        if "plot_representation_error" in included_plots:
            warnings.warn(
                "Looks like representation is image-like, haven't fully"
                " thought out how to best handle rescaling color ranges yet!"
            )
        # replace the bit of the title that specifies the range,
        # since we don't make any promises about that. we have to do
        # this here because we need the figure to have been created
        for ax in rep_error_axes:
            ax.set_title(re.sub(r"\n range: .* \n", "\n\n", ax.get_title()))

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
            if "display_metamer" in included_plots:
                artists.extend(
                    display.update_plot(
                        fig.axes[axes_idx["display_metamer"]],
                        data=metamer.saved_metamer[i],
                        batch_idx=batch_idx,
                    )
                )
            if "plot_representation_error" in included_plots:
                rep_error = _representation_error(
                    metamer,
                    iteration=min(i * metamer.store_progress, len(metamer.losses) - 1),
                )

                # we pass rep_error_axes to update, and we've grabbed
                # the right things above
                artists.extend(
                    display.update_plot(
                        rep_error_axes,
                        batch_idx=batch_idx,
                        model=metamer.model,
                        data=rep_error,
                    )
                )
                # again, we know that rep_error_axes contains all the axes
                # with the representation ratio info
                if (
                    (i + 1) % ylim_rescale_interval == 0
                    and metamer.target_representation.ndimension() == 3
                ):
                    display.rescale_ylim(rep_error_axes, rep_error)

            if "plot_pixel_values" in included_plots:
                # this is the dumbest way to do this, but it's simple --
                # clearing the axes can cause problems if the user has, for
                # example, changed the tick locator or formatter. not sure how
                # to handle this best right now
                fig.axes[axes_idx["plot_pixel_values"]].clear()
                plot_pixel_values(
                    metamer,
                    batch_idx=batch_idx,
                    channel_idx=channel_idx,
                    iteration=min(i * metamer.store_progress, len(metamer.losses) - 1),
                    ax=fig.axes[axes_idx["plot_pixel_values"]],
                )
            if "plot_loss" in included_plots:
                # loss always contains values from every iteration, but everything
                # else will be subsampled.
                x_val = metamer._convert_iteration(i, False)
                scat.set_offsets((x_val, metamer.losses[x_val]))
                artists.append(scat)
            # as long as blitting is True, need to return a sequence of artists
            return artists

    # don't need an init_func, since we handle initialization ourselves
    anim = mpl.animation.FuncAnimation(
        fig,
        movie_plot,
        frames=len(metamer.saved_metamer),
        blit=True,
        interval=1000.0 / framerate,
        repeat=False,
    )
    plt.close(fig)
    return anim
