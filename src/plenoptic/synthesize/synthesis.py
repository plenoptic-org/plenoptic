"""abstract synthesis super-class."""

import abc
import warnings

import numpy as np
import torch


def _get_name(x):
    """Get the name of an object, for saving/loading purposes"""
    if x is None:
        return None
    try:
        # if this passes, attr is a function
        name = f"{x.__module__}.{x.__name__}"
    except AttributeError:
        # if we're here, then it's an object
        cls = x.__class__
        name = f"{cls.__module__}.{cls.__name__}"
    return name


class Synthesis(abc.ABC):
    r"""Abstract super-class for synthesis objects.

    All synthesis objects share a variety of similarities and thus need
    to have similar methods. Some of these can be implemented here and
    simply inherited, some of them will need to be different for each
    sub-class and thus are marked as abstract methods here

    """

    @abc.abstractmethod
    def synthesize(self):
        r"""Synthesize something."""
        pass

    def save(
        self,
        file_path: str,
        save_attrs: list[str],
        save_io_attrs: list[str] = [],
        save_state_dict_attrs: list[str] = [],
    ):
        r"""Save all relevant variables in .pt file.

        If you leave attrs as None, we grab vars(self) and exclude '_model'.
        This is probably correct, but the option is provided to override it
        just in case.

        Parameters
        ----------
        file_path :
            The path to save the synthesis object to
        save_attrs :
            Names of the attributes to save directly.
        save_io_attrs :
            Names of attributes that we save as tuples of (name, inputs, outputs). On
            load, we check that the initialized object's name hasn't changed, and that
            when called on the same inputs, we get the same outputs. Intended for
            models, metrics, loss functions. Used to avoid saving callable, which is
            brittle and unsafe.
        save_state_dict_attrs :
            Names of attributes that we save as tuples of (name, state_dict).
            Corresponding attribute can be None, in which case we save an empty
            dictionary as state_dict. On load, we check that the initialized object's
            name hasn't changed, and load the state_dict. Intended for optimizers,
            schedulers. Used to avoid saving callables, which is brittle and unsafe.

        """
        save_dict = {}
        for k in save_attrs:
            if k in ["_model", "model"]:
                warnings.warn(
                    "Models can be quite large and they don't change"
                    " over synthesis. Please be sure that you "
                    "actually want to save the model."
                )
            attr = getattr(self, k)
            # detaching the tensors avoids some headaches like the
            # tensors having extra hooks or the like
            if isinstance(attr, torch.Tensor):
                attr = attr.detach()
            save_dict[k] = attr
        for k, tensors in save_io_attrs:
            attr = getattr(self, k)
            name = _get_name(attr)
            save_dict[k] = (name, tensors, attr(*tensors))
        for k in save_state_dict_attrs:
            attr = getattr(self, k)
            name = _get_name(attr)
            try:
                state_dict = attr.state_dict()
            except AttributeError:
                # then we assume that attr is None
                state_dict = {}
            save_dict[k] = (name, state_dict)
        torch.save(save_dict, file_path)

    def load(
        self,
        file_path: str,
        check_attr_for_new: str,
        map_location: str | None = None,
        weights_only: bool = True,
        check_attributes: list[str] = [],
        check_io_attributes: list[str] = [],
        state_dict_attributes: list[str] = [],
        **pickle_load_args,
    ):
        r"""Load all relevant attributes from a .pt file.

        This should be called by ``Synthesis`` object that has just been initialized.

        Note this operates in place and so doesn't return anything.

        Parameters
        ----------
        file_path :
            The path to load the synthesis object from
        check_attr_for_new :
            The name of an attribute that will either be None or have length 0 if the
            Synthesis object has just been initialized.
        map_location :
            map_location argument to pass to ``torch.load``. If you save
            stuff that was being run on a GPU and are loading onto a
            CPU, you'll need this to make sure everything lines up
            properly. This should be structured like the str you would
            pass to ``torch.device``
        weights_only :
            Indicates whether unpickler should be restricted to loading only tensors,
            primitive types, dictionaries and any types added via
            torch.serialization.add_safe_globals(). See :ref:`saveload` for more
            details.
        check_attributes :
            List of strings we ensure are identical in the current ``Synthesis`` object
            and the loaded one.
        check_io_attributes :
            Names of attributes whose input/output behavior we should check (i.e., if we
            call them on identical inputs, do we get identical outputs). In the loaded
            dictionary, these can either be callables that have been saved (if ``save``
            was called with ``save_objects=True``) or a tuple of three values: the name
            of the callable, the input to check, and the output we expect.
        state_dict_attributes :
            Names of attributes that were callables, saved as a tuple with the name of
            the callable and their state_dict. We will ensure the name of the attributes
            are identical and then load the state_dict. If the attribute is None on the
            initialized Synthesis object, then we set the tuple, and count on the
            Synthesis object to properly handle it when needed.
        pickle_load_args :
            any additional kwargs will be added to ``pickle_module.load`` via
            ``torch.load``, see that function's docstring for details.

        """
        check_attr_for_new = getattr(self, check_attr_for_new)
        if check_attr_for_new is not None and len(check_attr_for_new) > 0:
            raise ValueError(
                "load can only be called with a just-initialized"
                f" {self.__class__.__name__} object"
            )
        tmp_dict = torch.load(
            file_path,
            map_location=map_location,
            weights_only=weights_only,
            **pickle_load_args,
        )
        if map_location is not None:
            device = map_location
        else:
            for v in tmp_dict.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
        for k in check_attributes:
            # The only hidden attributes we'd check are those like
            # range_penalty_lambda, where this function is checking the
            # hidden version (which starts with '_'), but during
            # initialization, the user specifies the version without
            # the initial underscore. This is because this function
            # needs to be able to set the attribute, which can only be
            # done with the hidden version.
            display_k = k[1:] if k.startswith("_") else k
            if not hasattr(self, k):
                raise AttributeError(
                    "All values of `check_attributes` should be "
                    "attributes set at initialization, but got "
                    f"attr {display_k}!"
                )
            if isinstance(getattr(self, k), torch.Tensor):
                # there are two ways this can fail -- the first is if they're
                # the same shape but different values and the second (in the
                # except block) are if they're different shapes/dtypes.
                try:
                    if not torch.allclose(
                        getattr(self, k).to(tmp_dict[k].device),
                        tmp_dict[k],
                        rtol=5e-2,
                    ):
                        raise ValueError(
                            f"Saved and initialized {display_k} are different!"
                            f"\nSaved: {tmp_dict[k]}"
                            f"\nInitialized: {getattr(self, k)}"
                            f"\ndifference: {getattr(self, k) - tmp_dict[k]}"
                        )
                except RuntimeError as e:
                    # we end up here if dtype or shape don't match
                    if "The size of tensor a" in e.args[0]:
                        raise RuntimeError(
                            f"Attribute {display_k} have different shapes in"
                            " saved and initialized versions!"
                            f"\nSaved: {tmp_dict[k].shape}"
                            f"\nInitialized: {getattr(self, k).shape}"
                        )
                    elif "did not match" in e.args[0]:
                        raise RuntimeError(
                            f"Attribute {display_k} has different dtype in "
                            "saved and initialized versions!"
                            f"\nSaved: {tmp_dict[k].dtype}"
                            f"\nInitialized: {getattr(self, k).dtype}"
                        )
                    else:
                        raise e
            elif isinstance(getattr(self, k), float):
                if not np.allclose(getattr(self, k), tmp_dict[k]):
                    raise ValueError(
                        f"Saved and initialized {display_k} are different!"
                        f"\nSaved: {tmp_dict[k]}"
                        f"\nInitialized: {getattr(self, k)}"
                    )
            else:
                if getattr(self, k) != tmp_dict[k]:
                    raise ValueError(
                        f"Saved and initialized {display_k} are different!"
                        f"\nSaved: {tmp_dict[k]}"
                        f"\nInitialized: {getattr(self, k)}"
                    )
        for k in check_io_attributes:
            # same as above
            display_k = k[1:] if k.startswith("_") else k
            init_name = _get_name(getattr(self, k))
            try:
                saved_loss = tmp_dict[k][-1]
                error_str = "saved test"
                init_loss = getattr(self, k)(*tmp_dict[k][1])
                saved_name = tmp_dict[k][0]
            except TypeError:
                # then we saved the actual object, not its behavior, and need to do the
                # check live.
                # this way, we know it's the right shape
                tensor_a, tensor_b = torch.rand(2, *self._image_shape).to(device)
                saved_loss = tmp_dict[k](tensor_a, tensor_b)
                init_loss = getattr(self, k)(tensor_a, tensor_b)
                error_str = "two random"
                saved_name = _get_name(tmp_dict[k])
            try:
                # there are two ways this can fail -- the first is if they're
                # the same shape but different values and the second (in the
                # except block) are if they're different shapes/dtypes.
                if not torch.allclose(saved_loss, init_loss, rtol=1e-2):
                    raise ValueError(
                        f"Saved and initialized {display_k} behavior is "
                        f"different!"
                        f"\nSaved ({saved_name}) output on {error_str} tensors: "
                        f"{saved_loss}"
                        f"\nInitialized ({init_name}) output on "
                        f"{error_str} tensors: {init_loss}"
                        f"\nDifference: {init_loss-saved_loss}"
                    )
            except RuntimeError as e:
                # we end up here if dtype or shape don't match
                if "The size of tensor a" in e.args[0]:
                    raise RuntimeError(
                        f"Saved and initialized {display_k} output shape is "
                        f"different!"
                        f"\nSaved ({saved_name}) shape: {saved_loss.shape}"
                        f"\nInitialized ({init_name}) shape: {init_loss.shape}"
                    )
                elif "did not match" in e.args[0]:
                    raise RuntimeError(
                        f"Saved and initialized {display_k} output dtype is "
                        f"different!"
                        f"\nSaved ({saved_name}) dtype: {saved_loss.dtype}"
                        f"\nInitialized ({init_name}) dtype: {init_loss.dtype}"
                    )
                else:
                    raise e
        for k, v in tmp_dict.items():
            if k in check_io_attributes + state_dict_attributes:
                display_k = k[1:] if k.startswith("_") else k
                init_attr = getattr(self, k, None)
                if init_attr is not None:
                    init_name = _get_name(init_attr)
                    try:
                        saved_name = v[0]
                        if init_name != saved_name:
                            raise ValueError(
                                f"Saved and initialized {display_k} "
                                "have different names! Initialized: "
                                f"{init_name}, saved: {saved_name}"
                            )
                    except TypeError:
                        # then we don't have a name to check because we had saved the
                        # actual object
                        pass
                    if k in state_dict_attributes:
                        getattr(self, k).load_state_dict(v[1])
                    continue
                # if init_attr is None, then we haven't set it yet, so we set the saved
                # tuple as the attribute, and handle this later
            setattr(self, k, v)

    @abc.abstractmethod
    def to(self, *args, attrs: list[str] = [], **kwargs):
        r"""Moves and/or casts the parameters and buffers.
        Similar to ``save``, this is an abstract method only because you
        need to define the attributes to call to on.

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
        pinned memory to CUDA devices. When calling this method to move tensors
        to a CUDA device, items in ``attrs`` that start with "saved_" will not
        be moved.
        .. note::
            This method modifies the module in-place.
        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            attrs (:class:`list`): list of strs containing the attributes of
                this object to move to the specified device/dtype
        """
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        def move(a, k):
            move_device = None if k.startswith("saved_") else device
            if memory_format is not None and a.dim() == 4:
                return a.to(
                    move_device,
                    dtype,
                    non_blocking,
                    memory_format=memory_format,
                )
            else:
                return a.to(move_device, dtype, non_blocking)

        for k in attrs:
            if hasattr(self, k):
                attr = getattr(self, k)
                if isinstance(attr, torch.Tensor):
                    attr = move(attr.data, k)
                    if isinstance(getattr(self, k), torch.nn.Parameter):
                        attr = torch.nn.Parameter(attr)
                    if getattr(self, k).requires_grad:
                        attr = attr.requires_grad_()
                    setattr(self, k, attr)
                elif isinstance(attr, list):
                    setattr(self, k, [move(a, k) for a in attr])
                elif attr is not None:
                    setattr(self, k, move(attr, k))


class OptimizedSynthesis(Synthesis):
    r"""Abstract super-class for synthesis objects that use optimization.

    The primary difference between this and the generic Synthesis class is that
    these will use an optimizer object to iteratively update their output.

    """

    def __init__(
        self,
        range_penalty_lambda: float = 0.1,
        allowed_range: tuple[float, float] = (0, 1),
    ):
        """Initialize the properties of OptimizedSynthesis."""
        self._losses = []
        self._gradient_norm = []
        self._pixel_change_norm = []
        self._store_progress = None
        self._optimizer = None
        if range_penalty_lambda < 0:
            raise Exception("range_penalty_lambda must be non-negative!")
        self._range_penalty_lambda = range_penalty_lambda
        self._allowed_range = allowed_range

    @abc.abstractmethod
    def _initialize(self):
        r"""What to start synthesis with."""
        pass

    @abc.abstractmethod
    def objective_function(self):
        r"""How good is the current synthesized object.

        See ``plenoptic.tools.optim`` for some examples.
        """
        pass

    @abc.abstractmethod
    def _check_convergence(self):
        r"""How to determine if synthesis has finished.

        See ``plenoptic.tools.convergence`` for some examples.
        """
        pass

    def _closure(self) -> torch.Tensor:
        r"""An abstraction of the gradient calculation, before the optimization step.

        This enables optimization algorithms that perform several evaluations
        of the gradient before taking a step (ie. second order methods like
        LBFGS).

        Additionally, this is where ``loss`` is calculated and
        ``loss.backward()`` is called.

        Returns
        -------
        loss
            Loss of the current objective function

        """
        self.optimizer.zero_grad()
        loss = self.objective_function()
        loss.backward(retain_graph=False)
        return loss

    def _initialize_optimizer(
        self,
        optimizer: torch.optim.Optimizer | None,
        synth_name: str,
        learning_rate: float = 0.01,
    ):
        """Initialize optimizer.

        First time this is called, optimizer can be:

        - None, in which case we create an Adam optimizer with amsgrad=True and
          ``lr=learning_rate`` with a single parameter, the synthesis attribute

        - torch.optim.Optimizer, in which case it must already have the
          synthesis attribute (e.g., metamer) as its only parameter.

        The synthesis attribute is the one with the name ``synth_name``

        Every subsequent time (so, when resuming synthesis), optimizer must be
        None (and we use the original optimizer object).

        If we have loaded from a save state, self.optimizer will be a tuple with the
        name of the optimizer class (e.g., torch.optim.adam.Adam) and the
        optimizer's state_dict. In that case:

        - If optimizer is None, the saved optimizer must be Adam, and we load the
          state_dict.

        - else, the saved and user-specified optimizers must have the same class name
          and we load the state_dict.

        """
        synth_attr = getattr(self, synth_name)
        if optimizer is None:
            if self.optimizer is None:
                self._optimizer = torch.optim.Adam(
                    [synth_attr], lr=learning_rate, amsgrad=True
                )
            elif isinstance(self.optimizer, tuple):
                # then this comes from loading
                if self._optimizer[0] != _get_name(torch.optim.Adam):
                    raise TypeError(
                        "Don't know how to initialize saved optimizer "
                        f"'{self._optimizer[0]}'! Pass an initialized version of "
                        "this optimizer to `synthesize`, and we will update its "
                        "state_dict."
                    )
                state_dict = self.optimizer[1]
                self._optimizer = torch.optim.Adam([synth_attr])
                self._optimizer.load_state_dict(state_dict)
        else:
            if self.optimizer is not None and not isinstance(self.optimizer, tuple):
                raise TypeError("When resuming synthesis, optimizer arg must be None!")
            params = optimizer.param_groups[0]["params"]
            if len(params) != 1 or not torch.equal(params[0], synth_attr):
                # then the optimizer is not updating the right target, which means they
                # initialized it wrong.
                if not isinstance(self.optimizer, tuple):
                    raise ValueError(
                        f"For {synth_name} synthesis, optimizer must have one "
                        f"parameter, the {synth_name} we're synthesizing."
                    )
                else:
                    # a totally possible way this could happen is they initialized the
                    # synthesis object, initialized the optimizer, then called load. if
                    # you do it in that order, the optimizer target is incorrect
                    raise ValueError(
                        f"Optimizer parameter does not match self.{synth_name}. Did "
                        "you initialize this optimizer object before load? Optimizer "
                        "objects must be initialized after calling load()"
                    )
            if isinstance(self.optimizer, tuple):
                if self._optimizer[0] != _get_name(optimizer):
                    raise ValueError(
                        "User-specified optimizer must have same type as saved "
                        f"optimizer, but got: Saved: {self.optimizer[0]}, "
                        f"User-specified: {_get_name(optimizer)}."
                    )
                state_dict = self.optimizer[1]
                self._optimizer = optimizer
                self._optimizer.load_state_dict(state_dict)
            self._optimizer = optimizer

    def _initialize_scheduler(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        """Initialize scheduler.

        First time this is called, scheduler can be:

        - None, in which case we do nothing.

        - torch.optim.lr_scheduler.LRScheduler in which case its optimizer must match
          self.optimizer.

        Every subsequent time (so, when resuming synthesis), scheduler must be None (and
        we use the original scheduler object).

        If we have loaded from a save state, self.scheduler will be a tuple with the
        name of the scheduler class (e.g., torch.optim.lr_scheduler.ConstantLR) and the
        scheduler's state_dict. In that case:

        - If scheduler is None, we raise a ValueError.

        - else, the saved and user-specified schedulers must have the same class name
          and we load the state_dict.

        """
        if scheduler is None:
            if isinstance(self.scheduler, tuple):
                if self.scheduler[0] is not None:
                    # then this comes from loading, and we don't know how to initialize
                    # the scheduler
                    raise TypeError(
                        "Don't know how to initialize saved scheduler "
                        f"'{self.scheduler[0]}'! Pass an initialized version of this "
                        "scheduler to `synthesize`, and we will update its state_dict."
                    )
                else:
                    self.scheduler = None
        else:
            if isinstance(self.scheduler, tuple):
                if self.scheduler[0] != _get_name(scheduler):
                    raise ValueError(
                        "User-specified scheduler must have same type as saved "
                        f"scheduler, but got:\nSaved: {self.scheduler[0]}, "
                        f"User-specified: {_get_name(scheduler)}."
                    )
                state_dict = self.scheduler[1]
                self.scheduler = scheduler
                self.scheduler.load_state_dict(state_dict)
            elif self.scheduler is not None:
                raise TypeError("When resuming synthesis, scheduler arg must be None!")
            if self.optimizer is not scheduler.optimizer:
                raise ValueError(
                    "Scheduler's optimizer must match that of this "
                    f"{self.__class__.__name__} but got two different optimizers! Did "
                    "you call initialize scheduler before calling load()?"
                )
            self._scheduler = scheduler

    @property
    def range_penalty_lambda(self):
        return self._range_penalty_lambda

    @property
    def allowed_range(self):
        return self._allowed_range

    @property
    def losses(self):
        """Synthesis loss over iterations."""
        return torch.as_tensor(self._losses)

    @property
    def gradient_norm(self):
        """Synthesis gradient's L2 norm over iterations."""
        return torch.as_tensor(self._gradient_norm)

    @property
    def pixel_change_norm(self):
        """L2 norm change in pixel values over iterations."""
        return torch.as_tensor(self._pixel_change_norm)

    @property
    def store_progress(self):
        return self._store_progress

    @store_progress.setter
    def store_progress(self, store_progress: bool | int):
        """Initialize store_progress.

        Sets the ``self.store_progress`` attribute, as well as changing the
        ``saved_metamer`` attibute to a list so we can append to them. finally,
        adds first value to ``saved_metamer`` if it's empty.

        Parameters
        ----------
        store_progress : bool or int, optional
            Whether we should store the metamer image in progress on every
            iteration. If False, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress`` iterations
            (note then that 0 is the same as False and 1 the same as True). If
            True or int>0, ``self.saved_metamer`` contains the stored images.

        """
        if store_progress and store_progress is True:
            store_progress = 1
        if self.store_progress is not None and store_progress != self.store_progress:
            # we require store_progress to be the same because otherwise the
            # subsampling relationship between attrs that are stored every
            # iteration (loss, gradient, etc) and those that are stored every
            # store_progress iteration (e.g., saved_metamer) changes partway
            # through and that's annoying
            raise Exception(
                "If you've already run synthesize() before, must "
                "re-run it with same store_progress arg. You "
                f"passed {store_progress} instead of "
                f"{self.store_progress} (True is equivalent to 1)"
            )
        self._store_progress = store_progress

    @property
    def optimizer(self):
        return self._optimizer
