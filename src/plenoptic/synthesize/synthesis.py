"""
Abstract synthesis super-class.

Users should not interact with this file, but any concrete synthesis methods should
inherit one of these classes, to provide a unified interface.
"""

import abc
import functools
import importlib
import inspect
import math
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch

from ..tools import examine_saved_synthesis
from ..tools.data import _check_tensor_equality
from ..tools.regularization import penalize_range


def _get_name(x: object) -> str:
    """
    Get the name of an object ``x``, for saving/loading purposes.

    Parameters
    ----------
    x
        A python object or function.

    Returns
    -------
    name
        The name of that object, with full module path (e.g.,
        ``torch.optim.optimizer.Adam``).
    """  # numpydoc ignore=ES01
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
    r"""
    Abstract super-class for synthesis objects.

    All synthesis objects share a variety of similarities and thus need
    to have similar methods. Some of these can be implemented here and
    simply inherited, some of them will need to be different for each
    sub-class and thus are marked as abstract methods here.
    """

    def __init__(self):
        # flag to raise more informative error message if setup and load are both called
        self._loaded = False

    @abc.abstractmethod
    def synthesize(self):
        r"""Synthesize something."""  # numpydoc ignore=ES01
        pass

    def save(
        self,
        file_path: str,
        save_io_attrs: list[tuple[str]] = [],
        save_state_dict_attrs: list[str] = [],
    ):
        r"""
        Save all attributes in .pt file.

        Note that there are two special categories of attributes, as described below.
        All other attributes will be pickled directly and so should be either tensors or
        primitives (e.g., not a function or callable torch object). We do not check this
        explicitly, but load will fail if that's not the case.

        Parameters
        ----------
        file_path
            The path to save the synthesis object to.
        save_io_attrs
            List with tuples of form (str, (str, ...)). The first element is the name of
            the attribute to save, and the second element is a tuple of attributes of
            the Synthesis object, which we can pass as inputs to the attribute. We save
            them as tuples of (name, input_names, outputs). On load, we check that the
            initialized object's name hasn't changed, and that when called on the same
            inputs, we get the same outputs. Intended for models, metrics, loss
            functions. Used to avoid saving callables, which is brittle and unsafe.
        save_state_dict_attrs
            Names of attributes that we save as tuples of (name, state_dict).
            Corresponding attribute can be None, in which case we save an empty
            dictionary as state_dict. On load, we check that the initialized object's
            name hasn't changed, and load the state_dict. Intended for optimizers,
            schedulers. Used to avoid saving callables, which is brittle and unsafe.

        Raises
        ------
        ValueError
            If any of the strings specified in ``save_io_attrs`` as inputs for the
            callable attribute-to-save are not attributes of ``self``.
        """
        save_dict = {}
        save_dict["save_metadata"] = {
            "plenoptic_version": importlib.metadata.version("plenoptic"),
            "torch_version": importlib.metadata.version("torch"),
            "synthesis_object": _get_name(self),
        }
        save_io_attr_names = [k[0] for k in save_io_attrs]
        save_attrs = [
            k for k in vars(self) if k not in save_io_attr_names + save_state_dict_attrs
        ]
        for k in save_attrs:
            attr = getattr(self, k)
            # detaching the tensors avoids some headaches like the
            # tensors having extra hooks or the like
            if isinstance(attr, torch.Tensor):
                attr = attr.detach()
            save_dict[k] = attr
        for k, input_names in save_io_attrs:
            attr = getattr(self, k)
            name = _get_name(attr)
            tensors = [getattr(self, t) for t in input_names]
            save_dict[k] = (name, input_names, attr(*tensors))
            if any([n not in save_dict for n in input_names]):
                raise ValueError(
                    "input_name must be included in save dictionary, "
                    f"but got {input_names}!"
                )
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
        empty_on_init_attr: str,
        map_location: str | None = None,
        check_attributes: list[str] = [],
        check_io_attributes: list[str] = [],
        state_dict_attributes: list[str] = [],
        tensor_equality_atol: float = 1e-8,
        tensor_equality_rtol: float = 1e-5,
        **pickle_load_args: Any,
    ):
        r"""
        Load all relevant attributes from a .pt file.

        This should be called by ``Synthesis`` object that has just been initialized.

        Note this operates in place and so doesn't return anything.

        Parameters
        ----------
        file_path
            The path to load the synthesis object from.
        empty_on_init_attr
            The name of an attribute that will either be None or have length 0 if the
            Synthesis object has just been initialized.
        map_location
            Argument to pass to :func:`torch.load` as ``map_location``. If you save
            stuff that was being run on a GPU and are loading onto a
            CPU, you'll need this to make sure everything lines up
            properly. This should be structured like the str you would
            pass to :class:`torch.device`.
        check_attributes
            List of strings we ensure are identical in the current ``Synthesis`` object
            and the loaded one.
        check_io_attributes
            Names of attributes whose input/output behavior we should check (i.e., if we
            call them on identical inputs, do we get identical outputs). In the loaded
            dictionary, these are a tuple of three values: the name of the callable, the
            name of the attribute to use as input, and the output we expect.
        state_dict_attributes
            Names of attributes that were callables, saved as a tuple with the name of
            the callable and their state_dict. We will ensure the name of the attributes
            are identical and then load the state_dict. If the attribute is None on the
            initialized Synthesis object, then we set the tuple, and count on the
            Synthesis object to properly handle it when needed.
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
            If the loading object has not just been initialized.
        ValueError
            If the object saved at ``file_path`` is not the same type as the loading
            object.
        ValueError
            If either the saved or loading object has attributes not found in the
            other.
        ValueError
            If the saved and loading objects have a different value for one of the
            ``check_attributes``.
        ValueError
            If the behavior of one of the ``check_io_attributes`` is different between
            the saved and loading objects.

        Warns
        -----
        UserWarning
            If :func:`setup` will need to be called after load, to finish initializing
            one of the ``state_dict_attributes``
        """
        check_str = (
            "\n\nIf this is confusing, try calling "
            f"{_get_name(examine_saved_synthesis)}('{file_path}'),"
            " to examine saved object"
        )
        empty_on_init_attr = getattr(self, empty_on_init_attr)
        if empty_on_init_attr is not None and len(empty_on_init_attr) > 0:
            raise ValueError(
                "load can only be called with a just-initialized"
                f" {self.__class__.__name__} object"
            )
        tmp_dict = torch.load(
            file_path,
            map_location=map_location,
            **pickle_load_args,
        )
        metadata = tmp_dict.pop("save_metadata")
        if metadata["synthesis_object"] != _get_name(self):
            raise ValueError(
                f"Saved object was a {metadata['synthesis_object']}"
                f", but initialized object is {_get_name(self)}! "
                f"{check_str}"
            )
        # all attributes set at initialization should be present in the saved dictionary tmp_dict
        init_not_save = set(vars(self)) - set(tmp_dict)
        if len(init_not_save):
            compat_attrs = {"_current_loss", "penalty_function", "_penalty_lambda"}
            if not init_not_save <= compat_attrs:
                init_not_save_str = "\n ".join(
                    [f"{k}: {getattr(self, k)}" for k in init_not_save]
                )
                raise ValueError(
                    f"Initialized object has {len(init_not_save)} attribute(s) "
                    f"not present in the saved object!\n {init_not_save_str}"
                )
            if "_current_loss" in init_not_save:
                # in PR #370 (release 1.3.1), added _current_loss attribute, which we'll
                # handle for now, but warn about.
                tmp_dict["_current_loss"] = None
                warnings.warn(
                    "The saved object was saved with plenoptic 1.3.0 or earlier and "
                    "will not be compatible with future releases. Save this object "
                    "with current version of plenoptic or see the 'Reproducibility "
                    "and Compatibility' page of the documentation for how to make the "
                    "saved object futureproof and avoid this warning.",
                    category=FutureWarning,
                )
            penalty_missing = {"penalty_function", "_penalty_lambda"} & init_not_save
            if penalty_missing:
                # in PR #383, we added penalty_function and penalty_lambda attributes,
                # which we'll handle for now, but warn about.
                # Remove allowed_range and range_penalty_lambda so there's no extra key
                # in saved dictionary
                allowed_range = tmp_dict.pop(
                    "_allowed_range", tmp_dict.pop("allowed_range", None)
                )
                penalty_fn = functools.partial(
                    penalize_range, allowed_range=allowed_range
                )
                tmp_dict["penalty_function"] = (
                    _get_name(penalty_fn),
                    ("_image",),
                    penalty_fn(tmp_dict["_image"]),
                )
                range_penalty_lambda = tmp_dict.pop(
                    "_range_penalty_lambda",
                )
                tmp_dict["_penalty_lambda"] = range_penalty_lambda
                warnings.warn(
                    "The saved object was saved before penalty_function and "
                    "penalty_lambda existed and will not be compatible with future "
                    "releases. Save this object with the current version of plenoptic "
                    "or see the 'Reproducibility and Compatibility' page of the "
                    "documentation for how to make the saved object futureproof and "
                    "avoid this warning.",
                    category=FutureWarning,
                )
        # there shouldn't be any extra keys in the saved dictionary (we removed
        # save_metadata above)
        save_not_init = set(tmp_dict) - set(vars(self))
        if len(save_not_init):
            save_not_init_str = "\n ".join(
                [f"{k}: {tmp_dict[k]}" for k in save_not_init]
            )
            raise ValueError(
                f"Saved object has {len(save_not_init)} attribute(s) "
                f"not present in the initialized object!\n {save_not_init_str}"
            )
        for k in check_attributes:
            # The only hidden attributes we'd check are those like
            # penalty_lambda, where this function is checking the
            # hidden version (which starts with '_'), but during
            # initialization, the user specifies the version without
            # the initial underscore. This is because this function
            # needs to be able to set the attribute, which can only be
            # done with the hidden version.
            display_k = k[1:] if k.startswith("_") else k
            if isinstance(getattr(self, k), torch.Tensor):
                _check_tensor_equality(
                    tmp_dict[k],
                    getattr(self, k),
                    "Saved",
                    "Initialized",
                    error_prepend_str=(
                        f"Saved and initialized attribute {display_k} have "
                        f"different {{error_type}}!"
                    ),
                    error_append_str=check_str,
                    atol=tensor_equality_atol,
                    rtol=tensor_equality_rtol,
                )
            elif isinstance(getattr(self, k), float):
                if not np.allclose(getattr(self, k), tmp_dict[k]):
                    raise ValueError(
                        f"Saved and initialized {display_k} are different!"
                        f"\nSaved: {tmp_dict[k]}"
                        f"\nInitialized: {getattr(self, k)}"
                        f"{check_str}"
                    )
            else:
                if getattr(self, k) != tmp_dict[k]:
                    raise ValueError(
                        f"Saved and initialized {display_k} are different!"
                        f"\nSaved: {tmp_dict[k]}"
                        f"\nInitialized: {getattr(self, k)}"
                        f"{check_str}"
                    )
        for k, input_names in check_io_attributes:
            # same as above
            display_k = k[1:] if k.startswith("_") else k
            tensors = [tmp_dict[t] for t in tmp_dict[k][1]]
            init_name = _get_name(getattr(self, k))
            saved_name = tmp_dict[k][0]
            init_loss = getattr(self, k)(*tensors)
            saved_loss = tmp_dict[k][-1]
            _check_tensor_equality(
                saved_loss,
                init_loss,
                f"Saved ({saved_name})",
                f"Initialized ({init_name})",
                error_prepend_str=(
                    f"Saved and initialized {display_k} output have "
                    f"different {{error_type}}!"
                ),
                error_append_str=check_str,
                atol=tensor_equality_atol,
                rtol=tensor_equality_rtol,
            )
        for k, v in tmp_dict.items():
            # check_io_attributes is a tuple
            if k in [a[0] for a in check_io_attributes] + state_dict_attributes:
                display_k = k[1:] if k.startswith("_") else k
                init_attr = getattr(self, k, None)
                # then check their name has the same final part (e.g.,
                # "plenoptic.simulate.PortillaSimoncelli" or
                # "__main__.PortillaSimoncelli"), since we've already checked
                # the behavior, keep going (don't update the object's attribute)
                if init_attr is not None:
                    init_name = _get_name(init_attr)
                    saved_name = v[0]
                    init_name = init_name.split(".")[-1]
                    saved_name = saved_name.split(".")[-1]
                    if init_name != saved_name:
                        raise ValueError(
                            f"Saved and initialized {display_k} "
                            "have different names!"
                            f"\nSaved: {saved_name}"
                            f"\nInitialized: {init_name}"
                            f"{check_str}"
                        )
                    continue
                # if init_attr is None, then we haven't set it yet, so we set the saved
                # tuple as the attribute, and handle this later
            setattr(self, k, v)
        setup_attrs = []
        optim = tmp_dict.get("_optimizer", None)
        if isinstance(optim, tuple) and optim[0] != _get_name(torch.optim.Adam):
            setup_attrs.append("optimizer")
        sched = tmp_dict.get("_scheduler", None)
        if isinstance(sched, tuple) and sched[0] is not None:
            setup_attrs.append("scheduler")
        if setup_attrs:
            warnings.warn(
                f"You will need to call setup() to instantiate {', '.join(setup_attrs)}"
            )
        # Make sure we specify that we have loaded and setup has not been called
        self._loaded = True

    @abc.abstractmethod
    def to(self, *args: Any, attrs: list[str] = [], **kwargs: Any):
        r"""
        Move and/or cast the parameters and buffers.

        Similar to :func:`save`, this is an abstract method only because you
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
        pinned memory to CUDA devices.

        When calling this method to move tensors to a CUDA device, items in ``attrs``
        that start with ``"saved_"`` will not be moved.

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
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        def move(a: torch.Tensor, k: str) -> torch.Tensor:
            # numpydoc ignore=RT01,ES01,PR01,GL08
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
    r"""
    Abstract super-class for synthesis objects that use optimization.

    The primary difference between this and the generic Synthesis class is that
    these will use an optimizer object to iteratively update their output.

    Parameters
    ----------
    penalty_function
        A penalty function to help constrain the synthesized
        image by penalizing specific image properties.
    penalty_lambda
        Strength of the regularizer. Must be non-negative.
    """

    def __init__(
        self,
        penalty_function: Callable[[torch.Tensor], torch.Tensor] = penalize_range,
        penalty_lambda: float = 0.1,
    ):
        super().__init__()
        self._losses = []
        self._gradient_norm = []
        self._pixel_change_norm = []
        self._store_progress = None
        self._optimizer = None
        self._current_loss = None
        self.penalty_function = penalty_function
        if penalty_lambda < 0:
            raise Exception("penalty_lambda must be non-negative!")
        self._penalty_lambda = penalty_lambda

    @abc.abstractmethod
    def setup(self):
        r"""Initialize relevant attributes."""  # numpydoc ignore=ES01
        pass

    @abc.abstractmethod
    def objective_function(self):
        r"""
        How good is the current synthesized object.

        See ``plenoptic.tools.optim`` for some examples.
        """
        pass

    @abc.abstractmethod
    def _check_convergence(self):
        r"""
        How to determine if synthesis has finished.

        See ``plenoptic.tools.convergence`` for some examples.
        """
        pass

    def _closure(self) -> torch.Tensor:
        r"""
        Calculate the gradient, before the optimization step.

        This enables optimization algorithms that perform several evaluations
        of the gradient before taking a step (ie. second order methods like
        LBFGS).

        Additionally, this is where ``loss`` is calculated and
        ``loss.backward()`` is called.

        Returns
        -------
        loss
            Loss of the current objective function.
        """
        self.optimizer.zero_grad()
        loss = self.objective_function()
        loss.backward(retain_graph=False)
        return loss.item()

    def _initialize_optimizer(
        self,
        optimizer: torch.optim.Optimizer | None,
        synth_attr: torch.Tensor,
        optimizer_kwargs: dict | None = None,
        learning_rate: float = 0.01,
    ):
        """
        Initialize optimizer.

        optimizer can be:

        - None, in which case we create an Adam optimizer with amsgrad=True and
          ``lr=learning_rate``

        - uninitialized torch.optim.Optimizer, in which case it must already have the
          synthesis attribute (e.g., metamer) as its only parameter.

        In either case, we then initialize the optimizer with a single parameter, the
        synthesis attribute, ``synth_attr``

        If we have loaded from a save state, self.optimizer will be a tuple with the
        name of the optimizer class (e.g., torch.optim.adam.Adam) and the
        optimizer's state_dict. In that case:

        - If optimizer is None, the saved optimizer must be Adam, and we load the
          state_dict.

        - else, the saved and user-specified optimizers must have the same class name
          and we load the state_dict. in this case, ``optimizer_kwargs`` must be None

        Parameters
        ----------
        optimizer
            The (un-initialized) optimizer object to use. If ``None``, we use
            :class:`torch.optim.Adam`.
        synth_attr
            The Tensor that we are updating as part of optimization.
        optimizer_kwargs
            The keyword arguments to pass to the optimizer on initialization. If
            ``None``, we use ``{"lr": .01}`` and, if optimizer is ``None``,
            ``{"amsgrad": True}``.
        learning_rate
            The learning rate for the optimizer.

        Raises
        ------
        ValueError
            If ``optimizer_kwargs`` is not ``None`` and ``self.optimizer`` is a tuple
            (thus, :meth:`load` was called before this).
        TypeError
            If ``self.optimizer`` is a tuple (thus, :meth:`load` was called before
            this), was not :class`torch.optim.Adam`, and ``optimizer`` arg is ``None``.
        ValueError
            If ``self.optimizer`` is a tuple (thus, :meth:`load` was called before this)
            but ``optimizer`` arg has a different type than the saved optimizer.
        """
        if isinstance(self.optimizer, tuple):
            # then we're calling this after load()
            if optimizer_kwargs is not None:
                raise ValueError(
                    "When initializing optimizer after load, optimizer_kwargs"
                    " must be None!"
                )
            if optimizer is None:
                optimizer = torch.optim.Adam
                err_str = (
                    "Don't know how to initialize saved optimizer "
                    f"'{self._optimizer[0]}'! Pass an un-initialized version of "
                    "this optimizer to `setup`, and we will update its "
                    "state_dict."
                )
            else:
                err_str = (
                    "User-specified optimizer must have same type as saved "
                    "optimizer, but got:"
                    f"\nSaved: {self.optimizer[0]}, "
                    f"\nUser-specified: {_get_name(optimizer)}."
                )

            if self._optimizer[0] != _get_name(optimizer):
                raise ValueError(err_str)
            state_dict = self.optimizer[1]
            self._optimizer = optimizer([synth_attr])
            self._optimizer.load_state_dict(state_dict)
        else:
            if optimizer_kwargs is None:
                optimizer_kwargs = {"lr": learning_rate}
            else:
                optimizer_kwargs.setdefault("lr", learning_rate)
            if optimizer is None:
                optimizer_kwargs.setdefault("amsgrad", True)
                self._optimizer = torch.optim.Adam([synth_attr], **optimizer_kwargs)
            else:
                self._optimizer = optimizer([synth_attr], **optimizer_kwargs)

    def _initialize_scheduler(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        optimizer: torch.optim.Optimizer,
        scheduler_kwargs: dict | None = None,
    ):
        """
        Initialize scheduler.

        scheduler can be:

        - None, in which case we do nothing.

        - uninitialized torch.optim.lr_scheduler.LRScheduler in which case we initialize
          it using self.optimizer.

        If we have loaded from a save state, self.scheduler will be a tuple with the
        name of the scheduler class (e.g., torch.optim.lr_scheduler.ConstantLR) and the
        scheduler's state_dict. In that case:

        - If scheduler is None, we raise a ValueError.

        - else, the saved and user-specified schedulers must have the same class name
          and we load the state_dict. In this case, ``scheduler_kwargs`` must be None

        We also check if scheduler.step takes any arguments beyond self and epoch. If
        so, we set the `self._scheduler_step_arg` flag to True (it was set to False at
        object initialization). Then, we will pass the loss to scheduler.step when we
        call it.

        Parameters
        ----------
        scheduler
            The un-initialized learning rate scheduler object to use. If ``None``, we
            don't use one.
        optimizer
            The initialized optimizer whose learning rate ``scheduler`` will be
            adjusting.
        scheduler_kwargs
            The keyword arguments to pass to the scheduler on initialization.

        Raises
        ------
        ValueError
            If ``scheduler_kwargs`` is not ``None`` and ``self.scheduler`` is a tuple
            (thus, :meth:`load` was called before this).
        TypeError
            If ``self.scheduler`` is a tuple (thus, :meth:`load` was called before this)
            but ``scheduler`` arg is ``None``.
        ValueError
            If ``self.scheduler`` is a tuple (thus, :meth:`load` was called before this)
            but ``scheduler`` arg has a different type than the saved scheduler.
        """
        if isinstance(self.scheduler, tuple):
            # then we're calling this after load()
            if scheduler_kwargs is not None:
                raise ValueError(
                    "When initializing scheduler after load, scheduler_kwargs"
                    " must be None!"
                )
            if scheduler is None:
                if self.scheduler[0] is not None:
                    # then this comes from loading, and we don't know how to initialize
                    # the scheduler
                    raise TypeError(
                        "Don't know how to initialize saved scheduler "
                        f"'{self.scheduler[0]}'! Pass an initialized version of this "
                        "scheduler to `synthesize`, and we will update its state_dict."
                    )
                self._scheduler = None
            else:
                if self.scheduler[0] != _get_name(scheduler):
                    raise ValueError(
                        "User-specified scheduler must have same type as saved "
                        f"scheduler, but got:\nSaved: {self.scheduler[0]}, "
                        f"\nUser-specified: {_get_name(scheduler)}."
                    )
                state_dict = self.scheduler[1]
                self._scheduler = scheduler(optimizer)
                self._scheduler.load_state_dict(state_dict)
        else:
            if scheduler_kwargs is None:
                scheduler_kwargs = {}
            if scheduler is not None:
                self._scheduler = scheduler(optimizer, **scheduler_kwargs)
                step_args = set(inspect.getfullargspec(self.scheduler.step).args)
                if len(step_args - {"self", "epoch"}):
                    # then we do want to pass the loss to scheduler.step
                    self._scheduler_step_arg = True

    def _convert_iteration(
        self,
        iteration: int,
        iteration_to_progress: bool = True,
        iteration_selection: Literal["floor", "ceiling", "round"] = "round",
    ) -> int:
        """
        Convert between synthesis iteration and ``store_progress``'s iteration.

        Several of the ``OptimizedSynthesis`` attributes are not updated every
        single iteration but every ``self.store_progress`` iterations. This
        converts between the two.

        Parameters
        ----------
        iteration
            Synthesis iteration to summarize.
        iteration_to_progress
            Whether to convert from synthesis iteration to ``store_progress``
            iteration (in which case, behavior is controlled by
            ``iteration_selection``) or vice-versa (in which case we return
            ``iteration * self.store_progress``).
        iteration_selection

            How to select the relevant iteration from the saved synthesis attribute
            when the request iteration wasn't stored.

            When synthesis was run with ``store_progress=n`` (where ``n>1``),
            synthesis outputs are only saved every ``n`` iterations. If you request an
            iteration where synthesis wasn't saved, this determines which available
            iteration is used instead:

            * ``"floor"``: use the closest saved iteration **before** the
              requested one.

            * ``"ceiling"``: use the closest saved iteration **after** the
              requested one.

            * ``"round"``: use the closest saved iteration.

        Returns
        -------
        converted_iteration
            Converted iteration.
        """
        if iteration_to_progress:
            # round and ceiling may be one greater than e.g., len(self._saved_metamer).
            # however, self.saved_metamer always has the current metamer appended, and
            # so this will be okay
            if iteration_selection == "floor":
                iter = math.floor(iteration / self.store_progress)
            elif iteration_selection == "round":
                iter = round(iteration / self.store_progress)
            elif iteration_selection == "ceiling":
                iter = math.ceil(iteration / self.store_progress)
        else:
            iter = iteration * self.store_progress
            # this might go off the end
            if iter >= len(self.losses):
                iter = len(self.losses) - 1
        return iter

    @abc.abstractmethod
    def get_progress(
        self,
        iteration: int | None,
        iteration_selection: Literal["floor", "ceiling", "round"] = "round",
        addt_every_iter_attributes: list[str] = [],
        store_progress_attributes: list[str] = [],
    ) -> dict[str, torch.Tensor | None | int]:
        """
        Return dictionary summarizing synthesis progress at ``iteration``.

        Note that for the most recent iteration (``iteration=-1`` or ``iteration=None``
        or ``iteration==len(self.losses)-1``), we do not have values for
        :attr:`pixel_change_norm`, :attr:`gradient_norm` or
        ``addt_every_iter_attributes``, since in this case we are showing the loss and
        value for the current synthesis output.

        Parameters
        ----------
        iteration
            Synthesis iteration to summarize. If ``None``, grab the most recent.
            Negative values are allowed.
        iteration_selection

            How to select the relevant iteration from the saved synthesis attribute
            when the request iteration wasn't stored.

            When synthesis was run with ``store_progress=n`` (where ``n>1``),
            synthesis outputs are only saved every ``n`` iterations. If you request an
            iteration where synthesis wasn't saved, this determines which available
            iteration is used instead:

            * ``"floor"``: use the closest saved iteration **before** the
              requested one.

            * ``"ceiling"``: use the closest saved iteration **after** the
              requested one.

            * ``"round"``: use the closest saved iteration.

        addt_every_iter_attributes
            Additional attributes that have values appended every iteration.
            ``losses``, ``pixel_change_norm`` and ``gradient_norm`` are always
            included.
        store_progress_attributes
            Attributes that have values appended every ``self.store_progress``
            iterations (and may thus be ``None``).

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
            If the iteration used for ``store_progress_attributes`` is not the same as
            the argument ``iteration`` (because e.g., you set ``iteration=3`` but
            ``self.store_progress=2``).
        """
        if iteration is None:
            iter = len(self.losses) - 1
        elif iteration < 0:
            iter = len(self.losses) + iteration
        else:
            iter = iteration

        if iter < 0:
            # if this is negative after our remapping above, then it was too large
            # and it wrapped around again (e.g., -100 when there were only 90
            # iterations)
            raise IndexError(
                f"{iteration=} out of bounds with "
                f"{len(self.losses)} iterations of synthesis"
            )

        # len(self.losses) is the number of synthesis iterations plus 1 (for the current
        # loss), so we grab the hidden version, which has the proper length
        try:
            loss = self.losses[iter]
        except IndexError as e:
            raise IndexError(
                f"{iteration=} out of bounds with "
                f"{len(self.losses)} iterations of synthesis"
            ) from e
        progress_info = {"losses": loss, "iteration": iter}
        # then this is the most recent one, which we don't have pixel_change_norm or
        # gradient_norm for
        if iter == len(self.losses) - 1:
            progress_info.update({"pixel_change_norm": None, "gradient_norm": None})
        else:
            progress_info.update(
                {
                    "pixel_change_norm": self.pixel_change_norm[iter],
                    "gradient_norm": self.gradient_norm[iter],
                }
            )
        progress_info.update(
            {k: getattr(self, k)[iter] for k in addt_every_iter_attributes}
        )
        if self.store_progress:
            # treat None special: always grab current one
            if iteration is None:
                store_progress_iter = -1
            else:
                store_progress_iter = self._convert_iteration(
                    iter, iteration_selection=iteration_selection
                )
            progress_info.update(
                {
                    k: getattr(self, k)[store_progress_iter]
                    for k in store_progress_attributes
                }
            )
            # treat None special: always grab current one
            if iteration is None:
                store_progress_iter = iter
            else:
                store_progress_iter = self._convert_iteration(
                    store_progress_iter, False, iteration_selection
                )
            if store_progress_iter != iter:
                warnings.warn(
                    f"loss iteration and iteration for {store_progress_attributes} are"
                    " not the same"
                )
            progress_info.update({"store_progress_iteration": store_progress_iter})
        return progress_info

    @property
    def penalty_lambda(self) -> float:
        """Magnitude of the regularization weight."""
        # numpydoc ignore=RT01,ES01
        return self._penalty_lambda

    @property
    def losses(self) -> torch.Tensor:
        """
        Optimization loss over iterations.

        Will have ``length=num_iter+1``, where ``num_iter`` is the number of
        iterations of synthesis run so far.

        This tensor always lives on the CPU.
        """  # numpydoc ignore=RT01
        current_loss = self._current_loss
        # this will happen if we haven't run synthesize() yet or got
        # interrupted
        if current_loss is None:
            try:
                # compute current loss, no need to compute gradient
                with torch.no_grad():
                    current_loss = self.objective_function().item()
            except RuntimeError as e:
                exp_msg = "a Tensor with 0 elements cannot be converted to Scalar"
                if e.args[0] != exp_msg:
                    raise e
                # this will happen if setup() has not been called and so we can't
                # compute loss because synthesis hasn't been initialized.
                return torch.empty(0)
        return torch.as_tensor([*self._losses, current_loss])

    @property
    def gradient_norm(self) -> torch.Tensor:
        """Optimization gradient's L2 norm over iterations."""
        # numpydoc ignore=RT01,ES01
        return torch.as_tensor(self._gradient_norm)

    @property
    def pixel_change_norm(self) -> torch.Tensor:
        """L2 norm change in pixel values over iterations."""
        # numpydoc ignore=RT01,ES01
        return torch.as_tensor(self._pixel_change_norm)

    @property
    def store_progress(self) -> bool | int:
        """
        How often we are caching progress.

        If ``False``, we don't save anything. If ``True``, we save every iteration. If
        an int, we save every ``store_progress`` iterations (note then that 0 is the
        same as ``False`` and 1 the same as ``True``).
        """  # numpydoc ignore=RT01
        return self._store_progress

    @store_progress.setter
    def store_progress(self, store_progress: bool | int):
        """
        Initialize store_progress.

        Parameters
        ----------
        store_progress : bool or int, optional
            Whether we should store the synthesis output in progress on every
            iteration. If ``False``, we don't save anything. If ``True``, we save
            every iteration. If an int, we save every ``store_progress`` iterations
            (note then that 0 is the same as ``False`` and 1 the same as ``True``).

        Raises
        ------
        ValueError
            If ``store_progress`` has already been set and you are trying to change the
            value.
        """
        # numpydoc ignore=RT01,ES01
        if store_progress and store_progress is True:
            store_progress = 1
        if self.store_progress is not None and store_progress != self.store_progress:
            # we require store_progress to be the same because otherwise the
            # subsampling relationship between attrs that are stored every
            # iteration (loss, gradient, etc) and those that are stored every
            # store_progress iteration (e.g., saved_metamer) changes partway
            # through and that's annoying
            raise ValueError(
                "If you've already run synthesize() before, must "
                "re-run it with same store_progress arg. You "
                f"passed {store_progress} instead of "
                f"{self.store_progress} (True is equivalent to 1)"
            )
        self._store_progress = store_progress

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Torch optimizer object which updates the synthesis target."""
        # numpydoc ignore=RT01,ES01
        return self._optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Learning rate scheduler which adjusts optimizer learning rate."""
        # numpydoc ignore=RT01,ES01
        return self._scheduler
