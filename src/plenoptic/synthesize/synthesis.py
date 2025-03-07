"""abstract synthesis super-class."""

import abc
import importlib
import inspect

import numpy as np
import torch

from ..tools import examine_saved_synthesis
from ..tools.data import _check_tensor_equality


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
        save_io_attrs: list[tuple[str]] = [],
        save_state_dict_attrs: list[str] = [],
    ):
        r"""Save all attributes in .pt file.

        Note that there are two special categories of attributes, as described below.
        All other attributes will be pickled directly and so should be either tensors or
        primitives (e.g., not a function or callable torch object). We do not check this
        explicitly, but load will fail if that's not the case.

        Parameters
        ----------
        file_path :
            The path to save the synthesis object to
        save_io_attrs :
            List with tuples of form (str, (str, ...)). The first element is the name of
            the attribute to we save, and the second element is a tuple of attributes of
            the Synthesis object, which we can pass as inputs to the attribute. We save
            them as tuples of (name, input_names, outputs). On load, we check that the
            initialized object's name hasn't changed, and that when called on the same
            inputs, we get the same outputs. Intended for models, metrics, loss
            functions. Used to avoid saving callable, which is brittle and unsafe.
        save_state_dict_attrs :
            Names of attributes that we save as tuples of (name, state_dict).
            Corresponding attribute can be None, in which case we save an empty
            dictionary as state_dict. On load, we check that the initialized object's
            name hasn't changed, and load the state_dict. Intended for optimizers,
            schedulers. Used to avoid saving callables, which is brittle and unsafe.

        """
        save_dict = {}
        save_dict["save_metadata"] = {
            "plenoptic_version": importlib.metadata.version("plenoptic"),
            "torch_version": importlib.metadata.version("torch"),
            "synthesis_object": _get_name(self),
        }
        save_attrs = [
            k
            for k in vars(self)
            if k not in [k[0] for k in save_io_attrs] + save_state_dict_attrs
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
        **pickle_load_args,
    ):
        r"""Load all relevant attributes from a .pt file.

        This should be called by ``Synthesis`` object that has just been initialized.

        Note this operates in place and so doesn't return anything.

        Parameters
        ----------
        file_path :
            The path to load the synthesis object from
        empty_on_init_attr :
            The name of an attribute that will either be None or have length 0 if the
            Synthesis object has just been initialized.
        map_location :
            map_location argument to pass to ``torch.load``. If you save
            stuff that was being run on a GPU and are loading onto a
            CPU, you'll need this to make sure everything lines up
            properly. This should be structured like the str you would
            pass to ``torch.device``
        check_attributes :
            List of strings we ensure are identical in the current ``Synthesis`` object
            and the loaded one.
        check_io_attributes :
            Names of attributes whose input/output behavior we should check (i.e., if we
            call them on identical inputs, do we get identical outputs). In the loaded
            dictionary, these are a tuple of three values: the name of the callable, the
            name of the attribute to use as input, and the output we expect.
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
        empty_on_init_attr = getattr(self, empty_on_init_attr)
        check_str = (
            "\n\nIf this is confusing, try calling "
            f"{_get_name(examine_saved_synthesis)}('{file_path}'),"
            " to examine saved object"
        )
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
        # all attributes set at initialization should be present in the saved dictionary
        init_not_save = set(vars(self)) - set(tmp_dict)
        if len(init_not_save):
            init_not_save_str = "\n ".join(
                [f"{k}: {getattr(self,k)}" for k in init_not_save]
            )
            raise ValueError(
                f"Initialized object has {len(init_not_save)} attribute(s) "
                f"not present in the saved object!\n {init_not_save_str}"
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
            # range_penalty_lambda, where this function is checking the
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
                    rtol=5e-2,
                    error_prepend_str=(
                        f"Saved and initialized attribute {display_k} have "
                        f"different {{error_type}}!"
                    ),
                    error_append_str=check_str,
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
                rtol=1e-2,
                error_prepend_str=(
                    f"Saved and initialized {display_k} output have "
                    f"different {{error_type}}!"
                ),
                error_append_str=check_str,
            )
        for k, v in tmp_dict.items():
            # check_io_attributes is a tuple
            if k in [a[0] for a in check_io_attributes] + state_dict_attributes:
                display_k = k[1:] if k.startswith("_") else k
                init_attr = getattr(self, k, None)
                # then check they have the same name and, since we've already checked
                # the behavior, keep going (don't update the object's attribute)
                if init_attr is not None:
                    init_name = _get_name(init_attr)
                    saved_name = v[0]
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
                        "you initialize this optimizer object before calling load? "
                        "Optimizer objects must be initialized after calling load"
                    )
            if isinstance(self.optimizer, tuple):
                if self._optimizer[0] != _get_name(optimizer):
                    raise ValueError(
                        "User-specified optimizer must have same type as saved "
                        "optimizer, but got:"
                        f"\nSaved: {self.optimizer[0]}, "
                        f"\nUser-specified: {_get_name(optimizer)}."
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

        We also check if scheduler.step takes any arguments beyond self and epoch. If
        so, we set the `_scheduler_step_arg` flag to True (it was set to False at object
        initialization). Then, we will pass the loss to scheduler.step when we call it.

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
                    self._scheduler = None
        else:
            if isinstance(self.scheduler, tuple):
                if self.scheduler[0] != _get_name(scheduler):
                    raise ValueError(
                        "User-specified scheduler must have same type as saved "
                        f"scheduler, but got:\nSaved: {self.scheduler[0]}, "
                        f"\nUser-specified: {_get_name(scheduler)}."
                    )
                state_dict = self.scheduler[1]
                self._scheduler = scheduler
                self._scheduler.load_state_dict(state_dict)
            elif self.scheduler is not None:
                raise TypeError("When resuming synthesis, scheduler arg must be None!")
            if self.optimizer is not scheduler.optimizer:
                raise ValueError(
                    "Scheduler's optimizer must match that of this "
                    f"{self.__class__.__name__} but got two different optimizers! Did "
                    "you initialize this scheduler object before calling load? "
                    "Schedulers and optimizers must be optimized after calling load"
                )
            self._scheduler = scheduler
            step_args = set(inspect.getfullargspec(self.scheduler.step).args)
            if len(step_args - {"self", "epoch"}):
                # then we do want to pass the loss to scheduler.step
                self._scheduler_step_arg = True

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

    @property
    def scheduler(self):
        return self._scheduler
