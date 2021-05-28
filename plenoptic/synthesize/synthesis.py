"""abstract synthesis super-class
"""
import abc
import warnings
import torch


class Synthesis(metaclass=abc.ABCMeta):
    r"""Abstract super-class for synthesis methods

    All synthesis methods share a variety of similarities and thus need
    to have similar methods. Some of these can be implemented here and
    simply inherited, some of them will need to be different for each
    sub-class and thus are marked as abstract methods here

    """

    def __init__(self,model):
        # this initializes all the attributes that are shared, though
        # they can be overwritten in the individual __init__() if
        # necessary
        super().__init__()
        self.model = model


    @abc.abstractmethod
    def synthesize(self):
        r"""Generate an image (or set/sequence of images) given a
        base signal and a model"""
        pass


    def save(self, file_path,
             attrs=None):
        r"""Save all relevant (non-model) variables in .pt file.
        
        Parameters
        ----------
        file_path : str
            The path to save the synthesis object to
        attrs : list
            List of strs containing the names of the attributes of this
            object to save.
        """
        
        if attrs is None:
            attrs = vars(self)
            attrs.pop('model')

        save_dict = {}
        for k in attrs:
            if k=='model':
                warnings.warn("Models can be quite large and they don't change over synthesis. Please be sure that you actually want to save the model.")
            attr = getattr(self, k)
            # detaching the tensors avoids some headaches like the
            # tensors having extra hooks or the like
            if isinstance(attr, torch.Tensor):
                attr = attr.detach()
            save_dict[k] = attr
        torch.save(save_dict, file_path, pickle_module=dill)


    def load(self, file_path, map_location=None,
             check_attributes=[],
             **pickle_load_args):
        r"""Load all relevant attributes from a .pt file.
        This should be called by an initialized ``Synthesis`` object -- we will
        ensure that the attributes in the ``check_attributes`` arg all match in
        the current and loaded object.
        Note that we check a ``loss_function`` in a special way (because
        comparing two python callables if very difficult): we compare the
        outputs on some random images.
        Note this operates in place and so doesn't return anything.
        Parameters
        ----------
        file_path : str
            The path to load the synthesis object from
        map_location : str, optional
            map_location argument to pass to ``torch.load``. If you save
            stuff that was being run on a GPU and are loading onto a
            CPU, you'll need this to make sure everything lines up
            properly. This should be structured like the str you would
            pass to ``torch.device``
        check_attributes : list, optional
            List of strings we ensure are identical in the current
            ``Synthesis`` object and the loaded one. Checking the model is
            generally not recommended, since it can be hard to do (checking
            callable objects is hard in Python) -- instead, checking the
            ``base_representation`` should ensure the model hasn't functinoally
            changed.
        pickle_load_args :
            any additional kwargs will be added to ``pickle_module.load`` via
            ``torch.load``, see that function's docstring for details.
        Examples
        --------
        >>> metamer = po.synth.Metamer(img, model)
        >>> metamer.synthesize(max_iter=10, store_progress=True)
        >>> metamer.save('metamers.pt')
        >>> metamer_copy = po.synth.Metamer(img, model)
        >>> metamer_copy.load('metamers.pt')
        Note that you must create a new instance of the Synthesis object and
        *then* load.
        """
        tmp_dict = torch.load(file_path, pickle_module=dill, **pickle_load_args)
        for k in check_attributes:
            if not hasattr(self, k):
                raise Exception("All values of `check_attributes` should be attributes set at"
                                f" initialization, but got attr {k}!")
            if isinstance(getattr(self, k), torch.Tensor):
                # there are two ways this can fail -- the first is if they're
                # the same shape but different values and the second (in the
                # except block) are if they're different shapes.
                try:
                    if not torch.allclose(getattr(self, k).to(tmp_dict[k].device), tmp_dict[k]):
                        raise Exception(f"Saved and initialized {k} are different! Initialized: {getattr(self, k)}"
                                        f", Saved: {tmp_dict[k]}, difference: {getattr(self, k) - tmp_dict[k]}")
                except RuntimeError:
                    raise Exception(f"Attribute {k} have different shapes in saved and initialized versions!"
                                    f" Initialized: {getattr(self, k).shape}, Saved: {tmp_dict[k].shape}")
            else:
                if getattr(self, k) != tmp_dict[k]:
                    raise Exception(f"Saved and initialized {k} are different! Self: {getattr(self, k)}"
                                    f", Saved: {tmp_dict[k]}")
        for k, v in tmp_dict.items():
            setattr(self, k, v)
        self.to(device=map_location)
    

    @abc.abstractmethod
    def to(self, *args, attrs=[], **kwargs):
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
        Returns:
            Module: self
        """
        
        try:
            self.model = self.model.to(*args, **kwargs)
        except AttributeError:
            warnings.warn("model has no `to` method, so we leave it as is...")
        
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        def move(a, k):
            move_device = None if k.startswith("saved_") else device
            if memory_format is not None and a.dim() == 4:
                return a.to(move_device, dtype, non_blocking, memory_format=memory_format)
            else:
                return a.to(move_device, dtype, non_blocking)
        
        for k in attrs:
            if hasattr(self, k):
                attr = getattr(self, k)
                if isinstance(attr, torch.Tensor):
                    attr = move(attr, k)
                    if isinstance(getattr(self, k), torch.nn.Parameter):
                        attr = torch.nn.Parameter(attr)
                    setattr(self, k, attr)
                elif isinstance(attr, list):
                    setattr(self, k, [move(a, k) for a in attr])
        return self