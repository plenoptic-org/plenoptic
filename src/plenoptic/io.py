"""Helper functions for saving/loading."""  # numpydoc ignore=ES01

import importlib

import torch

__all__ = [
    "examine_saved_synthesis",
    "LoadWarning",
]


def __dir__() -> list[str]:
    return __all__


class LoadWarning(UserWarning):
    """
    Custom warning to raise if there's an issue with loading.

    And we do not want it to result in an error.

    Examples
    --------
    >>> import plenoptic as po
    >>> import warnings
    >>> model = po.models.Gaussian((31, 31))
    >>> model.eval()
    Gaussian()
    >>> po.remove_grad(model)
    >>> met = po.Metamer(po.data.einstein(), model)
    >>> met.synthesize(2)
    >>> met.save("load_warning_example.pt")
    >>> # this loss function has a different name but the same behavior
    >>> met = po.Metamer(po.data.einstein(), model, lambda *args: po.loss.mse(*args))
    >>> with warnings.catch_warnings(record=True) as warned:
    ...     met.load("load_warning_example.pt", raise_on_checks=False)
    ...     print(len(warned), warned[0].category)
    1 <class 'plenoptic.io.LoadWarning'>
    """

    pass


def _parse_save_io_attr_name(
    synth_object: dict, input_names: tuple[str]
) -> tuple[list[torch.Tensor], list[str]]:
    """
    Parse names of save_io_attrs, allowing for more complex behavior.

    The strings specified in ``input_names`` must either be the names of this
    object's attributes or of the form ``x * name``, where ``x`` is a float and
    ``name`` is a string as above, in which case we multiply that attribute by
    ``x``.

    Parameters
    ----------
    synth_object
        Dictionary containing tensors corresponding to ``input_names`` from.
    input_names
        The second element from the tuple ``save_io_attrs`` input to
        :func:`save`.

    Returns
    -------
    tensors
        The tensors to pass to the corresponding ``save_io_attr``.
    input_names_test
        List of strings of attributes that we ensure we save.
    """
    tensors = []
    input_names_test = []
    for t in input_names:
        t = t.split("*")
        if len(t) == 2:
            name = t[1].strip()
            scale = float(t[0].strip())
        else:
            name = t[0]
            scale = 1
        input_names_test.append(name)
        tensors.append(scale * synth_object[name])
    return tensors, input_names_test


def examine_saved_synthesis(file_path: str, map_location: str | None = None):
    """
    Examine saved synthesis object.

    This is used for debugging, it will print out information about the versions used,
    names of the callable attributes, shapes of tensor attributes, etc.

    .. warning::
       This will call :func:`torch.load` on the specified path; it will not try to
       initialize any plenoptic objects. We set ``weights_only=True``, so it should
       be safe (it will not run any saved code), but will load the tensors into memory,
       which may be a problem if they are large.

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

    Raises
    ------
    ValueError
        If we are unable to parse the saved object, in which case we believe it is not
        a saved plenoptic synthesis object.

    Examples
    --------
    The function prints out information about the versions of plenoptic and pytorch used
    when saving the object (and the currently installed versions), as well the object
    type, and then information about all of the attributes of the object, split into
    separate groups for callables, tensors, and everything else.

    >>> import plenoptic as po
    >>> # Download one of our cached examples
    >>> saved_obj = po.data.fetch_data("example_mad.pt")
    >>> po.io.examine_saved_synthesis(saved_obj)
    Metadata
    --------
    plenoptic version : ...
    torch version     : ...
    Saved object type : plenoptic.MADCompetition
    <BLANKLINE>
    Callables attributes
    --------------------
    optimized_metric : ...
    reference_metric : ...
    penalty_function : ...
    optimizer        : ...
    scheduler        : ...
    <BLANKLINE>
    Tensor attributes
    -----------------
    image         : torch.float64, shape torch.Size([1, 1, 256, 256])
    mad_image     : torch.float64, shape torch.Size([1, 1, 256, 256])
    initial_image : torch.float64, shape torch.Size([1, 1, 256, 256])
    <BLANKLINE>
    Other attributes
    ----------------
    loaded                  : True
    losses                  : <class 'list'> with length ...
    penalties               : <class 'list'> with length ...
    gradient_norm           : <class 'list'> with length ...
    pixel_change_norm       : <class 'list'> with length ...
    store_progress          : ...
    current_loss            : ...
    current_penalty         : ...
    penalty_lambda          : ...
    image_shape             : <class 'torch.Size'> with length 4
    scheduler_step_arg      : False
    optimized_metric_loss   : <class 'list'> with length ...
    reference_metric_loss   : <class 'list'> with length ...
    reference_metric_target : ...
    metric_tradeoff_lambda  : ...
    minmax                  : max
    saved_mad_image         : <class 'list'> with length ...
    current_ref_metric      : ...
    current_opt_metric      : ...

    Because we must load in the torch object in order to retrieve the above information,
    ``map_location`` must be set in order to examine an object that was saved from a
    CUDA device if none is available on your current machine:

    >>> saved_obj = po.data.fetch_data("example_eigendistortion_color.pt")
    >>> po.io.examine_saved_synthesis(saved_obj)
    Traceback (most recent call last):
    RuntimeError: Attempting to deserialize object on a CUDA device but
    torch.cuda.is_available() is False...

    Note that the information about the attributes printed depends on the object type:

    >>> po.io.examine_saved_synthesis(saved_obj, map_location="cpu")
    Metadata
    --------
    plenoptic version : ...
    torch version     : ...
    Saved object type : plenoptic.Eigendistortion
    <BLANKLINE>
    Callables attributes
    --------------------
    model : ...
    <BLANKLINE>
    Tensor attributes
    -----------------
    image_flat          : torch.float64, shape torch.Size([1200, 1])
    image               : torch.float64, shape torch.Size([1, 3, 20, 20])
    representation_flat : torch.float64, shape torch.Size([1296, 1])
    eigendistortions    : torch.float64, shape torch.Size([2, 3, 20, 20])
    eigenvalues         : torch.float64, shape torch.Size([2])
    eigenindex          : torch.int64, shape torch.Size([2])
    <BLANKLINE>
    Other attributes
    ----------------
    loaded      : False
    image_shape : <class 'torch.Size'> with length 3
    jacobian    : None

    This only works for plenoptic synthesis objects, not arbitrary torch objects:

    >>> import torch
    >>> torch.save(torch.rand(1, 1, 32, 32), "test.pt")
    >>> po.io.examine_saved_synthesis("test.pt")
    Traceback (most recent call last):
    ValueError: Unable to parse saved object -- is this a plenoptic synthesis object?...
    """
    load_dict = torch.load(file_path, map_location=map_location, weights_only=True)
    try:
        metadata = load_dict.pop("save_metadata")
    except AttributeError as e:
        raise ValueError(
            "Unable to parse saved object -- is this a plenoptic"
            f" synthesis object?\n\nOriginal exception: {e}"
        )
    print("Metadata\n--------")
    print(
        f"plenoptic version : {metadata['plenoptic_version']} "
        f"(installed: {importlib.metadata.version('plenoptic')})"
    )
    print(
        f"torch version     : {metadata['torch_version']} "
        f"(installed: {importlib.metadata.version('torch')})"
    )
    print(f"Saved object type : {metadata['synthesis_object']}")
    print("\nCallables attributes\n--------------------")
    callables = [
        (k, v)
        for k, v in load_dict.items()
        if isinstance(v, tuple) and (isinstance(v[0], str) or v[0] is None)
    ]
    pad_len = max([len(k[1:] if k.startswith("_") else k) for k, v in callables]) + 1
    for k, v in callables:
        display_k = k[1:] if k.startswith("_") else k
        load_dict.pop(k)
        # then this is state_dict attribute
        if len(v) == 2:
            print(f"{display_k:<{pad_len}}: {v[0]}")
        # then this is an io attribute
        else:
            tensors, _ = _parse_save_io_attr_name(load_dict, v[1])
            print(
                f"{display_k:<{pad_len}}: {v[0]}, "
                f"{[t.shape for t in tensors]} -> {v[2].shape}"
            )
    print("\nTensor attributes\n-----------------")
    tensors = [(k, v) for k, v in load_dict.items() if isinstance(v, torch.Tensor)]
    pad_len = max([len(k[1:] if k.startswith("_") else k) for k, v in tensors]) + 1
    for k, v in tensors:
        display_k = k[1:] if k.startswith("_") else k
        load_dict.pop(k)
        print(f"{display_k:<{pad_len}}: {v.dtype}, shape {v.shape}")
    print("\nOther attributes\n----------------")
    pad_len = max([len(k[1:] if k.startswith("_") else k) for k in load_dict]) + 1
    for k, v in load_dict.items():
        display_k = k[1:] if k.startswith("_") else k
        if hasattr(v, "__len__") and not isinstance(v, str):
            print(f"{display_k:<{pad_len}}: {type(v)} with length {len(v)}")
        else:
            print(f"{display_k:<{pad_len}}: {v}")
