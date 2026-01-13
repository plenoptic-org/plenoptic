"""
Functions to validate synthesis inputs.

These are intended to be useful both for developers and users: users can use them to
ensure their models / images will work with our methods, and developers should use them
to ensure a standard interface to the synthesis objects.
"""

import itertools
import warnings
from collections.abc import Callable

import torch
from torch import Tensor


def validate_input(
    input_tensor: Tensor,
    no_batch: bool = False,
):
    """
    Determine whether ``input_tensor`` can be used for synthesis.

    In particular, this function:

    - Checks if input_tensor has a float or complex dtype (``TypeError``).

    - If ``no_batch`` is ``True``, check whether ``input_tensor.shape[0] == 1`` or
      ``input_tensor.ndimension()==1`` (``ValueError``).

    Additionally, if input_tensor is not 4d, or of it is outside the range
    `(0, 1)`, raises a ``UserWarning``.

    Parameters
    ----------
    input_tensor
        The tensor to validate.
    no_batch
        If ``True``, raise a ValueError if the batch dimension of ``input_tensor``
        is greater than 1.

    Raises
    ------
    ValueError
        If ``no_batch`` is ``True`` and the batch dimension of ``input_tensor``
        is greater than 1.
    TypeError
        If ``input_tensor`` does not have a float or complex dtype.

    Warns
    -----
    UserWarning
        If ``input_tensor`` is not 4d, or if ``input_tensor`` has values
        outside (0, 1).

    Examples
    --------
    Check that our built-in images work:

    >>> import plenoptic as po
    >>> po.tools.validate.validate_input(po.data.einstein())

    Raise warning:

    >>> import plenoptic as po
    >>> img = po.data.einstein() * 5.0
    >>> po.tools.validate.validate_input(img)  # doctest: +ELLIPSIS
    """
    # validate dtype
    if input_tensor.dtype not in [
        torch.float16,
        torch.complex32,
        torch.float32,
        torch.complex64,
        torch.float64,
        torch.complex128,
    ]:
        raise TypeError(
            "Only float or complex dtypes are"
            + f" allowed but got type {input_tensor.dtype}"
        )
    if input_tensor.ndimension() != 4:
        warnings.warn(
            "plenoptic's methods have mostly been tested on 4d inputs with shape "
            "torch.Size([n_batch, n_channels, im_height, im_width]). They should "
            "theoretically work with different dimensionality; if you have any "
            "problems, please open an issue at https://github.com/plenoptic-org/"
            "plenoptic/issues/new?template=bug_report.md"
        )
    # if input is 1d, then it satisfies no_batch
    if no_batch and input_tensor.ndimension() > 1 and input_tensor.shape[0] != 1:
        # numpy raises ValueError when operands cannot be broadcast together,
        # so it seems reasonable here
        raise ValueError("input_tensor batch dimension must be 1.")
    allowed_range = (0.0, 1.0)
    if input_tensor.min() < allowed_range[0] or input_tensor.max() > allowed_range[1]:
        input_range = (input_tensor.min().item(), input_tensor.max().item())
        warnings.warn(
            f"input_tensor range is {input_range}, which is outside"
            f" the tested range (0, 1). Synthesis should still work, but if"
            " you have any problems, please open an issue."
        )


def validate_model(
    model: torch.nn.Module,
    image_shape: tuple[int, int, int, int] | None = None,
    image_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
):
    """
    Determine whether model can be used for synthesis.

    In particular, this function checks the following (with their associated
    errors raised):

    - If ``model`` adds a gradient to an input tensor, which implies that some
      of it is learnable (``ValueError``).

    - If ``model`` returns a tensor when given a tensor, failure implies that
      not all computations are done using torch (``ValueError``).

    - If ``model`` strips gradient from an input with gradient attached
      (``ValueError``).

    - If ``model`` casts an input tensor to something else and returns it to a
      tensor before returning it (``ValueError``).

    - If ``model`` changes the precision of the input tensor (``TypeError``).

    - If ``model`` changes the device of the input (``RuntimeError``).

    Finally, we raise a ``UserWarning``:

    - If ``model`` is in training mode. Note that this is different from having
      learnable parameters, see `pytorch docs
      <https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc>`_.

    - If ``model`` returns an output with other than 3 or 4 dimensions when given a
      tensor with shape ``image_shape``.

    Parameters
    ----------
    model
        The model to validate.
    image_shape
        Some models (e.g., the steerable pyramid) can only accept inputs of a
        certain shape. If that's the case for ``model``, use this to
        specify the expected shape. If ``None``, we use an image of shape
        ``(1,1,16,16)``.
    image_dtype
        What dtype to validate against.
    device
        What device to place test image on.

    Raises
    ------
    ValueError
        If ``model`` fails one of the checks listed above.
    TypeError
        If ``model`` changes the precision of the input tensor.
    RuntimeError
        If ``model`` changes the device of the input tensor.

    Warns
    -----
    UserWarning
       If ``model`` is in training mode or returns an output with other than 3 or 4
       dimensions.

    See Also
    --------
    remove_grad
        Helper function for detaching all parameters (in place).

    Examples
    --------
    Check that one of our built-in models work:

    >>> import plenoptic as po
    >>> model = po.simul.PortillaSimoncelli((256, 256))
    >>> po.tools.validate.validate_model(model, image_shape=(1, 1, 256, 256))

    Intentionally fail:

    >>> import plenoptic as po
    >>> import torch
    >>> class FailureModel(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...
    ...     def forward(self, x):
    ...         x = x.detach().numpy()
    ...         return torch.as_tensor(x)
    >>> po.tools.validate.validate_model(FailureModel())  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: model strips gradient from input, ...
    """
    if image_shape is None:
        image_shape = (1, 1, 16, 16)
    test_img = torch.rand(
        image_shape, dtype=image_dtype, requires_grad=False, device=device
    )
    try:
        if model(test_img).requires_grad:
            raise ValueError(
                "model adds gradient to input, at least one of its parameters"
                " is learnable. Try calling plenoptic.tools.remove_grad()"
                " on it."
            )
    # in particular, numpy arrays lack requires_grad attribute
    except AttributeError:
        raise ValueError(
            "model does not return a torch.Tensor object -- are you sure all"
            " computations are performed using torch?"
        )
    test_img.requires_grad_()
    try:
        if not model(test_img).requires_grad:
            raise ValueError(
                "model strips gradient from input, do you detach it somewhere?"
            )
    # this gets raised if something tries to cast a tensor with requires_grad
    # to an array, which can happen explicitly or if they try to use a numpy /
    # scipy / etc function. This gets reached (rather than the first
    # AttributeError) if they cast it to an array in the middle of forward()
    # and then try to cast it back to a tensor
    except RuntimeError:
        raise ValueError(
            "model tries to cast the input into something other than"
            " torch.Tensor object -- are you sure all computations are"
            " performed using torch?"
        )
    if image_dtype in [torch.float16, torch.complex32]:
        allowed_dtypes = [torch.float16, torch.complex32]
    elif image_dtype in [torch.float32, torch.complex64]:
        allowed_dtypes = [torch.float32, torch.complex64]
    elif image_dtype in [torch.float64, torch.complex128]:
        allowed_dtypes = [torch.float64, torch.complex128]
    else:
        raise TypeError(
            f"Only float or complex dtypes are allowed but got type {image_dtype}"
        )
    if model(test_img).dtype not in allowed_dtypes:
        raise TypeError("model changes precision of input, don't do that!")
    if model(test_img).ndimension() not in [3, 4]:
        warnings.warn(
            "plenoptic's methods have mostly been tested on models which produce 3d"
            " or 4d outputs. They should theoretically work with different "
            "dimensionality; if you have any problems, please open an issue at "
            "https://github.com/plenoptic-org/plenoptic/issues/new?"
            "template=bug_report.md"
        )
    if model(test_img).device != test_img.device:
        # pytorch device errors are RuntimeErrors
        raise RuntimeError("model changes device of input, don't do that!")
    if hasattr(model, "training") and model.training:
        warnings.warn(
            "model is in training mode, you probably want to call eval()"
            " to switch to evaluation mode"
        )


def validate_coarse_to_fine(
    model: torch.nn.Module,
    image_shape: tuple[int, int, int, int] | None = None,
    device: str | torch.device = "cpu",
):
    """
    Determine whether a model can be used for coarse-to-fine synthesis.

    In particular, this function checks the following (with associated errors):

    - Whether ``model`` has a ``scales`` attribute (``AttributeError``).

    - Whether ``model.forward`` accepts a ``scales`` keyword argument (``TypeError``).

    - Whether the output of ``model.forward`` changes shape when the ``scales``
      keyword argument is set (``ValueError``).

    Parameters
    ----------
    model
        The model to validate.
    image_shape
        Some models (e.g., the steerable pyramid) can only accept inputs of a
        certain shape. If that's the case for ``model``, use this to
        specify the expected shape. If ``None``, we use an image of shape
        ``(1,1,16,16)``.
    device
        Which device to place the test image on.

    Raises
    ------
    AttributeError
        If ``model`` does not have a ``scales`` attribute.
    TypeError
        If ``model.forward`` does not accept a ``scales`` keyword argument.
    ValueError
        If ``model.forward`` output does not change shape when ``scales`` keyword
        argument is set.

    Examples
    --------
    Check that one of our built-in models work:

    >>> import plenoptic as po
    >>> model = po.simul.PortillaSimoncelli((256, 256))
    >>> po.tools.validate.validate_coarse_to_fine(model, image_shape=(1, 1, 256, 256))

    Intentionally fail:

    >>> import plenoptic as po
    >>> import torch
    >>> # this fails because it's missing the scales attribute
    >>> class FailureModel(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.model = po.simul.PortillaSimoncelli((256, 256))
    ...
    ...     def forward(self, x):
    ...         return self.model(x)
    >>> shape = (1, 1, 256, 256)
    >>> model = FailureModel()
    >>> po.tools.validate.validate_coarse_to_fine(model, shape)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    AttributeError: model has no scales attribute ...
    """
    warnings.warn(
        "Validating whether model can work with coarse-to-fine synthesis --"
        " this can take a while!"
    )
    msg = "and therefore we cannot do coarse-to-fine synthesis"
    if not hasattr(model, "scales"):
        raise AttributeError(f"model has no scales attribute {msg}")
    if image_shape is None:
        image_shape = (1, 1, 16, 16)
    test_img = torch.rand(image_shape, device=device)
    model_output_shape = model(test_img).shape
    for len_val in range(1, len(model.scales)):
        for sc in itertools.combinations(model.scales, len_val):
            try:
                if model_output_shape == model(test_img, scales=sc).shape:
                    raise ValueError(
                        "Output of model forward method doesn't change"
                        f" shape when scales keyword arg is set to {sc} {msg}"
                    )
            except TypeError:
                raise TypeError(
                    f"model forward method does not accept scales argument {sc} {msg}"
                )


def validate_metric(
    metric: torch.nn.Module | Callable[[Tensor, Tensor], Tensor],
    image_shape: tuple[int, int, int, int] | None = None,
    image_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
):
    """
    Determine whether a metric can be used for MADCompetition synthesis.

    In particular, this functions checks the following (with associated
    exceptions):

    - Whether ``metric`` is callable and accepts two tensors of shape ``image_shape`` as
      input (``TypeError``).

    - Whether ``metric`` returns a scalar when called with two tensors of shape
      ``image_shape`` as input (``ValueError``).

    - Whether ``metric`` returns a value less than 5e-7 when with two identical tensors
      of shape ``image_shape`` as input (``ValueError``). (This threshold was chosen
      because 1-SSIM of two identical images is 5e-8 on GPU).

    Parameters
    ----------
    metric
        The metric to validate.
    image_shape
        Some models (e.g., the steerable pyramid) can only accept inputs of a
        certain shape. If that's the case for ``model``, use this to
        specify the expected shape. If ``None``, we use an image of shape
        ``(1,1,16,16)``.
    image_dtype
        What dtype to validate against.
    device
        What device to place the test images on.

    Raises
    ------
    TypeError
        If ``metric`` cannot be called with two tensors of specified shape.
    ValueError
        If ``metric`` does not return a scalar or doesn't return a value less than
        5e-7 when given two identical tensors.

    Examples
    --------
    Check that 1-SSIM works:

    >>> import plenoptic as po
    >>> po.tools.validate.validate_metric(lambda x, y: 1 - po.metric.ssim(x, y))

    Check that SSIM doesn't work (because SSIM=0 means that images are *different*,
    whereas we need metric=0 to mean *identical*):

    >>> import plenoptic as po
    >>> po.tools.validate.validate_metric(po.metric.ssim)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: metric should return ...
    """
    if image_shape is None:
        image_shape = (1, 1, 16, 16)
    test_img = torch.rand(image_shape, dtype=image_dtype, device=device)
    try:
        same_val = metric(test_img, test_img).item()
    except TypeError:
        raise TypeError("metric should be callable and accept two tensors as input")
    # as of torch 2.0.0, this is a RuntimeError (a Tensor with X elements
    # cannot be converted to Scalar); previously it was a ValueError (only one
    # element tensors can be converted to Python scalars)
    except (ValueError, RuntimeError):
        raise ValueError(
            "metric should return a scalar value but"
            + f" output had shape {metric(test_img, test_img).shape}"
        )
    # on gpu, 1-SSIM of two identical images is 5e-8, so we use a threshold
    # of 5e-7 to check for zero
    if same_val > 5e-7:
        raise ValueError(
            "metric should return <= 5e-7 on"
            + f" two identical images but got {same_val}"
        )
    # this is hard to test
    for i in range(20):
        second_test_img = torch.rand_like(test_img)
        if metric(test_img, second_test_img).item() < 0:
            raise ValueError("metric should always return non-negative numbers!")


def validate_penalty(
    penalty_function: torch.nn.Module | Callable[[Tensor], Tensor],
    image_shape: tuple[int, int, int, int] | None = None,
    image_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
):
    """
    Determine whether ``penalty_function`` can be used for regularization in synthesis.

    In particular, this function checks the following (with their associated
    errors raised):

    - Whether ``penalty_function`` is callable and accepts a single tensor
      of shape ``image_shape`` as input (``TypeError``).

    - Whether ``penalty_function`` returns a scalar when called with a tensor
      of shape ``image_shape`` as input (``ValueError``).

    - If ``penalty_function`` adds a gradient to an input tensor, which implies
      that learnable parameters are being used (``ValueError``).

    - If ``penalty_function`` returns a tensor when given a tensor, failure
      implies that not all computations are done using torch (``ValueError``).

    - If ``penalty_function`` strips gradient from an input with gradient attached
      (``ValueError``).

    - If ``penalty_function`` casts an input tensor to something else and returns
      it to a tensor before returning it (``ValueError``).

    - If ``penalty_function`` changes the precision of the input tensor, or
      doesn't return a real output (``TypeError``).

    - If ``penalty_function`` changes the device of the input (``RuntimeError``).

    Parameters
    ----------
    penalty_function
        The penalty function to validate.
    image_shape
        Some models (e.g., the steerable pyramid) can only accept inputs of a
        certain shape. If that's the case for ``model``, use this to
        specify the expected shape. If ``None``, we use an image of shape
        ``(1,1,16,16)``.
    image_dtype
        What dtype to validate against.
    device
        What device to place test image on.

    Raises
    ------
    ValueError
        If ``penalty_function`` fails one of the checks listed above.
    TypeError
        If ``penalty_function`` changes the precision of the input tensor.
    RuntimeError
        If ``penalty_function`` changes the device of the input tensor.

    Examples
    --------
    Check that one of our built-in penalty functions work:

    >>> import plenoptic as po
    >>> penalty_fun = po.tools.regularization.penalize_range
    >>> po.tools.validate.validate_penalty(penalty_fun)

    Intentionally fail:

    >>> import plenoptic as po
    >>> import torch
    >>> def failure_penalty(synth_img):
    ...     non_scalar = synth_img**2
    ...     return non_scalar
    >>> po.tools.validate.validate_penalty(failure_penalty)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: penalty_function should return a scalar value but...
    """
    if image_shape is None:
        image_shape = (1, 1, 16, 16)
    test_img = torch.rand(
        image_shape, dtype=image_dtype, requires_grad=False, device=device
    )
    try:
        penalty = penalty_function(test_img)
    except TypeError:
        raise TypeError(
            "penalty_function should be callable and accept a tensor as input"
        )
    try:
        if penalty.requires_grad:
            raise ValueError(
                "penalty_function adds gradient to input, it is using learnable"
                " parameters. Try calling plenoptic.tools.remove_grad()"
                " on it."
            )
    # in particular, numpy arrays lack requires_grad attribute
    except AttributeError:
        raise ValueError(
            "penalty_function does not return a torch.Tensor object -- are you sure"
            " all computations are performed using torch?"
        )
    if penalty.numel() != 1:
        raise ValueError(
            "penalty_function should return a scalar value but"
            + f" output had shape {penalty.shape}"
        )
    test_img.requires_grad_()
    try:
        if not penalty_function(test_img).requires_grad:
            raise ValueError(
                "penalty_function strips gradient from input, do you detach"
                " it somewhere?"
            )
    # this gets raised if something tries to cast a tensor with requires_grad
    # to an array, which can happen explicitly or if they try to use a numpy /
    # scipy / etc function. This gets reached (rather than the first
    # AttributeError) if they cast it to an array in the middle of forward()
    # and then try to cast it back to a tensor
    except RuntimeError:
        raise ValueError(
            "penalty_function tries to cast the input into something other than"
            " torch.Tensor object -- are you sure all computations are"
            " performed using torch?"
        )
    if image_dtype in [torch.float16, torch.complex32]:
        allowed_dtypes = [torch.float16]
    elif image_dtype in [torch.float32, torch.complex64]:
        allowed_dtypes = [torch.float32]
    elif image_dtype in [torch.float64, torch.complex128]:
        allowed_dtypes = [torch.float64]
    else:
        raise TypeError(
            f"Only float or complex dtypes are allowed for the input, but got"
            f" type {image_dtype}"
        )
    output_dtype = penalty_function(test_img).dtype
    if output_dtype not in allowed_dtypes:
        raise TypeError(
            "penalty_function should return a real output with the same precision"
            " as the input, but got type {output_dtype} instead of {image_dtype}"
        )
    if penalty_function(test_img).device != test_img.device:
        # pytorch device errors are RuntimeErrors
        raise RuntimeError("penalty_function changes device of input, don't do that!")


def remove_grad(model: torch.nn.Module):
    """
    Detach all parameters and buffers of model (in place).

    Because models in plenoptic are fixed (i.e., we don't change their parameters),
    we want to remove the gradients from their parameters to avoid unnecessary
    computation.

    Parameters
    ----------
    model
        Torch Module with learnable parameters.

    Examples
    --------
    >>> import plenoptic as po
    >>> model = po.simul.OnOff(31, pretrained=True, cache_filt=True).eval()
    >>> po.tools.validate.validate_model(model)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: model adds gradient to input, ...
    >>> po.tools.remove_grad(model)
    >>> po.tools.validate.validate_model(model)
    """
    for p in model.parameters():
        if p.requires_grad:
            p.detach_()
    for p in model.buffers():
        if p.requires_grad:
            p.detach_()
