"""Tools related to optimization, such as objective functions."""
# numpydoc ignore=ES01

# to avoid circular import error:
# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import PortillaSimoncelli

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "mse",
    "l2_norm",
    "relative_sse",
    "penalize_range",
    "portilla_simoncelli_loss_factory",
]


def __dir__() -> list[str]:
    return __all__


def set_seed(seed: int | None = None) -> None:
    """
    Set the seed.

    We call both :func:`torch.manual_seed()` and :func:`numpy.random.seed()`.

    Parameters
    ----------
    seed
        The seed to set. If ``None``, do nothing.
    """
    if seed is not None:
        # random initialization
        torch.manual_seed(seed)
        np.random.seed(seed)


def mse(synth_rep: Tensor, ref_rep: Tensor, **kwargs: Any) -> Tensor:
    r"""
    Calculate the MSE between ``synth_rep`` and ``ref_rep``.

    For two tensors, :math:`x` and :math:`y`, with :math:`n` values
    each:

    .. math::

        MSE = \frac{1}{n}\sum_{i=1}^n (x_i - y_i)^2

    The two images must have a float dtype.

    Parameters
    ----------
    synth_rep
        The first tensor to compare, model representation of the
        synthesized image.
    ref_rep
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``.
    **kwargs
        Ignored, only present to absorb extra arguments.

    Returns
    -------
    loss
        The mean-squared error between ``synth_rep`` and ``ref_rep``.
    """
    return torch.pow(synth_rep - ref_rep, 2).mean()


def l2_norm(synth_rep: Tensor, ref_rep: Tensor, **kwargs: Any) -> Tensor:
    r"""
    Calculate the L2-norm of the difference between ``ref_rep`` and ``synth_rep``.

    For two tensors, :math:`x` and :math:`y`, with :math:`n` values
    each:

    .. math::

        L2 = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}

    Parameters
    ----------
    synth_rep
        The first tensor to compare, model representation of the
        synthesized image.
    ref_rep
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``.
    **kwargs
        Ignored, only present to absorb extra arguments.

    Returns
    -------
    loss
        The L2-norm of the difference between ``ref_rep`` and ``synth_rep``.
    """
    return torch.linalg.vector_norm(ref_rep - synth_rep, ord=2)


def relative_sse(synth_rep: Tensor, ref_rep: Tensor, **kwargs: Any) -> Tensor:
    r"""
    Calculate the relative sum of squared errors between two tensors.

    This is the squared L2-norm of the difference between reference representation and
    synthesized representation relative to the squared L2-norm of the reference
    representation:

    For two tensors, :math:`x` and :math:`y`:

    .. math::

        \frac{||x - y||_2^2}{||x||_2^2}

    where :math:`x` is ``ref_rep``, :math:`x` is ``synth_rep``, and :math:`||x||_2` is
    the L2-norm.

    Parameters
    ----------
    synth_rep
        The first tensor to compare, model representation of the
        synthesized image.
    ref_rep
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``.
    **kwargs
        Ignored, only present to absorb extra arguments.

    Returns
    -------
    loss
        Ratio of the squared l2-norm of the difference between ``ref_rep`` and
        ``synth_rep`` to the squared l2-norm of ``ref_rep``.
    """
    return (
        torch.linalg.vector_norm(ref_rep - synth_rep, ord=2) ** 2
        / torch.linalg.vector_norm(ref_rep, ord=2) ** 2
    )


def penalize_range(
    synth_img: Tensor,
    allowed_range: tuple[float, float] = (0.0, 1.0),
    **kwargs: Any,
) -> Tensor:
    r"""
    Calculate quadratic penalty on values outside of ``allowed_range``.

    Instead of clamping values to exactly fall in a range, this provides
    a 'softer' way of doing it, by imposing a quadratic penalty on any
    values outside the allowed_range. All values within the
    allowed_range have a penalty of 0.

    Parameters
    ----------
    synth_img
        The tensor to penalize. the synthesized image.
    allowed_range
        2-tuple of values giving the (min, max) allowed values.
    **kwargs
        Ignored, only present to absorb extra arguments.

    Returns
    -------
    penalty
        Penalty for values outside range.
    """
    # Using clip like this is equivalent to using boolean indexing (e.g.,
    # synth_img[synth_img < allowed_range[0]]) but much faster
    below_min = torch.clip(synth_img - allowed_range[0], max=0).pow(2).sum()
    above_max = torch.clip(synth_img - allowed_range[1], min=0).pow(2).sum()
    return below_min + above_max


def _groupwise_l2_norm_weights(
    model: torch.nn.Module,
    image: Tensor,
    reweighting_dict: dict[str, Tensor | float] | None = None,
) -> dict[str, Tensor]:
    r"""
    Compute groupwise L2 norm, as a tensor for reweighting.

    This function returns a tensor that can be used to perform a groupwise reweighting
    of a model's representation. It is used by
    :func:`~plenoptic.tools.optim.groupwise_relative_l2_norm_factory` and similar
    functions, which normalize model representations so that all statistics are roughly
    the same scale, which makes optimization easier.

    This requires that ``model`` has a ``convert_to_dict`` method, which converts the
    representation from a tensor (as returned by ``forward``) to a dictionary. The
    dictionary representation should have keys that define the different groups within
    the representation, and its values should be tensors (of any shape).

    The optional ``reweighting_dict`` argument allows users to further tweak the
    weights, if necessary. If not ``None``, keys should be a subset of those found in
    the output of ``model.convert_to_dict``, and whose values are Tensors (broadcastable
    to the shape of the corresponding values in ``model.convert_to_dict`` output) which
    will be multiplied by the corresponding group *after* normalization. Thus, a number
    greater than 1 will increase its weight in the loss, a number less than 1 will
    decrease the weight, and 0 will remove it from the calculation entirely.

    For an example of a compliant model, see the
    :class:`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli` model.

    Parameters
    ----------
    model
        An instantiated model.
    image
        The target image for metamer synthesis.
    reweighting_dict
        Dictionary specifying further reweighting. See above for details.

    Returns
    -------
    weights
        Dictionary containing the L2 norm of each statistic group. Should probably be
        passed to ``model.convert_to_tensor``, but left in this form in case any further
        reweighting needs to be done (e.g., zeroing out some component).

    Raises
    ------
    ValueError
        If ``reweighting_dict`` contains keys not found in the model representation
        (``model.convert_to_dict(model(image))``).

    Warns
    -----
    UserWarning
        If ``model.convert_to_dict()`` does not return an
        :class:`~collections.OrderedDict`. ``convert_to_dict`` and
        ``convert_to_tensor`` need to invert each other, which means you should probably
        use an :class:`~collections.OrderedDict`, which guarantees that the order of the
        keys is preserved. You can use
        :func:`~plenoptic.tools.validate.validate_convert_tensor_dict` to heuristically
        check whether your model satisfies this constraint.
    """
    if reweighting_dict is None:
        reweighting_dict = {}
    weights = {}
    rep = model.convert_to_dict(model(image))
    if not isinstance(rep, OrderedDict):
        warnings.warn(
            "model.convert_to_dict did not return an OrderedDict. This might "
            "not be a problem, but convert_to_dict and convert_to_tensor must"
            " invert each other. Calling "
            "plenoptic.tools.validate.validate_convert_tensor_dict(model)"
            " will attempt to validate this constraint."
        )
    if extra_keys := set(reweighting_dict.keys()) - set(rep.keys()):
        raise ValueError(
            "reweighting_dict contains keys not found in model representation!"
            f" {extra_keys}"
        )
    for k, v in rep.items():
        wt = torch.linalg.vector_norm(v[~v.isnan()], ord=2)
        weights[k] = reweighting_dict.get(k, 1) * torch.ones_like(v) / wt
    return weights


def groupwise_relative_l2_norm_factory(
    model: torch.nn.Module,
    image: Tensor,
    reweighting_dict: dict[str, Tensor | float] | None = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    r"""
    Create loss function that computes groupwise relative L2 norm for synthesis.

    This loss factory returns a callable which should make optimization easier when
    used as the ``loss_function`` when initializing
    :class:`~plenoptic.synthesize.metamer.Metamer` for synthesizing metamers. The
    resulting loss function will normalize each group within the representation by the
    L2 norm of that group on ``image``, which should be the target image for that
    synthesis.

    This requires that ``model`` has two methods, ``convert_to_dict`` and
    ``convert_to_tensor``, which convert the representation between a tensor (as
    returned by ``forward``) and an :class:`~collections.OrderedDict`. The dictionary
    representation should have keys that define the different groups within the
    representation, and its values should be tensors (of any shape).

    The optional ``reweighting_dict`` argument allows users to further tweak the
    weights, if necessary. If not ``None``, keys should be a subset of those found in
    the output of ``model.convert_to_dict``, and whose values are Tensors (broadcastable
    to the shape of the corresponding values in ``model.convert_to_dict`` output) which
    will be multiplied by the corresponding group *after* normalization. Thus, a number
    greater than 1 will increase its weight in the loss, a number less than 1 will
    decrease the weight, and 0 will remove it from the calculation entirely.

    For an example of a compliant model, see the
    :class:`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli` model.

    Parameters
    ----------
    model
        An instantiated model.
    image
        The target image for metamer synthesis.
    reweighting_dict
        Dictionary specifying further reweighting. See above for details.

    Returns
    -------
    loss_func
        A callable to use as your loss function for metamer synthesis.

    Raises
    ------
    ValueError
        If ``reweighting_dict`` contains keys not found in the model representation
        (``model.convert_to_dict(model(image))``).

    Warns
    -----
    UserWarning
        If ``model.convert_to_dict()`` does not return an
        :class:`~collections.OrderedDict`. ``convert_to_dict`` and
        ``convert_to_tensor`` need to invert each other, which means you should probably
        use an :class:`~collections.OrderedDict`, which guarantees that the order of the
        keys is preserved. You can use
        :func:`~plenoptic.tools.validate.validate_convert_tensor_dict` to heuristically
        check whether your model satisfies this constraint.

    Examples
    --------
    Create the loss function with a simple model.

    >>> import plenoptic as po
    >>> from collections import OrderedDict
    >>> import torch
    >>> po.tools.set_seed(0)
    >>> class TestModel(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.kernel = torch.nn.Conv2d(1, 2, (5, 5), bias=False)
    ...         self.kernel.weight.detach_()
    ...
    ...     def forward(self, x):
    ...         return self.kernel(x)
    ...
    ...     def convert_to_dict(self, rep):
    ...         return OrderedDict({f"channel_{i}": rep[:, i] for i in range(2)})
    ...
    ...     def convert_to_tensor(self, rep_dict):
    ...         return torch.stack(list(rep_dict.values()), axis=1)
    >>> img = po.data.einstein()
    >>> img2 = torch.rand_like(img)
    >>> model = TestModel()
    >>> loss = po.tools.optim.groupwise_relative_l2_norm_factory(model, img)
    >>> loss(model(img), model(img2))
    tensor(0.6512)
    >>> po.tools.optim.l2_norm(model(img), model(img2))
    tensor(78.5674)

    Use ``reweighting_dict`` to further tweak weighting.

    >>> import plenoptic as po
    >>> from collections import OrderedDict
    >>> import torch
    >>> po.tools.set_seed(0)
    >>> class TestModel(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.kernel = torch.nn.Conv2d(1, 2, (5, 5), bias=False)
    ...         self.kernel.weight.detach_()
    ...
    ...     def forward(self, x):
    ...         return self.kernel(x)
    ...
    ...     def convert_to_dict(self, rep):
    ...         return OrderedDict({f"channel_{i}": rep[:, i] for i in range(2)})
    ...
    ...     def convert_to_tensor(self, rep_dict):
    ...         return torch.stack(list(rep_dict.values()), axis=1)
    >>> img = po.data.einstein()
    >>> img2 = torch.rand_like(img)
    >>> model = TestModel()
    >>> reweighting_dict = {"channel_0": 0.5}
    >>> loss = po.tools.optim.groupwise_relative_l2_norm_factory(
    ...     model, img, reweighting_dict
    ... )
    >>> loss(model(img), model(img2))
    tensor(0.4822)
    >>> # channel_0 is of shape (1, 256, 256)
    >>> channel_0 = torch.ones_like(model.convert_to_dict(model(img))["channel_0"])
    >>> channel_0[..., 128:] = 0
    >>> reweighting_dict = {"channel_0": channel_0}
    >>> loss = po.tools.optim.groupwise_relative_l2_norm_factory(
    ...     model, img, reweighting_dict
    ... )
    >>> loss(model(img), model(img2))
    tensor(0.5612)
    """
    weights = _groupwise_l2_norm_weights(model, image, reweighting_dict)
    weights = model.convert_to_tensor(weights)

    def loss(x: Tensor, y: Tensor) -> Tensor:  # numpydoc ignore=GL08
        return l2_norm(weights * x, weights * y)

    return loss


def portilla_simoncelli_loss_factory(
    model: "PortillaSimoncelli",
    image: Tensor,
    reweighting_dict: dict[str, Tensor | float] | None = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """
    Create the loss function required for ``PortillaSimoncelli`` metamer synthesis.

    This loss factory returns a callable which should be used as the ``loss_function``
    when initializing :class:`~plenoptic.synthesize.metamer.Metamer` for synthesizing
    metamers with the
    :class:`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli` model. It
    zeroes the model's representation of the images' min/max pixel values and increases
    the weight on the variance of the highpass residuals before computing the L2-norm.

    The optional ``reweighting_dict`` argument allows users to tweak the weights. If not
    ``None``, keys should be a subset of those found in the output of
    :func:`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli.convert_to_dict`
    and whose values are Tensors (broadcastable to the shape of the corresponding values
    in ``convert_to_dict`` output) which will be multiplied by the corresponding
    group. Thus, a number greater than 1 will increase its weight in the loss, a number
    less than 1 will decrease the weight, and 0 will remove it from the calculation
    entirely. ``reweighting_dict`` takes precedence, so e.g., if it includes a
    ``"pixel_statistics"`` key, that will dictate how min/max pixel values are weighted.

    To understand how the returned loss works and see how to write your own loss
    factory, see :ref:`ps-optimization`.

    Parameters
    ----------
    model
        An instantiated
        :class:`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli`
        model.
    image
        The target image for metamer synthesis, or an image with the same shape, dtype,
        and device.
    reweighting_dict
        Dictionary specifying reweighting. See above for details.

    Returns
    -------
    loss_func
        A callable to use as your loss function for ``PortillaSimoncelli`` metamer
        synthesis.

    Raises
    ------
    ValueError
        If ``reweighting_dict`` contains keys not found in the model representation
        (``model.convert_to_dict(model(image))``).
    ValueError
        If model representation (``model.convert_to_dict(model(image))``) includes the
        key ``"pixel_statistics"`` but the corresponding tensor does not have
        ``shape[-1] == 6`` or if it includes the key ``"var_highpass_residual"`` but the
        corresponding tensor does not have ``shape[-1] == 1`` and the corresponding key
        is not included explicitly in ``reweighting_dict``.

    Warns
    -----
    UserWarning
        If model representation (``model.convert_to_dict(model(image))``) does not
        include the keys ``"pixel_statistics"`` or ``"var_highpass_residual"``.

    Examples
    --------
    Create the loss function.

    >>> import plenoptic as po
    >>> import torch
    >>> po.tools.set_seed(0)
    >>> img = po.data.einstein()
    >>> img2 = torch.rand_like(img)
    >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
    >>> loss = po.tools.optim.portilla_simoncelli_loss_factory(model, img)
    >>> loss(model(img), model(img2))
    tensor(31.9155)
    >>> po.tools.optim.l2_norm(model(img), model(img2))
    tensor(31.5433)

    Use the loss function for metamer synthesis.

    >>> import plenoptic as po
    >>> img = po.data.einstein()
    >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
    >>> loss = po.tools.optim.portilla_simoncelli_loss_factory(model, img)
    >>> met = po.synth.Metamer(img, model, loss_function=loss)

    Use ``reweighting_dict`` to increase weight on image pixel moments, while keeping
    min/max out of the loss. The model includes 6 pixel stats (see :ref:`ps-model-stats`
    for details)

    >>> import plenoptic as po
    >>> import torch
    >>> po.tools.set_seed(0)
    >>> img = po.data.einstein()
    >>> img2 = torch.rand_like(img)
    >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
    >>> rep = model.convert_to_dict(model(img))
    >>> pixel_stats = torch.as_tensor([10, 10, 10, 10, 0, 0])
    >>> pixel_stats = pixel_stats * torch.ones_like(rep["pixel_statistics"])
    >>> reweighting_dict = {"pixel_statistics": pixel_stats}
    >>> loss = po.tools.optim.portilla_simoncelli_loss_factory(
    ...     model, img, reweighting_dict
    ... )
    >>> loss(model(img), model(img2))
    tensor(35.9753)

    Use ``reweighting_dict`` to include min/max in the loss and increase the importance
    of the standard deviations of the magnitude bands.

    >>> import plenoptic as po
    >>> import torch
    >>> po.tools.set_seed(0)
    >>> img = po.data.einstein()
    >>> img2 = torch.rand_like(img)
    >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
    >>> reweighting_dict = {"pixel_statistics": 1, "magnitude_std": 100}
    >>> loss = po.tools.optim.portilla_simoncelli_loss_factory(
    ...     model, img, reweighting_dict
    ... )
    >>> loss(model(img), model(img2))
    tensor(251.5188)
    """
    if reweighting_dict is None:
        reweighting_dict = {}
    weights = model.convert_to_dict(torch.ones_like(model(image)))
    # do this before adding defaults for pixel_stats and var_highpass_residual
    if extra_keys := set(reweighting_dict.keys()) - set(weights.keys()):
        raise ValueError(
            "reweighting_dict contains key(s) not found in model representation! "
            f"{extra_keys}"
        )
    if "pixel_statistics" in weights:
        pixel_stats = torch.ones_like(weights["pixel_statistics"])
        pixel_stats[..., -2:] = 0
        if pixel_stats.shape[-1] != 6 and "pixel_statistics" not in reweighting_dict:
            raise ValueError(
                "Expected model's 'pixel_statistics' representation "
                f"to have 6 values, but it has {pixel_stats.shape[-1]}"
                " values instead! Unsure what corresponds to the "
                "min/max, set this directly in reweighting_dict"
            )
        reweighting_dict.setdefault("pixel_statistics", pixel_stats)
    else:
        warnings.warn(
            "pixel_statistics not found in your model representation, "
            "continuing without removing them. Hope you know what "
            "you're doing..."
        )
    if "var_highpass_residual" in weights:
        n_highpass = weights["var_highpass_residual"].shape[-1]
        if n_highpass != 1 and "var_highpass_residual" not in reweighting_dict:
            raise ValueError(
                "Expected model's 'var_highpass_residual' representation "
                f"to have 1 value, but it has {n_highpass}"
                " values instead! Unsure how to handle this,"
                " set directly in reweighting_dict"
            )
        reweighting_dict.setdefault("var_highpass_residual", 100)
    else:
        warnings.warn(
            "var_highpass_residual not found in your model representation, "
            "continuing without reweighting them. Hope you know what "
            "you're doing..."
        )
    for k in weights:
        weights[k] *= reweighting_dict.get(k, 1)
    weights = model.convert_to_tensor(weights)

    def loss(x: Tensor, y: Tensor) -> Tensor:  # numpydoc ignore=GL08
        return l2_norm(weights * x, weights * y)

    return loss
