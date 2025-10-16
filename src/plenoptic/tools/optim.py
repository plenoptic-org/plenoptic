"""Tools related to optimization, such as objective functions."""
# numpydoc ignore=ES01

# to avoid circular import error:
# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..simulate.models import PortillaSimoncelli

import numpy as np
import torch
from torch import Tensor


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


def portilla_simoncelli_loss_factory(
    model: "PortillaSimoncelli",
    image: Tensor,
    minmax_weight: float = 0,
    highpass_weight: float = 100,
) -> Callable[[Tensor, Tensor], Tensor]:
    """
    Create the loss function required for ``PortillaSimoncelli`` metamer synthesis.

    This loss factory returns a callable which should be used as the ``loss_function``
    when initializing :class:`~plenoptic.synthesize.metamer.Metamer` for synthesizing
    metamers with the
    :class:`~plenoptic.simulate.portilla_simoncelli.PortillaSimoncelli` model. It
    reweights the model's representation of the images' min/max pixel values and the
    variance of the highpass residuals before computing the L2-norm.

    To understand how the returned loss works and see how to write your own loss
    factory, see the documentation (INSERT LINK).

    Parameters
    ----------
    model
        An instantiated
        :class:`~plenoptic.simulate.portilla_simoncelli.PortillaSimoncelli` model.
    image
        The target image for metamer synthesis, or an image with the same shape, dtype,
        and device.
    minmax_weight
        How to reweight the images' min/max for optimization purposes. It is recommended
        to set this to 0, and to allow the range penalty to match that property.
    highpass_weight
        How to reweight the variance of the highpass residuals in the model
        representation. It is recommended to set this around 100: too low and they will
        not be matched precisely enough, too high and the other statistics will not be
        well-matched.

    Returns
    -------
    loss_func
        A callable to use as your loss function for ``PortillaSimoncelli`` metamer
        synthesis.

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
    >>> loss(model(img), img2)
    tensor(31.9155)
    >>> po.tools.optim.l2_norm(model(img), img2)
    tensor(31.5433)

    Use the loss function for metamer synthesis. See documentation (INSERT LINK)
    for more details.

    >>> import plenoptic as po
    >>> img = po.data.einstein()
    >>> model = po.simul.PortillaSimoncelli(img.shape[-2:])
    >>> loss = po.tools.optim.portilla_simoncelli_loss_factory(model, img)
    >>> met = po.synth.Metamer(model, img, loss_function=loss)
    """
    weights = model.convert_to_dict(torch.ones_like(model(image)))
    if "pixel_statistics" in weights:
        # reweight the pixel min/max and the variance of the highpass residuals, since
        # they're weird.
        weights["pixel_statistics"][..., -2:] = minmax_weight
    k = "var_highpass_residual"
    if k in weights:
        weights[k] = highpass_weight * torch.ones_like(weights[k])
    weights = model.convert_to_tensor(weights)

    def loss(x: Tensor, y: Tensor) -> Tensor:  # numpydoc ignore=GL08
        return l2_norm(weights * x, weights * y)

    return loss
