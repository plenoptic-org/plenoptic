"""Tools related to optimization such as more objective functions.
"""
import torch
from torch import Tensor
from typing import Optional, Tuple
import numpy as np


def set_seed(seed: Optional[int] = None) -> None:
    """Set the seed.

    We call both ``torch.manual_seed()`` and ``np.random.seed()``.

    Parameters
    ----------
    seed
        The seed to set. If None, do nothing.
    """
    if seed is not None:
        # random initialization
        torch.manual_seed(seed)
        np.random.seed(seed)


def mse(synth_rep: Tensor, ref_rep: Tensor, **kwargs) -> Tensor:
    r"""return the MSE between synth_rep and ref_rep

    For two tensors, :math:`x` and :math:`y`, with :math:`n` values
    each:

    .. math::

        MSE &= \frac{1}{n}\sum_i=1^n (x_i - y_i)^2

    The two images must have a float dtype

    Parameters
    ----------
    synth_rep
        The first tensor to compare, model representation of the
        synthesized image
    ref_rep
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``,
    kwargs
        Ignored, only present to absorb extra arguments

    Returns
    -------
    loss
        The mean-squared error between ``synth_rep`` and ``ref_rep``
    """
    return torch.pow(synth_rep - ref_rep, 2).mean()


def l2_norm(synth_rep: Tensor, ref_rep: Tensor, **kwargs) -> Tensor:
    r"""l2-norm of the difference between ref_rep and synth_rep

    Parameters
    ----------
    synth_rep
        The first tensor to compare, model representation of the
        synthesized image.
    ref_rep
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``.
    kwargs
        Ignored, only present to absorb extra arguments.

    Returns
    -------
    loss
        The L2-norm of the difference between ``ref_rep`` and ``synth_rep``.
    """
    return torch.linalg.matrix_norm(ref_rep - synth_rep, ord=2)


def relative_MSE(synth_rep: Tensor, ref_rep: Tensor, **kwargs) -> Tensor:
    r"""Squared l2-norm of the difference between reference representation
    and synthesized representation relative to the squared l2-norm of the
    reference representation:

    $$\frac{||x - \hat{x}||_2^2}{||x||_2^2}$$

    Parameters
    ----------
    synth_rep
        The first tensor to compare, model representation of the
        synthesized image.
    ref_rep
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``.
    kwargs
        Ignored, only present to absorb extra arguments

    Returns
    -------
    loss
        Ratio of the squared l2-norm of the difference between ``ref_rep`` and
        ``synth_rep`` to the squared l2-norm of ``ref_rep``
    """
    return torch.linalg.matrix_norm(ref_rep - synth_rep, p=2) ** 2 / torch.linalg.matrix_norm(ref_rep, p=2) ** 2


def penalize_range(
    synth_img: Tensor, allowed_range: Tuple[float, float] = (0.0, 1.0), **kwargs
) -> Tensor:
    r"""penalize values outside of allowed_range

    instead of clamping values to exactly fall in a range, this provides
    a 'softer' way of doing it, by imposing a quadratic penalty on any
    values outside the allowed_range. All values within the
    allowed_range have a penalty of 0

    Parameters
    ----------
    synth_img
        The tensor to penalize. the synthesized image.
    allowed_range
        2-tuple of values giving the (min, max) allowed values
    kwargs
        Ignored, only present to absorb extra arguments

    Returns
    -------
    penalty
        Penalty for values outside range
    """
    # the indexing should flatten it
    below_min = synth_img[synth_img < allowed_range[0]]
    below_min = torch.pow(below_min - allowed_range[0], 2)
    above_max = synth_img[synth_img > allowed_range[1]]
    above_max = torch.pow(above_max - allowed_range[1], 2)
    return torch.sum(torch.cat([below_min, above_max]))
