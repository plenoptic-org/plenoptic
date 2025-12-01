"""Tools for regularizing image synthesis."""
# numpydoc ignore=ES01

from typing import Any

import torch
from torch import Tensor


def penalize_range(
    synth_img: Tensor,
    allowed_range: tuple[float, float] = (0.0, 1.0),
    **kwargs: Any,
) -> Tensor:
    r"""
    Calculate quadratic penalty on values outside of ``allowed_range``.

    Provides a 'soft' pixel-range regularization by imposing a
    quadratic penalty on any values outside the allowed_range.
    All values within the allowed_range have a penalty of 0.

    To use as a `penalty_function` in synthesis methods,
    `functools.partial` must be used to fix the `allowed_range`
    (see Examples).

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

    Examples
    --------
    Synthesize a metamer with pixel range (-1, 1):

    >>> import functools
    >>> import plenoptic as po
    >>> img = po.data.einstein()
    >>> model = po.simul.Gaussian(30).eval()
    >>> po.tools.remove_grad(model)
    >>> # Make function penalizing values outside (-1, 1)
    >>> custom_range = functools.partial(
    ...   po.tools.penalize_range, allowed_range=(-1, 1)
    ... )
    >>> met = po.synth.Metamer(img, model, penalty_function=custom_range)
    >>> met.synthesize(10)
    """
    # Using clip like this is equivalent to using boolean indexing (e.g.,
    # synth_img[synth_img < allowed_range[0]]) but much faster
    below_min = torch.clip(synth_img - allowed_range[0], max=0).pow(2).sum()
    above_max = torch.clip(synth_img - allowed_range[1], min=0).pow(2).sum()
    return below_min + above_max
