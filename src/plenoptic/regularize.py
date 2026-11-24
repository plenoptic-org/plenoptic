"""Regularization functions for image synthesis."""
# numpydoc ignore=ES01

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

__all__ = ["penalize_range", "Penalties"]


def __dir__() -> list[str]:
    return __all__


def penalize_range(
    img: Tensor,
    allowed_range: tuple[float, float] = (0.0, 1.0),
    **kwargs: Any,
) -> Tensor:
    r"""
    Calculate quadratic penalty on values outside of ``allowed_range``.

    Provides a 'soft' pixel-range regularization by imposing a
    quadratic penalty on any values outside the allowed_range.
    All values within the allowed_range have a penalty of 0.

    To use as a ``penalty_function`` in synthesis methods,
    ``functools.partial`` must be used to fix the ``allowed_range``
    (see Examples).

    Parameters
    ----------
    img
        The tensor to penalize.
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
    Initialize metamer with allowed range (0.2, 0.8):

    >>> import plenoptic as po
    >>> img = po.data.einstein()
    >>> model = po.models.Gaussian(30).eval()
    >>> po.remove_grad(model)
    >>> # Make function penalizing values outside (0.2, 0.8)
    >>> def custom_range(x):
    ...     penalty = po.regularize.penalize_range(x, allowed_range=(0.2, 0.8))
    ...     return penalty
    >>> met_default = po.Metamer(img, model)
    >>> met_custom = po.Metamer(img, model, penalty_function=custom_range)
    >>> # Compare the value of the penalties
    >>> met_default.penalty_function(img)
    tensor(0.)
    >>> met_custom.penalty_function(img)
    tensor(49.3881)
    """
    # Using clip like this is equivalent to using boolean indexing (e.g.,
    # img[img < allowed_range[0]]) but much faster
    below_min = torch.clip(img - allowed_range[0], max=0).pow(2).sum()
    above_max = torch.clip(img - allowed_range[1], min=0).pow(2).sum()
    return below_min + above_max


class Penalties(torch.nn.Module):
    """
    Apply arbitrary number of penalty functions to image.

    This Module combines an arbitrary number of penalty functions and applies
    them separately to a single image, returning a single 1d tensor with the
    appropriate number of values.

    All appropriate synthesis methods will automatically construct this object
    with specified ``penalty_functions``. Users only need to construct it
    themselves if they wish to set ``names``.

    Parameters
    ----------
    penalty_functions
        A callable or list of callables applied during stimulus synthesis that
        returns a scalar penalty to be minimized. By penalizing certain
        properties of the image, like pixels values outside an allowed range, we
        can constrain those image properties. See :ref:`how-to-penalty` in the
        documentation for details and examples.
    names
        Either ``None``, a single string, or a list of the same length as
        ``penalty_functions``, containing the names of each member of
        ``penalty_functions``. If ``None``, we grab the member's name using the
        ``__name__`` attribute (for functions) or ``__class__.__name__``
        attribute (for :class:`torch.nn.Modules`). See Examples section.

    Raises
    ------
    ValueError
        If ``penalty_functions`` and ``names`` have different numbers of elements.

    Examples
    --------
    >>> import plenoptic as po
    >>> import torch
    >>> po.set_seed(0)
    >>> img = torch.randn((1, 1, 256, 256))
    >>> penalties = po.regularize.Penalties(po.regularize.penalize_range)
    >>> penalties(img)
    tensor([37770.3047])
    >>> penalties
    [
      0: penalize_range,
    ]

    We can grab names from the object automatically, but this is not always
    informative, e.g., if you use a ``lambda`` function. In that case, you can
    set ``names`` to override automatic names:

    >>> penalty_func = lambda x: po.regularize.penalize_range(x, (-1, 0))
    >>> po.regularize.Penalties(penalty_func)
    [
      0: <lambda>,
    ]
    >>> names = "encourage negative range"
    >>> po.regularize.Penalties(penalty_func, names=names)
    [
      0: encourage negative range,
    ]

    When multiple ``penalty_functions`` are specified, names can be a list of
    mixed names and ``None``, and we will automatically grab object names for
    each ``None``:

    >>> penalty_funcs = [
    ...     po.regularize.penalize_range,
    ...     lambda x: po.loss.mse(x, po.data.einstein()),
    ... ]
    >>> po.regularize.Penalties(penalty_funcs, names=[None, "We want Einstein"])
    [
      0: penalize_range,
      1: We want Einstein,
    ]

    ``penalty_functions`` can also include :class:`torch.nn.Module`, in which case we
    can move all sub-modules, buffers, etc. when calling ``to``

    >>> class GaussianPenalty(torch.nn.Module):
    ...     def __init__(self, std=3.0):
    ...         super().__init__()
    ...         self.model = po.models.Gaussian(31, std=std).eval()
    ...         po.remove_grad(self.model)
    ...         self.register_buffer("penalty_target", self.model(po.data.curie()))
    ...
    ...     def forward(self, x):
    ...         return po.loss.mse(self.model(x), self.penalty_target)
    >>> penalty_funcs = [po.regularize.penalize_range, GaussianPenalty()]
    >>> penalties = po.regularize.Penalties(penalty_funcs)
    >>> penalties(img)
    tensor([3.7770e+04, 2.1924e-01])
    >>> penalties.to(torch.float64)
    [
      0: penalize_range,
      1: GaussianPenalty,
    ]
    >>> penalties(img.to(torch.float64))
    tensor([3.7770e+04, 2.1924e-01], dtype=torch.float64)
    """

    def __init__(
        self,
        penalty_functions: Callable[[Tensor], Tensor]
        | list[Callable[[Tensor], Tensor]],
        names: None | list[str | None] = None,
    ):
        super().__init__()
        self.penalty_functions = []
        self.names = []
        if not isinstance(penalty_functions, list):
            penalty_functions = [penalty_functions]
        if names is None:
            names = len(penalty_functions) * [None]
        elif isinstance(names, str):
            names = [names]
        if len(penalty_functions) != len(names):
            raise ValueError(
                "penalty_functions and names must have same number of values!"
            )
        for i, (pen, name) in enumerate(zip(penalty_functions, names)):
            if name is None:
                try:
                    name = pen.__name__
                except AttributeError:
                    name = pen.__class__.__name__
            try:
                self.add_module(f"_penalty_module_{i}", pen)
                self.penalty_functions.append(getattr(self, f"_penalty_module_{i}"))
            except TypeError:
                self.penalty_functions.append(pen)
            self.names.append(name)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute all penalties on input tensor.

        Parameters
        ----------
        x
            The tensor to penalize.

        Returns
        -------
        penalty
            The penalty value(s), as a 1d tensor with one value per
            penalty function.

        Examples
        --------
        >>> import plenoptic as po
        >>> import torch
        >>> po.set_seed(0)
        >>> img = torch.randn((1, 1, 256, 256))
        >>> penalties = po.regularize.Penalties(po.regularize.penalize_range)
        >>> penalties(img)
        tensor([37770.3047])

        The number of returned elements equals the number of elements in
        ``penalty_functions`` passed at initialization:

        >>> penalty_funcs = [
        ...     po.regularize.penalize_range,
        ...     lambda x: po.loss.mse(x, po.data.einstein()),
        ... ]
        >>> penalties = po.regularize.Penalties(penalty_funcs)
        >>> penalties(img)
        tensor([3.7770e+04, 1.2359e+00])
        """  # numpydoc ignore=ES01
        return torch.stack([p(x) for p in self.penalty_functions])

    def __repr__(self) -> str:  # noqa: D105
        # numpydoc ignore=GL08
        repr_str = "[\n"
        tab = "  "
        for i, (pen, name) in enumerate(zip(self.penalty_functions, self.names)):
            repr_str += f"{tab}{i}: {name},\n"
        repr_str += "]"
        return repr_str
