"""Functions for computing image statistics on multi-dimensional tensors."""

# numpydoc ignore=ES01
import torch
from torch import Tensor


def variance(
    x: Tensor,
    mean: float | Tensor | None = None,
    dim: int | list[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    r"""
    Calculate sample variance.

    Note that this is the uncorrected, or sample, variance, corresponding to
    ``torch.var(*, correction=0)``.

    Parameters
    ----------
    x
        The input tensor.
    mean
        Reuse a precomputed mean.
    dim
        The dimension or dimensions to reduce.
    keepdim
        Whether to retain the reduced dimensions (as singletons) or not.

    Returns
    -------
    out
        The variance tensor.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if mean is None:
        mean = torch.mean(x, dim=dim, keepdim=True)
    return torch.mean((x - mean).pow(2), dim=dim, keepdim=keepdim)


def skew(
    x: Tensor,
    mean: float | Tensor | None = None,
    var: float | Tensor | None = None,
    dim: int | list[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    r"""
    Calculate sample estimate of *asymmetry* about input's mean.

    To help with interpretation:

    - Skew of normal distribution is 0.

    - Negative skew, also known as left-skewed: the left tail is longer. Distribution
      appears as a right-leaning curve.

    - Positive skew, also known as right-skewed: the right tail is longer. Distribution
      appears as a left-leaning curve.

    Parameters
    ----------
    x
        The input tensor.
    mean
        Reuse a precomputed mean.
    var
        Reuse a precomputed variance.
    dim
        The dimension or dimensions to reduce.
    keepdim
        Whether to retain the reduced dimensions (as singletons) or not.

    Returns
    -------
    out
        The skewness tensor.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if mean is None:
        mean = torch.mean(x, dim=dim, keepdim=True)
    if var is None:
        var = variance(x, mean=mean, dim=dim, keepdim=keepdim)
    return torch.mean((x - mean).pow(3), dim=dim, keepdim=keepdim) / var.pow(1.5)


def kurtosis(
    x: Tensor,
    mean: float | Tensor | None = None,
    var: float | Tensor | None = None,
    dim: int | list[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    r"""
    Calculate sample estimate of *tailedness* (presence of outliers).

    To help with interpretation:

    - Kurtosis of univariate normal is 3.

    - Smaller than 3: *platykurtic* (e.g. uniform distribution).

    - Greater than 3: *leptokurtic* (e.g. Laplace distribution).

    Parameters
    ----------
    x
        The input tensor.
    mean
        Reuse a precomputed mean.
    var
        Reuse a precomputed variance.
    dim
        The dimension or dimensions to reduce.
    keepdim
        Whether to retain the reduced dimensions (as singletons) or not.

    Returns
    -------
    out
        The kurtosis tensor.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if mean is None:
        mean = torch.mean(x, dim=dim, keepdim=True)
    if var is None:
        var = variance(x, mean=mean, dim=dim, keepdim=keepdim)
    return torch.mean(torch.abs(x - mean).pow(4), dim=dim, keepdim=keepdim) / var.pow(2)
