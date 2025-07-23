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

    Examples
    --------
    >>> import torch
    >>> from plenoptic.tools.stats import variance
    >>> x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    >>> v = variance(x)
    >>> v
    tensor(1.6667)

    If you have precomputed the mean, you can pass it and avoid recomputing it:

    >>> precomputed_mean = torch.mean(x)
    >>> v = variance(x, mean=precomputed_mean)
    >>> v
    tensor(1.6667)

    If you want to compute along a specific dimension, you can specify it:

    >>> v = variance(x, dim=0)
    >>> v
    tensor([1., 1., 1.])

    This function differs from ``torch.var`` in that it does not apply a correction:

    >>> plenoptic_v_corrected = v * x.shape[0] / (x.shape[0] - 1)
    >>> torch_v = torch.var(x, dim=0)
    >>> torch.isclose(plenoptic_v_corrected, torch_v)
    tensor([True, True, True])
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

    Examples
    --------
    >>> import torch
    >>> from plenoptic.tools.stats import skew, variance
    >>> x = torch.tensor([[1.0, 2.0, 3.0, 2.0], [3.0, 4.0, 5.0, 3.0]])
    >>> s = skew(x)
    tensor(0.2440)

    If you have precomputed the mean and/or variance,
    you can pass them and avoid recomputing:

    >>> precomputed_mean = torch.mean(x)
    >>> precomputed_var = variance(x)
    >>> s = skew(x, mean=precomputed_mean, var=precomputed_var)
    >>> s
    tensor(0.2440)

    If you want to compute along a specific dimension, you can specify it:

    >>> s = skew(x, dim=0)
    >>> s
    tensor([0., 0., 0., 0.])
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

    Examples
    --------
    >>> import torch
    >>> from plenoptic.tools.stats import kurtosis, variance
    >>> x = torch.tensor([[1.0, 2.0, 3.0, 2.0], [3.0, 4.0, 5.0, 3.0]])
    >>> k = kurtosis(x)
    tensor(2.4031)

    If you have precomputed the mean and/or variance,
    you can pass them and avoid recomputing:

    >>> precomputed_mean = torch.mean(x)
    >>> precomputed_var = variance(x)
    >>> k = kurtosis(x, mean=precomputed_mean, var=precomputed_var)
    >>> k
    tensor(2.4031)

    If you want to compute along a specific dimension, you can specify it:

    >>> k = kurtosis(x, dim=0)
    >>> k
    tensor([1., 1., 1., 1.])
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if mean is None:
        mean = torch.mean(x, dim=dim, keepdim=True)
    if var is None:
        var = variance(x, mean=mean, dim=dim, keepdim=keepdim)
    return torch.mean(torch.abs(x - mean).pow(4), dim=dim, keepdim=keepdim) / var.pow(2)
