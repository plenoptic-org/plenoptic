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

    See Also
    --------
    :func:`~plenoptic.tools.stats.skewness`
        Calculate sample skewness.
    :func:`~plenoptic.tools.stats.kurtosis`
        Calculate sample kurtosis.

    Examples
    --------
    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import torch
        >>> from plenoptic.tools.stats import variance
        >>> torch.manual_seed(42)
        >>> x1 = torch.normal(mean=0, std=1, size=(10000,))
        >>> v1 = variance(x1)
        >>> x2 = torch.normal(mean=0, std=3, size=(10000,))
        >>> v2 = variance(x2)
        >>> fig, (ax1, ax2) = plt.subplots(
        ...     1, 2, sharex=True, sharey=True, figsize=(8, 4)
        ... )
        >>> ax1.hist(x1, bins=50)
        >>> ax1.set_title(f"Variance: {v1:.4f}")
        >>> ax1.set_ylabel("Frequency")
        >>> ax2.hist(x2, bins=50)
        >>> ax2.set_title(f"Variance: {v2:.4f}")
        >>> plt.show()

    If you have precomputed the mean, you can pass it and avoid recomputing it:

    >>> precomputed_mean = torch.mean(x1)
    >>> v = variance(x1, mean=precomputed_mean)
    >>> v
    tensor(1.0088)

    If you want to compute along a specific dimension, you can specify it:

    >>> x = torch.normal(mean=torch.zeros(100, 2), std=torch.tensor([1.0, 2.0]))
    >>> v = variance(x, dim=0)
    >>> v
    tensor([0.9067, 2.9203])

    This function differs from :func:`torch.var` in that it does not apply a correction:

    >>> plenoptic_v_corrected = v * x.shape[0] / (x.shape[0] - 1)
    >>> torch_v = torch.var(x, dim=0)
    >>> torch.isclose(plenoptic_v_corrected, torch_v)
    tensor([True, True])
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

    See Also
    --------
    :func:`~plenoptic.tools.stats.variance`
        Calculate sample variance.

    Examples
    --------
    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import torch
        >>> from plenoptic.tools.stats import variance
        >>> torch.manual_seed(42)
        >>> x1 = torch.normal(mean=0, std=1, size=(10000,))
        >>> v1 = variance(x1)
        >>> x2 = torch.normal(mean=0, std=3, size=(10000,))
        >>> v2 = variance(x2)
        >>> fig, (ax1, ax2) = plt.subplots(
        ...     1, 2, sharex=True, sharey=True, figsize=(8, 4)
        ... )
        >>> ax1.hist(x1, bins=50)
        >>> ax1.set_title(f"Variance: {v1:.4f}")
        >>> ax1.set_ylabel("Frequency")
        >>> ax2.hist(x2, bins=50)
        >>> ax2.set_title(f"Variance: {v2:.4f}")
        >>> plt.show()

    >>> import torch
    >>> from plenoptic.tools.stats import skew, variance
    >>> x = torch.tensor([[1.0, 2.0, 3.0, 2.0], [3.0, 4.0, 5.0, 3.0]])
    >>> s = skew(x)
    >>> s
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

    See Also
    --------
    :func:`~plenoptic.tools.stats.variance`
        Calculate sample variance.

    Examples
    --------
    >>> import torch
    >>> from plenoptic.tools.stats import kurtosis, variance
    >>> x = torch.tensor([[1.0, 2.0, 3.0, 2.0], [3.0, 4.0, 5.0, 3.0]])
    >>> k = kurtosis(x)
    >>> k
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
