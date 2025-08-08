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
        >>> from plenoptic.tools import set_seed
        >>> set_seed(42)
        >>> x = torch.randn(10000)
        >>> v = variance(x)
        >>> x_more = x * 3
        >>> v_more = variance(x_more)
        >>> x_less = x * 0.3
        >>> v_less = variance(x_less)
        >>> fig, (ax_less, ax, ax_more) = plt.subplots(
        ...     1, 3, sharex=True, sharey=True, figsize=(12, 4)
        ... )
        >>> _ = ax_less.hist(x_less, bins=50)
        >>> _ = ax_less.set(title=f"σ=0.3\nVariance: {v_less:.4f}", ylabel="Frequency")
        >>> _ = ax.hist(x, bins=50)
        >>> _ = ax.set(title=f"Standard Gaussian, σ=1\nVariance: {v:.4f}")
        >>> _ = ax_more.hist(x_more, bins=50)
        >>> _ = ax_more.set(title=f"σ=3\nVariance: {v_more:.4f}")

    If you have precomputed the mean, you can pass it and avoid recomputing it:

    >>> precomputed_mean = torch.mean(x)
    >>> v = variance(x, mean=precomputed_mean)
    >>> v
    tensor(1.0088)

    If you want to compute along a specific dimension, you can specify it:

    >>> x = torch.randn(10000, 2)
    >>> v = variance(x, dim=0)
    >>> v
    tensor([1.0127, 1.0045])

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
        >>> from plenoptic.tools.stats import skew
        >>> from plenoptic.tools import set_seed
        >>> set_seed(42)
        >>> x = torch.randn(10000)
        >>> s = skew(x)
        >>> x_right = torch.exp(x / 2)
        >>> s_right = skew(x_right)
        >>> x_left = -torch.exp(x / 2)
        >>> s_left = skew(x_left)
        >>> fig, (ax_left, ax, ax_right) = plt.subplots(
        ...     1, 3, sharex=True, figsize=(12, 4)
        ... )
        >>> _ = ax_left.hist(x_left, bins=50)
        >>> _ = ax_left.set(
        ...     title=f"Left skew: {s_left:.4f}", ylabel="Frequency", xlim=(-5, 5)
        ... )
        >>> _ = ax.hist(x, bins=50)
        >>> _ = ax.set(title=f"Standard Gaussian\nSkew: {s:.4f}")
        >>> _ = ax_right.hist(x_right, bins=50)
        >>> _ = ax_right.set(title=f"Right skew: {s_right:.4f}")

    If you have precomputed the mean and/or variance,
    you can pass them and avoid recomputing:

    >>> precomputed_mean = torch.mean(x)
    >>> precomputed_var = variance(x)
    >>> s = skew(x, mean=precomputed_mean, var=precomputed_var)
    >>> s
    tensor(-0.0010)

    If you want to compute along a specific dimension, you can specify it:

    >>> x = torch.randn(10000, 2)
    >>> s = skew(x, dim=0)
    >>> s
    tensor([-0.0257, -0.0063])
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
    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import torch
        >>> from plenoptic.tools.stats import kurtosis
        >>> from plenoptic.tools import set_seed
        >>> set_seed(42)
        >>> x = torch.randn(10000)
        >>> k = kurtosis(x)
        >>> x_platy = torch.rand(10000) * 10 - 5
        >>> k_platy = kurtosis(x_platy)
        >>> x_lepto = torch.distributions.Laplace(loc=0.0, scale=1.0).sample((10000,))
        >>> k_lepto = kurtosis(x_lepto)
        >>> fig, (ax_platy, ax, ax_lepto) = plt.subplots(
        ...     1, 3, sharex=True, figsize=(12, 4)
        ... )
        >>> _ = ax_platy.hist(x_platy.numpy(), bins=50)
        >>> _ = ax_platy.set(
        ...     title=f"Platykurtic (Uniform)\nKurtosis: {k_platy:.4f}",
        ...     ylabel="Frequency",
        ...     xlim=(-5, 5),
        ... )
        >>> _ = ax.hist(x.numpy(), bins=50)
        >>> _ = ax.set(title=f"Standard Gaussian\nKurtosis: {k:.4f}")
        >>> _ = ax_lepto.hist(x_lepto.numpy(), bins=50)
        >>> _ = ax_lepto.set(title=f"Leptokurtic (Laplace)\nKurtosis: {k_lepto:.4f}")

    If you have precomputed the mean and/or variance,
    you can pass them and avoid recomputing:

    >>> precomputed_mean = torch.mean(x)
    >>> precomputed_var = variance(x)
    >>> k = kurtosis(x, mean=precomputed_mean, var=precomputed_var)
    >>> k
    tensor(2.9354)

    If you want to compute along a specific dimension, you can specify it:

    >>> x = torch.randn(10000, 2)
    >>> k = kurtosis(x, dim=0)
    >>> k
    tensor([3.0057, 2.9506])
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if mean is None:
        mean = torch.mean(x, dim=dim, keepdim=True)
    if var is None:
        var = variance(x, mean=mean, dim=dim, keepdim=keepdim)
    return torch.mean(torch.abs(x - mean).pow(4), dim=dim, keepdim=keepdim) / var.pow(2)
