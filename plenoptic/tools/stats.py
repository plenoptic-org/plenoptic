import torch


def variance(x, mean=None, dim=None, keepdim=False):
    r"""sample estimate of `x` *variability*
    Arguments
    ----------
    x: torch.Tensor
        the input tensor
    mean: torch.Tensor or None, optional
        reuse a precomputed mean
    dim: `int`, `list` of `int`, or None, optional
        the dimension or dimensions to reduce.
    keepdim: `bool`, optional
        whether the output tensor has dim retained or not.
    Returns
    -------
    out: torch.Tensor
        the output tensor.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if mean is None:
        mean = torch.mean(x, dim=dim, keepdim=True)
    return torch.mean((x - mean).pow(2), dim=dim, keepdim=keepdim)


def skew(x, mean=None, var=None, dim=None, keepdim=False):
    r"""sample estimate of `x` *asymmetry* about its mean
    Arguments
    ----------
    x: torch.Tensor
        the input tensor
    mean: torch.Tensor or None, optional
        reuse a precomputed mean
    var: torch.Tensor or None, optional
        reuse a precomputed variance
    dim: `int`, `list` of `int`, or None, optional
        the dimension or dimensions to reduce.
    keepdim: `bool`, optional
        whether the output tensor has dim retained or not.
    Returns
    -------
    out: torch.Tensor
        the output tensor.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if mean is None:
        mean = torch.mean(x, dim=dim, keepdim=True)
    if var is None:
        var = variance(x, mean=mean, dim=dim, keepdim=keepdim)
    return torch.mean((x - mean).pow(3),
                      dim=dim, keepdim=keepdim)/var.pow(1.5)


def kurtosis(x, mean=None, var=None, dim=None, keepdim=False):
    r"""sample estimate of `x` *tailedness* (presence of outliers)
    kurtosis of univariate noral is 3
    smaller than 3: *platykurtic* (eg. uniform distribution)
    greater than 3: *leptokurtic* (eg. Laplace distribution)
    Arguments
    ----------
    x: torch.Tensor
        the input tensor
    mean: torch.Tensor or None, optional
        reuse a precomputed mean
    var: torch.Tensor or None, optional
        reuse a precomputed variance
    dim: `int`, `list` of `int`, or None, optional
        the dimension or dimensions to reduce.
    keepdim: `bool`, optional
        whether the output tensor has dim retained or not.
    Returns
    -------
    out: torch.Tensor
        the output tensor.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if mean is None:
        mean = torch.mean(x, dim=dim, keepdim=True)
    if var is None:
        var = variance(x, mean=mean, dim=dim, keepdim=keepdim)
    return torch.mean(torch.abs(x - mean).pow(4),
                      dim=dim, keepdim=keepdim)/var.pow(2)