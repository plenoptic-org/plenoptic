import torch


def skew(x, dim=None, keepdim=False):
    r"""sample estimate of 'x' *asymmetry* about its mean
    """
    return torch.mean((x - x.mean(dim=dim, keepdim=keepdim)).pow(3), dim=dim, keepdim=keepdim)/(x.var(dim=dim, keepdim=keepdim).pow(1.5))


def kurtosis(x, dim=None, keepdim=False):
    r"""sample estimate of 'x' *tailedness* (outliers)
    kurtosis of univariate noral is 3
    smaller than 3: *platykurtic* (eg. uniform distribution)
    greater than 3: *leptokurtic* (eg. Laplace distribution)
    """
    # implementation is only for real components
    return torch.mean(torch.abs(x - x.mean(dim=dim, keepdim=keepdim)).pow(4), dim=dim, keepdim=keepdim)/(x.var(dim=dim, keepdim=keepdim).pow(2))
