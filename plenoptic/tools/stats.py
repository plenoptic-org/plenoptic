import torch


def kurtosis(x):
    r"""sample estimate of 'x' *tailedness* (outliers)
    kurtosis of univariate noral is 3
    smaller than 3: *platykurtic* (eg. uniform distribution)
    greater than 3: *leptokurtic* (eg. Laplace distribution)
    """
    # implementation is only for real components
    return torch.mean(torch.abs(x-x.mean()).pow(4))/(x.var().pow(2))


def skew(x):
    r"""sample estimate of 'x' *asymmetry* about its mean
    """
    return torch.mean((x-x.mean()).pow(3))/(x.var().pow(1.5))
