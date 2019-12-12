import torch


def kurtosis(mtx):
    # implementation is only for real components
    return torch.mean(torch.abs(mtx-mtx.mean()).pow(4))/(mtx.var().pow(2))

def skew(mtx):
    return torch.mean((mtx-mtx.mean()).pow(3))/(mtx.var().pow(1.5))
