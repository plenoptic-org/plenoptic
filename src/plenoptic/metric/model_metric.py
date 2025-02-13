import torch

# the list of functions that are safe for torch loader. these are the functions here
# that can be used as metrics for synthesis
_SAFE_FUNCS = ["model_metric"]
__all__ = _SAFE_FUNCS


def __dir__() -> list[str]:
    return __all__


def model_metric(x, y, model):
    """
    Calculate distance between x and y in model space root mean squared error

    Parameters
    ----------
    image: torch.Tensor
        image, (B x C x H x W)
    model: torch class
        torch model with defined forward and backward operations

    Notes
    -----


    """

    repx = model(x)
    repy = model(y)

    # for optimization purpose (stabilizing the gradient around zero)
    epsilon = 1e-10

    dist = torch.sqrt(torch.mean((repx - repy) ** 2) + epsilon)

    return dist
