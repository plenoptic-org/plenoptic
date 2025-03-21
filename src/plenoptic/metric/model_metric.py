import torch


def model_metric(x, y, model):
    """
    Calculate distance between x and y in model space root mean squared error

    Parameters
    ----------
    x, y:
        images with shape (batch, channel, height, width)
    model:
        torch model with defined forward and backward operations

    Examples
    --------
    >>> import plenoptic as po
    >>> einstein_img = po.data.einstein()
    >>> curie_img = po.data.curie()
    >>> model = po.simul.Gaussian(30)
    >>> model_metric = po.metric.model_metric(einstein_img, curie_img, model)
    >>> model_metric
    tensor(0.3128, grad_fn=<SqrtBackward0>)
    >>> # calculate this model metric manually:
    >>> torch.mean((model(einstein_img) - model(curie_img)).pow(2)).sqrt()
    tensor(0.3128, grad_fn=<SqrtBackward0>)
    """

    repx = model(x)
    repy = model(y)

    # for optimization purpose (stabilizing the gradient around zero)
    epsilon = 1e-10

    dist = torch.sqrt(torch.mean((repx - repy) ** 2) + epsilon)

    return dist
