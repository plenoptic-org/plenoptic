"""
Model metrics.

Simple functions to convert models, which can return a tensor of arbitrary shape, to
metrics, which must return a tensor.
"""  # numpydoc ignore=EX01

import torch


def model_metric(
    x: torch.Tensor, y: torch.Tensor, model: torch.nn.Module
) -> torch.Tensor:
    r"""
    Calculate distance between x and y in model space root mean squared error.

    For two images, :math:`x` and :math:`y`, and model :math:`M`.

    .. math::

        metric = \sqrt{\frac{1}{n}\sum_i (M(x)_i - M(y)_i)^2+\epsilon}

    where :math:`M(x)` and :math:`M(y)` are the model representations of ``x`` and
    ``y``, with :math:`n` elements, and :math:`\epsilon=1e-10` is to stabilize the
    gradient around zero.

    Parameters
    ----------
    x, y
        Images to pass to ``model``.
    model
        Torch model with defined forward operation.

    Returns
    -------
    model_error
        Root mean-squared error between the model representation of ``x`` and ``y``.

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
