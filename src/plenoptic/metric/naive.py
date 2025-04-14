import torch


def mse(img1, img2):
    r"""Return the MSE between img1 and img2.

    Our baseline metric to compare two images is often mean-squared
    error, MSE. This is not a good approximation of the human visual
    system, but is handy to compare against.

    For two images, :math:`x` and :math:`y`, with :math:`n` pixels
    each:

    .. math::

        MSE &= \frac{1}{n}\sum_i=1^n (x_i - y_i)^2

    The two images must have a float dtype

    Parameters
    ----------
    img1 : torch.Tensor
        The first image to compare
    img2 : torch.Tensor
        The second image to compare, must be same size as ``img1``

    Returns
    -------
    mse : torch.float
        the mean-squared error between ``img1`` and ``img2``

    Examples
    --------
    >>> import plenoptic as po
    >>> import torch
    >>> einstein = po.data.einstein()
    >>> po.tools.set_seed(0)
    >>> einstein_noisy = einstein + 0.1 * torch.randn_like(einstein)
    >>> po.metric.mse(einstein, einstein_noisy)
    tensor([[0.0100]])
    """
    return torch.pow(img1 - img2, 2).mean((-1, -2))
