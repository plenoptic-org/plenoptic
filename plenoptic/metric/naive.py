import torch


def mse(img1, img2):
    r"""return the MSE between img1 and img2

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
    img1 : torch.tensor
        The first image to compare
    img2 : torch.tensor
        The second image to compare, must be same size as ``img1``

    Returns
    -------
    mse : torch.float
        the mean-squared error between ``img1`` and ``img2``

    """
    return torch.pow(img1 - img2, 2).mean((-1, -2))
