"""tools related to optimization

such as more objective functions
"""
import torch


def l2_norm(x, y):
    r"""L2-norm of the difference between x and y

    good default objective function

    Parameters
    ----------
    x : torch.tensor
        The first tensor to compare
    y : torch.tensor
        The second tensor to compare, must be same size as ``y``

    Returns
    -------
    loss : torch.float
        the L2-norm of the differencebetween ``x`` and ``y``

    """
    return torch.norm(x - y, p=2)


def penalize_range(x, allowed_range=(0, 1)):
    r"""penalize values outside of allowed_range

    instead of clamping values to exactly fall in a range, this provides
    a 'softer' way of doing it, by imposing a quadratic penalty on any
    values outside the allowed_range. All values within the
    allowed_range have a penalty of 0

    Parameters
    ----------
    x : torch.tensor
        the tensor to penalize. probably an image.
    allowed_range : tuple, optional
        2-tuple of values giving the (min, max) allowed values

    """
    # the indexing should flatten it
    below_min = x[x < allowed_range[0]]
    below_min = torch.pow(below_min - allowed_range[0], 2)
    above_max = x[x > allowed_range[1]]
    above_max = torch.pow(above_max - allowed_range[1], 2)
    return torch.sum(torch.cat([below_min, above_max]))
