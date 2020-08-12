import torch


def penalize_range(x, allowed_range=(0, 1)):
    r"""penalize values outside of allowed_range
    instead of clamping values to exactly fall in a range, this provides
    a 'softer' way of doing it, by imposing a quadratic penalty on any
    values outside the allowed_range. All values within the
    allowed_range have a penalty of 0
    Parameters
    ----------
    x : torch.Tensor
        the tensor to penalize. the synthesized image.
    allowed_range : tuple, optional
        2-tuple of values giving the (min, max) allowed values
    kwargs :
        ignored, only present to absorb extra arguments
    Returns
    -------
    penalty : torch.float
        penalty for values outside range
    """
    # the indexing should flatten it
    below_min = x[x < allowed_range[0]]
    below_min = torch.pow(below_min - allowed_range[0], 2)
    above_max = x[x > allowed_range[1]]
    above_max = torch.pow(above_max - allowed_range[1], 2)
    return torch.sum(torch.cat([below_min, above_max]))


def pretty_print(i, max_iter, t, loss, g_norm,lr=[]):
    if lr==[]:
        print('iter', i + 1, '/', max_iter,
              '\tloss',      round(loss, 4),
              '\ttime',      round(t, 3),
              '\tgrad norm', round(g_norm, 3))

    else:
        print('iter', i + 1, '/', max_iter,
              '\tloss',      round(loss, 4),
              '\ttime',      round(t, 3),
              '\tgrad norm', round(g_norm, 3),
              '\tlr', round(lr,6))
