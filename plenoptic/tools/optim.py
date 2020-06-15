"""tools related to optimization

such as more objective functions
"""
import torch


def mse(synth_rep, ref_rep, **kwargs):
    r"""return the MSE between synth_rep and ref_rep

    For two tensors, :math:`x` and :math:`y`, with :math:`n` values
    each:

    .. math::

        MSE &= \frac{1}{n}\sum_i=1^n (x_i - y_i)^2

    The two images must have a float dtype

    Parameters
    ----------
    synth_rep : torch.tensor
        The first tensor to compare, model representation of the
        synthesized image
    ref_rep : torch.tensor
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``,
    kwargs :
        ignored, only present to absorb extra arguments

    Returns
    -------
    loss : torch.float
        the mean-squared error between ``synth_rep`` and ``ref_rep``

    """
    return torch.pow(synth_rep - ref_rep, 2).mean()


def l2_norm(synth_rep, ref_rep, **kwargs):
    r"""L2-norm of the difference between ref_rep and synth_rep

    good default objective function

    Parameters
    ----------
    synth_rep : torch.tensor
        The first tensor to compare, model representation of the
        synthesized image
    ref_rep : torch.tensor
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``,
    kwargs :
        ignored, only present to absorb extra arguments

    Returns
    -------
    loss : torch.float
        the L2-norm of the difference between ``ref_rep`` and ``synth_rep``

    """
    return torch.norm(ref_rep - synth_rep, p=2)


def penalize_range(synth_img, allowed_range=(0, 1), **kwargs):
    r"""penalize values outside of allowed_range

    instead of clamping values to exactly fall in a range, this provides
    a 'softer' way of doing it, by imposing a quadratic penalty on any
    values outside the allowed_range. All values within the
    allowed_range have a penalty of 0

    Parameters
    ----------
    synth_img : torch.tensor
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
    below_min = synth_img[synth_img < allowed_range[0]]
    below_min = torch.pow(below_min - allowed_range[0], 2)
    above_max = synth_img[synth_img > allowed_range[1]]
    above_max = torch.pow(above_max - allowed_range[1], 2)
    return torch.sum(torch.cat([below_min, above_max]))


def log_barrier(synth_img, allowed_range=(0, 1), epsilon=1., **kwargs):
    r"""use a log barrier to prevent values outside allowed_range

    this returns: ``epsilon * torch.mean(torch.log((synth_img -
    allowed_range[0]) * (allowed_range[1] * synth_img)))``

    everything outside of ``allowed_range`` has an infinite penalty.

    NOTE: this currently seems to not work unless you're very careful,
    because the optimizer can easily adjust one of the pixels to outside
    the allowed_range (unless the step size is small), resulting in
    infinite loss

    Parameters
    ----------
    synth_img : torch.tensor
        the tensor to penalize. the synthesized image.
    allowed_range : tuple, optional
        2-tuple of values giving the (min, max) allowed values
    epsilon : float, optional
        parameter to control magnitude of penalty. the smaller this is,
        the closer this approximation is to the step function that
        provides a penalty of 0 to everything within ``allowed_range``
        and infinite penalty to everything outside it, but the harder it
        is to optimize
    kwargs :
        ignored, only present to absorb extra arguments

    Returns
    -------
    penalty : torch.float
        penalty for values outside range

    """
    img = synth_img.flatten()
    penalty = torch.log((img - allowed_range[0]) * (allowed_range[1] - img))
    return epsilon * -torch.mean(penalty)


def l2_and_penalize_range(synth_rep, ref_rep, synth_img, allowed_range=(0, 1), beta=.5, **kwargs):
    """loss that combines L2-norm of the difference and range penalty

    this function returns a weighted average of the L2-norm of the
    difference between ``ref_rep`` and ``synth_rep`` (as calculated by
    ``l2_norm()``) and the range penalty of ``synth_img`` (as calculated
    by ``penalize_range()``).

    The loss is: ``beta * l2_norm(ref_rep, synth_rep) + (1-beta) *
    penalize_range(synth_img, allowed_range)``

    Parameters
    ----------
    synth_rep : torch.tensor
        The first tensor to compare, model representation of the
        synthesized image
    ref_rep : torch.tensor
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``,
    synth_img : torch.tensor
        the tensor to penalize. the synthesized image.
    allowed_range : tuple, optional
        2-tuple of values giving the (min, max) allowed values
    beta : float, optional
        parameter that gives the tradeoff between L2-norm of the
        difference and the range penalty
    kwargs :
        ignored, only present to absorb extra arguments

    Returns
    -------
    loss : torch.float
        the loss

    """
    l2_loss = l2_norm(ref_rep, synth_rep)
    range_penalty = penalize_range(synth_img, allowed_range)
    return beta * l2_loss + (1-beta) * range_penalty


def mse_and_penalize_range(synth_rep, ref_rep, synth_img, allowed_range=(0, 1), beta=.5, **kwargs):
    """loss that combines MSE of the difference and range penalty

    this function returns a weighted average of the MSE of the
    difference between ``ref_rep`` and ``synth_rep`` (as calculated by
    ``mse()``) and the range penalty of ``synth_img`` (as calculated by
    ``penalize_range()``).

    The loss is: ``beta * mse(ref_rep, synth_rep) + (1-beta) *
    penalize_range(synth_img, allowed_range)``

    Parameters
    ----------
    synth_rep : torch.tensor
        The first tensor to compare, model representation of the
        synthesized image
    ref_rep : torch.tensor
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``,
    synth_img : torch.tensor
        the tensor to penalize. the synthesized image.
    allowed_range : tuple, optional
        2-tuple of values giving the (min, max) allowed values
    beta : float, optional
        parameter that gives the tradeoff between MSE of the difference
        and the range penalty
    kwargs :
        ignored, only present to absorb extra arguments

    Returns
    -------
    loss : torch.float
        the loss

    """
    mse_loss = mse(ref_rep, synth_rep)
    range_penalty = penalize_range(synth_img, allowed_range)
    return beta * mse_loss + (1-beta) * range_penalty


def l2_and_log_barrier(synth_rep, ref_rep, synth_img, allowed_range=(0, 1), epsilon=1, **kwargs):
    """loss the combines L2-norm of the difference and range penalty

    this function returns the sum of the L2-norm of the difference
    between ``ref_rep`` and ``synth_rep`` (as calculated by
    ``l2_norm()``) and the log-barrier penalty of ``synth_img`` (as
    calculated by ``log_barrier()``).

    The loss is: ``l2_norm(ref_rep, synth_rep) + log_barrier(synth_img,
    allowed_range, epsilon)``

    Parameters
    ----------
    synth_rep : torch.tensor
        The first tensor to compare, model representation of the
        synthesized image
    ref_rep : torch.tensor
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``,
    synth_img : torch.tensor
        the tensor to penalize. the synthesized image.
    allowed_range : tuple, optional
        2-tuple of values giving the (min, max) allowed values
    epsilon : float, optional
        parameter to control magnitude of penalty. the smaller this is,
        the closer this approximation is to the step function that
        provides a penalty of 0 to everything within ``allowed_range``
        and infinite penalty to everything outside it, but the harder it
        is to optimize
    kwargs :
        ignored, only present to absorb extra arguments

    Returns
    -------
    loss : torch.float
        the loss

    """
    l2_loss = l2_norm(synth_rep, ref_rep)
    range_penalty = log_barrier(synth_img, allowed_range, epsilon)
    return l2_loss + range_penalty
