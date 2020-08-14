"""tools related to optimization

such as more objective functions
"""
import torch
import warnings
import numpy as np
from skimage import color
import os.path as op
import imageio
from glob import glob


def mse(synth_rep, ref_rep, **kwargs):
    r"""return the MSE between synth_rep and ref_rep

    For two tensors, :math:`x` and :math:`y`, with :math:`n` values
    each:

    .. math::

        MSE &= \frac{1}{n}\sum_i=1^n (x_i - y_i)^2

    The two images must have a float dtype

    Parameters
    ----------
    synth_rep : torch.Tensor
        The first tensor to compare, model representation of the
        synthesized image
    ref_rep : torch.Tensor
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
    synth_rep : torch.Tensor
        The first tensor to compare, model representation of the
        synthesized image
    ref_rep : torch.Tensor
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
    synth_img : torch.Tensor
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


def l2_and_penalize_range(synth_rep, ref_rep, synth_img, allowed_range=(0, 1),
                          lmbda=.1, **kwargs):
    """Loss the combines L2-norm of the difference and range penalty.

    this function returns a weighted average of the L2-norm of the difference
    between ``ref_rep`` and ``synth_rep`` (as calculated by ``l2_norm()``) and
    the range penalty of ``synth_img`` (as calculated by ``penalize_range()``).

    The loss is: ``l2_norm(synth_rep, ref_rep) + lmbda *
    penalize_range(synth_img, allowed_range)``

    Parameters
    ----------
    synth_rep : torch.Tensor
        The first tensor to compare, model representation of the
        synthesized image
    ref_rep : torch.Tensor
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``,
    synth_img : torch.Tensor
        the tensor to penalize. the synthesized image.
    allowed_range : tuple, optional
        2-tuple of values giving the (min, max) allowed values
    lmbda : float, optional
        parameter that gives the tradeoff between L2-norm of the
        difference and the range penalty, as described above
    kwargs :
        ignored, only present to absorb extra arguments

    Returns
    -------
    loss : torch.float
        the loss

    """
    l2_loss = l2_norm(synth_rep, ref_rep)
    range_penalty = penalize_range(synth_img, allowed_range)
    return l2_loss + lmbda * range_penalty


def mse_and_penalize_range(synth_rep, ref_rep, synth_img, allowed_range=(0, 1),
                           lmbda=.1, **kwargs):
    """Loss the combines MSE of the difference and range penalty.

    this function returns a weighted average of the MSE of the difference
    between ``ref_rep`` and ``synth_rep`` (as calculated by ``mse()``) and
    the range penalty of ``synth_img`` (as calculated by ``penalize_range()``).

    The loss is: ``mse(synth_rep, ref_rep) + lmbda * penalize_range(synth_img,
    allowed_range)``

    Parameters
    ----------
    synth_rep : torch.Tensor
        The first tensor to compare, model representation of the
        synthesized image
    ref_rep : torch.Tensor
        The second tensor to compare, model representation of the
        reference image. must be same size as ``synth_rep``,
    synth_img : torch.Tensor
        the tensor to penalize. the synthesized image.
    allowed_range : tuple, optional
        2-tuple of values giving the (min, max) allowed values
    lmbda : float, optional
        parameter that gives the tradeoff between MSE of the
        difference and the range penalty, as described above
    kwargs :
        ignored, only present to absorb extra arguments

    Returns
    -------
    loss : torch.float
        the loss

    """
    mse_loss = mse(synth_rep, ref_rep)
    range_penalty = penalize_range(synth_img, allowed_range)
    return mse_loss + lmbda * range_penalty


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


def generate_norm_stats(model, input_dir, save_path=None, img_shape=None, as_gray=True,
                        index=None):
    r"""Generate the statistics we want to use for normalization in models

    We sometimes want to normalize our models by whitening their
    internal representation (i.e., setting their mean to 0 and their
    covariance to the identity matrix). In practice, you need many
    samples to do this, but at the very least you can approximate this
    by z-scoring them, subtracting their mean and dividing by their
    standard deviation. In either case, to do this you need to get the
    model's statistics across a variety of images.

    This function will help you do that: by taking a model and an input
    directory, will load every image we find in that directory
    (non-recursively), and combine them into a giant 4d tensor. We then
    pass this to the model (so it must be able to work on batched
    images) to get its representation of all of these. If the model has
    a ``to_normalize`` attribute (a list of strings specifying which
    attributes you want to normalize), we'll go through and grab those
    attributes; otherwise we'll use the value returned by
    ``model(images)`` (i.e., from its forward method). This will be a
    dictionary with keys for each value of ``to_normalize`` or just
    ``"representation"`` if the model has no ``to_normalize``
    attribute. We then average so that we have a single value per batch
    and per channel (we should be able to figure out how many dimensions
    that requires averaging over), and save the resulting dictionary at
    save_path (if it's not None) and return it.

    Caveats / notes:

    - Since we're combining all the images into one tensor, they have to
      be the same shape. This is specified using ``img_shape`` (so it
      should be a tuple of ints) or, if it's None, inferred from the
      first image we load

    - The attributes contained within ``to_normalize`` or the value
      returned by ``model(images)`` must either be a tensor (with 2 or
      more dimensions) or a dictionary containing tensors (with 2 or
      more dimensions)

    - If you want to run this on a whole bunch of images, you may not be
      able to do it all at once for memory reasons. If that's the case,
      you can use the ``index`` arg. If you set that to a 2-tuple, we'll
      glob to find all the files in the folder (getting a giant list)
      and only go from index[0] to index[1] of them. In this case, make
      sure you do something yourself to save them separately and later
      concatenate them, because this function won't.

    In order to use this dictionary to actually z-score your statistics,
    use the ``zscore_stats`` function.

    Parameters
    ----------
    model : torch.nn.Module
        The model we want to generate normalization statistics for.
    input_dir : str
        Path to a directory that contains the images we want to use for
        generating the statistics.
    save_path : str or None, optional
        If a str, the path (should end in '.pt') to save the statistics
        at. If None, we don't save.
    img_shape : tuple or None, optional
        The image shape we want to require that all images have. Since
        we're concatenating all the images into one big tensor, they
        need to have the same dimensions. If a tuple, we only add those
        images that match this shape. If None, we grab that shape from
        the first image we load in.
    as_gray : bool, optional
        If True, we convert any 3d images to grayscale using
        skimage.color.rgb2gray. If False, we do nothing
    index : tuple or None, optional
        If a tuple, must be a 2-tuple of ints. Then, after globbing to
        find all the files in a folder, we only go from index[0] to
        index[1] of them. If None, we go through all files

    Returns
    -------
    stats : dict
        A dictionary containing the statistics to use for normalization.

    """
    images = []
    paths = glob(op.join(input_dir, '*'))
    if index is not None:
        paths = paths[index[0]:index[1]]
    for p in paths:
        try:
            im = imageio.imread(p)
        except ValueError:
            warnings.warn("Unable to load in file %s, it's probably not an image, skipping..." %
                          p)
            continue
        if img_shape is None:
            img_shape = im.shape
        im = im / np.iinfo(im.dtype).max
        # we don't actually use the as_gray argument because that
        # converts the dtype to float32 and we want to make sure to
        # properly set its range between 0 and 1 first
        if as_gray and im.ndim == 3:
            # then it's a color image, and we need to make it grayscale
            im = color.rgb2gray(im)
        if im.shape == img_shape:
            images.append(im)
            if im.max() > 1 or im.min() < 0:
                raise Exception("Somehow we ended up with an image with a max greater than 1 or a"
                                " min less than 0 even after we tried to normalize it! Max: %s, "
                                "min: %s, file: %s" %
                                (im.max(), im.min(), p))
    images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    stats = {'representation': model(images)}
    if hasattr(model, 'to_normalize'):
        stats = {}
        for attr in model.to_normalize:
            stats[attr] = getattr(model, attr)
    for k, v in stats.items():
        if isinstance(v, dict):
            for l, w in v.items():
                ndim_to_avg = [-(i+1) for i in range(w.ndimension() - 2)]
                stats[k][l] = w.mean(ndim_to_avg)
        else:
            ndim_to_avg = [-(i+1) for i in range(v.ndimension() - 2)]
            stats[k] = v.mean(ndim_to_avg)
    if save_path is not None:
        torch.save(stats, save_path)
    return stats


def zscore_stats(stats_dict, model=None, **to_normalize):
    r"""zscore the model's statistics based on stats_dict

    We'd like to use the dictionary of statistics generated by
    ``generate_norm_stats`` to actually normalize some of the statistics
    in our model. This will take a model and the ``stats_dict``
    dictionary and z-score the appropriate attribute of the model
    (whether it's a dictionary or a tensor), subtacting off the mean of
    the value that's in ``stats_dict`` and dividing by the standard
    deviation.

    There are two (mutually-exclusive) ways to use this function:

    1. Pass a model, in which case we assume that the keys in
       ``stats_dict`` correspond to attributes of this model and we'll
       normalize any of them that are present in both ``stats_dict`` and
       as attributes of ``model``. In this case, we return a modified
       version of ``model``.

    2. Pass keyword arguments, in which case we'll assume these
       correspond to the keys in ``stats_dict``. In this case, we return
       a dictionary, ``normalized``, whose keys are the keywords passed
       to this function.

    NOTE: There's no clever way to figure out *when* you want this to be
    called, so you'll have to decide where to insert this in your
    ``model.forward()`` call based on *your* knowledge of the contents
    of ``stats_dict``

    Parameters
    ----------
    stats : dict
        A dictionary containing the statistics to use for normalization
        (as returned/saved by the ``generate_norm_stats`` function).
    model : torch.nn.Module or None, optional
        The model we want to normalize statistics for. If None,
        to_normalize keywords must be set and vice versa

    Returns
    -------
    model : torch.nn.Module
        The normalized model.
    normalized : dict
        Dictionary with the keywords passed to this function.

    """
    if to_normalize:
        if model is not None:
            raise Exception("keywords were passed to normalize, so model must be None!")
        normalized = {}
        for k, v in to_normalize.items():
            mean_stats = stats_dict[k].mean(0)
            std_stats = stats_dict[k].std(0)
            normalized[k] = (v - mean_stats) / std_stats
        return normalized
    for k, v in stats_dict.items():
        if k not in model.to_normalize:
            warnings.warn("stats_dict key %s not found in model.to_normalize, skipping!" % k)
            continue
        if isinstance(v, dict):
            attr = getattr(model, k)
            for l, w in v.items():
                mean_w = w.mean(0)
                std_w = w.std(0)
                if l in attr.keys():
                    val = (attr[l] - mean_w) / std_w
                    if isinstance(attr[l], torch.nn.Parameter):
                        val = torch.nn.Parameter(val)
                    attr[l] = val
                else:
                    warnings.warn("stats_dict key %s not found in model.%s, skipping!" % (l, k))
            setattr(model, k, attr)
        else:
            mean_v = v.mean(0)
            std_v = v.std(0)
            val = (getattr(model, k) - mean_v) / std_v
            if isinstance(getattr(model, k), torch.nn.Parameter):
                val = torch.nn.Parameter(val)
            setattr(model, k, val)
    return model
