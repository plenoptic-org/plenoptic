import torch
import imageio
from glob import glob
import os.path as op
import warnings
from ...tools.conv import blur_downsample, upsample_blur
from ...tools.signal import rectangular_to_polar


def rectangular_to_polar_dict(coeff_dict, dim=-1, residuals=False):
    """Return the complex modulus and the phase of each complex tensor in a dictionary.

    Parameters
    ----------
    x : dictionary
       A dictionary containing complex tensors.
    dim : int
       The dimension that contains the real and imaginary components.
    residuals: boolean, optional
        An option to carry around residuals in the energy branch.

    Returns
    -------
    energy : dictionary
        The dictionary of torch.tensors containing the local complex
        modulus of ``x``.
    state: dictionary
        The dictionary of torch.tensors containing the local phase of
        ``x``.

    Note
    ----
    Since complex numbers are not supported by pytorch, we represent
    complex tensors as having an extra dimension with two slices, where
    one contains the real and the other contains the imaginary
    components. E.g., ``1+2j`` would be represented as
    ``torch.tensor([1, 2])`` and ``[1+2j, 4+5j]`` would be
    ``torch.tensor([[1, 2], [4, 5]])``. In the cases represented here,
    this "complex dimension" is the last one, and so the default
    argument ``dim=-1`` would work.

    Note that energy and state is not computed on the residuals.

    Computing the state is local gain control in disguise, see
    ``real_rectangular_to_polar`` and ``local_gain_control``.

    Example
    -------
    >>> complex_steerable_pyramid = Steerable_Pyramid_Freq(image_shape, is_complex=True)
    >>> pyr_coeffs = complex_steerable_pyramid(image)
    >>> complex_cell_responses = rect2pol_dict(pyr_coeffs)[0]

    """

    energy = {}
    state = {}
    for key in coeff_dict.keys():
        # ignore residuals

        if isinstance(key, tuple) or not key.startswith('residual'):
            energy[key], state[key] = rectangular_to_polar(coeff_dict[key].select(dim, 0),
                                                           coeff_dict[key].select(dim, 1))

    if residuals:
        energy['residual_lowpass'] = coeff_dict['residual_lowpass']
        energy['residual_highpass'] = coeff_dict['residual_highpass']

    return energy, state


def rectangular_to_polar_real(x, epsilon=1e-12):
    """This function is an analogue to rectangular_to_polar for real valued signals.

    Norm and direction (analogous to complex modulus and phase) are
    defined using blurring operator and division.  Indeed blurring the
    responses removes high frequencies introduced by the squaring
    operation. In the complex case adding the quadrature pair response
    has the same effect (note that this is most clearly seen in the
    frequency domain).  Here computing the direction (phase) reduces to
    dividing out the norm (modulus), indeed the signal only has one real
    component. This is a normalization operation (local unit vector),
    ehnce the connection to local gain control.

    Parameters
    ----------
    x : torch.tensor
        Tensor of shape (B,C,H,W)
    epsilon: float
        Small constant to avoid division by zero.

    Returns
    -------
    norm : torch.tensor
        The local energy of ``x``. Note that it is down sampled by a
        factor 2 in (unlike rect2pol).
    direction: torch.tensor
        The local phase of ``x`` (aka. local unit vector, or local
        state)

    """

    # these could be parameters, but no use case so far
    step = (2, 2)
    p = 2.0

    norm = torch.pow(blur_downsample(torch.abs(x ** p), step=step), 1 / p)
    direction = x / (upsample_blur(norm, step=step) + epsilon)

    return norm, direction


def local_gain_control(coeff_dict, residuals=False):
    """Spatially local gain control.

    This function is an analogue to rectangular_to_polar_dict for real
    valued signals.

    Parameters
    ----------
    coeff_dict : dictionary
        A dictionary containing tensors of shape (B,C,H,W)
    residuals: boolean, optional
        An option to carry around residuals in the energy dict.

    Returns
    -------
    energy : dictionary
        The dictionary of torch.tensors containing the local energy of
        ``x``.
    state: dictionary
        The dictionary of torch.tensors containing the local phase of
        ``x``.

    Note
    ----
    Note that energy and state is not computed on the residuals.

    See Also
    --------
    ``real_rectangular_to_polar``

    """
    energy = {}
    state = {}

    for key in coeff_dict.keys():
        # we don't want to do this on the residuals
        if isinstance(key, tuple) or not key.startswith('residual'):
            energy[key], state[key] = rectangular_to_polar_real(coeff_dict[key])

    if residuals:
        energy['residual_lowpass'] = coeff_dict['residual_lowpass']
        energy['residual_highpass'] = coeff_dict['residual_highpass']

    return energy, state


# def local_gain_control_ori(coeff_dict, residuals=True):
#     """local gain control in spatio-orientation neighborhood
#     """


def normalize(x, power=2, sum_dim=-1):
    r"""Compute the norm and direction of x

    We compute the norm as :math:`\sqrt[p]{\sum_i x_i^p}`, where
    :math:`p` is the ``power`` arg. We also return the direction, which
    we get by dividing ``x`` by this norm.

    We sum over only one dimension (by default, the final one). If you
    want to do this on a 2d tensor, throwing away all spatial
    information, you should flatten it before passing it here

    In comparison to ``rectangular_to_polar_real``, we do not do this in
    a spatially local manner; we assume that you have either already
    thrown out or want to ignore the spatial information in these
    tensors.

    ``rectangular_to_polar`` is very similar but is meant to operate on
    complex tensors and does not allow one to set the power used when
    computing the norm.

    Parameters
    ----------
    x : torch.Tensor
        The tensor to compute the norm of.
    power : float, optional
        What power to use when computing the norm. The default, 2, means
        we're computing the L2-norm
    sum_dim : int, optional
        The dimension to sum over

    Returns
    -------
    norm : torch.Tensor
        The norm / magnitude of the tensor x
    direction : torch.Tensor
        THe direction of the tensor x

    """
    norm = torch.pow(torch.sum(torch.abs(x ** power), sum_dim), 1 / power)
    direction = x / norm

    return norm, direction


def normalize_dict(coeff_dict, power=2, sum_dim=-1):
    r"""Normalize the tensors contained within a dictionary

    We do this with a call to ``normalize``, and so compute the
    norm/energy as :math:`\sqrt[p]{x^p}`, where :math:`p` is the
    ``power`` arg. We also return the direction, which we get by
    dividing ``x`` by this norm.

    We sum over only one dimension (by default, the final one). If you
    want to do this on a 2d tensor, throwing away all spatial
    information, you should flatten it before passing it here

    In comparison to ``local_gain_control`` and
    ``rectangular_to_polar_dict``, we do not do this in a spatially
    local manner; we assume that you have either already thrown out or
    want to ignore the spatial information in these tensors. Also unlike
    those functions, we compute normalize on every key in ``coeff_dict``
    (whereas they will ignore the residuals)

    Parameters
    ----------
    coeff_dict : dictionary
        A dictionary containing tensors
    power : float, optional
        What power to use when computing the norm. The default, 2, means
        we're computing the L2-norm
    sum_dim : int, optional
        The dimension to sum over

    Returns
    -------
    energy : dictionary
        The dictionary of torch.Tensors containing the energy/norm of
        each entry in ``coeff_dict``.
    state: dictionary
        The dictionary of torch.tensors containing the phase/magnitude of
        each entry in ``coeff_dict``.

    """
    energy = {}
    state = {}

    for key in coeff_dict.keys():
        energy[key], state[key] = normalize(coeff_dict[key], power, sum_dim)

    return energy, state


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
        The ``as_gray`` argument to pass to ``imageio.imread``; whether
        we want to load in the image as grayscale or not
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
    for im in paths:
        try:
            im = imageio.imread(im, as_gray=as_gray)
        except ValueError:
            warnings.warn("Unable to load in file %s, it's probably not an image, skipping..." %
                          im)
            continue
        if img_shape is None:
            img_shape == im.shape
        if im.max() > 1:
            im /= 255
        if im.shape == img_shape:
            images.append(im)
    images = torch.Tensor(images).unsqueeze(1)
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


def cone(x, power=1/3, epsilon=1e-10):
    """Simple function to model the effect of the cone's non-linearities

    The response of the human cone photoreceptors to photons is
    non-linear. There are probably many interestig nuances here, but a
    first-pass approximation is to treat ``response = log(photons)`` or
    ``response = (photons) ^ (1/3)``. Since ``log`` is poorly behaved
    near 0, we'll use the power-law approximation instead.

    We allow the user to set the power used, but 1/3, the default, is
    reasonable.

    Note that we're assuming the input to this function contains values
    proportional to photon counts; thus, it should be a raw image or
    other linearized / "de-gamma-ed" image (all images meant to be
    displayed on a standard display will have been gamma-corrected,
    which involves raising their values to a power, typically 1/2.2).

    Parameters
    ----------
    x : torch.tensor
        Tensor of shape (B,C,H,W), representing the photon counts
    power : float
        The power to raise all values of ``x`` to.
    epsilon : float
        We add a small epsilon to the input before raising it to the
        power, because torch.pow is a bit weird at 0 (sometimes 0 raised
        to a power less than 1 gives 1?) and this can mess up gradients

    Returns
    -------
    cone_response : torch.tensor
        Tensor, same shape as ``x``, representing the non-linear cone
        response

    """
    return torch.pow(x + epsilon, power)
