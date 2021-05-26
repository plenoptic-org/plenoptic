import torch
from ...tools.conv import blur_downsample, upsample_blur
from ...tools.signal import rectangular_to_polar, polar_to_rectangular


def rectangular_to_polar_dict(coeff_dict, residuals=True):
    """Wraps the rectangular to polar transform to act on all
    the values in a dictionary.

    Parameters
    ----------
    coeff_dict : dict
       A dictionary containing complex tensors.
    residuals: bool, optional
        An option to carry around residuals in the energy branch.

    Returns
    -------
    energy : dict
        The dictionary of torch.Tensors containing the local complex
        modulus of ``coeff_dict``.
    state: dict
        The dictionary of torch.Tensors containing the local phase of
        ``coeff_dict``.

    Note
    ----
    Computing the state is local gain control in disguise, see
    ``local_gain_control`` and ``local_gain_control_dict``.
    """
    energy = {}
    state = {}
    for key in coeff_dict.keys():
        # ignore residuals
        if isinstance(key, tuple) or not key.startswith('residual'):
            energy[key], state[key] = rectangular_to_polar(coeff_dict[key])

    if residuals:
        energy['residual_lowpass'] = coeff_dict['residual_lowpass']
        energy['residual_highpass'] = coeff_dict['residual_highpass']

    return energy, state


def polar_to_rectangular_dict(energy, state, residuals=True):
    """Wraps the polar to rectangular transform to act on all
    the values in a matching pair of dictionaries.

    Parameters
    ----------
    energy : dict
        The dictionary of torch.Tensors containing the local complex
        modulus.
    state : dict
        The dictionary of torch.Tensors containing the local phase.
    residuals: bool, optional
        An option to carry around residuals in the energy branch.

    Returns
    -------
    coeff_dict : dict
       A dictionary containing complex tensors of coefficients.
    """

    coeff_dict = {}
    for key in energy.keys():
        # ignore residuals

        if isinstance(key, tuple) or not key.startswith('residual'):
            coeff_dict[key] = polar_to_rectangular(energy[key], state[key])

    if residuals:
        coeff_dict['residual_lowpass'] = energy['residual_lowpass']
        coeff_dict['residual_highpass'] = energy['residual_highpass']

    return coeff_dict


def local_gain_control(x, epsilon=1e-8):
    """Spatially local gain control.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (B,C,H,W)
    epsilon: float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    norm : torch.Tensor
        The local energy of ``x``. Note that it is down sampled by a
        factor 2 in (unlike rect2pol).
    direction: torch.Tensor
        The local phase of ``x`` (aka. local unit vector, or local
        state)

    Note
    ----
    This function is an analogue to rectangular_to_polar for
    real valued signals.
    Norm and direction (analogous to complex modulus and phase) are
    defined using blurring operator and division.  Indeed blurring the
    responses removes high frequencies introduced by the squaring
    operation. In the complex case adding the quadrature pair response
    has the same effect (note that this is most clearly seen in the
    frequency domain).  Here computing the direction (phase) reduces to
    dividing out the norm (modulus), indeed the signal only has one real
    component. This is a normalization operation (local unit vector),
    hence the connection to local gain control.
    """

    # these could be parameters, but no use case so far
    step = (2, 2)
    p = 2.0

    norm = blur_downsample(torch.abs(x ** p), step=step).pow(1 / p)
    direction = x / (upsample_blur(norm, step=step) + epsilon)

    return norm, direction


def local_gain_release(norm, direction, epsilon=1e-8):
    """Spatially local gain release.

    Parameters
    ----------
    norm : torch.Tensor
        The local energy of ``x``. Note that it is down sampled by a
        factor 2 in (unlike rect2pol).
    direction: torch.Tensor
        The local phase of ``x`` (aka. local unit vector, or local
        state)
    epsilon: float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    x : torch.Tensor
        Tensor of shape (B,C,H,W)

    Note
    ----
    This function is an analogue to polar_to_rectangular for
    real valued signals.
    Norm and direction (analogous to complex modulus and phase) are
    defined using blurring operator and division.  Indeed blurring the
    responses removes high frequencies introduced by the squaring
    operation. In the complex case adding the quadrature pair response
    has the same effect (note that this is most clearly seen in the
    frequency domain).  Here computing the direction (phase) reduces to
    dividing out the norm (modulus), indeed the signal only has one real
    component. This is a normalization operation (local unit vector),
    hence the connection to local gain control.
    """
    step = (2, 2)
    x = direction * (upsample_blur(norm, step=step) + epsilon)
    return x


def local_gain_control_dict(coeff_dict, residuals=True):
    """Spatially local gain control, for each element in a dictionary.

    Parameters
    ----------
    coeff_dict : dict
        A dictionary containing tensors of shape (B,C,H,W)
    residuals: bool, optional
        An option to carry around residuals in the energy dict.

    Returns
    -------
    energy : dict
        The dictionary of torch.Tensors containing the local energy of
        ``x``.
    state: dict
        The dictionary of torch.Tensors containing the local phase of
        ``x``.
    Note
    ----
    Note that energy and state is not computed on the residuals.

    The inverse operation is achieved by `local_gain_release_dict`.
    This function is an analogue to rectangular_to_polar_dict for real
    valued signals. For more details, see :meth:`local_gain_control`
    """
    energy = {}
    state = {}

    for key in coeff_dict.keys():
        # we don't want to do this on the residuals
        if isinstance(key, tuple) or not key.startswith('residual'):
            energy[key], state[key] = local_gain_control(
                                      coeff_dict[key])

    if residuals:
        energy['residual_lowpass'] = coeff_dict['residual_lowpass']
        energy['residual_highpass'] = coeff_dict['residual_highpass']

    return energy, state


def local_gain_release_dict(energy, state, residuals=True):
    """Spatially local gain release, for each element in a dictionary.

    Parameters
    ----------
    energy : dict
        The dictionary of torch.Tensors containing the local energy of
        ``x``.
    state: dict
        The dictionary of torch.Tensors containing the local phase of
        ``x``.
    residuals: bool, optional
        An option to carry around residuals in the energy dict.

    Returns
    -------
    coeff_dict : dict
        A dictionary containing tensors of shape (B,C,H,W)

    Note
    ----
    The inverse operation to `local_gain_control_dict`.
    This function is  an analogue to polar_to_rectangular_dict for real
    valued signals. For more details, see :meth:`local_gain_release`
    """
    coeff_dict = {}

    for key in energy.keys():
        # we don't want to do this on the residuals
        if isinstance(key, tuple) or not key.startswith('residual'):
            coeff_dict[key] = local_gain_release(
                                      energy[key], state[key])

    if residuals:
        coeff_dict['residual_lowpass'] = energy['residual_lowpass']
        coeff_dict['residual_highpass'] = energy['residual_highpass']

    return coeff_dict
