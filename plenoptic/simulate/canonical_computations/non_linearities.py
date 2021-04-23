import torch
from ...tools.conv import blur_downsample, upsample_blur
from ...tools.signal import rectangular_to_polar, polar_to_rectangular


def rectangular_to_polar_dict(coeff_dict, dim=-1, residuals=False):
    """Return the complex modulus and the phase of each complex tensor
    in a dictionary.

    Parameters
    ----------
    coeff_dict : dict
       A dictionary containing complex tensors.
    dim : int, optional
       The dimension that contains the real and imaginary components.
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
    ``rectangular_to_polar_real`` and ``local_gain_control``.
    """
    energy = {}
    state = {}
    for key in coeff_dict.keys():
        # ignore residuals
        if isinstance(key, tuple) or not key.startswith('residual'):
            energy[key], state[key] = rectangular_to_polar(
                                        coeff_dict[key].select(dim, 0),
                                        coeff_dict[key].select(dim, 1))

    if residuals:
        energy['residual_lowpass'] = coeff_dict['residual_lowpass']
        energy['residual_highpass'] = coeff_dict['residual_highpass']

    return energy, state


def polar_to_rectangular_dict(energy, state, dim=-1, residuals=False):
    """Return the real and imaginary part  tensor in a dictionary.

    Parameters
    ----------
    energy : dict
        The dictionary of torch.Tensors containing the local complex
        modulus.
    state : dict
        The dictionary of torch.Tensors containing the local phase.
    dim : int, optional
       The dimension that contains the real and imaginary components.
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
            real, imag = polar_to_rectangular(energy[key], state[key])
            coeff_dict[key] = torch.stack((real, imag), dim=dim)

    if residuals:
        coeff_dict['residual_lowpass'] = energy['residual_lowpass']
        coeff_dict['residual_highpass'] = energy['residual_highpass']

    return coeff_dict


def rectangular_to_polar_real(x, epsilon=1e-8):
    """This function is an analogue to rectangular_to_polar for
    real valued signals.

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

    """

    # these could be parameters, but no use case so far
    step = (2, 2)
    p = 2.0

    norm = torch.pow(blur_downsample(torch.abs(x ** p), step=step), 1 / p)
    direction = x / (upsample_blur(norm, step=step) + epsilon)

    return norm, direction


def polar_to_rectangular_real(norm, direction, epsilon=1e-8):
    """This function is an analogue to polar_to_rectangular for
    real valued signals.

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

    """
    step = (2, 2)
    x = direction * (upsample_blur(norm, step=step) + epsilon)
    return x


def local_gain_control(coeff_dict, residuals=False):
    """Spatially local gain control.

    The inverse operation is achieved by `local_gain_release`.
    This function is an analogue to rectangular_to_polar_dict for real
    valued signals.

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

    See Also
    --------
    :meth:`rectangular_to_polar_real`

    """
    energy = {}
    state = {}

    for key in coeff_dict.keys():
        # we don't want to do this on the residuals
        if isinstance(key, tuple) or not key.startswith('residual'):
            energy[key], state[key] = rectangular_to_polar_real(
                                      coeff_dict[key])

    if residuals:
        energy['residual_lowpass'] = coeff_dict['residual_lowpass']
        energy['residual_highpass'] = coeff_dict['residual_highpass']

    return energy, state


def local_gain_release(energy, state, residuals=False):
    """Spatially local gain release.

    The inverse operation to `local_gain_control`.
    This function is  an analogue to polar_to_rectangular_dict for real
    valued signals.

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


    See Also
    --------
    :meth:`polar_to_rectangular_real`

    """
    coeff_dict = {}

    for key in energy.keys():
        # we don't want to do this on the residuals
        if isinstance(key, tuple) or not key.startswith('residual'):
            coeff_dict[key] = polar_to_rectangular_real(
                                      energy[key], state[key])

    if residuals:
        coeff_dict['residual_lowpass'] = energy['residual_lowpass']
        coeff_dict['residual_highpass'] = energy['residual_highpass']

    return coeff_dict
