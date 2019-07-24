import torch
from ..tools.conv import blur_downsample, upsample_blur
from ..tools.signal import rect2pol


def rect2pol_dict(coeff_dict, dim=-1):
    """Return the complex modulus and the phase of each complex tensor in a dictionary.

    Parameters
    ----------
    x : dictionary
       A dictionary containing complex tensors.
    dim : int
       The dimension that contains the real and imaginary components.
    Returns
    -------
    energy : dictionary
        The dictionary of torch.tensors containing the local complex modulus of ``x``.
    state: dictionary
        The dictionary of torch.tensors containing the local phase of ``x``.

    Note
    ----
    Note that energy and state is not computed on the residuals.

    Since complex numbers aren't implemented in torch, we represent complex tensors as having an
    extra dimension with two slices, where one contains the real and the other contains the
    imaginary components. E.g., ``1+2j`` would be represented as ``torch.tensor([1, 2])`` and
    ``[1+2j, 4+5j]`` would be ``torch.tensor([[1, 2], [4, 5]])``. In the cases represented here,
    this "complex dimension" is the last one, and so the default argument ``dim=-1`` would work.

    This is local gain control in disguise, see 'real_rectangular_to_polar' and 'local_gain_control'.
    """

    energy = {}
    state = {}
    for key in coeff_dict.keys():
        # ignore residuals
        if isinstance(key, tuple):
            energy[key], state[key] = rect2pol(coeff_dict[key].select(dim, 0), coeff_dict[key].select(dim, 1))

    return energy, state


def real_rectangular_to_polar(x, epsilon=1e-12):
    """This function is an analogue to rect2pol for real valued signals.

    Norm and direction (analogous to complex modulus and phase) are defined using blurring operator and division.
    Indeed blurring the responses removes high frequencies introduced by the squaring operation. In the complex case
    adding the quadrature pair response has the same effect (note that this is most clearly seen in the frequency domain).
    Here computing the direction (phase) reduces to dividing out the norm (modulus), indeed the signal only has one
    real component. This is a normalization operation (local unit vector), ehnce the connection to local gain control.

    Parameters
    ----------
    x : torch.tensor
        Tensor of shape (B,C,H,W)
    epsilon: float
        Small constant to avoid division by zero.
    Returns
    -------
    norm : torch.tensor
        The local energy of ``x``. Note that it is down sampled by a factor 2 in  (unlike rect2pol).
    direction: torch.tensor
        The local phase of ``x`` (aka. local unit vector, or local state)
    """

    # these could be parameters, but no use case so far
    step = (2, 2)
    p = 2.0

    norm = torch.pow(blur_downsample(torch.abs(x ** p), step=step), 1 / p)
    direction = x / (upsample_blur(norm, step=step) + epsilon)

    return norm, direction


def local_gain_control(coeff_dict):
    """Spatially local gain control. This function is an analogue to rect2pol_dict for real valued signals.

    Parameters
    ----------
    coeff_dict : dictionary
       A dictionary containing tensors of shape (B,C,H,W)

    Returns
    -------
    energy : dictionary
        The dictionary of torch.tensors containing the local energy of ``x``.
    state: dictionary
        The dictionary of torch.tensors containing the local phase of ``x``.

    Note
    ----
    Note that energy and state is not computed on the residuals.

    see: `real_rectangular_to_polar`
    """
    energy = {}
    state = {}

    for key in coeff_dict.keys():
        if isinstance(key, tuple):
            energy[key], state[key] = real_rectangular_to_polar(coeff_dict[key])

    if residuals:
        energy['residual_lowpass'] = coeff_dict['residual_lowpass']
        energy['residual_highpass'] = coeff_dict['residual_highpass']

    return energy, state

# def local_gain_control_ori(coeff_dict, residuals=True):
#     """local gain control in spatio-orientation neighborhood
#     """