import torch
from ..tools.conv import blur_downsample, upsample_blur


def complex_modulus(x, dim=-1):
    """Return the complex modulus of a complex tensor
    Since complex numbers aren't implemented in torch, we represent complex tensors as having an
    extra dimension with two slices, where one contains the real and the other contains the
    imaginary components. E.g., ``1+2j`` would be represented as ``torch.tensor([1, 2])`` and
    ``[1+2j, 4+5j]`` would be ``torch.tensor([[1, 2], [4, 5]])``. In the cases represented here,
    this "complex dimension" is the last one, and so the default argument ``dim=-1`` would work.

    Parameters
    ----------
    x : torch.tensor
       The complex tensor to take the modulus of.
    dim : int
       The dimension that contains the real and iamginary components.
    Returns
    -------
    y : torch.tensor
        The tensor containing the complex modulus of ``x``.
    """
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim, keepdim=True))


def local_rectangular_to_polar(x, p=2.0, epsilon=1e-12):
    """Spatially local gain control
    Parameters
    ----------
    x : torch.tensor
       The complex tensor to take the modulus of.
    p : int, float
       The power.
    Returns
    -------
    norm : torch.tensor
        The local energy of ``x``.
    dicrection: torch.tensor
        The local phase of ``x`` (aka. local unit vector, local state)
    """
    norm = torch.pow(blur_downsample(torch.abs(x ** p)), 1 / p)
    direction = (x / (upsample_blur(norm) + epsilon))

    return norm, direction


def local_gain_control(coeff_dict, residuals=True):
    """Spatially local gain control

    local energy
    local phase / local unit vector
    """
    energy = {}
    normalized_energy = {}

    for key in coeff_dict.keys():
        if isinstance(key, tuple):
            energy[key], normalized_energy[key] = local_rectangular_to_polar(coeff_dict[key])

    if residuals:
        energy['residual_lowpass'] = coeff_dict['residual_lowpass']
        energy['residual_highpass'] = coeff_dict['residual_highpass']

    return energy, normalized_energy
