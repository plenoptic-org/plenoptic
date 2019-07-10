import torch
import numpy as np


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

def stretch(z):
    """soft rescaling, the inverse of squish
    used at initialization
    from (-1, 1) to (-inf,inf) - avoid infinity by clamping

    tanh(log((1+x)/(1-x))/2) = identity, for x in (-1,1)
    """
    epsilon = 10e-8
    z = torch.clamp(z, -1 + epsilon, 1 - epsilon)
    return torch.log((1 + z) / (1 - z)) / 2


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
        The tensor containing the complex modulus of ``x``. It will have one fewer dimension than
        ``x``

    """
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim))
