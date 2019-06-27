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
