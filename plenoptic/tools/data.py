import torch
import numpy as np
from pyrtools import synthetic_images
import matplotlib.pyplot as plt
import os.path as op
from .signal import rescale


DATA_PATH = op.join(op.dirname(op.realpath(__file__)), '..', '..', 'data')


def to_numpy(x):
    r"""cast tensor to numpy in the most conservative way possible
    """
    try:
        x = x.detach().cpu().numpy().astype(np.float32)
    except AttributeError:
        # in this case, it's already a numpy array
        pass
    return x

def torch_complex_to_numpy(x):
    r""" convert a torch complex tensor (written as two stacked real and imaginary tensors)
    to a numpy complex array
    x: assumes x is a torch tensor with last dimension of size 2 where first component is the real
    component and the second is the imaginary component
    """
    x_np = to_numpy(x)
    x_np = x_np[...,0] + 1j * x_np[...,1]
    return x_np

def make_basic_stimuli(size=256, requires_grad=True):
    impulse = np.zeros((size, size))
    impulse[size // 2, size // 2] = 1

    step_edge = synthetic_images.square_wave(size=size, period=size + 1, direction=0, amplitude=1,
                                             phase=0)

    ramp = synthetic_images.ramp(size=size, direction=np.pi / 2, slope=1)

    bar = np.zeros((size, size))
    bar[size // 2 - 20:size // 2 + 20, size // 2 - 2:size // 2 + 2] = 1

    curv_edge = synthetic_images.disk(size=size, radius=size / 1.2, origin=(size, size))

    sine_grating = synthetic_images.sine(size) * synthetic_images.gaussian(size, covariance=size)

    square_grating = synthetic_images.square_wave(size, frequency=(.5, .5), phase=2 * np.pi / 3.)
    square_grating *= synthetic_images.gaussian(size, covariance=size)

    polar_angle = synthetic_images.polar_angle(size)

    angular_sine = synthetic_images.angular_sine(size, 6)

    zone_plate = synthetic_images.zone_plate(size)

    fract = synthetic_images.pink_noise(size, fract_dim=.8)

    checkerboard = plt.imread(op.join(DATA_PATH, 'checkerboard.pgm')).astype(float)
    # checkerboard = pt.blurDn(checkerboard, 1, 'qmf9')

    sawtooth = plt.imread(op.join(DATA_PATH, 'sawtooth.pgm')).astype(float)
    # sawtooth = pt.blurDn(sawtooth, 1, 'qmf9')

    reptil_skin = plt.imread(op.join(DATA_PATH, 'reptil_skin.pgm')).astype(float)
    # reptil_skin = pt.blurDn(reptil_skin, 1, 'qmf9')

    # image = plt.imread('/Users/pe/Pictures/umbrella.jpg').astype(float)
    # image = image[500:500+2**11,1000:1000+2**11,0]
    # image = pt.blurDn(image, 4, 'qmf9')
    image = plt.imread(op.join(DATA_PATH, 'einstein.png')).astype(float)[:,:,0]
    # image = pt.blurDn(image, 1, 'qmf9')

    stim = [impulse, step_edge, ramp, bar, curv_edge,
            sine_grating, square_grating, polar_angle, angular_sine, zone_plate,
            checkerboard, sawtooth, reptil_skin, fract, image]
    stim = [rescale(s) for s in stim]

    stimuli = torch.cat(
        [torch.tensor(s, dtype=torch.float32, requires_grad=requires_grad).unsqueeze(0).unsqueeze(0) for s in stim],
        dim=0)

    return stimuli


def polar_radius(size, exponent=1, origin=None, device=None):
    '''make distance-from-origin (r) matrix

    Compute a matrix of given size containing samples of a radial ramp
    function, raised to given exponent, centered at given origin.

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size,
        size)`. if a tuple, must be a 2-tuple of ints specifying the
        dimensions
    exponent : `float`
        the exponent of the radial ramp function.
    origin : `int`, `tuple`, or None
        the center of the image. if an int, we assume the origin is at
        `(origin, origin)`. if a tuple, must be a 2-tuple of ints
        specifying the origin (where `(0, 0)` is the upper left).  if
        None, we assume the origin lies at the center of the matrix,
        `(size+1)/2`.
    device : str or torch.device
        the device to create this tensor on

    Returns
    -------
    res : torch.tensor
        the polar radius matrix

    '''
    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xramp, yramp = torch.meshgrid(torch.arange(1, size[1]+1, device=device)-origin[1],
                                  torch.arange(1, size[0]+1, device=device)-origin[0])

    if exponent <= 0:
        # zero to a negative exponent raises:
        # ZeroDivisionError: 0.0 cannot be raised to a negative power
        r = xramp ** 2 + yramp ** 2
        res = np.power(r, exponent / 2.0, where=(r != 0))
    else:
        res = (xramp ** 2 + yramp ** 2) ** (exponent / 2.0)
    return res


def polar_angle(size, phase=0, origin=None, device=None):
    '''make polar angle matrix (in radians)

    Compute a matrix of given size containing samples of the polar angle (in radians, CW from the
    X-axis, ranging from -pi to pi), relative to given phase, about the given origin pixel.

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must be
        a 2-tuple of ints specifying the dimensions
    phase : `float`
        the phase of the polar angle function (in radians, clockwise from the X-axis)
    origin : `int`, `tuple`, or None
        the center of the image. if an int, we assume the origin is at `(origin, origin)`. if a
        tuple, must be a 2-tuple of ints specifying the origin (where `(0, 0)` is the upper left).
        if None, we assume the origin lies at the center of the matrix, `(size+1)/2`.
    device : str or torch.device
        the device to create this tensor on

    Returns
    -------
    res : torch.tensor
        the polar angle matrix

    '''
    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    # for some reason, torch.meshgrid returns them in the opposite order
    # that np.meshgrid does. So, in order to get the same output, we
    # grab them as (yramp, xramp) instead of (xramp, yramp)
    yramp, xramp = torch.meshgrid(torch.arange(1, size[1]+1, device=device)-origin[1],
                                  torch.arange(1, size[0]+1, device=device)-origin[0])

    res = torch.atan2(yramp, xramp)

    res = ((res+(np.pi-phase)) % (2*np.pi)) - np.pi

    return res


if __name__ == '__main__':
    make_basic_stimuli()
