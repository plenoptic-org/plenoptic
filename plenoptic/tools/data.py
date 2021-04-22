import torch
import numpy as np
from pyrtools import synthetic_images, blurDn
import matplotlib.pyplot as plt
import os.path as op
from .signal import rescale
import imageio
from skimage import color
import warnings
from glob import glob


DATA_PATH = op.join(op.dirname(op.realpath(__file__)), '..', '..', 'data/256')


def to_numpy(x, squeeze=False):
    r"""cast tensor to numpy in the most conservative way possible

    Parameters
    ----------------
    x: `torch.Tensor`
       Tensor to be converted to `numpy.ndarray` on CPU.

    squeeze: bool, optional
        removes all dummy dimensions of the tensor
    """

    try:
        x = x.detach().cpu().numpy()
    except AttributeError:
        # in this case, it's already a numpy array
        pass
    if squeeze:
        x = x.squeeze()
    return x


def load_images(paths, as_gray=True):
    r"""Correctly load in images

    Our models and synthesis methods expect their inputs to be 4d
    float32 images: (batch, channel, height, width), where the batch
    dimension contains multiple images and channel contains something
    like RGB or color channel. This function helps you get your inputs
    into that format. It accepts either a single file, a list of files,
    or a single directory containing images, will load them in,
    normalize them to lie between 0 and 1, convert them to float32,
    optionally convert them to grayscale, make them tensors, and get
    them into the right shape.

    Parameters
    ----------
    paths : str or list
        A str or list of strs. If a list, must contain paths of image
        files. If a str, can either be the path of a single image file
        or of a single directory. If a directory, we try to load every
        file it contains (using imageio.imwrite) and skip those we
        cannot (thus, for efficiency you should not point this to a
        directory with lots of non-image files). This is NOT recursive.
    as_gray : bool, optional
        Whether to convert the images into grayscale or not after
        loading them. If False, we do nothing. If True, we call
        skimage.color.rgb2gray on them.

    Returns
    -------
    images : torch.Tensor
        4d tensor containing the images
    """
    if isinstance(paths, str):
        if op.isfile(paths):
            paths = [paths]
        elif op.isdir(paths):
            paths = glob(op.join(paths, '*'))
        else:
            raise Exception("paths must either a single file, a list of "
                            "files, or a single directory, unsure what "
                            "to do with %s!" % paths)
    images = []
    for p in paths:
        try:
            im = imageio.imread(p)
        except ValueError:
            warnings.warn("Unable to load in file %s, it's probably not "
                          "an image, skipping..." % p)
            continue
        # make it a float32 array with values between 0 and 1
        im = im / np.iinfo(im.dtype).max
        if as_gray:
            im = color.rgb2gray(im)
        else:
            # RGB dimension ends up on the last one, so we rearrange
            im = np.moveaxis(im, -1, 0)
        images.append(im)
    try:
        images = torch.tensor(images, dtype=torch.float32)
    except ValueError:
        raise Exception("Concatenating the images into a tensor raised"
                        "a ValueError! This probably"
                        " means that not all images are the same size.")
    if as_gray:
        if images.ndimension() != 3:
            raise Exception("For loading in images as grayscale, this"
                            "should be a 3d tensor!")
        images = images.unsqueeze(1)
    else:
        if images.ndimension() == 3:
            # either this was a single color image or multiple grayscale ones
            if len(paths) > 1:
                # then single color image, so add the batch dimension
                images = images.unsqueeze(0)
            else:
                # then multiple grayscales ones, so add channel dimension
                images = images.unsqueeze(1)
    if images.ndimension() != 4:
        raise Exception("Somehow ended up with other than 4 dimensions!"
                        "Not sure how we got here")
    return images


def convert_float_to_int(im, dtype=np.uint8):
    r"""Convert image from float to 8 or 16 bit image

    We work with float images that lie between 0 and 1, but for saving
    them (either as png or in a numpy array), we want to convert them to
    8 or 16 bit integers. This function does that by multiplying it by
    the max value for the target dtype (255 for 8 bit 65535 for 16 bit)
    and then converting it to the proper type.

    We'll raise an exception if the max is higher than 1, in which case
    we have no idea what to do.

    Parameters
    ----------
    im : np.ndarray
        The image to convert
    dtype : {np.uint8, np.uint16}
        The target data type

    Returns
    -------
    im : np.ndarray
        The converted image, now with dtype=dtype

    """
    if im.max() > 1:
        raise Exception("all values of im must lie between 0 and 1,"
                        f"but max is {im.max()}")
    return (im * np.iinfo(dtype).max).astype(dtype)


def torch_complex_to_numpy(x):
    r""" convert a torch complex tensor (written as two stacked real
     and imaginary tensors) to a numpy complex array

    Parameters
    ----------------------
    x: `torch.Tensor`
        Tensor whose last dimension is size 2 where first component is the
        real component and the second is the imaginary component.
    """

    x_np = to_numpy(x)
    if x.ndim not in [5, 6]:
        raise Exception(f"x has {x.ndim} dimensions, but a complex tensor"
                        "should have 5 (real and imaginary stacked along"
                        "the final dim) or 6 if it's a video!")
    x_np = x_np[..., 0] + 1j * x_np[..., 1]
    return x_np


def make_synthetic_stimuli(size=256, requires_grad=True):
    r""" Make a set of basic stimuli, useful for developping and debugging models

    Parameters
    ----------
    size: `int`
        the stimuli will have `torch.Size([size, size])`
    requires_grad: `bool`
        weather to initialize the simuli with gradients

    Returns
    -------
    stimuli: `torch.FloatTensor` of shape [B, 1, size, size]
        the set of basic stiuli: [impulse, step_edge, ramp, bar, curv_edge,
                sine_grating, square_grating, polar_angle, angular_sine,
                zone_plate, fractal]
    """

    impulse = np.zeros((size, size))
    impulse[size // 2, size // 2] = 1

    step_edge = synthetic_images.square_wave(size=size, period=size + 1,
                                             direction=0, amplitude=1, phase=0)

    ramp = synthetic_images.ramp(size=size, direction=np.pi / 2, slope=1)

    bar = np.zeros((size, size))
    bar[size // 2 - size//10:size // 2 + size//10,
        size // 2 - 1:size // 2 + 1] = 1

    curv_edge = synthetic_images.disk(size=size, radius=size / 1.2,
                                      origin=(size, size))

    sine_grating = (synthetic_images.sine(size) * 
        synthetic_images.gaussian(size, covariance=size))

    square_grating = synthetic_images.square_wave(size, frequency=(.5, .5),
                                                  phase=2 * np.pi / 3.)
    square_grating *= synthetic_images.gaussian(size, covariance=size)

    polar_angle = synthetic_images.polar_angle(size)

    angular_sine = synthetic_images.angular_sine(size, 6)

    zone_plate = synthetic_images.zone_plate(size)

    fract = synthetic_images.pink_noise(size, fract_dim=.8)

    stim = [impulse, step_edge, ramp, bar, curv_edge,
            sine_grating, square_grating, polar_angle, angular_sine,
            zone_plate, fract]
    stim = [rescale(s) for s in stim]

    stimuli = torch.cat(
        [torch.tensor(s, dtype=torch.float32,
                      requires_grad=requires_grad).unsqueeze(0).unsqueeze(0)
         for s in stim],
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
    res : torch.Tensor
        the polar radius matrix

    '''
    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    # for some reason, torch.meshgrid returns them in the opposite order
    # that np.meshgrid does. So, in order to get the same output, we
    # grab them as (yramp, xramp) instead of (xramp, yramp). similarly,
    # we have to reverse the order from (size[1], size[0]) to (size[0],
    # size[1])
    yramp, xramp = torch.meshgrid(torch.arange(1, size[0]+1, device=device)-origin[0],
                                  torch.arange(1, size[1]+1, device=device)-origin[1])

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
    res : torch.Tensor
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
    # grab them as (yramp, xramp) instead of (xramp, yramp). similarly,
    # we have to reverse the order from (size[1], size[0]) to (size[0],
    # size[1])
    yramp, xramp = torch.meshgrid(torch.arange(1, size[0]+1, device=device)-origin[0],
                                  torch.arange(1, size[1]+1, device=device)-origin[1])

    res = torch.atan2(yramp, xramp)

    res = ((res+(np.pi-phase)) % (2*np.pi)) - np.pi

    return res


def _find_min_int(vals):
    """Find the minimum non-negative int not in an iterable.

    Parameters
    ----------
    vals : iterable
        iterable of ints or iterables of ints

    Returns
    -------
    min_idx : int
        minimum non-negative int

    """
    flat_vals = []
    for v in vals:
        try:
            flat_vals.extend(v)
        except TypeError:
            flat_vals.append(v)
    flat_vals = set(flat_vals)
    try:
        poss_vals = set(np.arange(max(flat_vals)+1))
    except ValueError:
        # then this is empty sequence and thus we should return 0
        return 0
    try:
        min_int = min(poss_vals - flat_vals)
    except ValueError:
        min_int = max(flat_vals) + 1
    return min_int


if __name__ == '__main__':
    make_synthetic_stimuli()
