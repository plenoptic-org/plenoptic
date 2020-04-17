import torch
import numpy as np
from pyrtools import synthetic_images
import matplotlib.pyplot as plt
import os.path as op
from .signal import rescale
import imageio
from skimage import color
import warnings
from glob import glob


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
    images : torch.tensor
        4d tensor containing the images
    """
    if isinstance(paths, str):
        if op.isfile(paths):
            paths = [paths]
        elif op.isdir(paths):
            paths = glob(op.join(paths, '*'))
        else:
            raise Exception("paths must either a single file, a list of files, or a single "
                            "directory, unsure what to do with %s!" % paths)
    images = []
    for p in paths:
        try:
            im = imageio.imread(p)
        except ValueError:
            warnings.warn("Unable to load in file %s, it's probably not an image, skipping..." % p)
            continue
        # make it a float32 array with values between 0 and 1
        im = im / np.iinfo(im.dtype).max
        if as_gray:
            im = color.rgb2gray(im)
        images.append(im)
    try:
        images = torch.tensor(images, dtype=torch.float32)
    except ValueError:
        raise Exception("Concatenating the images into a tensor raised a ValueError! This probably"
                        " means that not all images are the same size.")
    if as_gray:
        if images.ndimension() != 3:
            raise Exception("For loading in images as grayscale, this should be a 3d tensor!")
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
        raise Exception("Somehow ended up with other than 4 dimensions! Not sure how we got here")
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
    im : numpy array
        The image to convert
    dtype : {np.uint8, np.uint16}
        The target data type

    Returns
    -------
    im : numpy array
        The converted image, now with dtype=dtype

    """
    if im.max() > 1:
        raise Exception("all values of im must lie between 0 and 1, but max is %s" % im.max())
    return (im * np.iinfo(dtype).max).astype(dtype)


def torch_complex_to_numpy(x):
    r""" convert a torch complex tensor (written as two stacked real and imaginary tensors)
    to a numpy complex array
    x: assumes x is a torch tensor with last dimension of size 2 where first component is the real
    component and the second is the imaginary component
    """
    x_np = to_numpy(x)
    x_np = x_np[..., 0] + 1j * x_np[..., 1]
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


if __name__ == '__main__':
    make_basic_stimuli()
