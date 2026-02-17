"""Helper functions for loading tensors, converting dtypes, etc."""
# numpydoc ignore=ES01

import contextlib
import pathlib
import warnings
from collections.abc import Callable
from typing import Literal

import imageio.v3 as iio
import numpy as np
import torch
from skimage import color
from torch import Tensor

NUMPY_TO_TORCH_TYPES = {
    bool: torch.bool,  # np.bool deprecated in fav of built-in
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

TORCH_TO_NUMPY_TYPES = {value: key for (key, value) in NUMPY_TO_TORCH_TYPES.items()}


def to_numpy(x: Tensor | np.ndarray, squeeze: bool = False) -> np.ndarray:
    r"""
    Cast tensor to numpy in the most conservative way possible.

    In order this will:

    - Detach the tensor.

    - Move it to the CPU.

    - Convert it to numpy (calling ``.numpy()``).

    - Convert to the corresponding datatype.

    Parameters
    ----------
    x
        Tensor to be converted to :class:`numpy.ndarray` on CPU. If already an array, do
        nothing.
    squeeze
        Whether to remove all dummy dimensions from input.

    Returns
    -------
    array
        The former tensor, now an array.
    """
    with contextlib.suppress(AttributeError):
        # if this fails, it's already a numpy array
        x = x.detach().cpu().numpy().astype(TORCH_TO_NUMPY_TYPES[x.dtype])
    if squeeze:
        x = x.squeeze()
    return x


def load_images(
    paths: str | list[str] | pathlib.Path | list[pathlib.Path],
    as_gray: bool = True,
    sorted_key: None | Callable = None,
) -> Tensor:
    r"""
    Load in images.

    Our models and synthesis methods generally expect their inputs to
    be 4d float32 images: ``(batch, channel, height, width)``, where the batch
    dimension contains multiple images and channel contains something
    like RGB or color channel. This function helps you get your inputs
    into that format. It accepts either a single file, a list of files,
    or a single directory containing images, will load them in,
    normalize them to lie between 0 and 1, convert them to float32,
    optionally convert them to grayscale, make them tensors, and get
    them into the right shape.

    Parameters
    ----------
    paths
        A str or list of strs. If a list, must contain paths of image
        files. If a str, can either be the path of a single image file
        or of a single directory. If a directory, we try to load every
        file it contains (using imageio.imread) and skip those we
        cannot (thus, for efficiency you should not point this to a
        directory with lots of non-image files). This is NOT recursive.
    as_gray
        Whether to convert the images into grayscale or not after
        loading them. If ``False``, we do nothing. If ``True``, we call
        skimage.color.rgb2gray on them, which will result in a single
        channel.
    sorted_key
        How to sort the images. If ``None`` and ``paths`` is a directory,
        will sort the paths alphabetically. If ``paths`` is a list of files,
        must be ``None`` and is ignored. See :ref:`python:sortinghowto`
        for details on other possible values, and note that the objects to sort
        are :class:`pathlib.Path` objects.

    Returns
    -------
    images
        4d tensor containing the images.

    Raises
    ------
    FileNotFoundError
        If any of the explicit image paths do not exist.
    ValueError
        If the images we attempt to load are not all the same shape.
    ValueError
        If ``paths`` is a single file or list of files and ``sorted_key`` is not
        ``None``.

    Warns
    -----
    UserWarning
        If ``paths`` is a directory and any of the files it contains
        are non-images.

    Examples
    --------
    When ``sorted_key=None``, images from a directory are sorted alphabetically
    by filename.

    .. plot::

      >>> import plenoptic as po
      >>> img_dir = po.data.fetch_data("test_images.tar.gz") / "256"
      >>> titles = ["color_wheel", "curie", "einstein", "metal", "nuts"]
      >>> imgs = po.load_images(img_dir)
      >>> po.imshow(imgs, title=titles)
      <PyrFigure size ... with 5 Axes>

    Sort the images by the second letter of their filename:

    .. plot::

      >>> import plenoptic as po
      >>> img_dir = po.data.fetch_data("test_images.tar.gz") / "256"
      >>> titles = ["metal", "einstein", "color_wheel", "curie", "nuts"]
      >>> imgs = po.load_images(img_dir, sorted_key=lambda x: x.name[1])
      >>> po.imshow(imgs, title=titles)
      <PyrFigure size ... with 5 Axes>
    """
    try:
        paths = pathlib.Path(paths)
        if paths.is_dir():
            paths = sorted(paths.iterdir(), key=sorted_key)
        else:
            paths = [paths]
            if sorted_key is not None:
                raise ValueError(
                    "When paths argument is a single file, sorted_key must be None!"
                )

    except TypeError:
        # assume it is an iterable of paths
        paths = [pathlib.Path(p) for p in paths]
        if sorted_key is not None:
            raise ValueError(
                "When paths argument is a list of paths, sorted_key must be None!"
            )

    images = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"File {p} not found!")
        try:
            # this makes sure we close the file properly on except
            with open(p, "rb") as f:
                im = iio.imread(f)
        except (ValueError, OSError):
            warnings.warn(
                f"Unable to load in file {p}, it's probably not an image, skipping..."
            )
            continue
        # make it a float32 array with values between 0 and 1
        im = im / np.iinfo(im.dtype).max
        if im.ndim > 2:
            if as_gray:
                # From scikit-image 0.19 on, it will treat 2d signals as 1d
                # images with 3 channels, so only call rgb2gray when it's more
                # than 2d
                try:
                    im = color.rgb2gray(im)
                except ValueError:
                    # then maybe this is an rgba image instead
                    im = color.rgb2gray(color.rgba2rgb(im))
            else:
                # RGB(A) dimension ends up on the last one, so we rearrange
                im = np.moveaxis(im, -1, 0)
        elif im.ndim == 2 and not as_gray:
            # then expand this grayscale image to the rgb representation
            im = np.expand_dims(im, 0).repeat(3, 0)
        images.append(im)
    if len(set([i.shape for i in images])) > 1:
        raise ValueError(
            "All images must be the same shape but got the following: "
            f"{[i.shape for i in images]}"
        )
    if not images:
        paths = [p.name for p in paths]
        raise ValueError(f"None of the files found at {paths} were images!")
    images = torch.as_tensor(np.array(images), dtype=torch.float32)
    if as_gray:
        if images.ndimension() != 3:
            raise ValueError(
                "For loading in images as grayscale, this should be a 3d tensor!"
            )
        images = images.unsqueeze(1)
    else:
        if images.ndimension() == 3:
            # either this was a single color image:
            # so add the batch dimension
            #  or multiple grayscale images:
            # so add channel dimension
            images = images.unsqueeze(0) if len(paths) > 1 else images.unsqueeze(1)
    if images.ndimension() != 4:
        raise ValueError(
            "Somehow ended up with other than 4 dimensions! Not sure how we got here"
        )
    return images


def convert_float_to_int(
    image: np.ndarray, dtype: Literal[np.uint8, np.uint16] = np.uint8
) -> np.ndarray:
    r"""
    Convert numpy array from float to 8 or 16 bit integer.

    We work with float images that lie between 0 and 1, but for saving them (either as
    png or in a numpy array), we typically want to convert them to 8 or 16 bit integers.
    This function does that by multiplying it by the max value for the target dtype (255
    for 8 bit 65535 for 16 bit) and then converting it to the proper type.

    Parameters
    ----------
    image
        The image to convert, with max less than or equal to 1.
    dtype
        The target data type.

    Returns
    -------
    image
        The converted image, now with specified dtype.

    Raises
    ------
    ValueError
        If ``image`` max is greater than 1.
    """
    if image.max() > 1:
        raise ValueError(
            f"all values of image must lie between 0 and 1, but max is {image.max()}"
        )
    return (image * np.iinfo(dtype).max).astype(dtype)


def _find_min_int(vals: list[int]) -> int:
    """
    Find the minimum non-negative int not in an iterable.

    Parameters
    ----------
    vals
        Iterable(s) of ints.

    Returns
    -------
    min_idx
        Minimum non-negative int.
    """  # numpydoc ignore=ES01
    flat_vals = []
    for v in vals:
        try:
            flat_vals.extend(v)
        except TypeError:
            flat_vals.append(v)
    flat_vals = set(flat_vals)
    try:
        poss_vals = set(np.arange(max(flat_vals) + 1))
    except ValueError:
        # then this is empty sequence and thus we should return 0
        return 0
    try:
        min_int = min(poss_vals - flat_vals)
    except ValueError:
        min_int = max(flat_vals) + 1
    return min_int


def _check_tensor_equality(
    x: Tensor,
    y: Tensor,
    xname: str = "x",
    yname: str = "y",
    rtol: float = 1e-5,
    atol: float = 1e-8,
    error_prepend_str: str = "Different {error_type}",
    error_append_str: str = "",
):
    """
    Check two tensors for equality: device, size, dtype, and values.

    Raises a ValueError (with informative error messages) if any of the above are not
    equal.

    Parameters
    ----------
    x, y
        The tensors to compare.
    xname, yname
        Names of the tensors, used in error messages.
    rtol, atol
        Relative and absolute tolerance for value comparison, passed to
        :func:`torch.allclose`.
    error_prepend_str
        String to start error message with, should contain the string-formatting field
        ``"{error_type}"``.
    error_append_str
        String to finish error message with.

    Raises
    ------
    ValueError
        If any of the device, dtype, size, or values differ.
    """
    error_str = (
        f"{error_prepend_str}"
        f"\n{xname}: {{xvalue}}"
        f"\n{yname}: {{yvalue}}"
        f"{{difference}}"
        f"{error_append_str}"
    )
    if x.device != y.device:
        error_str = error_str.format(
            error_type="device", xvalue=x.device, yvalue=y.device, difference=""
        )
        raise ValueError(error_str)
    # they're allowed to be different shapes if they both have 1 element (e.g., a scalar
    # and a 1-element tensor)
    if x.shape != y.shape and not (x.nelement() == y.nelement() == 1):
        error_str = error_str.format(
            error_type="shape", xvalue=x.shape, yvalue=y.shape, difference=""
        )
        raise ValueError(error_str)
    elif x.dtype != y.dtype:
        error_str = error_str.format(
            error_type="dtype", xvalue=x.dtype, yvalue=y.dtype, difference=""
        )
        raise ValueError(error_str)
    elif not torch.allclose(x, y, rtol=rtol, atol=atol):
        error_str = error_str.format(
            error_type="values", xvalue=x, yvalue=y, difference=f"\nDifference: {x - y}"
        )
        raise ValueError(error_str)
