"""Helper functions for creating images."""
# numpydoc ignore=ES01

from typing import Literal

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "polar_radius",
    "polar_angle",
    "disk",
]


def __dir__() -> list[str]:
    return __all__


def polar_radius(
    size: int | tuple[int, int],
    exponent: float = 1.0,
    origin: int | tuple[int, int] | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    """
    Make distance-from-origin (r) matrix.

    Compute a matrix of given size containing samples of a radial ramp
    function, raised to given exponent, centered at given origin.

    Parameters
    ----------
    size
        If an int, we assume the image should be of dimensions ``(size,
        size)``. if a tuple, must be a 2-tuple of ints specifying the
        dimensions.
    exponent
        The exponent of the radial ramp function.
    origin
        The center of the image. if an int, we assume the origin is at
        ``(origin, origin)``. if a tuple, must be a 2-tuple of ints
        specifying the origin (where ``(0, 0)`` is the upper left).  if
        ``None``, we assume the origin lies at the center of the matrix,
        ``(size+1)/2``.
    device
        The device to create this tensor on.

    Returns
    -------
    radius
        The polar radius matrix.
    """
    if not hasattr(size, "__iter__"):
        size = (size, size)

    if origin is None:
        origin = ((size[0] + 1) / 2.0, (size[1] + 1) / 2.0)
    elif not hasattr(origin, "__iter__"):
        origin = (origin, origin)

    xramp, yramp = torch.meshgrid(
        torch.arange(1, size[1] + 1, device=device) - origin[1],
        torch.arange(1, size[0] + 1, device=device) - origin[0],
        indexing="xy",
    )

    if exponent <= 0:
        # zero to a negative exponent raises:
        # ZeroDivisionError: 0.0 cannot be raised to a negative power
        r = xramp**2 + yramp**2
        radius = np.power(r, exponent / 2.0, where=(r != 0))
    else:
        radius = (xramp**2 + yramp**2) ** (exponent / 2.0)
    return radius


def polar_angle(
    size: int | tuple[int, int],
    phase: float = 0.0,
    origin: int | tuple[float, float] | None = None,
    direction: Literal["clockwise", "counter-clockwise"] = "clockwise",
    device: torch.device | None = None,
) -> Tensor:
    """
    Make polar angle matrix (in radians).

    Compute a matrix of given size containing samples of the polar angle (in radians,
    increasing in user-defined direction from the X-axis, ranging from -pi to pi),
    relative to given phase, about the given origin pixel.

    Parameters
    ----------
    size
        If an int, we assume the image should be of dimensions ``(size, size)``. If a
        tuple, must be a 2-tuple of ints specifying the dimensions.
    phase
        The phase of the polar angle function (in radians, clockwise from the X-axis).
    origin
        The center of the image. if an int, we assume the origin is at
        ``(origin, origin)``. if a tuple, must be a 2-tuple of ints specifying the
        origin (where ``(0, 0)`` is the upper left). If ``None``, we assume the origin
        lies at the center of the matrix, ``(size+1)/2``.
    direction
        Whether the angle increases in a clockwise or counter-clockwise direction from
        the x-axis. The standard mathematical convention is to increase
        counter-clockwise, so that 90 degrees corresponds to the positive y-axis.
    device
        The device to create this tensor on.

    Returns
    -------
    res
        The polar angle matrix.

    Raises
    ------
    ValueError
        If ``direction`` takes an illegal value.
    """
    if direction not in ["clockwise", "counter-clockwise"]:
        raise ValueError(
            "direction must be one of {'clockwise', 'counter-clockwise'}, "
            f"but received {direction}!"
        )

    if not hasattr(size, "__iter__"):
        size = (size, size)

    if origin is None:
        origin = ((size[0] + 1) / 2.0, (size[1] + 1) / 2.0)
    elif not hasattr(origin, "__iter__"):
        origin = (origin, origin)

    xramp, yramp = torch.meshgrid(
        torch.arange(1, size[1] + 1, device=device) - origin[1],
        torch.arange(1, size[0] + 1, device=device) - origin[0],
        indexing="xy",
    )
    if direction == "counter-clockwise":
        yramp = torch.flip(yramp, [0])

    res = torch.atan2(yramp, xramp)

    res = ((res + (np.pi - phase)) % (2 * np.pi)) - np.pi

    return res


def disk(
    img_size: int | tuple[int, int] | torch.Size,
    outer_radius: float | None = None,
    inner_radius: float | None = None,
) -> Tensor:
    r"""
    Create a circular mask with softened edges.

    All values within ``inner_radius`` will be 1, and all values from ``inner_radius``
    to ``outer_radius`` will decay smoothly to 0.

    Parameters
    ----------
    img_size
        Size of image in pixels.
    outer_radius
        Total radius of disk. Values from ``inner_radius`` to ``outer_radius``
        will decay smoothly to zero.
    inner_radius
        Radius of inner disk. All elements from the origin to ``inner_radius``
        will be set to 1.

    Returns
    -------
    mask
        Tensor mask with ``torch.Size(img_size)``.

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import torch
      >>> disk = po.data.disk((256, 256), outer_radius=50, inner_radius=25)
      >>> # we can add batch and color dimensions
      >>> # (this is equivalent to using .unsqueeze(0) twice)
      >>> disk = disk[None, None]
      >>> # we can use the disk as a mask to apply to an image
      >>> img = po.data.einstein()
      >>> masked_img = img * disk
      >>> po.plot.imshow(
      ...     [disk, img, masked_img], title=["disk", "image", "mask applied"]
      ... )
      <PyrFigure ...>
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    assert len(img_size) == 2

    if outer_radius is None:
        outer_radius = (min(img_size) - 1) / 2

    if inner_radius is None:
        inner_radius = outer_radius / 2

    mask = torch.empty(*img_size)
    i0, j0 = (img_size[0] - 1) / 2, (img_size[1] - 1) / 2  # image center

    for i in range(img_size[0]):  # height
        for j in range(img_size[1]):  # width
            r = np.sqrt((i - i0) ** 2 + (j - j0) ** 2)

            if r > outer_radius:
                mask[i][j] = 0
            elif r < inner_radius:
                mask[i][j] = 1
            else:
                radial_decay = (r - inner_radius) / (outer_radius - inner_radius)
                mask[i][j] = (1 + np.cos(np.pi * radial_decay)) / 2

    return mask


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
