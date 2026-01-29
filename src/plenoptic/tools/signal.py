"""Helper functions for modifying tensors in useful ways."""

# numpydoc ignore=ES01
import numpy as np
import torch
from pyrtools.pyramids.steer import steer_to_harmonics_mtx
from torch import Tensor


def rescale(x: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    r"""
    Linearly rescale the dynamic range of the input to ``[a, b]``.

    Parameters
    ----------
    x
        Input tensor to rescale.
    a, b
        Min and max values, respectively, for the output.

    Returns
    -------
    rescaled_x
        The rescaled tensor.

    Examples
    --------
    >>> import torch
    >>> x = torch.tensor([2.0, 4.0, 6.0, 8.0])
    >>> rescale(x)
    tensor([0.0000, 0.3333, 0.6667, 1.0000])
    >>> rescale(x, a=-1, b=1)
    tensor([-1.0000, -0.3333, 0.3333, 1.0000])
    """  # numpydoc ignore=ES01
    v = x.max() - x.min()
    g = x - x.min()
    if v > 0:
        g = g / v
    return a + g * (b - a)


def _raised_cosine(
    width: float = 1, position: float = 0, values: tuple[float, float] = (0, 1)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a lookup table containing a "raised cosine" soft threshold function.

    .. code::

       Y =  VALUES(1)
           + (VALUES(2)-VALUES(1))
           * cos^2( PI/2 * (X - POSITION + WIDTH)/WIDTH )

    This lookup table is suitable for use by :func:`_interpolate1d`.

    Parameters
    ----------
    width
        The width of the region over which the transition occurs.
    position
        The location of the center of the threshold.
    values
        2-tuple specifying the values to the left and right of the transition.

    Returns
    -------
    X
        The x values of this raised cosine.
    Y
        The y values of this raised cosine.
    """
    sz = 256  # arbitrary!

    X = np.pi * np.arange(-sz - 1, 2) / (2 * sz)

    Y = values[0] + (values[1] - values[0]) * np.cos(X) ** 2

    # make sure end values are repeated, for extrapolation...
    Y[0] = Y[1]
    Y[sz + 2] = Y[sz + 1]

    X = position + (2 * width / np.pi) * (X + np.pi / 4)

    return X, Y


def _interpolate1d(
    x_new: Tensor, Y: Tensor | np.ndarray, X: Tensor | np.ndarray
) -> Tensor:
    r"""
    One-dimensional linear interpolation.

    Returns the one-dimensional piecewise linear interpolant to a
    function with given discrete data points ``(X, Y)``, evaluated at
    ``x_new``.

    Note: this function is just a wrapper around :func:`np.interp()`.

    Parameters
    ----------
    x_new
        The x-coordinates at which to evaluate the interpolated values.
    Y
        The y-coordinates of the data points.
    X
        The x-coordinates of the data points, same length as X.

    Returns
    -------
    interp_x
        Interpolated values of shape identical to ``x_new``.
    """
    out = np.interp(x=x_new.flatten(), xp=X, fp=Y)
    return np.reshape(out, x_new.shape)


def rectangular_to_polar(x: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Rectangular to polar coordinate transform.

    If input is real-valued, ``amplitude`` will be identical to the input and ``phase``
    will be all 0s.

    Parameters
    ----------
    x
        Complex tensor.

    Returns
    -------
    amplitude
        Tensor containing the amplitude (aka. complex modulus).
    phase
        Tensor containing the phase.

    See Also
    --------
    :func:`~plenoptic.simulate.canonical_computations.non_linearities.rectangular_to_polar_dict`
        Same operation on dictionaries.
    polar_to_rectangular
        The inverse operation.
    :func:`~plenoptic.simulate.canonical_computations.non_linearities.local_gain_control`
        The analogous function for real-valued signals.

    Examples
    --------
    >>> import torch
    >>> x = torch.tensor([1 + 1j, 1 - 1j])
    >>> amplitude, phase = rectangular_to_polar(x)
    >>> amplitude
    tensor([1.4142, 1.4142])
    >>> phase
    tensor([ 0.7854, -0.7854])
    """
    amplitude = torch.abs(x)
    phase = torch.angle(x)
    return amplitude, phase


def polar_to_rectangular(amplitude: Tensor, phase: Tensor) -> Tensor:
    r"""
    Polar to rectangular coordinate transform.

    Parameters
    ----------
    amplitude
        Tensor containing the amplitude (aka. complex modulus). Must be >= 0.
    phase
        Tensor containing the phase.

    Returns
    -------
    image
        Complex tensor.

    Raises
    ------
    ValueError
        If ``amplitude`` is not non-negative.

    See Also
    --------
    :func:`~plenoptic.simulate.canonical_computations.non_linearities.polar_to_rectangular_dict`
        Same operation on dictionaries.
    rectangular_to_polar
        The inverse operation.
    :func:`~plenoptic.simulate.canonical_computations.non_linearities.local_gain_release`
        The analogous function for real-valued signals.

    Examples
    --------
    >>> import torch
    >>> amplitude = torch.tensor([1.4142, 1.4142])
    >>> phase = torch.tensor([0.7854, -0.7854])
    >>> polar_to_rectangular(amplitude, phase)
    tensor([1.+1.j, 1.-1.j])
    """  # numpydoc ignore=ES01
    if (amplitude < 0).any():
        raise ValueError("Amplitudes must be non-negative.")

    real = amplitude * torch.cos(phase)
    imaginary = amplitude * torch.sin(phase)
    return torch.complex(real, imaginary)


def _steer(
    basis: Tensor,
    angle: np.ndarray | Tensor | float,
    harmonics: list[int] | None = None,
    steermtx: Tensor | np.ndarray | None = None,
    even_phase: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Steer ``basis`` to the specified ``angle``.

    Parameters
    ----------
    basis
        Array whose columns are vectorized rotated copies of a steerable
        function, or the responses of a set of steerable filters.
    angle
        Scalar or column vector the size of the basis. Specifies the angle(s)
        (in radians) to steer to.
    harmonics
        A list of harmonic numbers indicating the angular harmonic content of
        the basis. If ``None``, will use N even or odd low frequencies, as for
        derivative filters.
    steermtx
        Matrix which maps the filters onto Fourier series components (ordered
        ``[cos0, cos1, sin1, cos2, sin2, ..., sinN]``). See
        :func:`pyrtools.pyramids.steer.steer_to_harmonics_mtx` function for more
        details. If ``None``, assumes cosine phase harmonic components, and
        filter positions at ``2pi*n/N``.
    even_phase
        Specifies whether the harmonics are cosine or sine phase aligned about
        those positions.

    Returns
    -------
    res
        The resteered basis.
    steervect
        The weights used to resteer the basis.

    Raises
    ------
    ValueError
        If ``angle`` is not a scalar or appropriately-sized column vector.
    ValueError
        If ``harmonics`` is not 1d.
    ValueError
        If ``harmonics`` is not compatible with the size of ``basis``.
    """  # numpydoc ignore=ES01
    num = basis.shape[-1]
    device = basis.device

    if isinstance(angle, int | float) or angle.ndim == 0:
        angle = np.array([angle])
    else:
        if angle.shape[0] != basis.shape[0] or angle.shape[1] != 1:
            raise ValueError(
                "ANGLE must be a scalar, or a column vector the"
                "size of the basis elements"
            )

    # If HARMONICS is not specified, assume derivatives.
    if harmonics is None:
        harmonics = np.arange(1 - (num % 2), num, 2)

    if len(harmonics.shape) == 1 or harmonics.shape[0] == 1:
        # reshape to column matrix
        harmonics = harmonics.reshape(harmonics.shape[0], 1)
    elif harmonics.shape[0] != 1 and harmonics.shape[1] != 1:
        raise ValueError("input parameter HARMONICS must be 1D!")

    if 2 * harmonics.shape[0] - (harmonics == 0).sum() != num:
        raise ValueError("harmonics list is incompatible with basis size!")

    # If STEERMTX not passed, assume evenly distributed cosine-phase filters:
    if steermtx is None:
        steermtx = steer_to_harmonics_mtx(
            harmonics, np.pi * np.arange(num) / num, even_phase=even_phase
        )

    steervect = np.zeros((angle.shape[0], num))
    arg = angle * harmonics[np.nonzero(harmonics)[0]].T
    if all(harmonics):
        steervect[:, range(0, num, 2)] = np.cos(arg)
        steervect[:, range(1, num, 2)] = np.sin(arg)
    else:
        steervect[:, 0] = np.ones((arg.shape[0], 1))
        steervect[:, range(1, num, 2)] = np.cos(arg)
        steervect[:, range(2, num, 2)] = np.sin(arg)

    steervect = np.dot(steervect, steermtx)

    steervect = torch.as_tensor(steervect, dtype=basis.dtype).to(device)
    if steervect.shape[0] > 1:
        tmp = basis @ steervect
        res = tmp.sum().t()
    else:
        res = basis @ steervect.t()
    return res, steervect.reshape(num)


def make_disk(
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
        Tensor mask with shape ``torch.Size(img_size)``.

    Examples
    --------
    .. plot::

      >>> import torch
      >>> from plenoptic.tools.signal import make_disk
      >>> import matplotlib.pyplot as plt
      >>> disk = make_disk((128, 128), outer_radius=50, inner_radius=25)
      >>> plt.imshow(disk, cmap="gray")
      <...>
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


def add_noise(img: Tensor, noise_mse: float | list[float]) -> Tensor:
    """
    Add normally distributed noise to an image.

    This adds normally-distributed noise to an image so that the resulting
    noisy version has the specified mean-squared error.

    Parameters
    ----------
    img
        The image to make noisy.
    noise_mse
        The target MSE value / variance of the noise. More than one value is
        allowed.

    Returns
    -------
    noisy_img
        The noisy image. If ``noise_mse`` contains only one element, this will be
        the same size as ``img``. Else, each separate value from ``noise_mse`` will
        be along the batch dimension.

    Examples
    --------
    Basic usage:

    .. plot::
      :context: reset

      >>> import matplotlib.pyplot as plt
      >>> import torch
      >>> from plenoptic.tools.signal import add_noise
      >>> img = torch.zeros(32, 32)
      >>> noisy = add_noise(img, noise_mse=0.1)
      >>> noisy.shape
      torch.Size([1, 1, 32, 32])
      >>> fig, ax = plt.subplots(1, 2)
      >>> ax[0].imshow(img, cmap="gray")
      >>> ax[0].set_title("Original")
      >>> ax[0].axis("off")
      >>> ax[1].imshow(noisy[0, 0], cmap="gray")
      >>> ax[1].set_title("Noisy")
      >>> ax[1].axis("off")

    With multiple elements in ``noise_mse``:

    .. plot::
      :context: reset

      >>> import matplotlib.pyplot as plt
      >>> import torch
      >>> from plenoptic.tools.signal import add_noise
      >>> img = torch.zeros(32, 32)
      >>> noisy_multi = add_noise(img, noise_mse=[0.01, 0.1, 1.0])
      >>> noisy_multi.shape
      torch.Size([3, 1, 32, 32])
      >>> fig, axs = plt.subplots(1, 4)
      >>> axs[0].imshow(img, cmap="gray")
      >>> axs[0].set_title("Original")
      >>> axs[0].axis("off")
      >>> for i, (ax, noisy) in enumerate(zip(axs[1:], noisy_multi), 1):
      ...     ax.imshow(noisy[0], cmap="gray")
      ...     ax.set_title(f"Noisy {i}")
      ...     ax.axis("off")
    """
    noise_mse = torch.as_tensor(
        noise_mse, dtype=img.dtype, device=img.device
    ).unsqueeze(0)
    noise_mse = noise_mse.view(noise_mse.nelement(), 1, 1, 1)
    noise = 200 * torch.randn(
        max(noise_mse.shape[0], img.shape[0]),
        *img.shape[1:],
        device=img.device,
    )
    noise = noise - noise.mean()
    noise = noise * torch.sqrt(
        noise_mse / (noise**2).mean((-1, -2)).unsqueeze(-1).unsqueeze(-1)
    )
    return img + noise


def modulate_phase(x: Tensor, phase_factor: float = 2.0) -> Tensor:
    """
    Modulate the phase of a complex signal.

    Doubling the phase of a complex signal allows you to, for example, take the
    correlation between steerable pyramid coefficients at two adjacent spatial
    scales.

    Parameters
    ----------
    x
        Complex tensor whose phase will be modulated.
    phase_factor
        Multiplicative factor to change phase by.

    Returns
    -------
    x_mod
        Phase-modulated complex tensor.

    Raises
    ------
    TypeError
        If ``x`` is not complex-valued.
    """
    try:
        angle = torch.atan2(x.imag, x.real)
    except RuntimeError:
        # then x is not complex-valued
        raise TypeError("x must be a complex-valued tensor!")
    amp = x.abs()
    real = amp * torch.cos(phase_factor * angle)
    imag = amp * torch.sin(phase_factor * angle)
    return torch.complex(real, imag)


def autocorrelation(x: Tensor) -> Tensor:
    r"""
    Compute the autocorrelation of ``x``.

    This uses the Fourier transform to compute the autocorrelation in an efficient
    manner (see Notes).

    Parameters
    ----------
    x
       N-dimensional tensor. We assume the last two dimension are height and
       width and compute you autocorrelation on these dimensions (independently
       on each other dimension).

    Returns
    -------
    ac
        Autocorrelation of ``x``.

    Notes
    -----
    - By the Einstein-Wiener-Khinchin theorem: The autocorrelation of a wide
      sense stationary (WSS) process is the inverse Fourier transform of its
      energy spectrum (ESD) - which itself is the multiplication between
      FT(x(t)) and FT(x(-t)). In other words, the auto-correlation is
      convolution of the signal ``x`` with itself, which corresponds to squaring
      in the frequency domain. This approach is computationally more efficient
      than brute force (:math:`n log(n)` vs :math:`n^2`).

    - By Cauchy-Swartz, the autocorrelation attains it is maximum at the center
      location (ie. no shift) - that maximum value is the signal's variance
      (assuming that the input signal is mean centered).
    """
    # Calculate the auto-correlation
    ac = torch.fft.rfft2(x)
    # this is equivalent to ac.abs().pow(2) or to ac multiplied by its complex
    # conjugate
    ac = ac.real.pow(2) + ac.imag.pow(2)
    ac = torch.fft.irfft2(ac)
    ac = torch.fft.fftshift(ac, dim=(-2, -1)) / torch.mul(*x.shape[-2:])
    return ac


def center_crop(x: Tensor, output_size: int) -> Tensor:
    """
    Crop out the center of a signal.

    If x has an even number of elements on either of those final two
    dimensions, we round up.

    Parameters
    ----------
    x
        N-dimensional tensor, we assume the last two dimensions are height and
        width.
    output_size
        The size of the output. Must be a positive int. Note that we only support a
        single number, so both dimensions are cropped identically.

    Returns
    -------
    cropped
        Tensor whose last two dimensions have each been cropped to
        ``output_size``.

    Raises
    ------
    TypeError
        If ``output_size`` is not a single int.
    ValueError
        If ``output_size is not positive or larger than the height/width of ``x``.

    Examples
    --------
    >>> import plenoptic as po
    >>> img = po.data.einstein()
    >>> img.shape
    torch.Size([1, 1, 256, 256])
    >>> img = po.tools.center_crop(img, 128)
    >>> img.shape
    torch.Size([1, 1, 128, 128])
    """
    h, w = x.shape[-2:]
    output_size = torch.as_tensor(output_size)
    if output_size.ndim > 0:
        raise TypeError("output_size must be a single number!")
    if torch.is_floating_point(output_size):
        raise TypeError("output_size must be an int!")
    if output_size > h or output_size > w:
        raise ValueError("output_size is bigger than image height/width!")
    if output_size <= 0:
        raise ValueError("output_size must be positive!")
    return x[
        ...,
        (h // 2 - output_size // 2) : (h // 2 + (output_size + 1) // 2),
        (w // 2 - output_size // 2) : (w // 2 + (output_size + 1) // 2),
    ]


def expand(x: Tensor, factor: float) -> Tensor:
    r"""
    Expand a signal by a factor.

    We do this in the frequency domain: pasting the Fourier contents of ``x``
    in the center of a larger empty tensor, and then taking the inverse FFT.

    Parameters
    ----------
    x
        The signal for expansion.
    factor
        Factor by which to resize image. Must be larger than 1 and ``factor *
        x.shape[-2:]`` must give integer values.

    Returns
    -------
    expanded
        The expanded signal.

    Raises
    ------
    ValueError
        If ``factor`` is less than or equal to 1.
    ValueError
        If ``factor`` times the height or width of ``x`` is not an integer.

    See Also
    --------
    shrink
        The inverse operation.
    :func:`~plenoptic.tools.conv.upsample_blur`
        An alternative upsampling operation.
    """
    if factor <= 1:
        raise ValueError("factor must be strictly greater than 1!")
    im_x = x.shape[-1]
    im_y = x.shape[-2]
    mx = factor * im_x
    my = factor * im_y
    if int(mx) != mx:
        raise ValueError(
            f"factor * x.shape[-1] must be an integer but got {mx} instead!"
        )
    if int(my) != my:
        raise ValueError(
            f"factor * x.shape[-2] must be an integer but got {my} instead!"
        )
    mx = int(mx)
    my = int(my)

    fourier = factor**2 * torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    fourier_large = torch.zeros(
        *x.shape[:-2],
        my,
        mx,
        device=fourier.device,
        dtype=fourier.dtype,
    )

    y1 = my / 2 + 1 - im_y / 2
    y2 = my / 2 + im_y / 2
    x1 = mx / 2 + 1 - im_x / 2
    x2 = mx / 2 + im_x / 2
    # when any of these numbers are non-integers, if you round down, the
    # resulting image will be off.
    y1 = int(np.ceil(y1))
    y2 = int(np.ceil(y2))
    x1 = int(np.ceil(x1))
    x2 = int(np.ceil(x2))

    fourier_large[..., y1:y2, x1:x2] = fourier[..., 1:, 1:]
    fourier_large[..., y1 - 1, x1:x2] = fourier[..., 0, 1:] / 2
    fourier_large[..., y2, x1:x2] = fourier[..., 0, 1:].flip(-1) / 2
    fourier_large[..., y1:y2, x1 - 1] = fourier[..., 1:, 0] / 2
    fourier_large[..., y1:y2, x2] = fourier[..., 1:, 0].flip(-1) / 2
    esq = fourier[..., 0, 0] / 4
    fourier_large[..., y1 - 1, x1 - 1] = esq
    fourier_large[..., y1 - 1, x2] = esq
    fourier_large[..., y2, x1 - 1] = esq
    fourier_large[..., y2, x2] = esq

    fourier_large = torch.fft.ifftshift(fourier_large, dim=(-2, -1))
    im_large = torch.fft.ifft2(fourier_large)

    # if input was real-valued, output should be real-valued, but
    # using fft/ifft above means im_large will always be complex,
    # so make sure they align.
    if not x.is_complex():
        im_large = torch.real(im_large)
    return im_large


def shrink(x: Tensor, factor: int) -> Tensor:
    r"""
    Shrink a signal by a factor.

    We do this in the frequency domain: cropping out the center of the Fourier
    transform of ``x``, putting it in a new tensor, and taking the IFFT.

    Parameters
    ----------
    x
        The signal for expansion.
    factor
        Factor by which to resize image. Must be larger than 1 and ``factor /
        x.shape[-2:]`` must give integer values.

    Returns
    -------
    expanded
        The expanded signal.

    Raises
    ------
    ValueError
        If ``factor`` is less than or equal to 1.
    ValueError
        If the height or width of ``x`` divided by ``factor`` is not an integer.

    See Also
    --------
    expand
        The inverse operation.
    :func:`~plenoptic.tools.conv.blur_downsample`
        An alternative downsampling operation.
    """
    if factor <= 1:
        raise ValueError("factor must be strictly greater than 1!")
    im_x = x.shape[-1]
    im_y = x.shape[-2]
    mx = im_x / factor
    my = im_y / factor

    if int(mx) != mx:
        raise ValueError(f"x.shape[-1]/factor must be an integer but got {mx} instead!")
    if int(my) != my:
        raise ValueError(f"x.shape[-2]/factor must be an integer but got {my} instead!")

    mx = int(mx)
    my = int(my)

    fourier = 1 / factor**2 * torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    fourier_small = torch.zeros(
        *x.shape[:-2],
        my,
        mx,
        device=fourier.device,
        dtype=fourier.dtype,
    )

    y1 = im_y / 2 + 1 - my / 2
    y2 = im_y / 2 + my / 2
    x1 = im_x / 2 + 1 - mx / 2
    x2 = im_x / 2 + mx / 2
    # when any of these numbers are non-integers, if you round down, the
    # resulting image will be off.
    y1 = int(np.ceil(y1))
    y2 = int(np.ceil(y2))
    x1 = int(np.ceil(x1))
    x2 = int(np.ceil(x2))

    # This line is equivalent to
    fourier_small[..., 1:, 1:] = fourier[..., y1:y2, x1:x2]
    fourier_small[..., 0, 1:] = (
        fourier[..., y1 - 1, x1:x2] + fourier[..., y2, x1:x2]
    ) / 2
    fourier_small[..., 1:, 0] = (
        fourier[..., y1:y2, x1 - 1] + fourier[..., y1:y2, x2]
    ) / 2
    fourier_small[..., 0, 0] = (
        fourier[..., y1 - 1, x1 - 1]
        + fourier[..., y1 - 1, x2]
        + fourier[..., y2, x1 - 1]
        + fourier[..., y2, x2]
    ) / 4

    fourier_small = torch.fft.ifftshift(fourier_small, dim=(-2, -1))
    im_small = torch.fft.ifft2(fourier_small)

    # if input was real-valued, output should be real-valued, but
    # using fft/ifft above means im_small will always be complex,
    # so make sure they align.
    if not x.is_complex():
        im_small = torch.real(im_small)
    return im_small
