import numpy as np
import torch
from pyrtools.pyramids.steer import steer_to_harmonics_mtx
from torchvision.transforms.functional import center_crop
from typing import Union, Tuple


def steer(basis, angle, harmonics=None, steermtx=None, return_weights=False, even_phase=True):
    """Steer BASIS to the specfied ANGLE.

    Parameters
    ----------
    basis : array_like
        array whose columns are vectorized rotated copies of a steerable function, or the responses
        of a set of steerable filters.
    angle : array_like or int
        scalar or column vector the size of the basis. specifies the angle(s) (in radians) to
        steer to
    harmonics : list or None
        a list of harmonic numbers indicating the angular harmonic content of the basis. if None
        (default), N even or odd low frequencies, as for derivative filters
    steermtx : array_like or None
        matrix which maps the filters onto Fourier series components (ordered [cos0 cos1 sin1 cos2
        sin2 ... sinN]). See steer_to_harmonics_mtx function for more details. If None (default),
        assumes cosine phase harmonic components, and filter positions at 2pi*n/N.
    return_weights : bool
        whether to return the weights or not.
    even_phase : bool
        specifies whether the harmonics are cosine or sine phase aligned about those positions.

    Returns
    -------
    res : np.ndarray
        the resteered basis
    steervect : np.ndarray
        the weights used to resteer the basis. only returned if ``return_weights`` is True
    """

    num = basis.shape[-1]
    device = basis.device

    if isinstance(angle, (int, float)):
        angle = np.array([angle])
    else:
        if angle.shape[0] != basis.shape[0] or angle.shape[1] != 1:
            raise Exception("""ANGLE must be a scalar, or a column vector the size of the basis elements""")

    # If HARMONICS is not specified, assume derivatives.
    if harmonics is None:
        harmonics = np.arange(1 - (num % 2), num, 2)

    if len(harmonics.shape) == 1 or harmonics.shape[0] == 1:
        # reshape to column matrix
        harmonics = harmonics.reshape(harmonics.shape[0], 1)
    elif harmonics.shape[0] != 1 and harmonics.shape[1] != 1:
        raise Exception('input parameter HARMONICS must be 1D!')

    if 2 * harmonics.shape[0] - (harmonics == 0).sum() != num:
        raise Exception('harmonics list is incompatible with basis size!')

    # If STEERMTX not passed, assume evenly distributed cosine-phase filters:
    if steermtx is None:
        steermtx = steer_to_harmonics_mtx(harmonics, np.pi * np.arange(num) / num, even_phase=even_phase)

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

    steervect = torch.tensor(steervect, dtype=basis.dtype).to(device)
    if steervect.shape[0] > 1:
        tmp = basis @ steervect
        res = tmp.sum().t()
    else:
        res = basis @ steervect.t()
    if return_weights:
        return res, steervect.reshape(num)
    else:
        return res

def minimum(x, dim=None, keepdim=False):
    r"""compute minimum in torch over any axis or combination of axes in tensor

    Parameters
    ----------
    x: torch.Tensor
        input tensor
    dim: list of ints
        dimensions over which you would like to compute the minimum
    keepdim: bool
        keep original dimensions of tensor when returning result

    Returns
    -------
    min_x : torch.Tensor
        Minimum value of x.

    """
    if dim is None:
        dim = tuple(range(x.ndim))
    dim = reversed(sorted(dim))
    min_x = x
    for i in dim:
        min_x, _ = min_x.min(i, keepdim)
    return min_x


def maximum(x, dim=None, keepdim=False):
    r"""compute maximum in torch over any dim or combination of axes in tensor

    Parameters
    ----------
    x: torch.Tensor
        input tensor
    dim: list of ints
        dimensions over which you would like to compute the minimum
    keepdim: bool
        keep original dimensions of tensor when returning result

    Returns
    -------
    max_x : torch.Tensor
        Maximum value of x.

    """
    if dim is None:
        dim = tuple(range(x.ndim))
    dim = reversed(sorted(dim))
    max_x = x
    for i in dim:
        max_x, _ = max_x.max(i, keepdim)
    return max_x


def rescale(x, a=0, b=1):
    r"""Linearly rescale the dynamic range of the input x to [a,b]
    """
    v = x.max() - x.min()
    g = (x - x.min())
    if v > 0:
        g = g / v
    return a + g * (b-a)


def roll_n(X, axis, n):
    r"""
    Performs circular shift by ``n`` indices along given axis.
    Helper for ``fftshift``.

    Parameters
    ----------
    X: torch.Tensor
        Signal or frequency domain
    axis: int
        Axis along which to roll
    n: int
        How many indices to circularly shift
    Returns
    -------
    rolled: torch.Tensor

    TODO: DEPRECATED use torch.fft.fftshift
    """
    f_idx = tuple(slice(None, None, None) if i !=
                  axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(
        n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    rolled = torch.cat([back, front], axis)
    return rolled


def batch_fftshift(x):
    r"""
    Shift the zero-frequency component to the center of the spectrum.
    The input x is expected to have real and imaginary parts along the
    last dimension.

    TODO: DEPRECATED use torch.fft.fftshift
    """
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    # preallocation is much faster than using stack
    shifted = torch.empty((*real.shape, 2), device=real.device)
    shifted[..., 0] = real
    shifted[..., 1] = imag
    return shifted  # last dim=2 (real&imag)


def batch_ifftshift(x):
    r"""
    The inverse of ``batch_fftshift``.
    The input x is expected to have real and imaginary parts along the last
    dimension.
    
    TODO: DEPRECATED use torch.fft.fftshift
    """
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    # preallocation is much faster than using stack
    shifted = torch.empty((*real.shape, 2), device=real.device)
    shifted[..., 0] = real
    shifted[..., 1] = imag
    return shifted  # last dim=2 (real&imag)


def fftshift(x, dims=None):
    r"""Shift the zero-frequency component to the center of the spectrum.

    Parameters
    ---------
    x: torch.Tensor
        spectrum

    dims: tuple, optional
        dimensions along which to shift the spectrum.
        by default it will shift all but the first dimension (batch dimension).

    Returns
    -------
    x: torch.Tensor
        shifted spectrum

    TODO: DEPRECATED use torch.fft.fftshift
    """
    if dims is None:
        dims = tuple(range(1, x.ndim))
    shifts = [(x.shape[d] + 1)//2 for d in dims]
    return torch.roll(x, shifts=shifts, dims=dims)


def autocorr(x, n_shifts=7):
    """ Compute the autocorrelation of `x` up to `n_shifts` shifts in Fourier space

    Notes:
    - By the Einstein-Wiener-Khinchin theorem:
    The autocorrelation of a WSS process is the inverse Fourier transform
    of its energy spectrum (ESD) - which itself is the multiplication between
    FT(x(t)) and FT(x(-t))
    aka. auto-corr is convolution with self, which is squaring in Fourier space
    This approach is computationally more efficient than brute force
    (n log(n) vs n^2).
    - By Cauchy-Swartz, the autocorrelation attains it is maximum
    at the center location - that maximum value is the signal's variance
    (assuming that the input signal is mean centered).

    Parameters
    ---------
    x: torch.Tensor
        input signal of shape [b, c, h, w]
    n_shifts: integer
        Sets the length scale of the auto-correlation
        (ie. maximum offset or lag)

    Returns
    -------
    autocorr: torch.tensor
        computed autocorrelation

    TODO
    ----
    use torch.fft.rfftn
    signal_ndim argument
    ESD PSD
    periodogram
    """
    n_batch, n_ch, h, w = x.shape

    spectrum = torch.rfft(x, signal_ndim=2, normalized=False)
    energy_spectrum = torch.sum(spectrum ** 2, dim=-1, keepdim=True)
    zero_phase = torch.zeros_like(energy_spectrum)
    energy_spectrum = torch.cat((energy_spectrum, zero_phase), dim=-1)

    autocorr = torch.irfft(energy_spectrum, signal_ndim=2,
                           normalized=False, signal_sizes=(h, w))
    autocorr = fftshift(autocorr, dims=(2, 3)) / (h*w)

    if n_shifts is not None:
        autocorr = center_crop(autocorr, output_size=(n_shifts, n_shifts))

    return autocorr


def raised_cosine(width=1, position=0, values=(0, 1)):
    """Return a lookup table containing a "raised cosine" soft threshold
    function

    Y =  VALUES(1)
        + (VALUES(2)-VALUES(1))
        * cos^2( PI/2 * (X - POSITION + WIDTH)/WIDTH )

    this lookup table is suitable for use by `interpolate1d`

    Parameters
    ---------
    width : float
        the width of the region over which the transition occurs
    position : float
        the location of the center of the threshold
    values : tuple
        2-tuple specifying the values to the left and right of the transition.

    Returns
    -------
    X : `np.ndarray`
        the x values of this raised cosine
    Y : `np.ndarray`
        the y values of this raised cosine
    """

    sz = 256   # arbitrary!

    X = np.pi * np.arange(-sz-1, 2) / (2*sz)

    Y = values[0] + (values[1]-values[0]) * np.cos(X) ** 2

    # make sure end values are repeated, for extrapolation...
    Y[0] = Y[1]
    Y[sz+2] = Y[sz+1]

    X = position + (2*width/np.pi) * (X + np.pi / 4)

    return X, Y


def interpolate1d(x_new, Y, X):
    r""" One-dimensional linear interpolation.

    Returns the one-dimensional piecewise linear interpolant to a
    function with given discrete data points (X, Y), evaluated at x_new.

    Note: this function is just a wrapper around ``np.interp()``.

    Parameters
    ----------
    x_new: torch.Tensor
        The x-coordinates at which to evaluate the interpolated values.
    Y: array_like
        The y-coordinates of the data points.
    X: array_like
        The x-coordinates of the data points, same length as X.

    Returns
    -------
    Interpolated values of shape identical to `x_new`.

    TODO
    ----
    rename and reorder arguments, refactor corresponding use in SteerablePyr
    """

    out = np.interp(x=x_new.flatten(), xp=X, fp=Y)

    return np.reshape(out, x_new.shape)


def rectangular_to_polar(real, imaginary):
    r"""Rectangular to polar coordinate transform

    Parameters
    --------
    real: torch.Tensor
        tensor containing the real component
    imaginary: torch.Tensor
        tensor containing the imaginary component

    Returns
    -------
    amplitude: torch.Tensor
        tensor containing the amplitude (aka. complex modulus)
    phase: torch.Tensor
        tensor containing the phase
    Note
    ----
    Since complex numbers are not supported by pytorch, this function expects
    two tensors of the same shape. One containing the real component, one
    containing the imaginary component. This means that if complex numbers are
    represented as an extra dimension in the tensor of interest, the user
    needs to index through that dimension.

    TODO: torch.abs(cmplx) torch.angle(cmplx)
    """

    amplitude = torch.sqrt(real ** 2 + imaginary ** 2)
    phase = torch.atan2(imaginary, real)
    return amplitude, phase


def polar_to_rectangular(amplitude, phase):
    r"""Polar to rectangular coordinate transform

    Parameters
    ----------
    amplitude: torch.Tensor
        tensor containing the amplitude (aka. complex modulus). Must be > 0.
    phase: torch.Tensor
        tensor containing the phase

    Returns
    -------
    real: torch.Tensor
        tensor containing the real component
    imaginary: torch.Tensor
        tensor containing the imaginary component

    Note
    ----
    Since complex numbers are not supported by pytorch, this function returns
    two tensors of the same shape. One containing the real component, one
    containing the imaginary component.

    TODO: torch.complex(abs, angle)
    """
    if (amplitude < 0).any():
        raise ValueError("Amplitudes must be strictly positive.")

    real = amplitude * torch.cos(phase)
    imaginary = amplitude * torch.sin(phase)
    return real, imaginary


# def power_spectrum(x, log=True):
#     """ Compute the power spectrum of a spatial signal.

#     Parameters
#     ----------
#     x: torch.Tensor
#         Signal tensor [B, C, H, W]
#     log: bool
#         Apply the non-linear contrast function `log(1 + power)`.
#     Returns
#     -------
#     power: torch.Tensor
#         Power spectrum of signal along H and W

#     TODO: with torch 1.8
#     """
#     import torch.fft
#     spectrum = torch.fft.rfftn(x, dim=(2, 3), norm='ortho')
#     spectrum = torch.fft.fftshift(spectrum, dim=(2, 3))
#     amplitude = torch.abs(spectrum)
#     power = amplitude ** 2
#     if log:
#         return torch.log(1 + power)
#     else:
#         return power


def steer(basis, angle, harmonics=None, steermtx=None, return_weights=False,
          even_phase=True):
    """Steer BASIS to the specfied ANGLE.

    Parameters
    ----------
    basis : array_like
        array whose columns are vectorized rotated copies of a steerable
        function, or the responses of a set of steerable filters.
    angle : array_like or int
        scalar or column vector the size of the basis. specifies the angle(s)
        (in radians) to steer to
    harmonics : list or None
        a list of harmonic numbers indicating the angular harmonic content of
        the basis. if None (default), N even or odd low frequencies, as for
        derivative filters
    steermtx : array_like or None
        matrix which maps the filters onto Fourier series components (ordered
        [cos0 cos1 sin1 cos2 sin2 ... sinN]). See steer_to_harmonics_mtx
        function for more details. If None (default), assumes cosine phase
        harmonic components, and filter positions at 2pi*n/N.
    return_weights : bool
        whether to return the weights or not.
    even_phase : bool
        specifies whether the harmonics are cosine or sine phase aligned about
        those positions.

    Returns
    -------
    res : np.ndarray
        the resteered basis
    steervect : np.ndarray
        the weights used to resteer the basis. only returned if
        ``return_weights`` is True
    """

    num = basis.shape[-1]
    device = basis.device

    if isinstance(angle, (int, float)):
        angle = np.array([angle])
    else:
        if angle.shape[0] != basis.shape[0] or angle.shape[1] != 1:
            raise Exception("ANGLE must be a scalar, or a column vector the"
                            "size of the basis elements")

    # If HARMONICS is not specified, assume derivatives.
    if harmonics is None:
        harmonics = np.arange(1 - (num % 2), num, 2)

    if len(harmonics.shape) == 1 or harmonics.shape[0] == 1:
        # reshape to column matrix
        harmonics = harmonics.reshape(harmonics.shape[0], 1)
    elif harmonics.shape[0] != 1 and harmonics.shape[1] != 1:
        raise Exception('input parameter HARMONICS must be 1D!')

    if 2 * harmonics.shape[0] - (harmonics == 0).sum() != num:
        raise Exception('harmonics list is incompatible with basis size!')

    # If STEERMTX not passed, assume evenly distributed cosine-phase filters:
    if steermtx is None:
        steermtx = steer_to_harmonics_mtx(
            harmonics, np.pi * np.arange(num) / num, even_phase=even_phase)

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

    steervect = torch.tensor(steervect, dtype=basis.dtype).to(device)
    if steervect.shape[0] > 1:
        tmp = basis @ steervect
        res = tmp.sum().t()
    else:
        res = basis @ steervect.t()
    if return_weights:
        return res, steervect.reshape(num)
    else:
        return res


def make_disk(img_size: Union[int, Tuple[int, int]],
              outer_radius: float = None,
              inner_radius: float = None) -> torch.Tensor:
    r""" Create a circular mask with softened edges to  an image.
    All values within ``inner_radius`` will be 1, and all values from ``inner_radius``
    to ``outer_radius`` will decay smoothly to 0.

    Parameters
    ----------
    img_size:
        Size of image in pixels.
    outer_radius:
        Total radius of disk. Values from ``inner_radius`` to ``outer_radius`` will
        decay smoothly to zero.
    inner_radius:
        Radius of inner disk. All elements from the origin to ``inner_radius`` will be
        set to 1.

    Returns
    -------
    mask:
        Tensor mask with torch.Size(img_size).

    """

    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    assert len(img_size) == 2

    if outer_radius is None:
        outer_radius = (min(img_size)-1) / 2

    if inner_radius is None:
        inner_radius = outer_radius / 2

    mask = torch.empty(*img_size)
    i0, j0 = (img_size[0] - 1) / 2, (img_size[1] - 1) / 2  # image center

    for i in range(img_size[0]):  # height
        for j in range(img_size[1]):  # width

            r = np.sqrt((i-i0)**2 + (j-j0)**2)

            if r > outer_radius:
                mask[i][j] = 0
            elif r < inner_radius:
                mask[i][j] = 1
            else:
                radial_decay = (r - inner_radius) / (outer_radius - inner_radius)
                mask[i][j] = (1 + np.cos(np.pi * radial_decay)) / 2

    return mask


def add_noise(img, noise_mse):
    """Add normally distributed noise to an image

    This adds normally-distributed noise to an image so that the resulting
    noisy version has the specified mean-squared error.

    Parameters
    ----------
    img : torch.Tensor
        the image to make noisy
    noise_mse : float or list
        the target MSE value / variance of the noise. More than one value is
        allowed

    Returns
    -------
    noisy_img : torch.Tensor
        the noisy image. If `noise_mse` contains only one element, this will be
        the same size as `img`. Else, each separate value from `noise_mse` will
        be along the batch dimension.

    TODO
    ----
    parametrize in terms of SNR

    """
    noise_mse = torch.tensor(noise_mse, dtype=torch.float32, device=img.device).unsqueeze(0)
    noise_mse = noise_mse.view(noise_mse.nelement(), 1, 1, 1)
    noise = 200 * torch.randn(max(noise_mse.shape[0], img.shape[0]), *img.shape[1:], device=img.device)
    noise = noise - noise.mean()
    noise = noise * \
        torch.sqrt(noise_mse / (noise**2).mean((-1, -2)
                                               ).unsqueeze(-1).unsqueeze(-1))
    return img + noise
