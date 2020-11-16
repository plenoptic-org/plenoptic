import numpy as np
import torch
from pyrtools.pyramids.steer import steer_to_harmonics_mtx


def minimum(x, dim=None, keepdim=False):
    r"""compute minimum in torch over any axis or combination of axes in tensor
    Parameters
    -----------
    x: torch.Tensor
        input tensor
    dim: list of ints
        dimensions over which you would like to compute the minimum
    keepdim: bool
        keep original dimensions of tensor when returning result
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
    -----------
    x: torch.Tensor
        input tensor
    dim: list of ints
        dimensions over which you would like to compute the minimum
    keepdim: bool
        keep original dimensions of tensor when returning result
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
    r"""DEPRECATED
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
    r"""DEPRECATED
    Shift the zero-frequency component to the center of the spectrum.
    The input x is expected to have real and imaginary parts along the
    last dimension.
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
    r"""DEPRECATED
    The inverse of ``batch_fftshift``.
    The input x is expected to have real and imaginary parts along the last
    dimension.
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

    """
    if dims is None:
        dims = tuple(range(1, x.ndim))
    shifts = [(x.shape[d] + 1)//2 for d in dims]
    return torch.roll(x, shifts=shifts, dims=dims)


def center_crop(x, n_shifts=7):
    """ crop to center

    expects even height and width
    """

    n_batch, n_ch, h, w = x.shape
    s = min(h, w)  # smallest size
    la = (n_shifts - 1) // 2  # autocorrelation length scale
    le = min((s - 1) // 2, la)  # making sure it fits at that scale
    ch = h // 2  # center pixel
    cw = w // 2

    return x[:, :, ch-le:ch+le+1, cw-le:cw+le+1]


def autocorr(x, n_shifts=7):
    """
    Compute the autocorrelation of `x` up to `n_shifts` shifts in Fourier space

    Notes:
    - By the Einstein-Wiener-Khinchin theorem:
    The autocorrelation of a WSS process is the inverse Fourier transform
    of its energy spectrum (ESD) - which itself is the multiplication between
    FT(x(t)) and FT(x(-t))
    aka. auto-corr is convolution with self, which is squaring in Fourier space
    This approach is computationally more efficient than brute force
    (n log(n) vs n^2).
    - By Cauchy Swartz, the autocorrelation attains it is maximum
    at the center location - that maximum value is the signal's variance
    (assuming that the input signal is mean centered).

    Parameters
    ---------
    x: torch.Tensor
        input signal of shape [b, c, h, w]

    n_shifts: integer
        length scale

    Returns
    -------
    autocorr: torch.tensor
        computed autocorrelation

    TODO
    ----
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
        autocorr = center_crop(autocorr, n_shifts=n_shifts)

    return autocorr


def rcosFn(width=1, position=0, values=(0, 1)):
    """Return a lookup table containing a "raised cosine" soft threshold
    function

    Y =  VALUES(1)
        + (VALUES(2)-VALUES(1))
        * cos^2( PI/2 * (X - POSITION + WIDTH)/WIDTH )

    this lookup table is suitable for use by `pointOp`

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
    X : `np.array`
        the x valuesof this raised cosine
    Y : `np.array`
        the y valuesof this raised cosine
    """

    sz = 256   # arbitrary!

    X = np.pi * np.arange(-sz-1, 2) / (2*sz)

    Y = values[0] + (values[1]-values[0]) * np.cos(X)**2

    # make sure end values are repeated, for extrapolation...
    Y[0] = Y[1]
    Y[sz+2] = Y[sz+1]

    X = position + (2*width/np.pi) * (X + np.pi/4)

    return X, Y


def pointOp(im, Y, X):
    r""" One-dimensional linear interpolation.

    Wrapper function to ``np.interp()`` Returns piecewise linear
    interpolant to function with given discrete datapoints (X, Y),
    evaluated at image.

    Parameters
    ----------
    im: torch.Tensor
    Y: array_like
    X: array_like

    Returns
    -------
    Interpolated image

    TODO
    ----
    rename and reorder arguments, refactor corresponding use in SteerablePyr
    """

    out = np.interp(x=im.flatten(), xp=X, fp=Y)

    return np.reshape(out, im.shape)


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
    """
    if (amplitude < 0).any():
        raise ValueError("Amplitudes must be strictly positive.")

    real = amplitude * torch.cos(phase)
    imaginary = amplitude * torch.sin(phase)
    return real, imaginary


def power_spectrum(x, log=True):
    """ Returns the fft shifted power spectrum or log power spectrum of a signal.

    Parameters
    ----------
    x: torch.Tensor
        Signal tensor
    log: bool
        Whether or not to take the log of the power. A small epsilon=1e-5 is
        added to avoid log(0) errors.
    Returns
    -------
    sp_power: torch.Tensor
        Power spectrum of signal

    """

    sp = torch.rfft(x, signal_ndim=2, onesided=False)
    sp = batch_fftshift(sp)
    amplitude, phase = rectangular_to_polar(sp[..., 0], sp[..., 1])
    sp_power = amplitude ** 2
    if log:
        sp_power[sp_power < 1e-5] += 1e-5
        sp_power = torch.log(sp_power)

    return sp_power


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


def make_disk(img_size, outer_radius=None, inner_radius=None):
    r""" Create a circularr mask with softened edges to element-wise multiply
    with an image. All values within ``inner_radius`` will be 1, and all
    values from ``inner_radius`` to ``outer_radius`` will decay smoothly to 0.

    Parameters
    ----------
    img_size: int
        Size of square image in pixels.
    outer_radius: float, optional
        Total radius of disk. Values from ``inner_radius`` to ``outer_radius``
        will decay smoothly to zero.
    inner_radius: float, optional
        Radius of inner disk. All elements from the origin to ``inner_radius``
        will be set to 1.

    Returns
    -------
    mask: torch.Tensor
        Mask with torch.Size([img_size, img_size]).

    """

    if outer_radius is None:
        outer_radius = (img_size-1) / 2

    if inner_radius is None:
        inner_radius = outer_radius / 2

    mask = torch.Tensor(img_size, img_size)
    img_center = (img_size - 1) / 2

    for i in range(img_size):
        for j in range(img_size):

            r = np.sqrt((i-img_center)**2 + (j-img_center)**2)

            if r > outer_radius:
                mask[i][j] = 0
            elif r < inner_radius:
                mask[i][j] = 1
            else:
                mask[i][j] = (
                    1 + np.cos(np.pi * (r - inner_radius) /
                               (outer_radius - inner_radius))) / 2

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
    noise_mse = torch.tensor(noise_mse, dtype=torch.float32).unsqueeze(0)
    noise_mse = noise_mse.view(noise_mse.nelement(), 1, 1, 1)
    noise = 200 * \
        torch.randn(max(noise_mse.shape[0], img.shape[0]), *img.shape[1:])
    noise = noise - noise.mean()
    noise = noise * \
        torch.sqrt(noise_mse / (noise**2).mean((-1, -2)
                                               ).unsqueeze(-1).unsqueeze(-1))
    return img + noise
