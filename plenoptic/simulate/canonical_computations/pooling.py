"""functions to perform spatial pooling, as seen in Freeman and Simoncelli, 2011

In addition the raised-cosine windows used in that paper, we also
provide support for an alternative window construction:
Gaussians. They're laid out in the same fashion as the raised-cosine
windows, but are wider and have values everywhere (whereas the
raised-cosine windows are clipped so that they're zero for most of the
image). Using the raised-cosine windows led to issues with aliasing in
metamer synthesis, visible as ringing, with the PrimaryVisualCortex
model, because of the interactions between the windows and the steerable
pyramid filters.

The Gaussian windows don't have these problems, but require more windows
to evenly tile the image in the radial direction (and thus
PoolingWindows.forward will take more memory and more time). Note as
well that, whereas the max amplitude of the raised-cosine windows is
always 1 (for all transition region widths), the Gaussian windows will
have their max amplitude scaled down as their standard deviation
increases; as the standard deviation increases, the windows overlap
more, so that the number of windows a given pixel lies in increases and
thus the weighting in each of them needs to decrease in order to make
sure the sum across all windows is still 1 for every pixel. The Gaussian
windows will always intersect at x=.5, but the interpretation of this
depends on its standard deviation. For Gaussian windows, we recommend
(and only expect) a standard deviation of 1, so that each window
intersects at half a standard deviation. We support larger windows, but
these are intended to be used as the surround in the
difference-of-gaussian windows (centers should still have standard
deviation fo 1)

"""
import math
import itertools
import torch
import warnings
import numpy as np
import pyrtools as pt
import matplotlib.pyplot as plt
import os.path as op
from torch import nn
from ...tools.data import to_numpy, polar_angle, polar_radius

# see docstring of gaussian function for explanation of this constant
GAUSSIAN_SUM = 2 * 1.753314144021452772415339526931980189073725635759454989253 - 1


def piecewise_log(x, transition_x=1):
    r"""piecewise log-like function for handling the fovea better

    this function is used to give us another way of handling the
    fovea. it is a piecewise function that is linear at the fovea and
    log-like beyond that, with the transition specified by the arg
    ``transition_x``

    Specifically, let :math:`x_0` be ``transition_x``, then this
    function returns:

    .. math::

       \[ f(x)= \begin{cases}
                   - \log(-x+1-x_0)-x_0 & x < -x_0 \\
                   x & -x_0 \leq x \leq x_0 \\
                   \log(x+1-x_0)+x_0 & x > x_0
                \end{cases}
       \]

    The log portions of this function is slightly complicated by the
    fact that we want the value and the first derivative of the two
    segments to be equal at the transitions, :math:`x=\pm x_0`, and you
    can see that the values are :math:`\pm x_0` and a derivative of 1
    at :math:`x=\pm x_0`

    Parameters
    ----------
    x : torch.tensor, float, or array_like
        the tensor to transform. if not a tensor, then we transform it
        to a tensor
    transition_x : float, optional
        the value to transition from linear to log at, :math:`x_0` in
        equation above, must be positive

    Returns
    -------
    y : torch.tensor
        the transformed tensor

    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if transition_x < 0:
        raise Exception("transition_x must be positive!")
    # because we're doing this on tensors, we can't do it in a clean way
    # (if we were doing it on arrays, we could use np.piecewise but
    # alas). note that we're only replacing the bit larger than
    # transition_x, everything else gets left alone
    x = x.clone()
    # do the positive side
    x[x > transition_x] = torch.log(x[x > transition_x] + 1 - transition_x) + transition_x
    # and the negative side
    x[x < -transition_x] = -torch.log(-x[x < -transition_x] + 1 - transition_x) - transition_x
    return x


def piecewise_log_inverse(y, transition_x=1):
    r"""the inverse of our ``piecewise_log`` function

    this is just the inverse of ``piecewise_log``, so that
    ``piecewise_log_inverse(piecewise_log(x))=x``. we need this to make
    some of the calculations about our windows.

    Specifically, let :math:`x_0` be ``transition_x``, then this
    function returns (note that :math:`y=x` at :math:`x_0`):

    .. math::

       \[ f(y)= \begin{cases}
                   \exp(-y-x_0)-x_0+1 & y < -x_0 \\
                   y & -x_0 \leq y \leq x_0 \\
                   \exp(y-x_0)+x_0-1 & y > x_0
                \end{cases}
       \]

    Parameters
    ----------
    y : torch.tensor, float, or array_like
        the transformed tensor to transform. if not a tensor, then we transform it
        to a tensor
    transition_x : float, optional
        the value to transition from linear to log at, :math:`x_0` in
        equation above, must be positive

    Returns
    -------
    x : torch.tensor
        the untransformed tensor

    Notes
    -----

    Deriving this is relatively straightforward. The linear section is
    its own inverse, so we just need to work through the positive log
    section:

    .. math::

       y &= \log(x+1-x_0)+x_0
       \exp(y) &= \exp(\log(x+1-x_0)+x_0)
       \exp(y) &= (x+1-x_0)\exp(x_0)
       \frac{\exp(y)}{\exp(x_0)} &= x+1-x_0
       \exp(y-x_0) &= x+1-x_0
       \exp(y-x_0)+x_0-1 &= x

    and the negative:

    .. math::

       y &= -\log(-x+1-x_0)-x_0
       \exp(y) &= \exp(\log((-x+1-x_0)^{-1})-x_0)
       \exp(y) &= \frac{1}{(-x+1-x_0)\exp(x_0)}
       \frac{1}{\exp(y+x_0)} &= -x+1-x_0
       \exp(-y-x_0) &= -x+1-x_0
       \exp(-y-x_0)-x_0+1 &= x

    """
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    if transition_x < 0:
        raise Exception("transition_x must be positive!")
    # because we're doing this on tensors, we can't do it in a clean way
    # (if we were doing it on arrays, we could use np.piecewise but
    # alas). note that we're only replacing the bit larger than
    # transition_x, everything else gets left alone
    y = y.clone()
    # do the positive side
    y[y > transition_x] = torch.exp(y[y > transition_x] - transition_x) + transition_x - 1
    # do the negative side
    y[y < -transition_x] = -torch.exp(-y[y < -transition_x] - transition_x) - transition_x + 1
    return y


def _calc_reflected_idx(multiplier):
    r"""calculate indices for summing windows output

    this helper function calculates the indices used by the
    sum_windows_output() function, so they can be calculated once and then
    cached

    we make strong assumptions about how the reflected windows are laid
    out and will raise an Exception if they look like they're broken

    Parameters
    ----------
    multiplier : torch.tensor
        the multiplier returned by log_eccentricity_windows(), a tensor
        containing 1 or -1 for each window (along the first dimension),
        which shows which windows are the reflected ones

    Returns
    -------
    reflected_idx : torch.tensor
        a 1d boolean tensor specifying which windows are the reflected ones
    normal_idx : torch.tensor
        a 1d boolean tensor specifying which windows are the normal
        (un-reflected) ones

    """
    reflected_idx = torch.where(multiplier==-1)[0]
    normal_idx = torch.where(multiplier==1)[0]
    if reflected_idx[0] != 1 or not (np.diff(reflected_idx)==2).all():
        raise Exception("Something went wrong in the construction of your multiplier! It should"
                        " have its first -1 at index 1 and then every other index")
    return normal_idx, reflected_idx


def sum_windows_output(windows_output, multiplier, dim=0):
    r"""sum output of reflected and corresponding normal windows together

    this function is used when we have reflected windows and so need to
    combine their output with the output of the corresponding non-reflected
    window (otherwise we overweight the output of the reflected window)

    the indices (as computed by ``_calc_reflected_idx()``) can be passed
    directly to this function (for speed). if not, we'll calculate them
    ourselves

    we make strong assumptions about how the reflected windows are laid
    out and will raise an Exception if they look like they're broken

    Parameters
    ----------
    windows_output : torch.tensor
        the tensor containing the windows outputs to sum.
    multiplier : torch.tensor
        the multiplier returned by log_eccentricity_windows(), a tensor
        containing 1 or -1 for each window (along the first dimension),
        which shows which windows are the reflected ones.
    dim : {0, -1}, optional
        whether to sum along the first or last dimension. For now, no other
        dimensions are supported

    """
    normal_idx, reflected_idx = _calc_reflected_idx(multiplier)
    summed = windows_output[normal_idx].clone()
    if dim == 0:
        summed[reflected_idx[0]:len(reflected_idx)+1] += windows_output[reflected_idx]
    elif dim == -1 or dim == windows_output.ndim - 1:
        summed[..., reflected_idx[0]:len(reflected_idx)+1] += windows_output[..., reflected_idx]
    else:
        raise Exception("Currently only implemented for dim=0 or dim=-1!")
    return summed


def calc_angular_window_spacing(n_windows):
    r"""calculate and return the window spacing for the angular windows

    this is the :math:`w_{\theta }` term in equation 10 of the paper's
    online methods, referred to as the angular window width.

    For both cosine and gaussian windows, this is the distance between
    the peaks of the windows. For cosine windows, this is also the same
    as the windows' widths, but gausian windows' widths are
    approximately ``window_spacing * std_dev * 3`` (since they're
    Gaussian, 99.73% of their mass lie within 3 standard deviations, but
    the Gaussians are technically infinite)

    Parameters
    ----------
    n_windows : `float`
        The number of windows to pack into 2 pi. Note that we don't
        require it to be an integer here, but the code that makes use of
        this does.

    Returns
    -------
    window_spacing : `float`
        The spacing of the polar angle windows.

    """
    return (2*np.pi) / n_windows


def calc_angular_n_windows(window_spacing):
    r"""calculate and return the number of angular windows

    this is the :math:`N_{\theta }` term in equation 10 of the paper's
    online method, which we've rearranged in order to get this.

    Parameters
    ----------
    window_spacing : `float`
        The spacing of the polar angle windows.

    Returns
    -------
    n_windows : `float`
        The number of windows that fit into 2 pi.

    """
    return (2*np.pi) / window_spacing


def calc_eccentricity_window_spacing(min_ecc=.5, max_ecc=15, n_windows=None, scaling=None,
                                     std_dev=None, transition_x=None):
    r"""calculate and return the window spacing for the eccentricity windows

    this is the :math:`w_e` term in equation 11 of the paper's online
    methods (referred to as the window width), which we also refer to as
    the radial spacing. Note that we take exactly one of ``n_windows``
    or ``scaling`` in order to determine this value.

    If scaling is set, ``min_ecc`` / ``transition_x`` and ``max_ecc``
    are ignored (the window width only depends on scaling, not also on
    the range of eccentricities; they only matter when determining the
    width using ``n_windows``)

    For both cosine and gaussian windows, this is the distance between
    the peaks of the windows. For cosine windows, this is also the same
    as the windows' widths, but gausian windows' widths are
    approximately ``window_spacing * std_dev * 3`` (since they're
    Gaussian, 99.73% of their mass lie within 3 standard deviations, but
    the Gaussians are technically infinite); but remember that these
    values are in log space not linear.

    One of ``min_ecc`` or ``transition_x`` must be set. They determine
    how we handle the fovea (see the docstring of
    ``log_eccentricity_windows()`` for more details).

    Parameters
    ----------
    min_ecc : float or None, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods. If set, ``transition_x`` must
        be None. If None, ``transition_x`` must be set
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    n_windows : `float` or `None`
        The number of log-eccentricity windows we create. ``n_windows``
        xor ``scaling`` must be set.
    scaling : `float` or `None`.
        The ratio of the eccentricity window's radial full-width at
        half-maximum to eccentricity (see the ``calc_scaling``
        function). ``n_windows`` xor ``scaling`` must be set.
    std_dev : float or None, optional
        The standard deviation of the Gaussian window. If this is set,
        we compute the scaling value for the Gaussian windows instead of
        for the cosine ones.
    transition_x : float or None, optional
        If set, the point at which the eccentricity transitions from
        linear to log, see ``log_eccentricity_windows()`` for more
        details. If set, ``min_ecc`` must be None. If None, ``min_ecc``
        must be set. If set, we assume ``n_windows`` is the *total*
        number of windows (including both the foveal/linear windows and
        the log windows)

    Returns
    -------
    window_spacing : `float`
        The spacing  of the log-eccentricity windows.

    Notes
    -----
    No equation was given in the paper to calculate the window spacing,
    :math:`w_e` from the scaling, :math:`s`, so we derived it
    ourselves. We start with the final equation for the scaling, given
    in the Notes for the ``calc_scaling`` function.

    .. math::

        s &= \exp(w_e x_h) - \exp(-w_e x_h) \\
        s &= \exp(w_e x_h) - \frac{1}{\exp(w_e x_h)} \\

    We then substitute :math:`t=\exp(w_e x_h)`

    .. math::

        s &= t - \frac{1}{t}
        0 &= t^2 - st - 1

    Then using the quadratic formula:

    .. math::

        t &= \frac{s \pm \sqrt{s^2+4}}{2}

    We then substitute back for :math:`t` and drop the negative root
    because the window spacing is strictly positive.

    .. math::

        \exp(w_e x_h) &= \frac{s + \sqrt{s^2 + 4}}{2}
        w_e &= \log(\frac{s+\sqrt{s^2+4}}{2}) / x_h

    If ``transition_x``, :math:`x_0` is set, then things are slightly
    more complicated because we'll have linear windows at the fovea
    (between eccentricity 0 and :math:`x_0`) and log windows (our
    standard windows) everywhere else, and we want the spacing to be the
    same for both. To do so, we start with the equation for the total
    number of windows, as given in the ends of the Notes section of the
    ``calc_eccentricity_n_windows`` docstring and rearrange, solving for
    :math:`w_e` (remembering again that we're using our
    ``piecewise_log`` function, :math:`plog(x,x_0)` instead of the
    regular log function)

    .. math::

       N_e &= \frac{plog(e_m,x_0)-plog(x_0,x_0)+x_0}{w_e}+1
       w_e &= \frac{plog(e_m,x_0)-plog(x_0,x_0)+x_0+w_e}{N_e}
       w_e-\frac{w_e}{N_e} &= \frac{plog(e_m,x_0)-plog(x_0,x_0)+x_0}{N_e}
       w_e(1 - \frac{1}{N_e}) &= \frac{plog(e_m,x_0)-plog(x_0,x_0)+x_0}{N_e}
       w_e(\frac{N_e-1}{N_e}) &= \frac{plog(e_m,x_0)-plog(x_0,x_0)+x_0}{N_e}
       w_e &= \frac{plog(e_m,x_0)-plog(x_0,x_0)+x_0}{N_e-1}

    If ``transition_x`` is set and scaling is set, then we compute
    :math:`w_e` from scaling as above. See Notes section of
    ``calc_scaling`` docstring for why, but scaling is identical as long
    as ``transition_x=1`` (and so we raise an Exception if this is not
    the case)

    """
    if scaling is not None:
        if std_dev is None:
            x_half_max = .5
        else:
            x_half_max = std_dev * np.sqrt(2 * np.log(2))
        spacing = np.log((scaling + np.sqrt(scaling**2+4))/2) / x_half_max
        if transition_x is not None and transition_x != 1:
            raise Exception(f"transition_x must be 1, but got value {transition_x}! If transition_"
                            "_x is not 1, scaling will not be constant across the image. See "
                            "calc_scaling docstring Notes section for more details")
    elif n_windows is not None:
        if transition_x is not None:
            if min_ecc is not None:
                raise Exception("Exactly one of transition_x or min_ecc must be set!")
            # see Notes section of docstring for explanation of this
            spacing = (piecewise_log(max_ecc) - piecewise_log(transition_x) +
                       transition_x) / (n_windows - 1)
        else:
            spacing = (np.log(max_ecc) - np.log(min_ecc)) / n_windows
    else:
        raise Exception("Exactly one of n_windows or scaling must be set!")
    return spacing


def calc_eccentricity_n_windows(window_spacing, min_ecc=.5, max_ecc=15, std_dev=None,
                                transition_x=None):
    r"""calculate and return the number of eccentricity windows

    this is the :math:`N_e` term in equation 11 of the paper's online
    method, which we've rearranged in order to get this.

    One of ``min_ecc`` or ``transition_x`` must be set. They determine
    how we handle the fovea (see the docstring of
    ``log_eccentricity_windows()`` for more details).

    Parameters
    ----------
    window_spacing : `float`
        The spacing of the log-eccentricity windows.
    min_ecc : float or None, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    std_dev : float or None, optional
        The standard deviation of the Gaussian window. Adds extra
        windows to account for the fact that Gaussian windows are
        larger. If using cosine windows, this should be None
    transition_x : float or None, optional
        If set, the point at which the eccentricity transitions from
        linear to log, see ``log_eccentricity_windows()`` for more
        details. If set, ``min_ecc`` must be None. If None, ``min_ecc``
        must be set

    Returns
    -------
    n_windows : `float`
        The number of log-eccentricity windows we create.

    Notes
    -----

    If ``transition_x``, :math:`x_0` is set, then things are slightly
    more complicated because we'll have linear windows at the fovea
    (between eccentricity 0 and :math:`x_0`) and log windows (our
    standard windows) everywhere else. Our total number of windows then
    is the sum of these two populations:

    .. math::

       N_e = N_f + N_l

    where :math:`N_f` are the foveal (linear) windows and :math:`N_l`
    log windows. The log windows are calculated as our standard windows
    are (see paper [1]_), starting from ``transition_x``, :math:`x_0`
    instead of ``min_ecc``, except that we replace :math:`\log` with our
    ``piecewise_log`` function, :math:`plog(x, x_0)`:

    .. math::

        N_l = \frac{plog(e_m, x_0)-plog(x_0, x_0)}{w_e}

    The linear windows are simple and, analogously to the angular
    windows, we have the following numbers of windows:

    .. math::

       N_f = \frac{x_0}{w_e} + 1

    we add the 1 because we have a window centered at eccentricity 0, so
    that if the linear region extends to .5 and our spacing :math:`w_e`
    is .5, we'll have 2 windows.

    Thus, our total number of windows is:

    .. math::

       N_e = \frac{plog(e_m, x_0)-plog(x_0, x_0)+x_0}{w_e}+1

    """
    if transition_x is not None:
        if min_ecc is not None:
            raise Exception("Exactly one of transition_x or min_ecc must be set!")
        # see notes in docstring for explanation of this
        n_linear_windows = (transition_x / window_spacing) + 1
        n_windows = (piecewise_log(max_ecc) - piecewise_log(transition_x)) / window_spacing
        n_windows += n_linear_windows
    else:
        n_windows = (np.log(max_ecc) - np.log(min_ecc)) / window_spacing
    # the Gaussians need extra windows in order to make sure that we're
    # summing to 1 across the whole image (because they're wider and
    # shorter). to make sure of this, we want to get all the windows
    # past it up til the one who is 5 standard deviations away from the
    # outermost window calculated above (this matters more for larger
    # values of std_dev / larger windows).
    if std_dev is not None:
        n_windows += 5 * std_dev
    return n_windows


def calc_scaling(n_windows, min_ecc=.5, max_ecc=15, std_dev=None, transition_x=None):
    r"""calculate and return the scaling value, as reported in the paper

    Scaling is the ratio of the eccentricity window's radial full-width
    at half-maximum to eccentricity. For eccentricity, we use the
    window's "central eccentricity", the one where the input to the
    mother window (:math:`x` in equation 9 in the online methods) is 0.

    One of ``min_ecc`` or ``transition_x`` must be set. They determine
    how we handle the fovea (see the docstring of
    ``log_eccentricity_windows()`` for more details). They also
    determine which function we use to invert our log-transform:
    ``np.exp`` (if ``min_ecc`` is set) or ``piecewise_log_inverse`` (if
    ``transition_x`` is set). The only supported value for
    ``transition_x`` is 1, see the Notes section for more details

    Parameters
    ----------
    n_windows : `float`
        The number of log-eccentricity windows we create.
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    std_dev : float or None, optional
        The standard deviation fo the Gaussian window. If this is set,
        we compute the scaling value for the Gaussian windows instead of
        for the cosine ones.
    transition_x : {1, None}, optional
        If set, the point at which the eccentricity transitions from
        linear to log, see ``log_eccentricity_windows()`` for more
        details. If set, ``min_ecc`` must be None. If None, ``min_ecc``
        must be set. If set, we assume ``n_windows`` is the *total*
        number of windows (including both the foveal/linear windows and
        the log windows). We only support 1, since that's the only value
        for which scaling is constnat across all (log) windows

    Returns
    -------
    scaling : `float`.
        The ratio of the eccentricity window's radial full-width at
        half-maximum to eccentricity

    Notes
    -----
    No equation for the scaling, :math:`s`, was included in the paper,
    so we derived this ourselves. To start, we note that the mother
    window equation (equation 9) reaches its half-max (.5) at
    :math:`x=\pm .5`, and that, as above, we treat :math:`x=0` as the
    central eccentricity of the window. Then we must solve for these,
    using the values given within the parentheses in equation 11 as the
    value for :math:`x`, and take their ratios.

    In the following equations, we'll use :math:`x_h` as the value at
    which the window reaches its half-max. For the cosine windows, this
    is always :math:`\pm .5`, but for the Gaussian windows, it's
    :math:`x_h=\sigma\sqrt{2\log 2}`.

    It turns out that this holds for all permissible values of
    ``transition_region_width`` (:math:`t` in the equations) (try
    playing around with some plots if you don't believe me).

    Full-width half-maximum, :math:`W`, the difference between the two
    values of :math:`e_h`:

    .. math::

        \pm x_h &= \frac{\log(e_h) - (log(e_0)+w_e(n+1))}{w_e} \\
        e_h &= e_0 \cdot \exp(w_e(\pm x_h+n+1)) \\
        W &= e_0 (\exp(w_e(n+1+x_h)) - \exp(w_e(n+1-x_h))

    Window's central eccentricity, :math:`e_c`:

    .. math::

        0 &= \frac{\log(e_c) -(log(e_0)+w_e(n+1))}{w_e} \\
        e_c &= e_0 \cdot \exp(w_e(n+1))

    Then the scaling, :math:`s` is the ratio :math:`\frac{W}{e_c}`:

    .. math::

        s &= \frac{e_0 (\exp(w_e(n+1+x_h)) -  \exp(w_e(n+1-x_h)))}{e_0 \cdot \exp(w_e(n+1))} \\
        s &= \frac{\exp(w_e(n+1+x_h))}{\exp(w_e(n+1))} -  \frac{\exp(w_e(n+1-x_h))}{\exp(w_e(n+1))} \\
        s &= \exp(w_e(n+1+x_h-n-1)) -  \exp(w_e(n+1-x_h-n-1)) \\
        s &= \exp(x_h\cdot w_e) -  \exp(-x_h\cdot w_e)

    Note that we don't actually use the value returned by
    ``calc_windows_central_eccentricity`` for :math:`e_c`; we simplify
    it away in the calculation above.

    If ``transition_x`` is set, things are different. Following similar
    logic above, we need to find the full-width half max, :math:`W`, and
    the central eccentricity, :math:`e_c`, then take their ratio. Let
    :math:`x_0` be ``transition_x`` and :math:`plog` be our
    piecewise-log function (and :math:`plog^{-1}` be its inverse), then
    to find :math:`W`:

    .. math::

        \pm x_h = \frac{plog(e_h)-w_e n}{w_e} \\
        plog(e_h)= w_e (n\pm x_h) \\
        e_h = plog^{-1}(w_e(n\pm x_h) \\
        W = plog^{-1}(w_e(n+x_h)) - plog^{-1}(w_e(n-x_h))

    and for :math:`e_c`:

    .. math::

        0=\frac{plog(e_c)-w_e n}{w_e} \\
        e_c = plog^{-1}(w_e n)

    And finally, scaling :math:`s`:

    .. math::

        s = \frac{plog^{-1}(w_e(n+x_h)) - plog^{-1}(w_e(n-x_h))}{plog^{-1}(w_e n)} \\

    in order to go any further, need to replace :math:`plog^{-1}(x)`
    with its full expression. Note that In order for this to work, we
    need the exponential section of the piecewise function, not the
    linaer, so we require all of:

    .. math::

        w_e(n+x_h) >  x_0  \\
        w_e n > x_0 \\
        w_e(n-x_h) > x_0

    With a bit of rearranging, we see that if we're in the region
    :math:`n>\frac{x_0}{w_e}+x_h`, then we're good. This is equivalent
    to being one of the log windows (not the linear foveal windows), and
    this makes sense because scaling cannot be constant in the region
    with linear windows, because the windows aren't changing in size,
    and thus the ratio between their size and their eccentricity cannot
    be constant. So we're only considering the region above, and thus
    :math:`plog^{-1}(y,x_0)=\exp(y-x_0)+x_0-1`.

    Substituting this in, we get:

    .. math::

        s = \frac{\exp(w_e n+w_e x_h-x_0) +x_0 - 1- \exp(w_e n-w_e x_h-x_0)-x_0+1}{\exp(w_e n - x_0)+x_0 -1} \\
        s(\exp(w_e n - x_0)+x_0 -1) = \exp(w_e n+w_e x_h-x_0)- \exp(w_e n-w_e x_h-x_0) \\
        s(1 +\frac{x_0 -1}{\exp(w_e n - x_0)}) = \frac{\exp(w_e n+w_e x_h-x_0)}{\exp(w_e n - x_0)}- \frac{\exp(w_e n-w_e x_h-x_0)}{\exp(w_e n - x_0)} \\
        s(1 +\frac{x_0 -1}{\exp(w_e n - x_0)}) = \exp(w_e n+w_e x_h-x_0 - w_e n + x_0) - \exp(w_e n-w_e x_h-x_0-w_e n + x_0) \\
        s(1 +\frac{x_0 -1}{\exp(w_e n - x_0)}) = \exp(w_e x_h) - \exp(-w_e x_h)

    And we see we have a problem: scaling is dependent upon
    :math:`n`. It will eventually converge to some value (as
    :math:`n\rightarrow \infty`, the left side will converge to
    :math:`s`), but this depends on :math:`w_e` and will take longer and
    longer as :math:`w_e` shrinks.

    However, we can see the solution: the term that we want to go to 0
    has a numerator of :math:`x_0-1`. Therefore, if :math:`x_0=1`, then
    this second part will be 0 and scaling :math:`s` will be constant
    for all eccentricities (*beyond* the linear ones). So that's what
    we'll do!

    And so scaling is:

    .. math::

        s=\exp(w_e x_h) - \exp(-w_e x_h)

    which is the same as for our regular windows! So that's nice.

    """
    if std_dev is not None:
        x_half_max = std_dev * np.sqrt(2 * np.log(2))
    else:
        x_half_max = .5
    window_spacing = calc_eccentricity_window_spacing(min_ecc, max_ecc, n_windows,
                                                      transition_x=transition_x)
    if transition_x is not None and transition_x != 1:
        raise Exception(f"transition_x must be 1, but got value {transition_x}! If transitoin_x "
                        "not 1, scaling will not be constant across the image. See docstring "
                        "Notes section for more details")
    return np.exp(x_half_max*window_spacing) - np.exp(-x_half_max*window_spacing)


def calc_windows_eccentricity(ecc_type, n_windows, window_spacing, min_ecc=.5,
                              transition_region_width=.5, std_dev=None, transition_x=None):
    r"""calculate a relevant eccentricity for each radial window

    These are the values :math:`e_c`, as referred to in ``calc_scaling``
    (for each of the n windows)

    Parameters
    ----------
    ecc_type : {'min', 'central', 'max'}
        Which eccentricity you want to calculate: the minimum one where
        x=-(1+t)/2, the central one where x=0, or the maximum one where
        x=(1+t)/2
    n_windows : `float`
        The number of log-eccentricity windows we create. n_windows can
        be a non-integer, in which case we round it up (thus one of our
        central eccentricities might be above the maximum eccentricity
        for the windows actually created)
    window_spacing : `float`
        The spacing of the log-eccentricity windows.
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. Must lie between 0 and 1.
    std_dev : float or None, optional
        The standard deviation fo the Gaussian window. If this is set,
        we compute the eccentricities for the Gaussian windows instead of
        for the cosine ones.
    transition_x : float or None, optional
        If set, the point at which the eccentricity transitions from
        linear to log, see ``log_eccentricity_windows()`` for more
        details. If set, ``min_ecc`` must be None. If None, ``min_ecc``
        must be set. If set, we assume ``n_windows`` is the *total*
        number of windows (including both the foveal/linear windows and
        the log windows) and we only support gaussian windows, so
        ``std_dev`` must be set

    Returns
    -------
    eccentricity : np.array
        A list of length ``n_windows``, containing the minimum, central,
        or maximum eccentricities of each window.

    Notes
    -----
    For the raised-cosine windows, to find 'min', we solve for the
    eccentricity where :math:`x=\frac{-(1+t)}{2}` in equation 9:

    .. math::

        \frac{-(1+t)}{2} &= \frac{\log(e_{min}) -(log(e_0)+w_e(n+1))}{w_e} \\
        e_{min} &= \exp{\frac{-w_e(1+t)}{2} + \log{e_0} + w_e(n+1)}

    To find 'max', we solve for the eccentricity where
    :math:`x=\frac{(1+t)}{2}` in equation 9:

    .. math::

        \frac{(1+t)}{2} &= \frac{\log(e_{max}) -(\log(e_0)+w_e(n+1))}{w_e} \\
        e_{max} &= \exp{\frac{w_e(1+t)}{2} + \log(e_0) + w_e(n+1)}

    For either raised-cosine or gaussian windows, to find 'central', we
    solve for the eccentricity where :math:`x=0` in equation 9:

    .. math::

        0 &= \frac{\log(e_c) -(log(e_0)+w_e(n+1))}{w_e} \\
        e_c &= e_0 \cdot \exp(w_e(n+1))

    For the gaussian windows, we say min and max are at :math:`x=\pm 3
    \sigma`, respectively:

    .. math::

        3 \sigma &= \frac{\log(e_{max}) - (\log(e_0) + w_e(n+1))}{w_e}
        e_{max} &= \exp{3\sigma w_e + \log(e_0) + w_e(n+1)}

    And, similarly:

    .. math::

        -3 \sigma &= \frac{\log(e_{min}) - (\log(e_0) + w_e(n+1))}{w_e}
        e_{min} &= \exp{-3\sigma w_e + \log(e_0) + w_e(n+1)}

    If ``transition_x`` is set, things are different. See the docstring
    of ``log_eccentricity_windows`` for an explanation, but the input to
    the window function has now changed to

    .. math::

       \frac{plog(e) - w_e n}{w_e}

    where :math:`plog` is our ``piecewise_log`` function and everything
    else is as above. we then step through the above steps to
    re-calculate the central, min, and max eccentricities, defining them
    as above for the gaussians.

    central:

    .. math::

       0 &= \frac{plog(e_c)-w_e n}{w_e}
       e_c &= plog^{-1}(w_e n)

    where :math:`plog^{-1}` is the inverse of the ``piecewise_log``
    function, which we've implemented as the ``piecewise_log_inverse``
    function

    max:

    .. math::

       3 \sigma &= \frac{plog(e_{max}) - w_e n}{w_e}
       e_{max} &= \plog^{-1}{3\sigma w_e + w_e n}

    max:

    .. math::

       -3 \sigma &= \frac{plog(e_{min}) - w_e n}{w_e}
       e_{max} &= \plog^{-1}{-3\sigma w_e + w_e n}

    """
    if ecc_type not in ['min', 'max', 'central']:
        raise Exception(f"Don't know how to handle ecc_type {ecc_type}")
    if transition_x is None:
        if ecc_type == 'central':
            ecc = [min_ecc * np.exp(window_spacing * (i+1)) for i in np.arange(np.ceil(n_windows))]
        elif ecc_type == 'min':
            if std_dev is None:
                ecc = [(np.exp(-window_spacing*(1+transition_region_width)) * min_ecc *
                        np.exp(window_spacing * (i+1))) for i in np.arange(np.ceil(n_windows))]
            else:
                ecc = [(np.exp(-3*std_dev*window_spacing) * min_ecc *
                        np.exp(window_spacing * (i+1))) for i in np.arange(np.ceil(n_windows))]
        elif ecc_type == 'max':
            if std_dev is None:
                ecc = [(np.exp(window_spacing*(1+transition_region_width)) * min_ecc *
                        np.exp(window_spacing * (i+1))) for i in np.arange(np.ceil(n_windows))]
            else:
                ecc = [(np.exp(3*std_dev*window_spacing) * min_ecc *
                        np.exp(window_spacing * (i+1))) for i in np.arange(np.ceil(n_windows))]
    else:
        if ecc_type != 'central' and std_dev is None:
            # std_dev doesn't matter for ecc_type = central, so it might
            # not be set
            raise Exception("Only gaussian windows are supported with transition_x!")
        if ecc_type == 'central':
            ecc = [piecewise_log_inverse(window_spacing * i) for i in
                   np.arange(np.ceil(n_windows))]
        elif ecc_type == 'min':
            ecc = [piecewise_log_inverse(window_spacing * (i - 3 * std_dev)) for i in
                   np.arange(np.ceil(n_windows))]
        elif ecc_type == 'max':
            ecc = [piecewise_log_inverse(window_spacing * (i + 3 * std_dev)) for i in
                   np.arange(np.ceil(n_windows))]
    return np.array(ecc)


def calc_window_widths_actual(angular_window_spacing, radial_window_spacing, min_ecc=.5,
                              max_ecc=15, window_type='cosine', transition_region_width=.5,
                              std_dev=None, transition_x=None, surround_std_dev=None):
    r"""calculate and return the actual widths of the windows, in angular and radial directions

    whereas ``calc_angular_window_spacing`` returns a term used in the
    equations to generate the windows, this returns the actual angular
    and radial widths of each set of windows (in degrees).

    We return four total widths, two by two for radial and angular by
    'top' and 'full'. By 'top', we mean the width of the flat-top region
    of each window (where the windows value is 1), and by 'full', we
    mean the width of the entire window

    Parameters
    ----------
    angular_window_spacing : float
        The width of the windows in the angular direction, as returned
        by ``calc_angular_window_spacing``
    radial_window_spacing : float
        The width of the windows in the radial direction, as returned by
        ``calc_eccentricity_window_spacing``
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    window_type : {'cosine', 'gaussian'}
        Whether to use the raised cosine function from [1]_ or a
        Gaussian that has approximately the same structure. If cosine,
        ``transition_region_width`` must be set; if gaussian, then
        ``std_dev`` must be set
    transition_region_width : `float` or None, optional
        The width of the cosine windows' transition region, parameter
        :math:`t` in equation 9 from the online methods.
    std_dev : float or None, optional
        The standard deviation of the Gaussian window.
    transition_x : float or None, optional
        If set, the point at which the eccentricity transitions from
        linear to log. If set, ``min_ecc`` must be None. If None,
        ``min_ecc`` must be set. Only supported for gaussian windows

    Returns
    -------
    radial_top_width : np.array
        The width of the flat-top region of the windows in the radial
        direction (each value corresponds to a different ring of
        windows, from the fovea to the periphery).
    radial_full_width : np.array
        The full width of the windows in the radial direction (each
        value corresponds to a different ring of windows, from the fovea
        to the periphery).
    angular_top_width : np.array
        The width of the flat-top region of the windows in the angular
        direction (each value corresponds to a different ring of
        windows, from the fovea to the periphery).
    angular_full_width : np.array
        The full width of the windows in the angular direction (each
        value corresponds to a different ring of windows, from the fovea
        to the periphery).

    Notes
    -----
    For raised-cosine windows:

    In order to calculate the width in the angular direction, we start
    with the angular window width (:math:`w_{\theta }`). The 'top' width
    is then :math:`w_{\theta}(1-t)` and the 'full' width is
    :math:`w_{\theta}(1+t)`, where :math:`t` is the
    ``transition_region_width``. This gives us the width in radians, so
    we convert it to degrees by finding the windows' central
    eccentricity (:math:`e_c`, as referred to in ``calc_scaling`` and
    returned by ``calc_windows_central_eccentricity``), and find the
    circumference (in degrees) of the circle that goes through that
    eccentricity. We then multiply our width in radians by
    :math:`\frac{2\pi e_c}{2\pi}=e_c`.

    Calculating the width in the radial direction is slightly more
    complicated, because they're not symmetric or constant across the
    visual field. We start by noting, based on equation 9 in the paper,
    that the flat-top region is the region between :math:`x=\frac{\pm
    (1-t)}{2}` and the whole window is the region between
    :math:`x=\frac{\pm (1+t)}{2}`. We can then do a little bit of
    rearranging to forms used in this function.

    For gaussian windows:

    The 'top' width in either direction is 0, because a gaussian has no
    flat top region.

    We consider the 'full' width to be 3 standard deviations out. That
    means that a given window's full extent goes from :math:`x=-3\sigma`
    to :math:`x=3\sigma`, where :math:`\sigma` is ``std_dev``, the
    window's standard deviation, and :math:`x` is the input to the
    ``gaussian`` function (analogous to the ``mother_window`` function).

    In the angular direction, for window :math:`n`,
    :math:`x=\frac{\theta-w_\theta n}{w_\theta}` (see equation 10, and
    we ignore the part of that equation that includes :math:`t`, because
    the gaussian windows have no transition region width). Rearranging,
    we see that the extent of the window in radians is thus :math:`\pm
    3\sigma w_\theta`, so the full width is then :math:`6\sigma w_\theta
    e_c`, where :math:`e_c` is the window's central eccentricity and
    necessary to convert it to degreess.

    We can follow similar logic for the radial direction, knowing that
    we want to find the difference between math:`\exp(\pm 3\sigma w_e +
    \log e_0 + w_e(n+1))` and rearranging to the forms used in this
    function.

    If ``transition_x`` is set, things are different. See the docstring
    of ``log_eccentricity_windows`` for an explanation, but the input to
    the eccentricity window function has now changed to

    .. math::

       \frac{plog(e) - w_e n}{w_e}

    where :math:`plog` is our ``piecewise_log`` function and everything
    else is as above. This only affects the eccentricity windows and
    from the above its easy to compute what their values at +/- 3 std
    devs would be and then take the difference (making use of the
    ``piecewise_log_inverse`` function we've implemented, see docstring
    of ``calc_windows_eccentricity`` for more details)

    """
    if surround_std_dev is None:
        n_radial_windows = np.ceil(calc_eccentricity_n_windows(radial_window_spacing, min_ecc,
                                                               max_ecc, std_dev, transition_x))
    else:
        # surround_std_dev increases the number of windows, but that's it
        n_radial_windows = np.ceil(calc_eccentricity_n_windows(radial_window_spacing, min_ecc,
                                                               max_ecc, surround_std_dev,
                                                               transition_x))
    if transition_x is not None:
        # because there's a window centered at zero that doesn't get
        # counted in the function above
        n_radial_windows += 1
    window_central_eccentricities = calc_windows_eccentricity('central', n_radial_windows,
                                                              radial_window_spacing, min_ecc,
                                                              transition_x=transition_x)
    if window_type == 'cosine':
        radial_top = [min_ecc*(np.exp((radial_window_spacing*(3+2*n-transition_region_width))/2) -
                               np.exp((radial_window_spacing*(1+2*n+transition_region_width))/2))
                      for n in np.arange(n_radial_windows)]
        radial_full = [min_ecc*(np.exp((radial_window_spacing*(3+2*n+transition_region_width))/2) -
                                np.exp((radial_window_spacing*(1+2*n-transition_region_width))/2))
                       for n in np.arange(n_radial_windows)]
        angular_top = [angular_window_spacing * (1-transition_region_width) * e_c for e_c in
                       window_central_eccentricities]
        angular_full = [angular_window_spacing * (1+transition_region_width) * e_c for e_c in
                        window_central_eccentricities]
    elif window_type == 'gaussian':
        # gaussian windows have no flat top region, so this is always 0
        radial_top = [0 for i in np.arange(n_radial_windows)]
        angular_top = [0 for i in np.arange(n_radial_windows)]
        if min_ecc is not None:
            if transition_x is not None:
                raise Exception("either min_ecc or transition_x should be set, not both")
            radial_full = [min_ecc*(np.exp(radial_window_spacing*(3*std_dev+n+1)) -
                                    np.exp(radial_window_spacing*(-3*std_dev+n+1)))
                           for n in np.arange(n_radial_windows)]
        if transition_x is not None:
            radial_full = [(piecewise_log_inverse(radial_window_spacing*(3*std_dev+n)) -
                            piecewise_log_inverse(radial_window_spacing*(-3*std_dev+n)))
                           for n in np.arange(n_radial_windows)]
        angular_full = [6 * std_dev * angular_window_spacing * e_c for e_c in
                        window_central_eccentricities]
    return (np.array(radial_top), np.array(radial_full), np.array(angular_top),
            np.array(angular_full))


def calc_deg_to_pix(img_res, max_eccentricity=15):
    r"""Calculate the degree-to-pixel conversion factor

    We assume ``img_res`` is the full resolution of the image and
    ``max_eccentricity`` is the radius of the image in degrees. Thus, we
    divide half of ``img_res`` by ``max_eccentricity``. However, we want
    to be able to handle non-square images, so we assume the value you
    want to use is the max of the two numbers in ``img_res`` (this is
    the way we construct the PoolingWindow objects; we want the windows
    to fill the full image).

    Parameters
    ----------
    img_res : tuple
        The resolution of our image (should therefore contains
        integers).
    max_eccentricity : float, optional
        The eccentricity (in degrees) of the edge of the image

    Returns
    -------
    deg_to_pix : float
        The factor to convert degrees to pixels (in
        pixels/degree). E.g., multiply the eccentricity (in degrees) by
        deg_to_pix to get it in pixels

    """
    return (np.max(img_res) / 2) / max_eccentricity


def calc_min_eccentricity(scaling, img_res, max_eccentricity=15, pixel_area_thresh=1,
                          radial_to_circumferential_ratio=2):
    r"""Calculate the eccentricity where window area exceeds a threshold

    The pooling windows are used primarily for metamer synthesis, and
    conceptually, if the pooling windows only include a single pixel, or
    smaller, then the only metamer for that window is exactly that
    pixel. Therefore, we don't need to compute these tiny windows, and
    we don't want to for computational reasons (it will make everything
    take much longer).

    Since ``scaling`` sets the size of the windows at eccentricity, by
    giving the slope between the diameter (in the radial direction) at
    half-max and the eccentricity, we can use it to determine the area
    of the window in each direction. What we calculate here is only an
    approximation and slightly arbitrary. Let :math:`s` be the scaling
    value, then we approximate the area :math:`(s \cdot e \cdot
    \frac{r_{pix}}{r_{deg}})^2 \cdot \frac{\pi}{4} \cdot \frac{1}{r}`,
    where :math:`r` is the ratio between the radial and circumferential
    widths, i.e. the ``radial_to_circumferential_ratio``
    arg. :math:`s\cdot e` is the diameter at half-max in degrees,
    multiplying it by :math:`\frac{r_{pix}}{r_{deg}}` (the radius of the
    image in pixels and degrees, respectively) converts it to pixels,
    and that and multiplying by :math:`\frac{pi}{4}` gives the area of a
    circle with that diameter; multiplying it by :math:`\frac{1}{r}`
    then converts this to a regular oval with this aspect ratio. This is
    a lower-bound on the area of our windows, which are actually
    elongated ovals with a larger radius than this, and thus a bit more
    complicated to compute.

    Note that, since we're using the scaling to figure this out, we're
    computing the area at approximately the windows' full-max
    half-width, and this is what we're doing for both gaussian and
    raised-cosine windows (though gaussian windows technically extend
    further beyond the FWHM than raised-cosine windows, the difference
    is not large for small windows).

    This computation works for the piecewise-log way of handling the
    fovea as well (since scaling is the same, assuming
    ``transition_x=1`` and you're in the region of log windows), though
    how you use the returned value will change.

    Parameters
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling
        windows. Other pooling windows parameters
        (``radial_to_circumferential_ratio``,
        ``transition_region_width``) cannot be set here. If that ends up
        being of interest, will change that.
    img_res : tuple
        The resolution of our image (should therefore contains
        integers).
    max_eccentricity : float, optional
        The eccentricity (in degrees) of the edge of the image
    pixel_area_thresh : float, optional
        What area (in square pixels) to check against our approximate
        pooling window area. This is slightly arbitrary, but should be
        consistent
    radial_to_circumferential_ratio : `float`, optional
        ``scaling`` determines the number of log-eccentricity windows we
        can create; this ratio gives us the number of polar angle
        ones. Based on `scaling`, we calculate the width of the windows
        in log-eccentricity, and then divide that by this number to get
        their width in polar angle. Because we require an integer number
        of polar angle windows, we round the resulting number of polar
        angle windows to the nearest integer, so the ratio in the
        generated windows approximate this. 2 (the default) is the value
        used in the paper [1]_.

    Returns
    -------
    min_ecc_deg : float
        The eccentricity (in degrees) where window area will definitely
        exceed ``pixel_area_thresh``
    min_ecc_pix : float
        The eccentricity (in pixels) where window area will definitely
        exceed ``pixel_area_thresh``

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    deg_to_pix = calc_deg_to_pix(img_res, max_eccentricity)
    # see docstring for why we use this formula, but we're computing the
    # coefficients of a quadratic equation as a function of eccentricity
    # and use np.roots to find its roots
    quad_coeff = (scaling * deg_to_pix) ** 2 * (np.pi/4) / radial_to_circumferential_ratio
    # we only want the positive root
    min_ecc_deg = np.max(np.roots([quad_coeff, 0, -pixel_area_thresh]))
    return min_ecc_deg, min_ecc_deg * deg_to_pix


def calc_dog_normalization_factor(center_surround_ratio=0.53):
    r"""calculate factor to properly normalize difference-of-gaussian windows

    Following [2]_, our difference of gaussian windows are: :math:`w_c
    g_c - (1-w_c) g_s`, where :math:`g_c` is the center gaussian,
    :math:`g_s` is the surround gaussian, and :math:`w_c` is the
    ``center_surround_ratio`` parameter.

    We still want our windows to be normalized such that they sum to 1
    everywhere in the image across all windows. We've normalized the
    center and surround gaussians individually to do this (see the
    ``gaussian`` function), and so now we just need to quickly correct
    their sum. To use, you can divide the windows by this factor either
    before or after taking the difference between the center and
    surround gaussians.

    Parameters
    ----------
    center_surround_ratio : float, optional
        ratio giving the relative weights of the center and surround
        gaussians. default is the value from [2]_

    Returns
    -------
    norm_factor : float
        the amount to divide each component of the DoG windows by so
        they properly sum to 1

    Notes
    -----

    This is simply :math:`w_c - (1 - w_c)`. To see why, let's step
    through the following. When we sum across all gaussians windows at a
    given location, we're doing:

    ..math::

        S &= D(0) + 2 \sum_{n=1}^\inf D(n)
        S &= [w_c g_c - (1-w_c)g_s](0) + 2 \sum_{n=1}^\inf [w_c g_c - (1-w_c)g_s](n)
        S &= w_c g_c(0) + 2 \sum_{n=1}^\inf w_c g_c(n) - [(1-w_c)g_s(0) + 2 \sum_{n=1}^\inf (1-w_c)g_s(n)]
        S &= w_c [g_c(0) + 2 \sum_{n=1}^\inf g_c(n)] - (1-w_c)[g_s(0) + 2 \sum_{n=1}^\inf g_s(n)]
        S &= w_c * 1 - (1-w_c) * 1

    where :math:`D(x)` is the difference of Gaussians evaluated at
    :math:`x`, and all others are as above; see Notes in the docstring
    of ``gaussian`` for a bit more explanation of this logic

    because we normalized each of our gaussians, and thus we know that
    each of those sums will be 1.

    References
    ----------
    .. [2] Bradley, C., Abrams, J., & Geisler, W. S. (2014). Retina-v1
       model of detectability across the visual field. Journal of
       Vision, 14(12), 22–22. http://dx.doi.org/10.1167/14.12.22

    """
    return center_surround_ratio - (1 - center_surround_ratio)


def gaussian(x, std_dev=1):
    r"""Simple gaussian with mean 0, and adjustable std dev

    Possible alternative mother window, giving the weighting in each
    direction for the spatial pooling performed during the construction
    of visual metamers

    Parameters
    ----------
    x : float or array_like
        The distance in a direction
    std_dev : float or None, optional
        The standard deviation of the Gaussian window.

    Returns
    -------
    array
        The value of the window at each value of `x`

    Notes
    -----
    We normalize in here in order to make sure that the windows sum to
    1. In order to do that, we note that each Gaussian is centered at
    integer x values: 0, 1, 2, 3, etc. If we're summing at ``x=0``, we
    then note that the first window will be centered there and so have
    its max amplitude, its two nearest neighbors will be 1 away from
    their center (these Gaussians are symmetric), their two nearest
    neighbors will be 2 away from their center, etc. Therefore, we'll
    have one Gaussian at max value (1), two at
    :math:`\exp(\frac{-1^2}{2\sigma^2})`, two at
    :math:`\exp(\frac{-2^2}{2\sigma^2})`, etc.

    Summing at this location will give us the value we need to normalize
    by, :math:`S`. We work through this with :math:`\sigma=1`:

    ..math::

        S &= 1 + 2 * \exp(\frac{-(1)^2}{2\sigma^2}) + 2 * \exp(\frac{-(2)^2}{2\sigma^2}) + ...
        S &= 1 + 2 * \sum_{n=1}^{\inf} \exp({-n^2}{2})
        S &= -1 + 2 * \sum_{n=0}^{\inf} \exp({-n^2}{2})

    And we've stored this number as the constant ``GAUSSIAN_SUM`` (the
    infinite sum computed in the equation above was using Wolfram Alpha,
    https://www.wolframalpha.com/input/?i=sum+0+to+inf+e%5E%28-n%5E2%2F2%29+)

    When ``std_dev>1``, the windows overlap more. As with the
    probability density function of a normal distribution, we divide by
    ``std_dev`` to keep the integral constant for different values of
    ``std_dev`` (though the integral is not 1). This means that summing
    across multiple windows will still give us a value of 1.

    """
    return torch.exp(-(x**2 / (2 * std_dev**2))) / (std_dev * GAUSSIAN_SUM)


def mother_window(x, transition_region_width=.5):
    r"""Raised cosine 'mother' window function

    Used to give the weighting in each direction for the spatial pooling
    performed during the construction of visual metamers

    Notes
    -----
    For ``x`` values outside the function's domain, we return 0

    Equation 9 from the online methods of [1]_.

    Parameters
    ----------
    x : `float` or `array_like`
        The distance in a direction
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. Must lie between 0 and 1.

    Returns
    -------
    array
        The value of the window at each value of ``x``.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature
       Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    if transition_region_width > 1 or transition_region_width < 0:
        raise Exception("transition_region_width must lie between 0 and 1!")
    # doing it in this array-ized fashion is much faster
    y = torch.zeros_like(x)
    # this creates a bunch of masks
    masks = [(-(1 + transition_region_width) / 2 < x) & (x <= (transition_region_width - 1) / 2),
             ((transition_region_width - 1) / 2 < x) & (x <= (1 - transition_region_width) / 2),
             ((1 - transition_region_width) / 2 < x) & (x <= (1 + transition_region_width) / 2)]
    # and this creates the values where those masks are
    vals = [torch.cos(np.pi/2 * ((x - (transition_region_width-1)/2) / transition_region_width))**2,
            torch.ones_like(x),
            (-torch.cos(np.pi/2 * ((x - (1+transition_region_width)/2) /
                                   transition_region_width))**2 + 1)]
    for m, v in zip(masks, vals):
        y[m] = v[m]
    return y


def polar_angle_windows(n_windows, resolution, window_type='cosine', transition_region_width=.5,
                        std_dev=None, device=None):
    r"""Create polar angle windows

    We require an integer number of windows placed between 0 and 2 pi.

    Notes
    -----
    Equation 10 from the online methods of [1]_.

    Parameters
    ----------
    n_windows : `int`
        The number of polar angle windows we create.
    resolution : int or tuple
        2-tuple of ints specifying the resolution of the 2d images to
        make. If an int, will only make the windows in 1d (this is
        mainly for testing purposes)
    window_type : {'cosine', 'gaussian'}
        Whether to use the raised cosine function from [1]_ or a
        Gaussian that has approximately the same structure. If cosine,
        ``transition_region_width`` must be set; if gaussian, then
        ``std_dev`` must be set
    transition_region_width : `float` or None, optional
        The width of the cosine windows' transition region, parameter
        :math:`t` in equation 9 from the online methods.
    std_dev : float or None, optional
        The standard deviation of the Gaussian window.
    device : str or torch.device
        the device to create this tensor on

    Returns
    -------
    windows : torch.tensor
        A 3d tensor containing the (2d) polar angle windows. Windows
        will be indexed along the first dimension. If resolution was an
        int, then this will be a 2d arra containing the 1d polar angle
        windows

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    if int(n_windows) != n_windows:
        raise Exception("n_windows must be an integer!")
    if n_windows == 1:
        raise Exception("We cannot handle one window correctly!")
    # this is `w_\theta` in the paper
    window_spacing = calc_angular_window_spacing(n_windows)
    max_angle = 2*np.pi - window_spacing
    if window_type == 'gaussian' and (std_dev * 8) > n_windows:
        raise Exception(f"In order for windows to tile the circle correctly, n_windows ({n_windows}"
                        f") must be greater than 8*std_dev ({8*std_dev})!")
    if hasattr(resolution, '__iter__') and len(resolution) == 2:
        theta = polar_angle(resolution, device=device).unsqueeze(0)
        theta = theta + (np.pi - torch.linspace(0, max_angle, n_windows, device=device).unsqueeze(-1).unsqueeze(-1))
    else:
        theta = torch.linspace(0, 2 * np.pi, resolution, device=device).unsqueeze(0)
        theta = theta + (np.pi - torch.linspace(0, max_angle, n_windows, device=device).unsqueeze(-1))
    theta = ((theta % (2 * np.pi)) - np.pi) / window_spacing
    if window_type == 'gaussian':
        windows = gaussian(theta, std_dev)
    elif window_type == 'cosine':
        windows = mother_window(theta, transition_region_width)
    return torch.stack([w for w in windows if (w != 0).any()])


def log_eccentricity_windows(resolution, n_windows=None, window_spacing=None, min_ecc=.5,
                             max_ecc=15, window_type='cosine', transition_region_width=.5,
                             std_dev=None, transition_x=None, device=None, linear=False):
    r"""Create log eccentricity windows in 2d

    Note that exactly one of ``n_windows`` or ``window_width`` must be
    set.

    In order to convert the polar radius array we create from pixels to
    degrees, we assume that ``max_ecc`` is the maximum eccentricity in
    the whichever is the larger dimension (i.e., to convert from pixels
    to degrees, we multiply by ``max_ecc / (max(resolution)/2)``)

    NOTE: if ``n_windows`` (rater than ``window_width``) is set, this is
    not necessarily the number of arrays we'll return. In order to get
    the full set of windows, we want to consider those that would show
    up in the corners as well, so it's probable that this function
    returns one more window there; we determine if this is necessary by
    calling ``calc_eccentricity_n_windows`` with
    ``np.sqrt(2)*max_ecc``.

    Exactly one of ``transition_x`` or ``min_ecc`` must be set, which
    determines how we handle the fovea. If ``min_ecc`` is set, we handle
    it like in [1]_: we log-transform all eccentricities values, with
    ``min_ecc`` determining where positive transformed values, such that
    the windows will sum to 1 everywhere except for a region with radius
    of approximately ``min_ecc`` (see equation 11). If ``transition_x``
    is set, we use our ``piecewise_log`` function to transform the
    eccentrity, which gives us a linear region at the fovea and a log
    region beyond that (with ``transition_x`` giving the value at which
    they transition); the windows therefore sum to 1 everywhere in the
    image and a mask must be applied later to mask out the fovea (this
    is handled by the ``PoolingWindows`` class automatically). See Notes
    section for some more details

    Notes
    -----
    Equation 11 from the online methods of [1]_.

    Parameters
    ----------
    resolution : int or tuple
        2-tuple of ints specifying the resolution of the 2d images to
        make. If an int, will only make the windows in 1d (this is
        mainly for testing purposes)
    n_windows : `float` or `None`
        The number of log-eccentricity windows from ``min_ecc`` to
        ``max_ecc``. ``n_windows`` xor ``window_width`` must be set.
    window_spacing : `float` or `None`
        The spacing of the log-eccentricity windows. ``n_windows`` xor
        ``window_spacing`` must be set.
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    window_type : {'cosine', 'gaussian'}
        Whether to use the raised cosine function from [1]_ or a
        Gaussian that has approximately the same structure. If cosine,
        ``transition_region_width`` must be set; if gaussian, then
        ``std_dev`` must be set
    transition_region_width : `float` or None, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    std_dev : float or None, optional
        The standard deviation of the Gaussian window. WARNING -- For
        now, we only support ``std_dev=1`` (in order to ensure that the
        windows tile correctly, intersect at the proper point, follow
        scaling, and have proper aspect ratio; not sure we can make that
        happen for other values).
    transition_x : float or None, optional
        If set, the point at which the eccentricity transitions from
        linear to log. If set, ``min_ecc`` must be None. If None,
        ``min_ecc`` must be set
    device : str or torch.device
        the device to create this tensor on
    linear : bool, optional
        if True, create linear windows instead of log-spaced (only if
        ``transition_x=None``). NOTE This is only for playing around
        with, it really is not supported or a good idea because the
        angular windows still grow in size as a function of eccentricity
        and none of the calculations will work.

    Returns
    -------
    windows : torch.tensor
        A 3d tensor containing the (2d) log-eccentricity
        windows. Windows will be indexed along the first dimension. If
        resolution was an int, then this will be a 2d array containing
        the 1d polar angle windows
    multiplier : torch.tensor
        same number of dimensions as windows, with either a 1 or -1 for
        each window (indexed along first dimension). This was used to
        determine whether that window saw positive or negative
        eccentricity and is used for constructing DoG windows. Should be
        ignored by user

    Notes
    -----

    If ``transition_x`` is set, these windows are slightly different
    than described in the paper. If it's not set, the input to the
    window function (either raised-cosine as described in the paper or
    gaussian) is:

    .. math::

       \frac{\log(e)-(\log(e_0)+w_e(n+1))}{w_e}

    where :math:`e` is the eccentricity, :math:`e_0` is the minimum
    eccentricity ``min_ecc``, :math:`w_e` is the window spacing, as
    either specified as an argument to this function or calculated based
    on ``n_windows, and :math:`n` is indexes the windows (and runs from
    0 to ``n_windows-1``).

    If ``transition_x`` is set, we only support gaussian windows
    (haven't checked the math yet for the raised-cosine ones), and the
    input is:

    .. math::

       \frac{plog(e) - w_e n}{w_e}

    where all variables are the same as above, but :math:`plog` is our
    ``piecewise_log`` function. There are several notable differences
    here: we no longer have :math:`e_0` (the ``transition_x`` variable
    now controls where the log windows start and ``piecewise_log`` is
    constructed to avoid the input to the function shooting off to
    negative infinity) and we don't index by :math:`n+1` but by
    :math:`n`. This means we include a window centered directly at
    :math:`e=0`, the center of the image.

    This changes many of the other calculations, including the number of
    windows, spacing, scaling, and how they all relate to each
    other. See the appropriate functions' docstrings for details.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    if transition_x is not None:
        log_func = piecewise_log
        if window_type == 'cosine':
            raise Exception("Only gaussian windows supported with transition_x! Haven't checked "
                            "the math for the cosine ones")
        if min_ecc is not None:
            raise Exception("Either transition_x or min_ecc must be None!")
        if window_spacing is None:
            window_spacing = calc_eccentricity_window_spacing(min_ecc, max_ecc, n_windows,
                                                              std_dev=std_dev,
                                                              transition_x=transition_x)
        n_windows = calc_eccentricity_n_windows(window_spacing, min_ecc, max_ecc*np.sqrt(2), std_dev,
                                                transition_x)
        # note that our shift_arg no longer includes min_ecc
        shift_arg = (window_spacing * torch.arange(1, math.ceil(n_windows)+1, device=device)).unsqueeze(-1)
        # we want our windows to uniformly tile the image. we'll have
        # some that are cut off at the literal fovea (that is, at
        # eccentricity 0), so we find those windows that are within 4
        # standard deviations of 0 and duplicate them.
        foveal_windows = shift_arg[(shift_arg / window_spacing) <= 4*std_dev].unsqueeze(-1)
        # sort this so we get all the windows close to the fovea at the
        # beginning
        shift_arg, idx = torch.sort(torch.cat([shift_arg, foveal_windows]), 0)
        # We use multiplier to make sure that these duplicates see a
        # *negative* eccentricity, so they'll represent the completion
        # of the windows.
        multiplier = torch.ones_like(shift_arg)
        # we find everywhere there's a repeat in shift_arg, which
        # corresponds to a duplicated foveal window, and replace the 1
        # with the -1 (this will put the negative first in every pair)
        multiplier[np.where(~np.diff(shift_arg, axis=0).astype(bool))] = -1
        # finally, add the window centered at 0. we do this here because
        # we don't want it duplicated in the above bit.
        shift_arg = torch.cat([torch.tensor([0], dtype=torch.float32, device=shift_arg.device).unsqueeze(-1), shift_arg])
        multiplier = torch.cat([torch.tensor([1], dtype=torch.float32, device=multiplier.device).unsqueeze(-1), multiplier])
    else:
        if not linear:
            log_func = torch.log
        else:
            log_func = lambda x: x
        if window_spacing is None:
            window_spacing = calc_eccentricity_window_spacing(min_ecc, max_ecc, n_windows,
                                                              std_dev=std_dev)
        n_windows = calc_eccentricity_n_windows(window_spacing, min_ecc, max_ecc*np.sqrt(2), std_dev)
        shift_arg = (log_func(torch.tensor(min_ecc, dtype=torch.float32)) + window_spacing * torch.arange(1, math.ceil(n_windows)+1, device=device)).unsqueeze(-1)
        multiplier = torch.ones_like(shift_arg)
    if hasattr(resolution, '__iter__') and len(resolution) == 2:
        ecc = log_func(polar_radius(resolution, device=device) / calc_deg_to_pix(resolution, max_ecc)).unsqueeze(0)
        shift_arg = shift_arg.unsqueeze(-1)
        multiplier = multiplier.unsqueeze(-1)
    else:
        ecc = log_func(torch.linspace(0, max_ecc, resolution, device=device)).unsqueeze(0)
    ecc = (multiplier*ecc - shift_arg) / window_spacing
    if window_type == 'gaussian':
        windows = gaussian(ecc, std_dev)
    elif window_type == 'cosine':
        windows = mother_window(ecc, transition_region_width)
    return torch.stack([w for w in windows if (w != 0).any()]), multiplier


def create_pooling_windows(scaling, resolution, min_eccentricity=.5, max_eccentricity=15,
                           radial_to_circumferential_ratio=2, window_type='cosine',
                           transition_region_width=.5, std_dev=None, device=None,
                           center_surround_ratio=.53, surround_std_dev=10.1,
                           transition_x=None):
    r"""Create two sets of 2d pooling windows (log-eccentricity and polar angle) that span the visual field

    This creates the pooling windows that we use to average image
    statistics for metamer generation as done in [1]_. This is returned
    as two 3d torch tensors for further use with a model.

    Note that these are returned separately as log-eccentricity and
    polar angle tensors and if you want the windows used in the paper
    [1]_, you'll need to call ``torch.einsum`` (see Examples section)
    or, better yet, use the ``PoolingWindows`` class, which is provided
    for this purpose.

    Because difference of gaussian (DoG) windows are non-polar
    separable, we return those windows as two dictionaries (with keys
    'center' and 'surround') instead of two tensors. because taking the
    difference between the windows and multiplying them by the image are
    both linear operations, you can separately apply the two gaussians
    and then take their difference (this is what the ``PoolingWindows``
    class does).

    Parameters
    ----------
    scaling : `float` or `None`.
        The ratio of the eccentricity window's radial full-width at
        half-maximum to eccentricity (see the `calc_scaling` function).
    resolution : tuple
        2-tuple of ints specifying the resolution of the 2d images to
        make.
    min_eccentricity : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_eccentricity : `float`, optional
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    radial_to_circumferential_ratio : `float`, optional
        ``scaling`` determines the number of log-eccentricity windows we
        can create; this ratio gives us the number of polar angle
        ones. Based on `scaling`, we calculate the width of the windows
        in log-eccentricity, and then divide that by this number to get
        their width in polar angle. Because we require an integer number
        of polar angle windows, we round the resulting number of polar
        angle windows to the nearest integer, so the ratio in the
        generated windows approximate this. 2 (the default) is the value
        used in the paper [1]_.
    window_type : {'cosine', 'gaussian', 'dog'}
        Whether to use the raised cosine function from [1]_, a Gaussian
        that has approximately the same structure, or a difference of
        two such gaussians (``'dog'``, as in [2]_). If cosine,
        ``transition_region_width`` must be set; if gaussian, then
        ``std_dev`` must be set; if dog, then ``std_dev``,
        ``center_surround_ratio``, and ``surround_std_dev`` must all be
        set.
    transition_region_width : `float` or None, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    std_dev : float or None, optional
        The standard deviation of the Gaussian window. WARNING -- if
        this is too small (say < 3/4), then the windows won't tile
        correctly
    device : str or torch.device
        the device to create these tensors on
    center_surround_ratio : float, optional
        ratio giving the relative weights of the center and surround
        gaussians. default is the value from [2]_ (this is parameter
        :math:`w_c` from that paper)
    surround_std_dev : float, optional
        the standard deviation of the surround Gaussian window. default
        is the value from [2]_ (assuming ``std_dev=1``, this is
        parameter :math:`k_s` from that paper).
    transition_x : float or None, optional
        If set, the point at which the eccentricity transitions from
        linear to log. If set, ``min_ecc`` must be None. If None,
        ``min_ecc`` must be set

    Returns
    -------
    angle_windows : torch.tensor or dict
        The 3d tensor of 2d polar angle windows. Its shape will be
        ``(n_angle_windows, *resolution)``, where the number of windows
        is inferred in this function based on the values of ``scaling``
        and ``radial_to_circumferential_width``. If
        ``window_type='dog'``, then we return a dictionary with two keys
        ('center' and 'surround') containing those windows instead.
    ecc_windows : torch.tensor or dict
        The 3d tensor of 2d log-eccentricity windows. Its shape will be
        ``(n_eccen_windows, *resolution)``, where the number of windows
        is inferred in this function based on the values of ``scaling``,
        ``min_ecc``, and ``max_ecc``. If ``window_type='dog'``, then we
        return a dictionary with two keys ('center' and 'surround')
        containing those windows instead; unlike angle_windows, center
        and surround will not be the same shape, because the broader a
        window is, the more windows we need to generate in order to
        ensure they uniformly tile the periphery. The ``PoolingWindows``
        class will handle this automatically, but you can also
        concatenate zeros onto the center tensor in order to make them
        the same shape (``torch.cat([ecc_windows['center'],
        torch.zeros((ecc_windows['surround'].shape[0] -
        ecc_windows['center'].shape[0],
        *ecc_windows['center'].shape[1:]))])``)

    Examples
    --------
    To use, simply call with the desired scaling (for the version seen
    in the paper, don't change any of the default arguments; compare
    this image to the right side of Supplementary Figure 1C)

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import plenoptic as po
       import pyrtools as pt
       angle_w, ecc_w = po.simul.pooling.create_pooling_windows(.87, (256, 256))
       fig = pt.imshow(ecc_w)
       fig = pt.imshow(angle_w)
       plt.show()

    If you wish to get the windows as shown in Supplementary Figure 1C
    in the paper [1]_, use ``torch.einsum`` (if you wish to apply these
    to images, use the ``PoolingWindows`` class instead, which has many
    more features):

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyrtools as pt
       import plenoptic as po
       angle_w, ecc_w = po.simul.pooling.create_pooling_windows(.87, (256, 256))
       # we ignore the last ring of eccentricity windows here because
       # they're all relatively small, which makes the following plot
       # look weird. for how to properly handle them, see the
       # PoolingWindows class
       windows = torch.einsum('ahw,ehw->eahw', [a, e[:-1]]).flatten(0, 1)
       fig, ax = plt.subplots(1, 1, figsize=(5, 5))
       for w in windows:
           ax.contour(w, [.5], colors='r')
       plt.show()

    """
    if window_type == 'dog':
        dog = True
        window_type = 'gaussian'
        warnings.warn("This will not tile correctly at the fovea, due to differences in the rate "
                      "at which the sums across the center and surround windows rise to 1. "
                      "The PoolingWindows object has a work-around for this, use that for any "
                      "application of these windows.")
    else:
        dog = False
    ecc_window_spacing = calc_eccentricity_window_spacing(min_eccentricity, max_eccentricity,
                                                          scaling=scaling, std_dev=std_dev,
                                                          transition_x=transition_x)
    n_polar_windows = calc_angular_n_windows(ecc_window_spacing / radial_to_circumferential_ratio)
    # we want to set the number of polar windows where the ratio of
    # widths is approximately what the user specified. the constraint
    # that it's an integer is more important
    n_polar_windows = int(round(n_polar_windows))
    angle_tensor = polar_angle_windows(n_polar_windows, resolution, window_type,
                                       transition_region_width=transition_region_width,
                                       std_dev=std_dev, device=device)
    if dog:
        angle_tensor = {'center': angle_tensor}
        surround = polar_angle_windows(round(n_polar_windows), resolution, window_type,
                                       transition_region_width=transition_region_width,
                                       std_dev=surround_std_dev, device=device)
        angle_tensor['surround'] = surround
    ecc_tensor, ctr_mult = log_eccentricity_windows(resolution, None, ecc_window_spacing,
                                                    min_eccentricity, max_eccentricity,
                                                    window_type, std_dev=std_dev,
                                                    transition_region_width=transition_region_width,
                                                    device=device, transition_x=transition_x)
    if dog:
        ecc_tensor = {'center': ecc_tensor}
        surround, surr_mult = log_eccentricity_windows(resolution, ecc_tensor['center'].shape[0],
                                                       ecc_window_spacing, min_eccentricity,
                                                       max_eccentricity, window_type,
                                                       std_dev=surround_std_dev,
                                                       transition_region_width=transition_region_width,
                                                       device=device, transition_x=transition_x)
        ecc_tensor['surround'] = surround
        if transition_x is not None:
            # now we need to make sure that center and surround are the same
            # size. we make use of the mult tensors for that
            idx = np.zeros(len(surr_mult)).astype(bool)
            idx[:len(ctr_mult)] = True
            # this will go through and find all the places where surr_mult
            # has a -1 that ctr_mult does not and shift the 1s forward by
            # one, leaving a 0 there. this corresponds to the reflected
            # windows that the surround has that center does not
            for i in set(np.where(surr_mult==-1)[0]).difference(set(np.where(ctr_mult==-1)[0])):
                idx[i:] = np.concatenate([[0], idx[i:-1]])
            new_ctr = torch.zeros_like(ecc_tensor['surround'])
            # now we go ahead and paste the center windows in the
            # appropriate place, leaving the zeros
            new_ctr[idx, ...] = ecc_tensor['center']
            ecc_tensor['center'] = new_ctr
            ecc_tensor = dict((k, sum_windows_output(v, surr_mult)) for k, v in ecc_tensor.items())
        else:
            # now we need to make sure that center and surround are the same
            # size. we make use of the mult tensors for that
            idx = np.zeros(len(surr_mult)).astype(bool)
            idx[:len(ctr_mult)] = True
            new_ctr = torch.zeros_like(ecc_tensor['surround'])
            # now we go ahead and paste the center windows in the
            # appropriate place, leaving the zeros
            new_ctr[idx, ...] = ecc_tensor['center']
            ecc_tensor['center'] = new_ctr
    return angle_tensor, ecc_tensor


def normalize_windows(angle_windows, ecc_windows, window_width_pixels, scale=0,
                      center_surround_ratio=None, linear=False):
    r"""normalize windows to have L1-norm of 1

    we calculate the L1-norm of single windows (that is, product of
    eccentricity and angular windows) for all angles, one middling
    eccentricity (third of the way thorugh), then average across angles
    (because of alignment with pixel grid, L1-norm will vary somewhat
    across angles).

    I think L1-norm scales linearly with area, which is proportional to
    window_width^2, so we use that to scale it for the different
    windows. only eccentricity windows is normalized (don't need to
    divide both).

    this works with either DoG-style windows or regular ones.

    Parameters
    ----------
    angle_windows : dict
        dictionary containing the angular windows
    ecc_windows : dict
        dictionary containing the eccentricity windows
    window_width_pixels : array_like
        array containing radial full widths (in pixels) of each
        window. therefore, must have an element for each eccentricity
        window
    scale : int, optional
        which scale to calculate norm for and modify
    center_surround_ratio : float or None, optional
        if windows are DoGs, then need this ratio to properly construct
        the window (weights the difference between center and surround)
    linaer : bool, optional
        if False, scale windows as described above. if True, scale all
        windows by same amount

    Returns
    -------
    ecc_windows : dict
        the normalized ecc_windows. only ``scale`` is modified

    """
    try:
        # pick some window with a middling eccentricity
        n = ecc_windows[scale].shape[0] // 5
        # get the l1 norm of a single window
        w = torch.einsum('ahw,hw->ahw', angle_windows[scale], ecc_windows[scale][n])
        l1 = torch.norm(w, 1, (-1, -2))
        l1 = l1.mean(0)
    except KeyError:
        # then these are dog windows with separate centers and
        # surrounds. pick some window with a middling eccentricity
        n = ecc_windows['center'][scale].shape[0] // 5
        ctr = torch.einsum('ahw,hw->ahw', angle_windows['center'][scale],
                           ecc_windows['center'][scale][n])
        sur = torch.einsum('ahw,hw->ahw', angle_windows['surround'][scale],
                           ecc_windows['surround'][scale][n])
        w = center_surround_ratio * ctr - (1 - center_surround_ratio) * sur
        # get the l1 norm of a single window
        l1 = torch.norm(w, 1, (-1, -2))
        l1 = l1.mean(0)
    # the l1 norm grows with eccentricity squared (because it's
    # proportional to the area of the windows)
    if not linear:
        scale_factor = (l1*(window_width_pixels / window_width_pixels[n])**2).to(torch.float32)
    else:
        scale_factor = (l1*torch.ones(len(window_width_pixels)))
    while scale_factor.ndim < 3:
        scale_factor = scale_factor.unsqueeze(-1)
    try:
        ecc_windows[scale] = ecc_windows[scale] / scale_factor
    except KeyError:
        ecc_windows['center'][scale] = ecc_windows['center'][scale] / scale_factor
        ecc_windows['surround'][scale] = ecc_windows['surround'][scale] / scale_factor
    return ecc_windows, scale_factor


class PoolingWindows(nn.Module):
    r"""Generic class to set up scaling windows for use with other models

    Note that we will calculate the minimum eccentricity at which the
    area of the windows at half-max exceeds one pixel (based on
    ``scaling``, ``img_res`` and ``max_eccentricity``) and, if
    ``min_eccentricity`` is below that, will throw an Exception.

    This just generates the pooling windows given a small number of
    parameters. One tricky thing we do is generate a set of scaling
    windows for each scale (appropriately) sized. For example, the V1
    model will have 4 scales, so for a 256 x 256 image, the coefficients
    will have shape (256, 256), (128, 128), (64, 64), and (32,
    32). Therefore, we need windows of the same size (could also
    up-sample the coefficient tensors, but since that would need to
    happen each iteration of the metamer synthesis, pre-generating
    appropriately sized windows is more efficient).

    We will calculate the minimum eccentricity at which the area of the
    windows at half-max exceeds one pixel at each scale. For scales
    beyond the first however, we will not throw an Exception if this
    value is below ``min_eccentricity``. We instead print a warning to
    alert the user and use this value as ``min_eccentricity`` when
    creating the plots. In order to see what this value was, see
    ``self.calculated_min_eccentricity_degrees``

    We can optionally cache the windows tensor we create, if
    ``cache_dir`` is not None. In that case, we'll also check to see if
    appropriate cached windows exist before creating them and load them
    if they do. The path we'll use is
    ``{cache_dir}/scaling-{scaling}_size-{img_res}_e0-{min_eccentricity}_
    em-{max_eccentricity}_w-{window_width}_{window_type}.pt``, where
    {window_width} is ``transition_region_width`` if
    ``window_type='cosine'``, and ``std_dev`` if it's
    ``'gaussian'``. We'll cache each scale separately, changing the
    img_res (and potentially min_eccentricity) values in that save path
    appropriately.

    Note that, for now, if we have ``window_type='dog'``, then all the
    various size parameters describe the center filter (not the
    surround, nor the center - surround). if it becomes useful, will
    extend this

    Exactly one of ``transition_x`` or ``min_ecc`` must be set, which
    determines how we handle the fovea. If ``min_ecc`` is set, we handle
    it like in [1]_: we log-transform all eccentricities values, with
    ``min_ecc`` determining where positive transformed values, such that
    the windows will sum to 1 everywhere except for a region with radius
    of approximately ``min_ecc`` (see equation 11). If ``transition_x``
    is set, we use our ``piecewise_log`` function to transform the
    eccentrity, which gives us a linear region at the fovea and a log
    region beyond that (with ``transition_x`` giving the value at which
    they transition); the windows therefore sum to 1 everywhere in the
    image and a mask is applied later to mask out the fovea. Currently
    ``transition_x`` is only supported for ``window_type='dog'`` and not
    supported for any others

    Parameters
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling
        windows. Other pooling windows parameters
        (``radial_to_circumferential_ratio``,
        ``transition_region_width``) cannot be set here. If that ends up
        being of interest, will change that.
    img_res : tuple
        The resolution of our image (should therefore contains
        integers). Will use this to generate appropriately sized pooling
        windows.
    min_eccentricity : float, optional
        The eccentricity at which the pooling windows start.
    max_eccentricity : float, optional
        The eccentricity at which the pooling windows end.
    num_scales : int, optional
        The number of scales to generate masks for. For the RGC model,
        this should be 1, otherwise should match the number of scales in
        the steerable pyramid.
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    window_type : {'cosine', 'gaussian', 'dog'}
        Whether to use the raised cosine function from [1]_, a Gaussian
        that has approximately the same structure, or a difference of
        two such gaussians (``'dog'``, as in [2]_). If cosine,
        ``transition_region_width`` must be set; if gaussian, then
        ``std_dev`` must be set; if dog, then ``std_dev``,
        ``center_surround_ratio``, and ``surround_std_dev`` must all be
        set.
    transition_region_width : float or None, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. 0.5 (the default) is the
        value used in the paper [1]_.
    std_dev : float or None, optional
        The standard deviation of the Gaussian window. WARNING -- For
        now, we only support ``std_dev=1`` (in order to ensure that the
        windows tile correctly, intersect at the proper point, follow
        scaling, and have proper aspect ratio; not sure we can make that
        happen for other values).
    center_surround_ratio : float, optional
        ratio giving the relative weights of the center and surround
        gaussians. default is the value from [2]_ (this is parameter
        :math:`w_c` from that paper)
    surround_std_dev : float, optional
        the standard deviation of the surround Gaussian window. default
        is the value from [2]_ (assuming ``std_dev=1``, this is
        parameter :math:`k_s` from that paper).
    transition_x : float or None, optional
        If set, the point at which the eccentricity transitions from
        linear to log. If set, ``min_ecc`` must be None. If None,
        ``min_ecc`` must be set. This is required for difference of
        Gaussian windows, and not allowed for any others


    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    img_res : tuple
        The resolution of our image in pixels.
    transition_region_width : float or None
        The width of the cosine windows' transition region, parameter
        :math:`t` in equation 9 from the online methods.
    std_dev : float or None
        The standard deviation of the Gaussian windows.
    center_surround_ratio : float or None
        ratio giving the relative weights of the center and surround
        gaussians. default is the value from [2]_ (this is parameter
        :math:`w_c` from that paper)
    surround_std_dev : float or None
        the standard deviation of the surround Gaussian window. default
        is the value from [2]_ (assuming ``std_dev=1``, this is
        parameter :math:`k_s` from that paper).
    transition_x : float or None
        If set, the point at which the eccentricity transitions from
        linear to log. If set, ``min_ecc`` must be None. If None,
        ``min_ecc`` must be set. This is required for difference of
        Gaussian windows, and not allowed for any others
    angle_windows : dict
        A dict of 3d tensors containing the angular pooling windows in
        which the model parameters are averaged. Each key corresponds to
        a different scale and thus is a different size. If you have
        called ``parallel()``, this will be strucuted in a slightly
        different way (see that method for details)
    ecc_windows : dict
        A dict of 3d tensors containing the log-eccentricity pooling
        windows in which the model parameters are averaged. Each entry
        in the list corresponds to a different scale and thus is a
        different size. If you have called ``parallel()``, this will be
        structured in a slightly different way (see that method for
        details)
    state_dict_reduced : dict
        A dictionary containing those attributes necessary to initialize
        the model, plus a 'model_name' field which the ``load_reduced``
        method uses to determine which model constructor to call. This
        is used for saving/loading the models, since we don't want to
        keep the (very large) representation and intermediate steps
        around. To save, use ``self.save_reduced(filename)``, and then
        load from that same file using the class method
        ``po.simul.VentralModel.load_reduced(filename)``
    window_width_degrees : dict
        Dictionary containing the widths of the windows in
        degrees. There are six keys, corresponding to a 2x2 for the
        widths in the radial and angular directions by the 'top',
        'half', and 'full' widths (top is the width of the flat-top
        region of each window, where the window's value is 1; full is
        the width of the entire window; half is the width at
        half-max). Each value is a list containing the widths for the
        windows in different eccentricity bands. To visualize these, see
        the ``plot_window_widths`` method.
    window_width_pixels : list
        List of dictionaries containing the widths of the windows in
        pixels; each entry in the list corresponds to the widths for a
        different scale, as in ``windows``. See above for explanation of
        the dictionaries. To visualize these, see the
        ``plot_window_widths`` method.
    n_polar_windows : int
        The number of windows we have in the polar angle dimension
        (within each eccentricity band)
    n_eccentricity_bands : int
        The number of eccentricity bands in our model
    calculated_min_eccentricity_degrees : list
        List of floats (one for each scale) that contain
        ``calc_min_eccentricity()[0]``, that is, the minimum
        eccentricity (in degrees) where the area of the window at
        half-max exceeds one pixel (based on the scaling, size of the
        image in pixels and in degrees).
    calculated_min_eccentricity_pixels : list
        List of floats (one for each scale) that contain
        ``calc_min_eccentricity()[1]``, that is, the minimum
        eccentricity (in pixels) where the area of the window at
        half-max exceeds one pixel (based on the scaling, size of the
        image in pixels and in degrees).
    central_eccentricity_degrees : np.array
        A 1d array with shape ``(self.n_eccentricity_bands,)``, each
        value gives the eccentricity of the center of each eccentricity
        band of windows (in degrees).
    central_eccentricity_pixels : list
        List of 1d arrays (one for each scale), each with shape
        ``(self.n_eccentricity_bands,)``, each value gives the
        eccentricity of the center of each eccentricity band of windows
        (in degrees).
    window_approx_area_degrees : dict
        Dictionary containing the approximate areas of the windows, in
        degrees. There are three keys: 'top', 'half', and 'full',
        corresponding to which width we used to calculate the area (top
        is the width of the flat-top region of each window, where the
        window's value is 1; full is the width of the entire window;
        half is the width at half-max). To get this approximate area, we
        multiply the radial and angular widths against each other and
        then by pi/4 to get the area of the regular ellipse that has
        those widths (our windows are elongated, so this is probably an
        under-estimate). To visualize these, see the
        ``plot_window_areas`` method
    window_approx_area_pixels : list
        List of dictionaries containing the approximate areasof the
        windows in pixels; each entry in the list corresponds to the
        areas for a different scale, as in ``windows``. See above for
        explanation of the dictionaries. To visualize these, see the
        ``plot_window_areas`` method.
    deg_to_pix : list
        List of floats containing the degree-to-pixel conversion factor
        at each scale
    cache_dir : str or None
        If str, this is the directory where we cached / looked for
        cached windows tensors
    cached_paths : list
        List of strings, one per scale, taht we either saved or loaded
        the cached windows tensors from
    num_scales : int
        Number of scales this object has windows for
    num_devices : int
        Number of devices this object is split across
    window_type : {'cosine', 'gaussian'}
        Whether to use the raised cosine function from [1]_ or a
        Gaussian that has approximately the same structure.
    window_max_amplitude : float
        The max amplitude of an individual window. This will always be 1
        for raised-cosine windows, but will depend on ``std_dev`` for
        gaussian ones (for ``std_dev=1``, the only value we support for
        now, it's approximately .16).
    window_intersecting_amplitude : float
        The amplitude at which two neighboring windows intersect. This
        will always be .5 for raised-cosine windows, but will depend on
        ``std_dev`` for gaussian ones (for ``std_dev=1``, the only value
        we support for now, it's half a standard deviation away from the
        center, approximately .14)

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5, max_eccentricity=15, num_scales=1,
                 cache_dir=None, window_type='cosine', transition_region_width=.5, std_dev=None,
                 center_surround_ratio=.53, surround_std_dev=10.1, transition_x=None):
        super().__init__()
        if len(img_res) != 2:
            raise Exception("img_res must be 2d!")
        self.scaling = scaling
        if min_eccentricity is not None:
            min_eccentricity = float(min_eccentricity)
        self.min_eccentricity = min_eccentricity
        self.max_eccentricity = float(max_eccentricity)
        self.img_res = img_res
        self.num_scales = num_scales
        self.window_type = window_type
        self.transition_x = transition_x
        if transition_x is not None and transition_x != 1:
            raise Exception("Only transition_x=1 results in constant scaling across the image")
        self.angle_windows = {}
        self.ecc_windows = {}
        if window_type == 'cosine':
            assert transition_region_width is not None, "cosine windows need transition region widths!"
            self.transition_region_width = float(transition_region_width)
            self.std_dev = None
            window_width_for_saving = self.transition_region_width
            self.surround_std_dev = None
            self.center_surround_ratio = None
            self.window_max_amplitude = 1
            self.window_intersecting_amplitude = .5
            if transition_x is not None:
                raise NotImplementedError("Currently, transition_x is only supported with DoG "
                                          "windows!")
        elif window_type == 'gaussian':
            assert std_dev is not None, "gaussian windows need standard deviations!"
            self.std_dev = float(std_dev)
            self.transition_region_width = None
            self.surround_std_dev = None
            self.center_surround_ratio = None
            if std_dev != 1:
                raise Exception("Only std_dev=1 allowed for Gaussian windows!")
            window_width_for_saving = self.std_dev
            # 1 / (std_dev * GAUSSIAN_SUM) is the max in a single
            # direction (radial or angular), so the max for a single
            # window is its square
            self.window_max_amplitude = (1 / (std_dev * GAUSSIAN_SUM)) ** 2
            self.window_intersecting_amplitude = self.window_max_amplitude * np.exp(-.25/2)
            if transition_x is not None:
                raise NotImplementedError("Currently, transition_x is only supported with DoG "
                                          "windows!")
        elif window_type == 'dog':
            if num_scales > 1:
                raise NotImplementedError("Currently only support DoG windows with single scale! "
                                          "If you want more scales, need to think about how to "
                                          "handle transition_x for higher scales -- should it "
                                          "change the same way min_eccentricity does?")
            assert std_dev is not None, "DoG windows need standard deviations!"
            assert surround_std_dev is not None, "DoG windows need surround standard deviations!"
            assert center_surround_ratio is not None, "DoG windows need center surround ratios!"
            # assert transition_x is not None, "DoG windows need transition_x!"
            # assert min_eccentricity is None, "DoG windows need to have min_eccentricity=None (use transition_x instead)"
            if std_dev != 1:
                raise Exception("DoG windows' center gaussian must have std_dev=1!")
            self.std_dev = float(std_dev)
            self.surround_std_dev = float(surround_std_dev)
            self.center_surround_ratio = float(center_surround_ratio)
            self.transition_region_width = None
            window_width_for_saving = f'{self.std_dev}_s-{self.surround_std_dev}_r-{self.center_surround_ratio}'
            # 1 / (std_dev * GAUSSIAN_SUM) is the max in a single
            # direction (radial or angular), so the max for a single
            # window is its square
            self.center_max_amplitude = ((1 / (std_dev * GAUSSIAN_SUM))) ** 2
            self.center_intersecting_amplitude = self.center_max_amplitude * np.exp(-.25/2)
            self.surround_max_amplitude = ((1 / (surround_std_dev * GAUSSIAN_SUM))) ** 2
            self.surround_intersecting_amplitude = self.surround_max_amplitude * np.exp(-.25/2)
            self.window_max_amplitude = ((center_surround_ratio * self.center_max_amplitude) -
                                         (1 - center_surround_ratio) * self.surround_max_amplitude)
            self.window_intersecting_amplitude = ((center_surround_ratio * self.center_max_amplitude) * np.exp(-.25/(2*std_dev**2)) -
                                                  ((1 - center_surround_ratio) * self.surround_max_amplitude) * np.exp(-.25/(2*surround_std_dev**2)))
            self.min_ecc_mask = {}
            # we have separate center and surround dictionaries:
            self.angle_windows = {'center': {}, 'surround': {}}
            self.ecc_windows = {'center': {}, 'surround': {}}
        self.norm_factor = {}
        self.num_devices = 1
        if cache_dir is not None:
            self.cache_dir = op.expanduser(cache_dir)
            cache_path_template = op.join(self.cache_dir, "scaling-{scaling}_size-{img_res}_"
                                          "e0-{min_eccentricity:.03f}_em-{max_eccentricity:.01f}_w"
                                          "-{window_width}_{window_type}.pt")
        else:
            self.cache_dir = cache_dir
        self.cache_paths = []
        self.calculated_min_eccentricity_degrees = []
        self.calculated_min_eccentricity_pixels = []
        self._window_sizes()
        self.state_dict_reduced = {'scaling': scaling, 'img_res': img_res,
                                   'min_eccentricity': self.min_eccentricity,
                                   'max_eccentricity': self.max_eccentricity,
                                   'transition_region_width': self.transition_region_width,
                                   'cache_dir': self.cache_dir, 'window_type': window_type,
                                   'std_dev': self.std_dev, 'transition_x': self.transition_x,
                                   'surround_std_dev': self.surround_std_dev,
                                   'center_surround_ratio': self.center_surround_ratio}
        for i in range(self.num_scales):
            scaled_img_res = [np.ceil(j / 2**i) for j in img_res]
            min_ecc, min_ecc_pix = calc_min_eccentricity(scaling, scaled_img_res, max_eccentricity)
            # TEMPORARY
            min_ecc = self.min_eccentricity
            self.calculated_min_eccentricity_degrees.append(min_ecc)
            self.calculated_min_eccentricity_pixels.append(min_ecc_pix)
            if self.min_eccentricity is not None and min_ecc > self.min_eccentricity:
                if i == 0:
                    raise Exception("Cannot create windows with scaling %s, resolution %s, and min"
                                    "_eccentricity %s, it will contain windows smaller than a "
                                    "pixel. min_eccentricity must be at least %s!" %
                                    (scaling, img_res, self.min_eccentricity, min_ecc))
                else:
                    warnings.warn("For scale %s, min_eccentricity set to %.2f in order to avoid "
                                  "windows smaller than 1 pixel in area" % (i, min_ecc))
                    # this makes sure that whatever that third decimal
                    # place is, we're always one above it. e.g., if
                    # min_ecc was 1.3442, we want to use 1.345, and this
                    # will ensure that. (and we care about third decimal
                    # place because that's we're using in the save
                    # string)
                    min_ecc *= 1e3
                    min_ecc -= min_ecc % 1
                    min_ecc = (min_ecc+1) / 1e3
            else:
                min_ecc = self.min_eccentricity
            if transition_x is not None:
                r = polar_radius(scaled_img_res) / self.deg_to_pix[i]
                self.min_ecc_mask[i] = r > transition_x
            angle_windows = None
            ecc_windows = None
            if cache_dir is not None:
                format_kwargs = dict(scaling=scaling, max_eccentricity=self.max_eccentricity,
                                     img_res=','.join([str(int(i)) for i in scaled_img_res]),
                                     window_width=window_width_for_saving,
                                     window_type=window_type)
                if transition_x is None:
                    format_kwargs['min_eccentricity'] = float(min_ecc)
                else:
                    format_kwargs['min_eccentricity'] = float(transition_x)
                self.cache_paths.append(cache_path_template.format(**format_kwargs))
                if op.exists(self.cache_paths[-1]):
                    warnings.warn("Loading windows from cache: %s" % self.cache_paths[-1])
                    windows = torch.load(self.cache_paths[-1])
                    angle_windows = windows['angle']
                    ecc_windows = windows['ecc']
            if angle_windows is None or ecc_windows is None:
                angle_windows, ecc_windows = create_pooling_windows(
                    scaling, scaled_img_res, min_ecc, max_eccentricity, std_dev=self.std_dev,
                    transition_region_width=self.transition_region_width, window_type=window_type,
                    surround_std_dev=self.surround_std_dev, transition_x=transition_x,
                    center_surround_ratio=self.center_surround_ratio)

                if cache_dir is not None:
                    warnings.warn("Saving windows to cache: %s" % self.cache_paths[-1])
                    torch.save({'angle': angle_windows, 'ecc': ecc_windows}, self.cache_paths[-1])
            if window_type == 'dog':
                for k in ['center', 'surround']:
                    self.angle_windows[k][i] = angle_windows[k]
                    self.ecc_windows[k][i] = ecc_windows[k] * self.min_ecc_mask[i]
            else:
                self.angle_windows[i] = angle_windows
                self.ecc_windows[i] = ecc_windows
            self.ecc_windows, norm_factor = normalize_windows(self.angle_windows, self.ecc_windows,
                                                              self.window_width_pixels[i]['radial_full'], i,
                                                              self.center_surround_ratio)
            self.norm_factor[i] = norm_factor

    def _window_sizes(self):
        r"""Calculate the various window size metrics

        helper function that gets called during construction, should not
        be used by user. Sets the following attribute: n_polar_windows,
        n_eccentricity_bands, window_width_degrees, central_eccentricity_degrees,
        window_approx_area_degrees, window_width_pixels, central_eccentricity_pixels,
        window_approx_area_pixels, deg_to_pix

        all of these are based on calling various helper functions (all
        of which start with ``calc_``) and doing simple calculations
        based on the attributes already set (largely min_eccentricity,
        max_eccentricity, scaling, and transition_region_width)

        """
        window_type = {'dog': 'gaussian'}.get(self.window_type, self.window_type)
        ecc_window_width = calc_eccentricity_window_spacing(scaling=self.scaling,
                                                            std_dev=self.std_dev)
        n_polar_windows = int(round(calc_angular_n_windows(ecc_window_width / 2)))
        self.n_polar_windows = n_polar_windows
        angular_window_width = calc_angular_window_spacing(self.n_polar_windows)
        # we multiply max_eccentricity by sqrt(2) here because we want
        # to go out to the corner of the image
        window_widths = calc_window_widths_actual(angular_window_width, ecc_window_width,
                                                  self.min_eccentricity,
                                                  self.max_eccentricity*np.sqrt(2),
                                                  window_type, self.transition_region_width,
                                                  self.std_dev, self.transition_x,
                                                  self.surround_std_dev)
        self.window_width_degrees = dict(zip(['radial_top', 'radial_full', 'angular_top',
                                              'angular_full'], window_widths))
        self.n_eccentricity_bands = len(self.window_width_degrees['radial_top'])
        # transition width and std dev don't matter for central
        # eccentricity, just min and max (but transition_x does)
        self.central_eccentricity_degrees = calc_windows_eccentricity(
            'central', self.n_eccentricity_bands, ecc_window_width, self.min_eccentricity,
            transition_x=self.transition_x)
        self.window_width_degrees['radial_half'] = self.scaling * self.central_eccentricity_degrees
        # the 2 we divide by here is the
        # radial_to_circumferential_ratio; if we ever allow that to be
        # set by the user will need to update
        self.window_width_degrees['angular_half'] = self.window_width_degrees['radial_half'] / 2
        self.window_approx_area_degrees = {}
        for k in ['full', 'top', 'half']:
            self.window_approx_area_degrees[k] = (self.window_width_degrees['radial_%s' % k] *
                                                  self.window_width_degrees['angular_%s' % k] *
                                                  (np.pi/4))
        self.window_width_pixels = []
        self.window_approx_area_pixels = []
        self.central_eccentricity_pixels = []
        self.deg_to_pix = []
        for i in range(self.num_scales):
            deg_to_pix = calc_deg_to_pix([j/2**i for j in self.img_res], self.max_eccentricity)
            self.deg_to_pix.append(deg_to_pix)
            self.window_width_pixels.append(dict((k, v*deg_to_pix) for k, v in
                                                 self.window_width_degrees.copy().items()))
            self.window_approx_area_pixels.append({})
            for k in ['full', 'top', 'half']:
                self.window_approx_area_pixels[-1][k] = (self.window_width_pixels[-1]['radial_%s' % k] *
                                                         self.window_width_pixels[-1]['angular_%s' % k] *
                                                         (np.pi/4))
            self.central_eccentricity_pixels.append(self.deg_to_pix[-1] *
                                                    self.central_eccentricity_degrees)

    def to(self, *args, **kwargs):
        r"""Moves and/or casts the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)

        .. function:: to(dtype, non_blocking=False)

        .. function:: to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module

        Returns:
            Module: self
        """
        if self.window_type != 'dog':
            for k, v in self.angle_windows.items():
                self.angle_windows[k] = v.to(*args, **kwargs)
            for k, v in self.ecc_windows.items():
                self.ecc_windows[k] = v.to(*args, **kwargs)
            for k, v in self.norm_factor.items():
                self.norm_factor[k] = v.to(*args, **kwargs)
        else:
            for s in ['center', 'surround']:
                for k, v in self.angle_windows[s].items():
                    self.angle_windows[s][k] = v.to(*args, **kwargs)
                for k, v in self.ecc_windows[s].items():
                    self.ecc_windows[s][k] = v.to(*args, **kwargs)
                for k, v in self.norm_factor[s].items():
                    self.norm_factor[s][k] = v.to(*args, **kwargs)
        if hasattr(self, 'meshgrid'):
            # we don't want to change the dtype of meshgrid
            args = [a for a in args if not isinstance(a, torch.dtype)]
            kwargs.pop('dtype', None)
            for k, v in self.meshgrid.items():
                # meshgrid's values are (X, Y) tuples, each of which
                # needs to be sent separately
                self.meshgrid[k] = (v[0].to(*args, **kwargs), v[1].to(*args, **kwargs))
        return self

    def merge(self, other_PoolingWindows, scale_offset=.5):
        """Merge with a second PoolingWindows object

        This combines the angle_windows, ecc_windows, and window_size
        dictionaries of two PoolingWindows objects. Since they will both
        have similarly-indexed keys (0, 1, 2,... based on
        self.num_scales), we need some offset to keep them separate,
        which scale_offset provides. We thus merge the dictionaries like
        so:

        ```
        for k, v  in other_PoolingWindows.angle_windows.items():
            self.angle_windows[k+scale_offset] = v
        ```

        and similarly for ecc_windows and window_size

        The intended use case for this is to create one PoolingWindows
        object for a steerable pyramid with some number of scales, and
        then a second one for a corresponding "half-octave" steerable
        pyramid, which is built on the original image down-sampled by a
        factor of sqrt(2) in order to sample the frequencies half-way
        between the scales of the original pyramid. You might want to
        slightly adjust the shape of the down-sampled image (e.g., to
        make its size even), so we don't provide support to
        automatically create the windows for the half-scales; instead
        you should create a new PoolingWindows object based on your
        intended size and merge it into the original.

        Note that we don't return anything, we modify in-place.

        Parameters
        ----------
        other_PoolingWindows : plenoptic.simulate.PoolingWindows
            A second instantiated PoolingWindows object
        scale_offset : float, optional
            The amount to offset all the keys of the second
            PoolingWindows object by (see above for greater explanation)

        """
        for k, v in other_PoolingWindows.angle_windows.items():
            self.angle_windows[k+scale_offset] = v
        for k, v in other_PoolingWindows.ecc_windows.items():
            self.ecc_windows[k+scale_offset] = v
        for k, v in other_PoolingWindows.norm_factor.items():
            self.norm_factor[k+scale_offset] = v
        if hasattr(self, 'meshgrid'):
            for k, v in other_PoolingWindows.meshgrid.items():
                self.meshgrid[k+scale_offset] = v

    @staticmethod
    def _get_slice_vals(scaled_window_res, scaled_img_res):
        r"""Helper function to find the values to use when slicing windows down to size

        If we have a non-square image, we must create the windows as a
        square array and then slice it down to the size of the image,
        retaining the center of the windows array.

        The one wrinkle on this is that we also sometimes need to do
        this for different scales, so we need to make sure that
        'down-sampled' windows we create have the same shape as those
        created by our pyramid methods. It looks like that's always the
        ceiling of shape/2**scale. NOTE: This means it will probably not
        work if you're using something else that has multiple scales
        that doesn't use our pyramid methods and thus ends up with
        slightly differently sized down-sampled components. On images
        that are a power of 2, this shouldn't be an issue regardless

        This will only be for one dimension; because of how we've
        constructed the windows, we know they only need to be cut down
        in a single dimension

        Parameters
        ----------
        scaled_window_res : float
            The size of the square 'down-sampled'/scaled window we
            created (in one dimension; this should not be a tuple).
        scaled_img_res : float
            The size of the 'down-sampled'/scaled image we want to match
            (in one dimension; this should not be a tuple).

        Returns
        -------
        slice_vals : list
            A list of ints, use this to slice the window down correctly, e.g.,
            ``window[..., slice_vals[0]:slice_vals[1]]``

        """
        slice_vals = (scaled_window_res - scaled_img_res) / 2
        return [int(np.floor(slice_vals)), -int(np.ceil(slice_vals))]

    def parallel(self, devices, num_batches=1):
        r"""Parallelize the pooling windows across multiple GPUs

        PoolingWindows objects can get very large -- so large, that it's
        impossible to put them all on one GPU during a forward call. In
        order to solve that issue, we can spread them across multiple
        GPUs (CPU will still work, but then things get very slow for
        synthesis). Unfortunately we can't use ``torch.nn.DataParallel``
        for this because that only spreads the input/output across
        multiple devices, not components of a module. Because each
        window acts independently, we can split the different windows
        across devices.

        For the user, they should notice no difference between the
        parallelized and normal versions of PoolingWindows *EXCEPT* if
        they try to access ``PoolingWindows.windows`` directly: in the
        normal version, this is a dictionary with keys for each scale;
        in the parallelized version, this is a dictionary with keys (i,
        j), where i is the scale and j is the device index. Otherwise,
        all functions should work as before except that the input's
        device no longer needs to match PoolingWindows's device; we pass
        it to the correct device.

        We attempt to split the windows roughly evenly. So if you have 3
        devices and 100 windows, we'll put 34 on the first, 34 on the
        second, and the final 32 on the last. If you have multiple
        scales, each scale will be split in the same manner (though,
        since scales can have different numbers of windows, there's no
        guarantee they'll all be the same).

        Parameters
        ----------
        devices : list
            List of torch.devices or ints (corresponding to cuda
            numbers) to spread windows across
        num_batches : int
            The number of batches to further split the windows up
            into. The larger this number, the less memory the forward
            call will take but the slower it will be. So therefore, it's
            recommended you first try this with num_batches=1 and only
            gradually increase it as necessary

        Returns
        -------
        self

        See also
        --------
        unparallel : undo this parallelization

        """
        if self.window_type == 'dog':
            raise NotImplementedError("DoG windows do not support parallel!")
        angle_windows_gpu = {}
        for k, v in self.angle_windows.items():
            num = int(np.ceil(len(v) / len(devices)))
            for j, d in enumerate(devices):
                if j*num > len(v):
                    break
                angle_windows_gpu[(k, j)] = v[j*num:(j+1)*num].to(d)
        self.angle_windows = angle_windows_gpu
        ecc_windows_gpu = {}
        for k, v in self.ecc_windows.items():
            for j, d in enumerate(devices):
                ecc_windows_gpu[(k, j)] = v.to(d)
        self.ecc_windows = ecc_windows_gpu
        self.num_devices = len(devices)
        self.num_batches = num_batches
        return self

    def unparallel(self, device=torch.device('cpu')):
        r"""Unparallelize this object, bringing everything onto one device

        If you no longer want this object parallelized and spread across
        multiple devices, this method will collect all the windows back
        onto ``device``

        Parameters
        ----------
        device : torch.device or int
            The torch device to put every window on (if an int, this is
            the index of the gpu)

        Returns
        -------
        self

        See also
        --------
        parallel : parallelize PoolingWindows across multiple devices

        """
        if self.window_type == 'dog':
            raise NotImplementedError("DoG windows do not support unparallel!")
        angle_windows = {}
        keys = set([k[0] for k in self.angle_windows.keys()])
        for i in keys:
            tmp = []
            for j in range(self.num_devices):
                tmp.append(self.angle_windows[(i, j)].to(device))
            angle_windows[i] = torch.cat(tmp, 0)
        self.angle_windows = angle_windows
        ecc_windows = {}
        keys = set([k[0] for k in self.ecc_windows.keys()])
        for i in keys:
            tmp = []
            ecc_windows[i] = self.ecc_windows[(i, 0)].to(device)
        self.ecc_windows = ecc_windows
        self.num_devices = 1
        self.num_batches = 1
        return self

    def forward(self, x, idx=0, windows_key=None):
        r"""Window and pool the input

        We take an input, either a 4d tensor or a dictionary of 4d
        tensors, and return a windowed version of it. If it's a 4d
        tensor, we return a 5d tensor, with windows indexed along the
        3rd dimension. If it's a dictionary, we return a dictionary with
        the same keys and have changed all the values to 5d tensors,
        with windows indexed along the 3rd dimension.

        If it's a 4d tensor, we use the ``idx`` entry in the ``windows``
        list. If it's a dictionary, we assume it's keys are ``(scale,
        orientation)`` tuples and so use ``windows[key[0]]`` to find the
        appropriately-sized window (this is the case for, e.g., the
        steerable pyramid). If we want to use differently-structured
        dictionaries, we'll need to restructure this

        This is equivalent to calling ``self.pool(self.window(x, idx),
        idx)``, however, we don't produce the intermediate products and
        so this is more efficient.

        Parameters
        ----------
        x : dict or torch.Tensor
            Either a 4d tensor or a dictionary of 4d tensors.
        idx : int, optional
            Which entry in the ``windows`` list to use. Only used if
            ``x`` is a tensor
        windows_key : str or None, optional
            If None, we assume the angle_windows, ecc_windows attributes
            are dicts of tensors that we should use as the windows (with
            keys corresponding to different scales). If str, we assume
            they're dicts of dicts, and windows_key tells us the key so
            that we use ``self.angle_windows[windows_key]`` and
            similarly for ecc_windows as dicts of tensors (with keys
            corresponding to different scales). This is used by the DoG
            filter version

        Returns
        -------
        pooled_x : dict or torch.Tensor
            Same type as ``x``, see above for how it's created.

        See also
        --------
        window : window the input
        pool : pool the windowed input (get the weighted average)
        project : the opposite of this, going from pooled values to
            image

        """
        try:
            output_device = x.device
        except AttributeError:
            output_device = list(x.values())[0].device
        if windows_key is None:
            if self.window_type == 'dog':
                ctr = self.forward(x, idx, 'center')
                sur = self.forward(x, idx, 'surround')
                return self.center_surround_ratio * ctr - (1 - self.center_surround_ratio) * sur
            else:
                angle_windows = self.angle_windows
                ecc_windows = self.ecc_windows
        else:
            angle_windows = self.angle_windows[windows_key]
            ecc_windows = self.ecc_windows[windows_key]
        if isinstance(x, dict):
            if self.num_devices == 1:
                pooled_x = dict((k, torch.einsum('bchw,ahw,ehw->bcea',
                                                 [v.to(angle_windows[0].device),
                                                  angle_windows[k[0]],
                                                  ecc_windows[k[0]]]).flatten(2, 3))
                                for k, v in x.items())
            else:
                pooled_x = {}
                for k, v in x.items():
                    tmp = []
                    for i in range(self.num_devices):
                        angles = angle_windows[(k[0], i)]
                        e = ecc_windows[(k[0], i)]
                        v = v.to(angles.device)
                        t = []
                        for j, a in enumerate(torch.split(angles, angles.shape[0] // self.num_batches)):
                            t.append(torch.einsum('bchw,ahw,ehw->bcea', [v, a, e]).flatten(2, 3))
                        tmp.append(torch.cat(t, -1).to(output_device))
                    pooled_x[k] = torch.cat(tmp, -1)
        else:
            if self.num_devices == 1:
                pooled_x = (torch.einsum('bchw,ahw,ehw->bcea', [x.to(angle_windows[0].device),
                                                                angle_windows[idx],
                                                                ecc_windows[idx]]).flatten(2, 3))
            else:
                pooled_x = []
                for i in range(self.num_devices):
                    angles = angle_windows[(idx, i)]
                    e = ecc_windows[(idx, i)]
                    x = x.to(angles.device)
                    tmp = []
                    for j, a in enumerate(torch.split(angles, angles.shape[0] // self.num_batches)):
                        tmp.append(torch.einsum('bchw,ahw,ehw->bcea', [x, a, e]).flatten(2, 3))
                    pooled_x.append(torch.cat(tmp, -1).to(output_device))
                pooled_x = torch.cat(pooled_x, -1)
        return pooled_x

    def window(self, x, idx=0):
        r"""Window the input

        We take an input, either a 4d tensor or a dictionary of 4d
        tensors, and return a windowed version of it. If it's a 4d
        tensor, we return a 5d tensor, with windows indexed along the
        3rd dimension. If it's a dictionary, we return a dictionary with
        the same keys and have changed all the values to 5d tensors,
        with windows indexed along the 3rd dimension

        If it's a 4d tensor, we use the ``idx`` entry in the ``windows``
        list. If it's a dictionary, we assume it's keys are ``(scale,
        orientation)`` tuples and so use ``windows[key[0]]`` to find the
        appropriately-sized window (this is the case for, e.g., the
        steerable pyramid). If we want to use differently-structured
        dictionaries, we'll need to restructure this

        If you've called ``parallel()`` and this object has been spread
        across multiple devices, then the ``windowed_x`` value we return
        will look a little different:

        - if ``x`` was a dictionary, ``windowed_x`` will still be a
          dictionary but instead of having the same keys as ``x``, its
          keys will be ``(k, i)``, where ``k`` is the keys from ``x``
          and ``i`` is the indices of the devices

        - if ``x`` was a tensor, ``windowed_x`` will be a list of length
          ``self.num_devices``.

        In both cases, the different entries are on different devices,
        as specified by key[1] / the index and may be different
        shapes. ``pool`` will correctly bring them back together,
        concatenating them and bringing them onto the same device.

        Parameters
        ----------
        x : dict or torch.Tensor
            Either a 4d tensor or a dictionary of 4d tensors.
        idx : int, optional
            Which entry in the ``windows`` list to use. Only used if
            ``x`` is a tensor

        Returns
        -------
        windowed_x : dict or torch.Tensor
            Same type as ``x``, see above for how it's created.

        See also
        --------
        pool : pool the windowed input (get the weighted average)
        forward : perform the windowing and pooling simultaneously

        """
        if self.window_type == 'dog':
            raise NotImplementedError("DoG windows do not support window()!")
        if isinstance(x, dict):
            if list(x.values())[0].ndimension() != 4:
                raise Exception("PoolingWindows input must be 4d tensors or a dict of 4d tensors!"
                                " Unsqueeze until this is true!")
            if self.num_devices == 1:
                # one way to make this more general: figure out the size of
                # the tensors in x and in self.windows, and intelligently
                # lookup which should be used.
                return dict((k, torch.einsum('bchw,ahw,ehw->bceahw',
                                             [v.to(self.angle_windows[0].device),
                                              self.angle_windows[k[0]],
                                              self.ecc_windows[k[0]]]).flatten(2, 3))
                            for k, v in x.items())
            else:
                # then this is a dict and we're splitting it over multiple devices
                windowed_x = {}
                for k, v in x.items():
                    for i in range(self.num_devices):
                        e = self.ecc_windows[(k[0], i)]
                        angles = self.angle_windows[(k[0], i)]
                        x = x.to(angles.device)
                        tmp = []
                        for j, a in enumerate(torch.split(angles, angles.shape[0] // self.num_batches)):
                            tmp.append(torch.einsum('bchw,ahw,ehw->bcea', [v, a, e]).flatten(2, 3))
                        windowed_x[(k, i)] = torch.cat(tmp, -1)
                return windowed_x
        else:
            if x.ndimension() != 4:
                raise Exception("PoolingWindows input must be 4d tensors or a dict of 4d tensors!"
                                " Unsqueeze until this is true!")
            if self.num_devices == 1:
                return torch.einsum('bchw,ahw,ehw->bceahw', [x.to(self.angle_windows[0].device),
                                                             self.angle_windows[idx],
                                                             self.ecc_windows[idx]]).flatten(2, 3)
            else:
                windowed_x = []
                for i in range(self.num_devices):
                    e = self.ecc_windows[(idx, i)]
                    angles = self.angle_windows[(idx, i)]
                    x = x.to(angles.device)
                    tmp = []
                    for j, a in enumerate(torch.split(angles, angles.shape[0] // self.num_batches)):
                        tmp.append(torch.einsum('bchw,ahw,ehw->bceahw', [x, a, e]).flatten(2, 3))
                    windowed_x.append(torch.cat(tmp, -1))
                return tmp

    def pool(self, windowed_x, idx=0, output_device=torch.device('cpu')):
        r"""Pool the windowed input

        We take the windowed input (as returned by ``self.window()``)
        and perform a weighted average, dividing each windowed statistic
        by the sum of the window that generated it.

        The input must either be a 5d tensor or a dictionary of 5d
        tensors and we collapse across the spatial dimensions, returning
        a 3d tensor or a dictionary of 3d tensors.

        Similar to ``self.window()``, if it's a tensor, we use the
        ``idx`` entry in the ``windows`` list. If it's a dictionary, we
        assume it's keys are ``(scale, orientation)`` tuples and so use
        ``windows[key[0]]`` to find the appropriately-sized window (this
        is the case for, e.g., the steerable pyramid). If we want to use
        differently-structured dictionaries, we'll need to restructure
        this

        Parameters
        ----------
        windowed_x : dict or torch.Tensor
            Either a 5d tensor or a dictionary of 5d tensors
        idx : int, optional
            Which entry in the ``windows`` list to use. Only used if
            ``windowed_x`` is a tensor
        output_device : torch.device, optional
            If parallel was called before this, all the windows and
            windowed_x will be spread across multiple devices, so we
            need to know what device to place the output on. If parallel
            has not been called (i.e., PoolingWindows is only on one
            device, this is ignored)

        Returns
        -------
        pooled_x : dict or torch.Tensor
            Same type as ``windowed_x``, see above for how it's created.

        See also
        --------
        window : window the input
        forward : perform the windowing and pooling simultaneously

        """
        if self.window_type == 'dog':
            raise NotImplementedError("DoG windows do not support pool())!")
        if isinstance(windowed_x, dict):
            if self.num_devices == 1:
                # one way to make this more general: figure out the size
                # of the tensors in x and in self.angle_windows, and
                # intelligently lookup which should be used.
                return dict((k, v.sum((-1, -2)) ) for k, v in windowed_x.items())
            else:
                tmp = {}
                orig_keys = set([k[0] for k in windowed_x])
                for k in orig_keys:
                    t = []
                    for i in range(self.num_devices):
                        t.append(windowed_x[(k, i)].sum((-1, -2)).to(output_device))
                    tmp[k] = torch.cat(t, -1)
                return tmp
        else:
            if self.num_devices == 1:
                return windowed_x.sum((-1, -2))
            else:
                tmp = []
                for i, v in enumerate(windowed_x):
                    tmp.append(v.sum((-1, -2)).to(output_device))
                return torch.cat(tmp, -1)

    def project(self, pooled_x, idx=0, output_device=torch.device('cpu'), windows_key=None):
        r"""Project pooled values back onto an image

        For visualization purposes, you may want to project the pooled
        values (or values that have been pooled and then transformed in
        other ways) back onto an image. This method will do that for
        you.

        It takes a 3d tensor or dictionary of 3d tensors (like the
        output of ``forward()`` / ``pool()``; the final dimension must
        have a value for each window) and returns a 4d tensor or
        dictionary of 4d tensors (like the input of ``forward()`` /
        ``window()``).

        For example, if we have 100 windows, you must pass a i x j x 100
        tensor. For each of the i batches and j channels, we'll then
        multiply each of the 100 values by the corresponding window to
        end up with an i x j x 100 x height x width tensor. We then sum
        across windows to get i x j x heigth x width and return that.

        Parameters
        ----------
        pooled_x : dict or torch.Tensor
            3d Tensor or a dictionary of 3d tensors
        idx : int, optional
            Which entry in the ``windows`` list to use. Only used if
            ``pooled_x`` is a tensor
        output_device : torch.device, optional
            If parallel was called before this, all the windows and
            windowed_x will be spread across multiple devices, so we
            need to know what device to place the output on. If parallel
            has not been called (i.e., PoolingWindows is only on one
            device, this is ignored)
        windows_key : str or None, optional
            If None, we assume the angle_windows, ecc_windows attributes
            are dicts of tensors that we should use as the windows (with
            keys corresponding to different scales). If str, we assume
            they're dicts of dicts, and windows_key tells us the key so
            that we use ``self.angle_windows[windows_key]`` and
            similarly for ecc_windows as dicts of tensors (with keys
            corresponding to different scales). This is used by the DoG
            filter version

        Returns
        -------
        x : dict or torch.Tensor
            4d tensor or dictionary of 4d tensors

        See also
        --------
        forward : the opposite of this, going from image to pooled
            values

        """
        if windows_key is None:
            if self.window_type == 'dog':
                raise NotImplementedError("Can't call project directly when using DoG windows! "
                                          "Must call project_dog()")
            angle_windows = self.angle_windows
            ecc_windows = self.ecc_windows
        else:
            angle_windows = self.angle_windows[windows_key]
            ecc_windows = self.ecc_windows[windows_key]
        if isinstance(pooled_x, dict):
            if list(pooled_x.values())[0].ndimension() != 3:
                raise Exception("PoolingWindows input must be 3d tensors or a dict of 3d tensors!"
                                " Squeeze until this is true!")
            if self.num_devices == 1:
                tmp = {}
                for k, v in pooled_x.items():
                    if isinstance(k, tuple):
                        # in this case our keys are (scale, orientation)
                        # tuples, so we want the scale index
                        window_key = k[0]
                    else:
                        # in this case, the key is a string, probably
                        # "mean_luminance" and this corresponds to the
                        # lowest/largest scale
                        window_key = 0
                    v = v.reshape((*v.shape[:2], ecc_windows[window_key].shape[0],
                                   angle_windows[window_key].shape[0]))
                    tmp[k] = torch.einsum('bcea,ahw,ehw->bchw',
                                          [v.to(angle_windows[0].device),
                                           angle_windows[window_key],
                                           ecc_windows[window_key] * self.norm_factor[window_key]])
                return tmp
            else:
                tmp = {}
                for k, v in pooled_x.items():
                    num = int(np.ceil(self.n_polar_windows / self.num_devices))
                    t = []
                    if isinstance(k, tuple):
                        # in this case our keys are (scale, orientation)
                        # tuples, so we want the scale index
                        window_key = k[0]
                    else:
                        # in this case, the key is a string, probably
                        # "mean_luminance" and this corresponds to the
                        # lowest/largest scale
                        window_key = 0
                    v = v.reshape((*v.shape[:2], ecc_windows[(window_key, 0)].shape[0],
                                   self.n_polar_windows))
                    for i in range(self.num_devices):
                        e = ecc_windows[(window_key, i)]
                        a = angle_windows[(window_key, i)] * self.norm_factor[window_key]
                        d = v[..., i*num:(i+1)*num].to(a.device)
                        t.append(torch.einsum('bcea,ahw,ehw->bchw', [d, a, e]).to(output_device))
                    tmp[k] = torch.cat(t, 0).sum(0)
                return tmp
        else:
            if pooled_x.ndimension() != 3:
                raise Exception("PoolingWindows input must be 3d tensors or a dict of 3d tensors!"
                                " Squeeze until this is true!")
            if self.num_devices == 1:
                pooled_x = pooled_x.reshape((*pooled_x.shape[:2], ecc_windows[idx].shape[0],
                                             self.n_polar_windows))
                return torch.einsum('bcea,ahw,ehw->bchw', [pooled_x.to(angle_windows[0].device),
                                                           angle_windows[idx], ecc_windows[idx] *
                                                           self.norm_factor[idx]])
            else:
                pooled_x = pooled_x.reshape((*pooled_x.shape[:2], ecc_windows[(idx, 0)].shape[0],
                                             self.n_polar_windows))
                tmp = []
                num = int(np.ceil(self.n_polar_windows / self.num_devices))
                for i in range(self.num_devices):
                    a = angle_windows[(idx, i)]
                    e = ecc_windows[(idx, i)] * self.norm_factor[idx]
                    d = pooled_x[..., i*num:(i+1)*num].to(a.device)
                    tmp.append(torch.einsum('bcea,ahw,ehw->bchw', [d, a, e]).to(output_device))
                return torch.cat(tmp, 0).sum(0)

    def project_dog(self, x, idx=0, ones_flag=False):
        r"""Project pooled values for DoG windows

        This function returns the same thing as ``project`` but works
        slightly differently, because we want to project the center and
        surround components separately, and then sum them together (this
        will be more efficient because we don't want to hold all the
        windows in memory).

        See docstring of ``project`` for more details, but note that the
        input of this function should be the same as the input of
        ``forward()``. That's because we call ``forward()`` on the input
        twice, separately, to get the center and surround components,
        separately project them, and then sum them correctly

        Parameters
        ----------
        x : dict or torch.Tensor
            Either a 4d tensor or a dictionary of 4d tensors.
        idx : int, optional
            Which entry in the ``windows`` list to use. Only used if
            ``x`` is a tensor
        ones_flag : bool, optional
            if True, we don't project x, but project a representation of
            all ones that has the same shape. This is used for figuring
            out which portion of the image the windows cover (you may
            want to then convert it to boolean, because it will be flat
            everywhere but not necessarily 1; the exact value will
            depend on the center_surround_ratio and note that it will be
            approximatley 0 if the ratio is .5)

        Returns
        -------
        x : dict or torch.Tensor
            4d tensor or dictionary of 4d tensors

        See also
        --------
        forward : the opposite of this, going from image to pooled
            values
        project : the version for non-DoG windows

        """
        if self.window_type != 'dog':
            raise NotImplementedError("This is only for DoG windows!")
        try:
            output_device = x.device
        except AttributeError:
            output_device = list(x.values())[0].device
        ctr = self.forward(x, idx, 'center')
        sur = self.forward(x, idx, 'surround')
        if ones_flag:
            ctr = torch.ones_like(ctr)
            sur = torch.ones_like(sur)
        ctr = self.project(ctr, idx, output_device, 'center')
        sur = self.project(sur, idx, output_device, 'surround')
        return self.center_surround_ratio * ctr - (1 - self.center_surround_ratio) * sur

    def plot_windows(self, ax=None, contour_levels=None, colors='r',
                     subset=True, windows_scale=0, **kwargs):
        r"""plot the pooling windows on an image.

        This is just a simple little helper to plot the pooling windows
        on an axis. The intended use case is overlaying this on top of
        the image we're pooling (as returned by ``pyrtools.imshow``).

        Any additional kwargs get passed to ``ax.contour``

        Parameters
        ----------
        ax : matplotlib.pyplot.axis or None, optional
            The axis to plot the windows on. If None, will create a new
            figure with 1 axis
        contour_levels : None, array-like, or int, optional
            The ``levels`` argument to pass to ``ax.contour``. From that
            documentation: "Determines the number and positions of the
            contour lines / regions. If an int ``n``, use ``n`` data
            intervals; i.e. draw ``n+1`` contour lines. The level
            heights are automatically chosen. If array-like, draw
            contour lines at the specified levels. The values must be in
            increasing order". If None, will plot the contour that gives
            the first intersection (.5 for raised-cosine windows,
            self.window_max_amplitude * np.exp(-.25/2) (half a standard
            deviation away from max) for gaussian windows), as this is
            the easiest to see.
        colors : color string or sequence of colors, optional
            The ``colors`` argument to pass to ``ax.contour``. If a
            single character, all will have the same color; if a
            sequence, will cycle through the colors in ascending order
            (repeating if necessary)
        subset : bool, optional
            If True, will only plot four of the angle window
            slices. This is to save time and memory. If False, will plot
            all of them
        windows_scale : int, optional
            Which scale of the windows to use. windows is a list with
            different scales, so this specifies which one to use

        Returns
        -------
        ax : matplotlib.pyplot.axis
            The axis with the windows

        """
        if ax is None:
            dummy_data = np.ones(self.img_res)
            fig = pt.imshow(dummy_data, cmap='gray_r', title=None)
            ax = fig.axes[0]
        if contour_levels is None:
            if self.window_type == 'dog':
                contour_levels = [self.center_intersecting_amplitude]
            else:
                contour_levels = [self.window_intersecting_amplitude]
        if self.num_devices == 1:
            # attempt to not have all the windows in memory at once...
            try:
                angle_windows = self.angle_windows[windows_scale]
                ecc_windows = self.ecc_windows[windows_scale]
            except KeyError:
                # then this is the DoG windows and so we grab the center
                angle_windows = self.angle_windows['center'][windows_scale]
                ecc_windows = self.ecc_windows['center'][windows_scale]
            if subset:
                angle_windows = angle_windows[:4]
            for a in angle_windows:
                windows = torch.einsum('hw,ehw->ehw', [a, ecc_windows])
                for w in windows:
                    try:
                        # if this isn't true, then this window will be
                        # plotted weird
                        if not (w > contour_levels[0]).any():
                            continue
                    except TypeError:
                        # in this case, it's an int
                        pass
                    ax.contour(to_numpy(w), contour_levels, colors=colors, **kwargs)
        else:
            counter = 0
            for device in range(self.num_devices):
                for a in self.angle_windows[(windows_scale, device)]:
                    if subset and counter >= 4:
                        break
                    # we have a version of the eccentricity windows on
                    # each device that the angle windows are on, in
                    # order to avoid a .to() call (which is slow)
                    windows = torch.einsum('hw,ehw->ehw', [a, self.ecc_windows[(windows_scale, device)]])
                    for w in windows:
                        try:
                            # if this isn't true, then this window will be
                            # plotted weird
                            if not (w > contour_levels[0]).any():
                                continue
                        except TypeError:
                            # in this case, it's an int
                            pass
                        ax.contour(to_numpy(w), contour_levels, colors=colors, **kwargs)
                    counter += 1
        return ax

    def plot_window_widths(self, units='degrees', scale_num=0, figsize=(5, 5), jitter=.25,
                           ax=None):
        r"""plot the widths of the windows, in degrees or pixels

        We plot the width of the window in both angular and radial
        direction, as well as showing the 'top', 'half', and 'full'
        widths (top is the width of the flat-top region of each window,
        where the window's value is 1; full is the width of the entire
        window; half is the width at the half-max value, which is what
        corresponds to the scaling value)

        We plot this as a stem plot against eccentricity, showing the
        windows at their central eccentricity

        If the unit is 'pixels', then we also need to know which
        ``scale_num`` to plot (the windows are created at different
        scales, and so come in different pixel sizes)

        Parameters
        ----------
        units : {'degrees', 'pixels'}, optional
            Whether to show the information in degrees or pixels (both
            the width and the window location will be presented in the
            same unit).
        scale_num : int, optional
            Which scale window we should plot
        figsize : tuple, optional
            The size of the figure to create
        jitter : float or None, optional
            Whether to add a little bit of jitter to the x-axis to
            separate the radial and angular widths. There are only two
            values we separate, so we don't add actual jitter, just move
            one up by the value specified by jitter, the other down by
            that much (we use the same value at each eccentricity)
        ax : matplotlib.pyplot.axis or None, optional
            The axis to plot the windows on. If None, will create a new
            figure with 1 axis

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot

        """
        if units == 'degrees':
            data = self.window_width_degrees
            central_ecc = self.central_eccentricity_degrees
        elif units == 'pixels':
            data = self.window_width_pixels[scale_num]
            central_ecc = self.central_eccentricity_pixels[scale_num]
        else:
            raise Exception("units must be one of {'pixels', 'degrees'}, not %s!" % units)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        if jitter is not None:
            jitter_vals = {'radial': -jitter, 'angular': jitter}
        else:
            jitter_vals = {'radial': 0, 'angular': 0}
        colors = {'radial': 'C0', 'angular': 'C1'}
        sizes = {'full': 5, 'half': 10, 'top': 15}
        for direc, height in itertools.product(['radial', 'angular'], ['top', 'half', 'full']):
            m, s, b = ax.stem(central_ecc+jitter_vals[direc], data[direc+"_"+height],
                              colors[direc], colors[direc]+'.', label=direc+"_"+height,
                              use_line_collection=True)
            m.set(markersize=sizes[height])
        ax.set_ylabel('Window width (%s)' % units)
        ax.set_xlabel('Window central eccentricity (%s)' % units)
        ax.legend(loc='upper left')
        return fig

    def plot_window_areas(self, units='degrees', scale_num=0, figsize=(5, 5), ax=None):
        r"""plot the approximate areas of the windows, in degrees or pixels

        We plot the approximate area of the window, calculated using
        'top', 'half', and 'full' widths (top is the width of the
        flat-top region of each window, where the window's value is 1;
        full is the width of the entire window; half is the width at the
        half-max value, which is what corresponds to the scaling
        value). To get the approximate area, we multiply the radial
        width against the corresponding angular width, then divide by pi
        / 4.

        The half area shown here is what we use to compare against a
        threshold value in the ``calc_min_eccentricity()`` in order to
        determine what the minimum eccentricity where the windows
        contain more than 1 pixel.

        We plot this as a stem plot against eccentricity, showing the
        windows at their central eccentricity

        If the unit is 'pixels', then we also need to know which
        ``scale_num`` to plot (the windows are created at different
        scales, and so come in different pixel sizes)

        Parameters
        ----------
        units : {'degrees', 'pixels'}, optional
            Whether to show the information in degrees or pixels (both
            the area and the window location will be presented in the
            same unit).
        scale_num : int, optional
            Which scale window we should plot
        figsize : tuple, optional
            The size of the figure to create
        ax : matplotlib.pyplot.axis or None, optional
            The axis to plot the windows on. If None, will create a new
            figure with 1 axis

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot

        """
        if units == 'degrees':
            data = self.window_approx_area_degrees
            central_ecc = self.central_eccentricity_degrees
        elif units == 'pixels':
            data = self.window_approx_area_pixels[scale_num]
            central_ecc = self.central_eccentricity_pixels[scale_num]
        else:
            raise Exception("units must be one of {'pixels', 'degrees'}, not %s!" % units)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        sizes = {'full': 5, 'half': 10, 'top': 15}
        for height in ['top', 'half', 'full']:
            m, s, b = ax.stem(central_ecc, data[height], 'C0', 'C0.', label=height,
                              use_line_collection=True)
            m.set(markersize=sizes[height])
        ax.set_ylabel('Window area (%s)' % units)
        ax.set_xlabel('Window central eccentricity (%s)' % units)
        ax.legend(loc='upper left')
        return fig
