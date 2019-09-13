"""functions to perform spatial pooling, as seen in Freeman and Simoncelli, 2011

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
from ...tools.data import to_numpy


def calc_angular_window_width(n_windows):
    r"""calculate and return the window width for the angular windows

    this is the :math:`w_{\theta }` term in equation 10 of the paper's
    online methods

    Parameters
    ----------
    n_windows : `float`
        The number of windows to pack into 2 pi. Note that we don't
        require it to be an integer here, but the code that makes use of
        this does.

    Returns
    -------
    window_width : `float`
        The width of the polar angle windows.

    """
    return (2*np.pi) / n_windows


def calc_angular_n_windows(window_width):
    r"""calculate and return the number of angular windows

    this is the :math:`N_{\theta }` term in equation 10 of the paper's
    online method, which we've rearranged in order to get this.

    Parameters
    ----------
    window_width : `float`
        The width of the polar angle windows.

    Returns
    -------
    n_windows : `float`
        The number of windows that fit into 2 pi.

    """
    return (2*np.pi) / window_width


def calc_eccentricity_window_width(min_ecc=.5, max_ecc=15, n_windows=None, scaling=None):
    r"""calculate and return the window width for the eccentricity windows

    this is the :math:`w_e` term in equation 11 of the paper's online
    methods, which we also refer to as the radial width. Note that we
    take exactly one of ``n_windows`` or ``scaling`` in order to
    determine this value.

    If scaling is set, ``min_ecc`` and ``max_ecc`` are ignored (the
    window width only depends on scaling, not also on the range of
    eccentricities; they only matter when determining the width using
    ``n_windows``)

    Parameters
    ----------
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
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

    Returns
    -------
    window_width : `float`
        The width of the log-eccentricity windows.

    Notes
    -----
    No equation was given in the paper to calculate the window width,
    :math:`w_e` from the scaling, :math:`s`, so we derived it
    ourselves. We start with the final equation for the scaling, given
    in the Notes for the ``calc_scaling`` function.

    .. math::

        s &= \exp(w_e)^{.5} -  \exp(w_e)^{-.5} \\
        s &= \frac{\exp(w_e)-1}{\sqrt{\exp(w_e)}} \\
        s^2 &= \frac{\exp(2w_e)-2\exp(w_e)+1}{\exp(w_e)} \\
        s^2 &= \exp(w_e)-2+\frac{1}{\exp(w_e)} \\
        s^2+2 &= e^{w_e}+e^{-w_e} \\
        s^2e^{w_e}+2e^{w_e}&=e^{2w_e}+1 \\
        e^{2w_e} - e^{w_e}(s^2+2) +1 &=0

    And then solve using the quadratic formula:

    .. math::

        e^{w_e} &= \frac{s^2+2\pm\sqrt{(s^2+2)^2-4}}{2} \\
        e^{w_e} &= \frac{s^2+2\pm\sqrt{s^4+4s^2}}{2} \\
        e^{w_e} &= \frac{s^2+2\pm s\sqrt{s^2+4}}{2} \\
        w_e &= \log\left(\frac{s^2+2\pm s\sqrt{s^2+4}}{2}\right)

    The window width is strictly positive, so we only return the
    positive quadratic root (the one with plus in the numerator).

    """
    if scaling is not None:
        return np.log((scaling**2+2+scaling * np.sqrt(scaling**2+4))/2)
    elif n_windows is not None:
        return (np.log(max_ecc) - np.log(min_ecc)) / n_windows
    else:
        raise Exception("Exactly one of n_windows or scaling must be set!")


def calc_eccentricity_n_windows(window_width, min_ecc=.5, max_ecc=15):
    r"""calculate and return the number of eccentricity windows

    this is the :math:`N_e` term in equation 11 of the paper's online
    method, which we've rearranged in order to get this.

    Parameters
    ----------
    window_width : `float`
        The width of the log-eccentricity windows.
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.

    Returns
    -------
    n_windows : `float`
        The number of log-eccentricity windows we create.

    """
    return (np.log(max_ecc) - np.log(min_ecc)) / window_width


def calc_scaling(n_windows, min_ecc=.5, max_ecc=15):
    r"""calculate and return the scaling value, as reported in the paper

    Scaling is the ratio of the eccentricity window's radial full-width
    at half-maximum to eccentricity. For eccentricity, we use the
    window's "central eccentricity", the one where the input to the
    mother window (:math:`x` in equation 9 in the online methods) is 0.

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

    It turns out that this holds for all permissible values of
    ``transition_region_width`` (:math:`t` in the equations) (try
    playing around with some plots if you don't believe me).

    Full-width half-maximum, :math:`W`, the difference between the two
    values of :math:`e_h`:

    .. math::

        \pm.5 &= \frac{\log(e_h) - (log(e_0)+w_e(n+1))}{w_e} \\
        e_h &= e_0 \cdot \exp(w_e(\pm.5+n+1)) \\
        W &= e_0 (\exp(w_e(n+1.5)) - \exp(w_e(n+.5))

    Window's central eccentricity, :math:`e_c`:

    .. math::

        0 &= \frac{\log(e_c) -(log(e_0)+w_e(n+1))}{w_e} \\
        e_c &= e_0 \cdot \exp(w_e(n+1))

    Then the scaling, :math:`s` is the ratio :math:`\frac{W}{e_c}`:

    .. math::

        s &= \frac{e_0 (\exp(w_e(n+1.5)) -  \exp(w_e(n+.5)))}{e_0 \cdot \exp(w_e(n+1))} \\
        s &= \frac{\exp(w_e(n+1.5))}{\exp(w_e(n+1))} -  \frac{\exp(w_e(n+.5))}{\exp(w_e(n+1))} \\
        s &= \exp(w_e(n+1.5-n-1)) -  \exp(w_e(n+.5-n-1)) \\
        s &= \exp(.5\cdot w_e) -  \exp(-.5\cdot w_e)

    Note that we don't actually use the value returned by
    ``calc_windows_central_eccentricity`` for :math:`e_c`; we simplify
    it away in the calculation above.

    """
    window_width = (np.log(max_ecc) - np.log(min_ecc)) / n_windows
    return np.exp(.5*window_width) - np.exp(-.5*window_width)


def calc_windows_eccentricity(ecc_type, n_windows, window_width, min_ecc=.5,
                              transition_region_width=.5):
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
    window_width : `float`
        The width of the log-eccentricity windows.
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. Must lie between 0 and 1.

    Returns
    -------
    eccentricity : np.array
        A list of length ``n_windows``, containing the minimum, central,
        or maximum eccentricities of each window.

    Notes
    -----
    To find 'min', we solve for the eccentricity where
    :math:`x=\frac{-(1+t)}{2}` in equation 9:

    .. math::

        \frac{-(1+t)}{2} &= \frac{\log(e_{min}) -(log(e_0)+w_e(n+1))}{w_e} \\
        e_{min} &= \exp{\frac{-w_e(1+t)}{2} + \log{e_0} + w_e(n+1)}

    To find 'central', we solve for the eccentricity where :math:`x=0`
    in equation 9:

    .. math::

        0 &= \frac{\log(e_c) -(log(e_0)+w_e(n+1))}{w_e} \\
        e_c &= e_0 \cdot \exp(w_e(n+1))

    To find 'max', we solve for the eccentricity where
    :math:`x=\frac{(1+t)}{2}` in equation 9:

    .. math::

        \frac{(1+t)}{2} &= \frac{\log(e_{max}) -(log(e_0)+w_e(n+1))}{w_e} \\
        e_{max} &= \exp{\frac{w_e(1+t)}{2} + \log{e_0} + w_e(n+1)}

    """
    if ecc_type == 'central':
        ecc = [min_ecc * np.exp(window_width * (i+1)) for i in np.arange(np.ceil(n_windows))]
    elif ecc_type == 'min':
        ecc = [(np.exp(-window_width*(1+transition_region_width)) * min_ecc *
                np.exp(window_width * (i+1))) for i in np.arange(np.ceil(n_windows))]
    elif ecc_type == 'max':
        ecc = [(np.exp(window_width*(1+transition_region_width)) * min_ecc *
                np.exp(window_width * (i+1))) for i in np.arange(np.ceil(n_windows))]
    else:
        raise Exception("Don't know how to handle ecc_type %s" % ecc_type)
    return np.array(ecc)


def calc_window_widths_actual(angular_window_width, radial_window_width, min_ecc=.5, max_ecc=15,
                              transition_region_width=.5):
    r"""calculate and return the actual widths of the windows, in angular and radial directions

    whereas ``calc_angular_window_width`` returns a term used in the
    equations to generate the windows, this returns the actual angular
    and radial widths of each set of windows (in degrees).

    We return four total widths, two by two for radial and angular by
    'top' and 'full'. By 'top', we mean the width of the flat-top region
    of each window (where the windows value is 1), and by 'full', we
    mean the width of the entire window

    Parameters
    ----------
    angular_window_width : float
        The width of the windows in the angular direction, as returned
        by ``calc_angular_window_width``
    radial_window_width : float
        The width of the windows in the radial direction, as returned by
        ``calc_eccentricity_window_width``
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.

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

    """
    n_radial_windows = np.ceil(calc_eccentricity_n_windows(radial_window_width, min_ecc, max_ecc))
    window_central_eccentricities = calc_windows_eccentricity('central', n_radial_windows,
                                                              radial_window_width, min_ecc)
    radial_top = [min_ecc*(np.exp((radial_window_width*(3+2*i-transition_region_width))/2) -
                           np.exp((radial_window_width*(1+2*i+transition_region_width))/2))
                  for i in np.arange(n_radial_windows)]
    radial_full = [min_ecc*(np.exp((radial_window_width*(3+2*i+transition_region_width))/2) -
                            np.exp((radial_window_width*(1+2*i-transition_region_width))/2))
                   for i in np.arange(n_radial_windows)]
    angular_top = [angular_window_width * (1-transition_region_width) * e_c for e_c in
                   window_central_eccentricities]
    angular_full = [angular_window_width * (1+transition_region_width) * e_c for e_c in
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
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    # doing it in this array-ized fashion is much faster
    y = np.zeros_like(x)
    # this creates a bunch of masks
    masks = [(-(1 + transition_region_width) / 2 < x) & (x <= (transition_region_width - 1) / 2),
             ((transition_region_width - 1) / 2 < x) & (x <= (1 - transition_region_width) / 2),
             ((1 - transition_region_width) / 2 < x) & (x <= (1 + transition_region_width) / 2)]
    # and this creates the values where those masks are
    vals = [np.cos(np.pi/2 * ((x - (transition_region_width-1)/2) / transition_region_width))**2,
            np.ones_like(x),
            (-np.cos(np.pi/2 * ((x - (1+transition_region_width)/2) /
                                transition_region_width))**2 + 1)]
    for m, v in zip(masks, vals):
        y[m] = v[m]
    return y


def polar_angle_windows(n_windows, resolution, transition_region_width=.5):
    r"""Create polar angle windows in 2d

    We require an integer number of windows placed between 0 and 2 pi.

    Notes
    -----
    Equation 10 from the online methods of [1]_.

    Parameters
    ----------
    n_windows : `int`
        The number of polar angle windows we create.
    resolution : tuple
        2-tuple of ints specifying the resolution of the 2d images to
        make.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.

    Returns
    -------
    windows : np.array
        A 3d array containing the (2d) polar angle windows. Windows will
        be indexed along the first dimension.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    theta = pt.synthetic_images.polar_angle(resolution)
    # we want theta to lie between 0 and 2 pi
    theta = np.mod(theta, 2*np.pi)
    if int(n_windows) != n_windows:
        raise Exception("n_windows must be an integer!")
    if n_windows == 1:
        raise Exception("We cannot handle one window correctly!")
    # this is `w_\theta` in the paper
    window_width = calc_angular_window_width(n_windows)
    windows = []
    for n in range(int(n_windows)):
        if n == 0:
            # otherwise this region of theta is discontinuous (it jumps
            # from 2 pi to 0)
            tmp_theta = pt.synthetic_images.polar_angle(resolution)
        else:
            tmp_theta = theta
        mother_window_arg = ((tmp_theta - (window_width * n +
                                           (window_width * (1-transition_region_width)) / 2)) /
                             window_width)
        windows.append(mother_window(mother_window_arg, transition_region_width))
    windows = [i for i in windows if not (i == 0).all()]
    return np.array(windows)


def log_eccentricity_windows(resolution, n_windows=None, window_width=None, min_ecc=.5, max_ecc=15,
                             transition_region_width=.5):
    r"""Create log eccentricity windows in 2d

    Note that exactly one of ``n_windows`` or ``window_width`` must be
    set.

    In order to convert the polar radius array we create from pixels to
    degrees, we assume that ``max_ecc`` is the maximum eccentricity in
    the horizontal direction (i.e., to convert from pixels to degrees,
    we multiply by ``max_ecc / (resolution[1]/2)``)

    NOTE: if ``n_windows`` (rater than ``window_width``) is set, this is
    not necessarily the number of arrays we'll return. In order to get
    the full set of windows, we want to consider those that would show
    up in the corners as well, so it's probable that this function
    returns one more window there; we determine if this is necessary by
    calling ``calc_eccentricity_n_windows`` with ``np.sqrt(2)*max_ecc``

    Notes
    -----
    Equation 11 from the online methods of [1]_.

    Parameters
    ----------
    resolution : tuple
        2-tuple of ints specifying the resolution of the 2d images to
        make.
    n_windows : `float` or `None`
        The number of log-eccentricity windows from ``min_ecc`` to
        ``max_ecc``. ``n_windows`` xor ``window_width`` must be set.
    window_width : `float` or `None`
        The width of the log-eccentricity windows. ``n_windows`` xor
        ``window_width`` must be set.
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    transition_region_width : `float`
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.

    Returns
    -------
    windows : `np.array`
        A 3d array containing the (2d) log-eccentricity windows. Windows
        will be indexed along the first dimension.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    ecc = pt.synthetic_images.polar_radius(resolution) * (max_ecc / (resolution[1]/2))
    if window_width is None:
        window_width = calc_eccentricity_window_width(min_ecc, max_ecc, n_windows)
    n_windows = calc_eccentricity_n_windows(window_width, min_ecc, max_ecc*np.sqrt(2))
    windows = []
    for n in range(math.ceil(n_windows)):
        mother_window_arg = (np.log(ecc) - (np.log(min_ecc) + window_width * (n+1))) / window_width
        windows.append(mother_window(mother_window_arg, transition_region_width))
    windows = [i for i in windows if not (i == 0).all()]
    return np.array(windows)


def create_pooling_windows(scaling, resolution, min_eccentricity=.5, max_eccentricity=15,
                           radial_to_circumferential_ratio=2, transition_region_width=.5):
    r"""Create two sets of 2d pooling windows (log-eccentricity and polar angle) that span the visual field

    This creates the pooling windows that we use to average image
    statistics for metamer generation as done in [1]_. This is returned
    as two 3d torch tensors for further use with a model.

    Note that these are returned separately as log-eccentricity and
    polar angle tensors and if you want the windows used in the paper
    [1]_, you'll need to call ``torch.einsum`` (see Examples section)
    or, better yet, use the ``PoolingWindows`` class, which is provided
    for this purpose.

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
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. 0.5 (the default) is the
        value used in the paper [1]_.

    Returns
    -------
    angle_windows : `torch.tensor`
        The 3d tensor of 2d polar angle windows. Its shape will be
        ``(n_angle_windows, *resolution)``, where the number of windows
        is inferred in this function based on the values of ``scaling``
        and ``radial_to_circumferential_width``.
    ecc_windows : `torch.tensor`
        The 3d tensor of 2d log-eccentricity windows. Its shape will be
        ``(n_eccen_windows, *resolution)``, where the number of windows
        is inferred in this function based on the values of ``scaling``,
        ``min_ecc``, and ``max_ecc``.

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
    ecc_window_width = calc_eccentricity_window_width(min_eccentricity, max_eccentricity,
                                                      scaling=scaling)
    n_polar_windows = calc_angular_n_windows(ecc_window_width / radial_to_circumferential_ratio)
    # we want to set the number of polar windows where the ratio of widths is approximately what
    # the user specified. the constraint that it's an integer is more important
    angle_tensor = polar_angle_windows(round(n_polar_windows), resolution, transition_region_width)
    angle_tensor = torch.tensor(angle_tensor, dtype=torch.float32)
    ecc_tensor = log_eccentricity_windows(resolution, None, ecc_window_width, min_eccentricity,
                                          max_eccentricity,
                                          transition_region_width=transition_region_width)
    ecc_tensor = torch.tensor(ecc_tensor, dtype=torch.float32)
    return angle_tensor, ecc_tensor


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
    em-{max_eccentricity}_t-{transition_region_width}.pt``. We'll cache
    each scale separately, changing the img_res (and potentially
    min_eccentricity) values in that save path appropriately.

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
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. 0.5 (the default) is the
        value used in the paper [1]_.
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.

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
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    angle_windows : list or dict
        A list of 3d tensors containing the angular pooling windows in
        which the model parameters are averaged. Each entry in the list
        corresponds to a different scale and thus is a different
        size. If you have called ``parallel()``, this will be a
        dictionary instead (see that method for details)
    ecc_windows : list or dict
        A list of 3d tensors containing the log-eccentricity pooling
        windows in which the model parameters are averaged. Each entry
        in the list corresponds to a different scale and thus is a
        different size. If you have called ``parallel()``, this will be
        a dictionary instead (see that method for details)
    window_sizes : list or dict
        A list of 1d tensors giving the size of the combined pooling
        windows, that is, the output of ``torch.einsum('ahw,ehw->ea',
        [a, e])``, where a and e are the angle and eccentricity windows
        at the corresponding scale.
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

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5, max_eccentricity=15, num_scales=1,
                 transition_region_width=.5, cache_dir=None):
        super().__init__()
        if len(img_res) != 2:
            raise Exception("img_res must be 2d!")
        self.scaling = scaling
        self.transition_region_width = transition_region_width
        self.min_eccentricity = float(min_eccentricity)
        self.max_eccentricity = float(max_eccentricity)
        self.img_res = img_res
        self.num_scales = num_scales
        self.num_devices = 1
        self.angle_windows = []
        self.ecc_windows = []
        self.window_sizes = []
        if cache_dir is not None:
            self.cache_dir = op.expanduser(cache_dir)
            cache_path_template = op.join(self.cache_dir, "scaling-{scaling}_size-{img_res}_"
                                          "e0-{min_eccentricity:.03f}_em-{max_eccentricity:.01f}_t"
                                          "-{transition_region_width}.pt")
        else:
            self.cache_dir = cache_dir
        self.cache_paths = []
        self.calculated_min_eccentricity_degrees = []
        self.calculated_min_eccentricity_pixels = []
        self._window_sizes()
        self.state_dict_reduced = {'scaling': scaling, 'img_res': img_res,
                                   'min_eccentricity': self.min_eccentricity,
                                   'max_eccentricity': self.max_eccentricity,
                                   'transition_region_width': transition_region_width,
                                   'cache_dir': self.cache_dir}
        for i in range(num_scales):
            scaled_img_res = [np.ceil(j / 2**i) for j in img_res]
            min_ecc, min_ecc_pix = calc_min_eccentricity(scaling, scaled_img_res, max_eccentricity)
            self.calculated_min_eccentricity_degrees.append(min_ecc)
            self.calculated_min_eccentricity_pixels.append(min_ecc_pix)
            if min_ecc > self.min_eccentricity:
                if i == 0:
                    raise Exception("Cannot create windows with scaling %s, resolution %s, and min"
                                    "_eccentricity %s, it will contain windows smaller than a "
                                    "pixel. min_eccentricity must be at least %s!" %
                                    (scaling, img_res, min_eccentricity, min_ecc))
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
            angle_windows = None
            ecc_windows = None
            if cache_dir is not None:
                format_kwargs = dict(scaling=scaling, min_eccentricity=float(min_ecc),
                                     max_eccentricity=self.max_eccentricity,
                                     img_res=','.join([str(int(i)) for i in scaled_img_res]),
                                     transition_region_width=transition_region_width)
                self.cache_paths.append(cache_path_template.format(**format_kwargs))
                if op.exists(self.cache_paths[-1]):
                    warnings.warn("Loading windows from cache: %s" % self.cache_paths[-1])
                    windows = torch.load(self.cache_paths[-1])
                    angle_windows = windows['angle']
                    ecc_windows = windows['ecc']
            if angle_windows is None or ecc_windows is None:
                angle_windows, ecc_windows = create_pooling_windows(
                    scaling, scaled_img_res, min_ecc, max_eccentricity,
                    transition_region_width=transition_region_width)

                if cache_dir is not None:
                    warnings.warn("Saving windows to cache: %s" % self.cache_paths[-1])
                    torch.save({'angle': angle_windows, 'ecc': ecc_windows}, self.cache_paths[-1])
            self.angle_windows.append(angle_windows)
            self.ecc_windows.append(ecc_windows)
            self.window_sizes.append(torch.einsum('ahw,ehw->ea', [angle_windows,
                                                                  ecc_windows]).flatten())

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
        ecc_window_width = calc_eccentricity_window_width(scaling=self.scaling)
        self.n_polar_windows = int(round(calc_angular_n_windows(ecc_window_width / 2)))
        angular_window_width = calc_angular_window_width(self.n_polar_windows)
        # we multiply max_eccentricity by sqrt(2) here because we want
        # to go out to the corner of the image
        window_widths = calc_window_widths_actual(angular_window_width, ecc_window_width,
                                                  self.min_eccentricity,
                                                  self.max_eccentricity*np.sqrt(2),
                                                  self.transition_region_width)
        self.window_width_degrees = dict(zip(['radial_top', 'radial_full', 'angular_top',
                                              'angular_full'], window_widths))
        self.n_eccentricity_bands = len(self.window_width_degrees['radial_top'])
        self.central_eccentricity_degrees = calc_windows_eccentricity(
            'central', self.n_eccentricity_bands, ecc_window_width, self.min_eccentricity)
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
            deg_to_pix = calc_deg_to_pix([j/2**(i+1) for j in self.img_res], self.max_eccentricity)
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
        for i, w in enumerate(self.angle_windows):
            self.angle_windows[i] = w.to(*args, **kwargs)
        for i, w in enumerate(self.ecc_windows):
            self.ecc_windows[i] = w.to(*args, **kwargs)
        for i, w in enumerate(self.window_sizes):
            self.window_sizes[i] = w.to(*args, **kwargs)
        return self

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

    def parallel(self, devices):
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
        normal version, this is a list with one entry per scale; in the
        parallelized version, this is a dictionary with keys (i, j),
        where i is the scale and j is the device index. Otherwise, all
        functions should work as before except that the input's device
        no longer needs to match PoolingWindows's device; we pass it to
        the correct device.

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

        Returns
        -------
        self

        See also
        --------
        unparallel : undo this parallelization

        """
        angle_windows_gpu = {}
        for i, w in enumerate(self.angle_windows):
            num = int(np.ceil(len(w) / len(devices)))
            for j, d in enumerate(devices):
                if j*num > len(w):
                    break
                angle_windows_gpu[(i, j)] = w[j*num:(j+1)*num].to(d)
        self.angle_windows = angle_windows_gpu
        self.num_devices = len(devices)
        self.window_sizes = [w.to(devices[0]) for w in self.window_sizes]
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
        angle_windows = []
        for i in range(self.num_scales):
            tmp = []
            for j in range(self.num_devices):
                tmp.append(self.angle_windows[(i, j)].to(device))
            angle_windows.append(torch.cat(tmp, 0))
        self.angle_windows = angle_windows
        self.num_devices = 1
        return self

    def forward(self, x, idx=0):
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
        torch.cuda.empty_cache()
        try:
            output_device = x.device
        except AttributeError:
            output_device = list(x.values())[0].device
        window_size_mask = [(w > 1).to(output_device) for w in self.window_sizes]
        if isinstance(x, dict):
            if isinstance(self.angle_windows, list):
                pooled_x = dict((k, torch.einsum('bchw,ahw,ehw->bcea',
                                                 [v.to(self.angle_windows[0].device),
                                                  self.angle_windows[k[0]],
                                                  self.ecc_windows[k[0]]]).flatten(2, 3))
                                for k, v in x.items())
                # need to do the indexing because otherwise we can get some
                # divide by 0 issues. this doesn't mess up the forward pass,
                # but it does mess up the backwards pass
                pooled_x = dict((k, v[..., window_size_mask[k[0]]] /
                                 self.window_sizes[k[0]][window_size_mask[k[0]]])
                                for k, v in pooled_x.items())
            else:
                pooled_x = {}
                for k, v in x.items():
                    e = self.ecc_windows[k[0]]
                    tmp = []
                    sizes = self.window_sizes[k[0]][window_size_mask[k[0]]].to(output_device)
                    for i in range(self.num_devices):
                        a = self.angle_windows[(k[0], i)]
                        val = torch.einsum('bchw,ahw,ehw->bcea',
                                           [v.to(a.device), a, e.to(a.device)]).flatten(2, 3)
                        tmp.append(val.to(output_device))
                    pooled_x[k] = torch.cat(tmp, -1)[..., window_size_mask[k[0]]] / sizes
        else:
            if isinstance(self.angle_windows, list):
                pooled_x = torch.einsum('bchw,ahw,ehw->bcea', [x.to(self.angle_windows[0].device),
                                                               self.angle_windows[idx],
                                                               self.ecc_windows[idx]]).flatten(2, 3)
                # need to do the indexing because otherwise we can get some
                # divide by 0 issues. this doesn't mess up the forward pass,
                # but it does mess up the backwards pass
                pooled_x = (pooled_x[..., window_size_mask[idx]] /
                            self.window_sizes[idx][window_size_mask[idx]])
            else:
                pooled_x = []
                e = self.ecc_windows[idx]
                sizes = self.window_sizes[idx][window_size_mask[idx]].to(output_device)
                for i in range(self.num_devices):
                    a = self.angle_windows[(idx, i)]
                    val = torch.einsum('bchw,ahw,ehw->bcea',
                                       [x.to(a.device), a, e.to(a.device)]).to(output_device)
                    pooled_x.append(val.flatten(2, 3))
                pooled_x = torch.cat(pooled_x, -1)[..., window_size_mask[idx]] / sizes
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
        torch.cuda.empty_cache()
        if isinstance(x, dict):
            if list(x.values())[0].ndimension() != 4:
                raise Exception("PoolingWindows input must be 4d tensors or a dict of 4d tensors!"
                                " Unsqueeze until this is true!")
            if isinstance(self.angle_windows, list):
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
                tmp = {}
                for k, v in x.items():
                    e = self.ecc_windows[k[0]]
                    for i in range(self.num_devices):
                        a = self.angle_windows[(k[0], i)]
                        tmp[(k, i)] = torch.einsum('bchw,ahw,ehw->bceahw',
                                                   [v.to(a.device), a, e.to(a.device)]).flatten(2, 3)
                return tmp
        else:
            if x.ndimension() != 4:
                raise Exception("PoolingWindows input must be 4d tensors or a dict of 4d tensors!"
                                " Unsqueeze until this is true!")
            if isinstance(self.angle_windows, list):
                return torch.einsum('bchw,ahw,ehw->bceahw', [x.to(self.angle_windows[0].device),
                                                             self.angle_windows[idx],
                                                             self.ecc_windows[idx]]).flatten(2, 3)
            else:
                tmp = []
                e = self.ecc_windows[idx]
                for i in range(self.num_devices):
                    a = self.angle_windows[(idx, i)]
                    tmp.append(torch.einsum('bchw,ahw,ehw->bceahw',
                                            [x.to(a.device), a, e.to(a.device)]).flatten(2, 3))
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
        torch.cuda.empty_cache()
        window_size_mask = [w > 1 for w in self.window_sizes]
        if isinstance(windowed_x, dict):
            if isinstance(self.angle_windows, list):
                # one way to make this more general: figure out the size
                # of the tensors in x and in self.angle_windows, and
                # intelligently lookup which should be used.
                return dict((k, v.sum((-1, -2))[..., window_size_mask[k[0]]] /
                             self.window_sizes[k[0]][window_size_mask[k[0]]])
                            for k, v in windowed_x.items())
            else:
                tmp = {}
                orig_keys = set([k[0] for k in windowed_x])
                for k in orig_keys:
                    t = []
                    sizes = self.window_sizes[k[0]][window_size_mask[k[0]]].to(output_device)
                    for i in range(self.num_devices):
                        t.append(windowed_x[(k, i)].sum((-1, -2)).to(output_device))
                    tmp[k] = torch.cat(t, -1)[..., window_size_mask[k[0]]] / sizes
                return tmp
        else:
            if isinstance(self.angle_windows, list):
                return (windowed_x.sum((-1, -2))[..., window_size_mask[idx]] /
                        self.window_sizes[idx][window_size_mask[idx]])
            else:
                tmp = []
                sizes = self.window_sizes[idx][window_size_mask[idx]].to(output_device)
                for i, v in enumerate(windowed_x):
                    tmp.append(v.sum((-1, -2)).to(output_device))
                return torch.cat(tmp, -1)[..., window_size_mask[idx]] / sizes

    def project(self, pooled_x, idx=0, output_device=torch.device('cpu')):
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

        Returns
        -------
        x : dict or torch.Tensor
            4d tensor or dictionary of 4d tensors

        See also
        --------
        forward : the opposite of this, going from image to pooled
            values

        """
        torch.cuda.empty_cache()
        window_size_mask = [w > 1 for w in self.window_sizes]
        if isinstance(pooled_x, dict):
            if list(pooled_x.values())[0].ndimension() != 3:
                raise Exception("PoolingWindows input must be 3d tensors or a dict of 3d tensors!"
                                " Squeeze until this is true!")
            if isinstance(self.angle_windows, list):
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
                    expanded_v = torch.zeros((*v.shape[:2], *window_size_mask[window_key].shape))
                    expanded_v[..., window_size_mask[window_key]] = v
                    expanded_v = expanded_v.reshape((*v.shape[:2],
                                                     self.ecc_windows[window_key].shape[0],
                                                     self.angle_windows[window_key].shape[0]))
                    tmp[k] = torch.einsum('bcea,ahw,ehw->bchw',
                                          [expanded_v.to(self.angle_windows[0].device),
                                           self.angle_windows[window_key],
                                           self.ecc_windows[window_key]])
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
                    expanded_v = torch.zeros((*v.shape[:2], *window_size_mask[window_key].shape))
                    expanded_v[..., window_size_mask[window_key]] = v
                    expanded_v = expanded_v.reshape((*v.shape[:2],
                                                     self.ecc_windows[window_key].shape[0],
                                                     self.n_polar_windows))
                    e = self.ecc_windows[window_key]
                    for i in range(self.num_devices):
                        a = self.angle_windows[(window_key, i)]
                        d = expanded_v[..., i*num:(i+1)*num]
                        t.append(torch.einsum('bcea,ahw,ehw->bchw',
                                              [d.to(a.device), a, e.to(a.device)]).to(output_device))
                    tmp[k] = torch.cat(t, 0).sum(0)
                return tmp
        else:
            if pooled_x.ndimension() != 3:
                raise Exception("PoolingWindows input must be 3d tensors or a dict of 3d tensors!"
                                " Squeeze until this is true!")
            expanded_x = torch.zeros((*pooled_x.shape[:2], *window_size_mask[idx].shape))
            expanded_x = expanded_x.to(pooled_x.device)
            expanded_x[..., window_size_mask[idx]] = pooled_x
            expanded_x = expanded_x.reshape((*pooled_x.shape[:2], self.ecc_windows[idx].shape[0],
                                             self.n_polar_windows))
            if isinstance(self.angle_windows, list):
                return torch.einsum('bcea,ahw,ehw->bchw', [expanded_x.to(self.angle_windows[0].device),
                                                           self.angle_windows[idx],
                                                           self.ecc_windows[idx]])
            else:
                tmp = []
                num = int(np.ceil(self.n_polar_windows / self.num_devices))
                e = self.ecc_windows[idx]
                for i in range(self.num_devices):
                    a = self.angle_windows[(idx, i)]
                    d = expanded_x[..., i*num:(i+1)*num]
                    tmp.append(torch.einsum('bcea,ahw,ehw->bchw',
                                            [d.to(a.device), a, e.to(a.device)]).to(output_device))
                return torch.cat(tmp, 0).sum(0)

    def plot_windows(self, ax, contour_levels=[.5], colors='r', subset=True, windows_scale=0,
                     **kwargs):
        r"""plot the pooling windows on an image.

        This is just a simple little helper to plot the pooling windows
        on an existing axis. The use case is overlaying this on top of
        the image we're pooling (as returned by ``pyrtools.imshow``),
        and so we require an axis to be passed

        Any additional kwargs get passed to ``ax.contour``

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            The existing axis to plot the windows on
        contour_levels : array-like or int, optional
            The ``levels`` argument to pass to ``ax.contour``. From that
            documentation: "Determines the number and positions of the
            contour lines / regions. If an int ``n``, use ``n`` data
            intervals; i.e. draw ``n+1`` contour lines. The level
            heights are automatically chosen. If array-like, draw
            contour lines at the specified levels. The values must be in
            increasing order". ``[.5]`` (the default) is recommended for
            these windows.
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
        if isinstance(self.angle_windows, list):
            # attempt to not have all the windows in memory at once...
            if subset:
                angle_windows = self.angle_windows[windows_scale][:4]
            else:
                angle_windows = self.angle_windows[windows_scale]
            for a in angle_windows:
                windows = torch.einsum('hw,ehw->ehw', [a, self.ecc_windows[windows_scale]])
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
                    windows = torch.einsum('hw,ehw->ehw', [a, self.ecc_windows[windows_scale].to(a.device)])
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

    def plot_window_widths(self, units='degrees', scale_num=0, figsize=(5, 5), jitter=.25):
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
        fig, ax = plt.subplots(1, 1, figsize=figsize)
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

    def plot_window_areas(self, units='degrees', scale_num=0, figsize=(5, 5)):
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
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sizes = {'full': 5, 'half': 10, 'top': 15}
        for height in ['top', 'half', 'full']:
            m, s, b = ax.stem(central_ecc, data[height], 'C0', 'C0.', label=height,
                              use_line_collection=True)
            m.set(markersize=sizes[height])
        ax.set_ylabel('Window area (%s)' % units)
        ax.set_xlabel('Window central eccentricity (%s)' % units)
        ax.legend(loc='upper left')
        return fig
