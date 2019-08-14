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
    For ``x`` values outside the function's domain, we return ``np.nan``
    (not 0)

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
    `float`
        The value of the window at each value of ``x``.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature
       Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    if transition_region_width > 1 or transition_region_width < 0:
        raise Exception("transition_region_width must lie between 0 and 1!")
    if hasattr(x, '__iter__'):
        return np.array([mother_window(i, transition_region_width) for i in x])
    if -(1 + transition_region_width) / 2 < x <= (transition_region_width - 1) / 2:
        return np.cos(np.pi/2 * ((x - (transition_region_width-1)/2) / transition_region_width))**2
    elif (transition_region_width - 1) / 2 < x <= (1 - transition_region_width) / 2:
        return 1
    elif (1 - transition_region_width) / 2 < x <= (1 + transition_region_width) / 2:
        return (-np.cos(np.pi/2 * ((x - (1+transition_region_width)/2) /
                                   transition_region_width))**2 + 1)
    else:
        return np.nan


def polar_angle_windows(n_windows, theta_n_steps=100, transition_region_width=.5):
    r"""Create polar angle windows

    We require an integer number of windows placed between 0 and 2 pi.

    Notes
    -----
    Equation 10 from the online methods of [1]_.

    Parameters
    ----------
    n_windows : `int`
        The number of polar angle windows we create.
    theta_n_steps : `int`, optional
        The number of times we sample theta.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.

    Returns
    -------
    theta : `np.array`
        A 1d array containing the samples of the polar angle:
        ``np.linspace(0, 2*np.pi, theta_n_steps)``
    windows : `np.array`
        A 2d array containing the (1d) polar angle windows. Windows will
        be indexed along the first dimension.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    def remap_theta(x):
        if x > np.pi:
            return x - 2*np.pi
        else:
            return x
    theta = np.linspace(0, 2*np.pi, theta_n_steps)
    if int(n_windows) != n_windows:
        raise Exception("n_windows must be an integer!")
    if n_windows == 1:
        raise Exception("We cannot handle one window correctly!")
    # this is `w_\theta` in the paper
    window_width = calc_angular_window_width(n_windows)
    windows = []
    for n in range(int(n_windows)):
        if n == 0:
            tmp_theta = np.array([remap_theta(t) for t in np.linspace(0, 2*np.pi, theta_n_steps)])
        else:
            tmp_theta = theta
        mother_window_arg = ((tmp_theta - (window_width * n +
                                           (window_width * (1-transition_region_width)) / 2)) /
                             window_width)
        windows.append(mother_window(mother_window_arg, transition_region_width))
    windows = [i for i in windows if not np.isnan(i).all()]
    return theta, np.array(windows)


def log_eccentricity_windows(n_windows=None, window_width=None, min_ecc=.5, max_ecc=15,
                             ecc_n_steps=100, transition_region_width=.5):
    r"""Create log eccentricity windows

    Note that exactly one of ``n_windows`` or ``window_width`` must be
    set.

    Notes
    -----
    Equation 11 from the online methods of [1]_.

    Parameters
    ----------
    n_windows : `float` or `None`
        The number of log-eccentricity windows we create. ``n_windows``
        xor ``window_width`` must be set.
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
    ecc_n_steps : `int`, optional
        The number of times we sample the eccentricity.
    transition_region_width : `float`
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.

    Returns
    -------
    eccentricity : `np.array`
        A 1d array containing the samples of eccentricity:
        ``np.linspace(0, max_ecc, ecc_n_steps)`` (note that the windows
        start having non-NaN values at ``min_ecc`` degrees, but are
        sampled all the way down to 0 degrees)
    windows : `np.array`
        A 2d array containing the (1d) log-eccentricity windows. Windows
        will be indexed along the first dimension.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    ecc = np.linspace(0, max_ecc, ecc_n_steps)
    if window_width is None:
        window_width = calc_eccentricity_window_width(min_ecc, max_ecc, n_windows)
    else:
        n_windows = calc_eccentricity_n_windows(window_width, min_ecc, max_ecc)
    windows = []
    for n in range(math.ceil(n_windows)):
        mother_window_arg = (np.log(ecc) - (np.log(min_ecc) + window_width * (n+1))) / window_width
        windows.append(mother_window(mother_window_arg, transition_region_width))
    windows = [i for i in windows if not np.isnan(i).all()]
    return ecc, np.array(windows)


def create_pooling_windows(scaling, min_eccentricity=.5, max_eccentricity=15,
                           radial_to_circumferential_ratio=2, transition_region_width=.5,
                           theta_n_steps=1000, ecc_n_steps=1000, flatten=True):
    r"""Create 2d pooling windows (log-eccentricity by polar angle) that span the visual field

    This creates the pooling windows that we use to average image
    statistics for metamer generation as done in [1]_. This is returned
    as a 3d torch tensor for further use with a model, and will also
    return the theta and eccentricity grids necessary for plotting, if
    desired.

    Note that this is returned in polar coordinates and so if you wish
    to apply it to an image in rectangular coordinates, you'll need to
    make use of pyrtools's ``project_polar_to_cartesian`` function (see
    Examples section)

    Parameters
    ----------
    scaling : `float` or `None`.
        The ratio of the eccentricity window's radial full-width at
        half-maximum to eccentricity (see the `calc_scaling` function).
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
    theta_n_steps : `int`, optional
        The number of times we sample theta.
    ecc_n_steps : `int`, optional
        The number of times we sample the eccentricity.
    flatten : bool, optional
        If True, the returned ``windows`` Tensor will be 3d, with
        different windows indexed along the first dimension. If False,
        it will be 4d, with the first dimension corresponding to
        different polar angle windows and the second to different
        eccentricity bands.

    Returns
    -------
    windows : `torch.tensor`
        The 3d or 4d array of 2d windows (in polar angle and
        eccentricity). Whether its 3d or 4d depends on the value of the
        ``flatten`` arg. If True, it will be 3d, with separate windows
        indexed along the first dimension and the shape
        ``(n_polar_windows*n_ecc_windows, ecc_n_steps,
        theta_n_steps)``. If False, it will be 4d, with the shape
        ``(n_polar_windows, n_ecc_windows, ecc_n_steps, theta_n_steps)``
        (where the number of windows is inferred in this function based
        on the values of ``scaling`` and
        ``radial_to_circumferential_width``)
    theta_grid : `torch.tensor`
        The 2d array of polar angle values, as returned by
        meshgrid. Will have the shape ``(ecc_n_steps, theta_n_steps)``
    eccentricity_grid : `torch.tensor`
        The 2d array of eccentricity values, as returned by
        meshgrid. Will have the shape ``(ecc_n_steps, theta_n_steps)``

    Examples
    --------
    To use, simply call with the desired scaling (for the version seen
    in the paper, don't change any of the default arguments; compare
    this image to the right side of Supplementary Figure 1C)

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import plenoptic as po
       windows, theta, ecc = po.simul.create_pooling_windows(.87, theta_n_steps=256,
                                                             ecc_n_steps=256)
       fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
       for w in windows:
           ax.contour(theta, ecc, w, [.5])
       plt.show()

    If you wish to convert this to the Cartesian coordinates, in order
    to apply to an image, for example, you must use pyrtools's
    ``project_polar_to_cartesian`` function

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyrtools as pt
       import plenoptic as po
       windows, theta, ecc = po.simul.create_pooling_windows(.87, theta_n_steps=256,
                                                             ecc_n_steps=256)
       windows_cart = [pt.project_polar_to_cartesian(w) for w in windows]
       pt.imshow(windows_cart, col_wrap=8, zoom=.5)
       plt.show()

    Notes
    -----
    These create the pooling windows, as seen in Supplementary Figure
    1C, in [1]_.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    ecc_window_width = calc_eccentricity_window_width(min_eccentricity, max_eccentricity,
                                                      scaling=scaling)
    n_polar_windows = calc_angular_n_windows(ecc_window_width / radial_to_circumferential_ratio)
    # we want to set the number of polar windows where the ratio of widths is approximately what
    # the user specified. the constraint that it's an integer is more important
    theta, angle_tensor = polar_angle_windows(round(n_polar_windows), theta_n_steps,
                                              transition_region_width)
    angle_tensor = torch.tensor(angle_tensor, dtype=torch.float32)
    ecc, ecc_tensor = log_eccentricity_windows(None, ecc_window_width, min_eccentricity,
                                               max_eccentricity, ecc_n_steps,
                                               transition_region_width=transition_region_width)
    ecc_tensor = torch.tensor(ecc_tensor, dtype=torch.float32)
    windows_tensor = torch.einsum('ik,jl->ijkl', [ecc_tensor, angle_tensor])
    if flatten:
        # just flatten the first two dimensions, so its now 3d instead of 4d
        windows_tensor = windows_tensor.flatten(0, 1)
    theta_grid, ecc_grid = np.meshgrid(theta, ecc)
    theta_grid = torch.tensor(theta_grid, dtype=torch.float32)
    ecc_grid = torch.tensor(ecc_grid, dtype=torch.float32)
    return windows_tensor, theta_grid, ecc_grid


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
    windows : list
        A list of 3d tensors containing the pooling windows in which the
        model parameters are averaged. Each entry in the list
        corresponds to a different scale and thus is a different size
        (though they should all have the same number of windows)
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

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5, max_eccentricity=15, num_scales=1,
                 transition_region_width=.5, cache_dir=None):
        super().__init__()
        if len(img_res) != 2:
            raise Exception("img_res must be 2d!")
        if img_res[0] != img_res[1]:
            warnings.warn("The windows must be created square initially, so we'll do that and then"
                          " crop them down to the proper size")
            window_res = 2*[np.max(img_res)]
        else:
            window_res = img_res
        # We construct the windows in polar space and then interpolate
        # them back into cartesian. therefore, we need to construct them
        # further out and then crop down. For example, if we want to
        # create windows for a 256 x 256 image that runs out to 15
        # degrees eccentricity, setting max_eccentricity to 15 and the
        # number of theta and eccentricity steps to 256 each would mean
        # we have no idea what to do in the far corners; we'd only
        # properly create windows that fill a circle with diameter
        # 256. so we create windows that go out to the furthest possible
        # distance, sqrt(2) times the width of the image. we then
        # similarly need to increase the max_eccentricity for this by
        # sqrt(2) for the same reason
        window_res = [int(np.ceil(i*np.sqrt(2))) for i in window_res]
        self.scaling = scaling
        self.transition_region_width = transition_region_width
        self.min_eccentricity = min_eccentricity
        self.max_eccentricity = max_eccentricity
        self.img_res = img_res
        self.windows = []
        if cache_dir is not None:
            self.cache_dir = op.expanduser(cache_dir)
            cache_path_template = op.join(self.cache_dir, "scaling-{scaling}_size-{img_res}_e0-"
                                          "{min_eccentricity}_em-{max_eccentricity}_t-{transition_"
                                          "region_width}.pt")
        else:
            self.cache_dir = cache_dir
        self.cache_paths = []
        self.calculated_min_eccentricity_degrees = []
        self.calculated_min_eccentricity_pixels = []
        self._window_sizes(num_scales)
        self.state_dict_reduced = {'scaling': scaling, 'img_res': img_res,
                                   'min_eccentricity': min_eccentricity,
                                   'max_eccentricity': max_eccentricity,
                                   'transition_region_width': transition_region_width,
                                   'cache_dir': self.cache_dir}
        for i in range(num_scales):
            scaled_window_res = [np.ceil(j / 2**i) for j in window_res]
            scaled_img_res = [np.ceil(j / 2**i) for j in img_res]
            # the first value returned is the min_ecc in degrees, the
            # second is in pixels
            min_ecc, min_ecc_pix = calc_min_eccentricity(scaling, scaled_window_res,
                                                         max_eccentricity)
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
            else:
                min_ecc = self.min_eccentricity
            windows = None
            if cache_dir is not None:
                self.cache_paths.append(cache_path_template.format(
                    scaling=scaling, min_eccentricity=min_ecc, max_eccentricity=max_eccentricity,
                    img_res=','.join([str(i) for i in scaled_img_res]),
                    transition_region_width=transition_region_width))
                if op.exists(self.cache_paths[-1]):
                    warnings.warn("Loading windows from cache: %s" % self.cache_paths[-1])
                    windows = torch.load(self.cache_paths[-1])
            if windows is None:
                windows, theta, ecc = create_pooling_windows(
                    # for why we're multiplying max_eccentricity by sqrt(2),
                    # see the long comment above window_res
                    scaling, min_ecc, max_eccentricity*np.sqrt(2),
                    ecc_n_steps=scaled_window_res[0], theta_n_steps=scaled_window_res[1],
                    transition_region_width=transition_region_width)

                # need this to be float32 so we can divide the representation by it.
                windows = torch.tensor([pt.project_polar_to_cartesian(w) for w in windows],
                                       dtype=torch.float32)
                if img_res[0] != window_res[0]:
                    slice_vals = PoolingWindows._get_slice_vals(scaled_window_res[0], scaled_img_res[0])
                    windows = windows[..., slice_vals[0]:slice_vals[1], :]
                if img_res[1] != window_res[1]:
                    slice_vals = PoolingWindows._get_slice_vals(scaled_window_res[1], scaled_img_res[1])
                    windows = windows[..., slice_vals[0]:slice_vals[1]]
                # this gets rid of the windows that are off the edge of the
                # image
                windows = windows[windows.sum((-1, -2)) > 1]
                if cache_dir is not None:
                    warnings.warn("Saving windows to cache: %s" % self.cache_paths[-1])
                    torch.save(windows, self.cache_paths[-1])
            self.windows.append(windows)

    def _window_sizes(self, num_scales=1):
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
        ecc_window_width = calc_eccentricity_window_width(self.min_eccentricity,
                                                          self.max_eccentricity,
                                                          scaling=self.scaling)
        self.n_polar_windows = int(round(calc_angular_n_windows(ecc_window_width / 2)))
        angular_window_width = calc_angular_window_width(self.n_polar_windows)
        window_widths = calc_window_widths_actual(angular_window_width, ecc_window_width,
                                                  self.min_eccentricity, self.max_eccentricity,
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
        for i in range(num_scales):
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
        for i, w in enumerate(self.windows):
            self.windows[i] = w.to(*args, **kwargs)
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
        idx)``

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

        """
        windowed_x = self.window(x, idx)
        return self.pool(windowed_x, idx)

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
        if isinstance(x, dict):
            if list(x.values())[0].ndimension() != 4:
                raise Exception("PoolingWindows input must be 4d tensors or a dict of 4d tensors!"
                                " Unsqueeze until this is true!")
            # one way to make this more general: figure out the size of
            # the tensors in x and in self.windows, and intelligently
            # lookup which should be used.
            return dict((k, torch.einsum('ijkl,wkl->ijwkl', [v, self.windows[k[0]]]))
                        for k, v in x.items())
        else:
            if x.ndimension() != 4:
                raise Exception("PoolingWindows input must be 4d tensors or a dict of 4d tensors!"
                                " Unsqueeze until this is true!")
            return torch.einsum('ijkl,wkl->ijwkl', [x, self.windows[idx]])

    def pool(self, windowed_x, idx=0):
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
            ``x`` is a tensor

        Returns
        -------
        pooled_x : dict or torch.Tensor
            Same type as ``windowed_x``, see above for how it's created.

        See also
        --------
        window : window the input
        forward : perform the windowing and pooling simultaneously

        """
        if isinstance(windowed_x, dict):
            # one way to make this more general: figure out the size of
            # the tensors in x and in self.windows, and intelligently
            # lookup which should be used.
            return dict((k, v.sum((-1, -2)) / self.windows[k[0]].sum((-1, -2)))
                        for k, v in windowed_x.items())
        else:
            return windowed_x.sum((-1, -2)) / self.windows[idx].sum((-1, -2))

    def plot_windows(self, ax, contour_levels=[.5], colors='r', **kwargs):
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

        Returns
        -------
        ax : matplotlib.pyplot.axis
            The axis with the windows

        """
        for w in self.windows[0]:
            ax.contour(w.detach(), contour_levels, colors=colors, **kwargs)
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
