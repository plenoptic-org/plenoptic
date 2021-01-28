"""functions to perform spatial pooling, as seen in Freeman and Simoncelli, 2011

In addition the raised-cosine windows used in that paper, we also
provide support for an alternative window construction:
Gaussians. They're laid out in the same fashion as the raised-cosine
windows, but are wider and have values everywhere (whereas the
raised-cosine windows are clipped so that they're zero for most of the
image). Using the raised-cosine windows led to issues with aliasing in
metamer synthesis, visible as ringing, with the PooledV1
model, because of the interactions between the windows and the steerable
pyramid filters.

The Gaussian windows don't have these problems, but require more windows to
evenly tile the image in the radial direction (and thus PoolingWindows.forward
will take more memory and more time). Note as well that, whereas the max
amplitude of the raised-cosine windows is always 1 (for all transition region
widths), the Gaussian windows will have their max amplitude scaled down as
their standard deviation increases; as the standard deviation increases, the
windows overlap more, so that the number of windows a given pixel lies in
increases and thus the weighting in each of them needs to decrease in order to
make sure the sum across all windows is still 1 for every pixel. The Gaussian
windows will always intersect at x=.5, but the interpretation of this depends
on its standard deviation. For Gaussian windows, we recommend (and only
support) a standard deviation of 1, so that each window intersects at half a
standard deviation.

pooling_windows.py contains the PoolingWindows class, which uses most of these
functions

"""
import math
import re
import torch
import numpy as np
from ...tools.data import polar_angle, polar_radius

# see docstring of gaussian function for explanation of this constant
GAUSSIAN_SUM = 2 * 1.753314144021452772415339526931980189073725635759454989253 - 1


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
                                     std_dev=None):
    r"""calculate and return the window spacing for the eccentricity windows

    this is the :math:`w_e` term in equation 11 of the paper's online
    methods (referred to as the window width), which we also refer to as
    the radial spacing. Note that we take exactly one of ``n_windows``
    or ``scaling`` in order to determine this value.

    If scaling is set, ``min_ecc`` and ``max_ecc`` are ignored (the window
    width only depends on scaling, not also on the range of eccentricities;
    they only matter when determining the width using ``n_windows``)

    For both cosine and gaussian windows, this is the distance between
    the peaks of the windows. For cosine windows, this is also the same
    as the windows' widths, but gausian windows' widths are
    approximately ``window_spacing * std_dev * 3`` (since they're
    Gaussian, 99.73% of their mass lie within 3 standard deviations, but
    the Gaussians are technically infinite); but remember that these
    values are in log space not linear.

    Parameters
    ----------
    min_ecc : float, optional
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
    std_dev : float or None, optional
        The standard deviation of the Gaussian window. If this is set,
        we compute the scaling value for the Gaussian windows instead of
        for the cosine ones.

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

    """
    if scaling is not None:
        if std_dev is None:
            x_half_max = .5
        else:
            x_half_max = std_dev * np.sqrt(2 * np.log(2))
        spacing = np.log((scaling + np.sqrt(scaling**2+4))/2) / x_half_max
    elif n_windows is not None:
        spacing = (np.log(max_ecc) - np.log(min_ecc)) / n_windows
    else:
        raise Exception("Exactly one of n_windows or scaling must be set!")
    return spacing


def calc_eccentricity_n_windows(window_spacing, min_ecc=.5, max_ecc=15, std_dev=None):
    r"""calculate and return the number of eccentricity windows

    this is the :math:`N_e` term in equation 11 of the paper's online
    method, which we've rearranged in order to get this.

    Parameters
    ----------
    window_spacing : `float`
        The spacing of the log-eccentricity windows.
    min_ecc : float, optional
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

    Returns
    -------
    n_windows : `float`
        The number of log-eccentricity windows we create.

    """
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


def calc_scaling(n_windows, min_ecc=.5, max_ecc=15, std_dev=None):
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
    std_dev : float or None, optional
        The standard deviation fo the Gaussian window. If this is set,
        we compute the scaling value for the Gaussian windows instead of
        for the cosine ones.

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
    ``calc_windows_eccentricity`` for :math:`e_c`; we simplify
    it away in the calculation above.

    """
    if std_dev is not None:
        x_half_max = std_dev * np.sqrt(2 * np.log(2))
    else:
        x_half_max = .5
    window_spacing = calc_eccentricity_window_spacing(min_ecc, max_ecc,
                                                      n_windows)
    return np.exp(x_half_max*window_spacing) - np.exp(-x_half_max*window_spacing)


def calc_windows_eccentricity(ecc_type, n_windows, window_spacing, min_ecc=.5,
                              transition_region_width=.5, std_dev=None):
    r"""calculate a relevant eccentricity for each radial window

    These are the values :math:`e_c`, as referred to in ``calc_scaling``
    (for each of the n windows)

    Parameters
    ----------
    ecc_type : {'min', 'central', 'max', '{n}std'}
        Which eccentricity you want to calculate: the minimum one where
        x=-(1+t)/2, the central one where x=0, or the maximum one where
        x=(1+t)/2. if std_dev is set, minimum and maximum are +/- 3
        std_dev. if '{n}std' (where n is a positive or negative
        integer), then we return the eccentricity at that many std_dev
        away from center (only std_dev is set).
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

    Returns
    -------
    eccentricity : np.ndarray
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

    """
    if ecc_type not in ['min', 'max', 'central'] and not ecc_type.endswith('std'):
        raise Exception(f"Don't know how to handle ecc_type {ecc_type}")
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
    elif ecc_type.endswith('std'):
        if std_dev is None:
            raise Exception(f"std_dev must be set if ecc_type == {ecc_type}")
        else:
            n = int(re.findall('([-0-9]+)std', ecc_type)[0])
            ecc = [(np.exp(n*std_dev*window_spacing) * min_ecc *
                    np.exp(window_spacing * (i+1))) for i in np.arange(np.ceil(n_windows))]
    return np.array(ecc)


def calc_window_widths_actual(angular_window_spacing, radial_window_spacing,
                              min_ecc=.5, max_ecc=15, window_type='cosine',
                              transition_region_width=.5, std_dev=None):
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

    Returns
    -------
    radial_top_width : np.ndarray
        The width of the flat-top region of the windows in the radial
        direction (each value corresponds to a different ring of
        windows, from the fovea to the periphery).
    radial_full_width : np.ndarray
        The full width of the windows in the radial direction (each
        value corresponds to a different ring of windows, from the fovea
        to the periphery).
    angular_top_width : np.ndarray
        The width of the flat-top region of the windows in the angular
        direction (each value corresponds to a different ring of
        windows, from the fovea to the periphery).
    angular_full_width : np.ndarray
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

    """
    n_radial_windows = np.ceil(calc_eccentricity_n_windows(radial_window_spacing, min_ecc,
                                                           max_ecc, std_dev))
    window_central_eccentricities = calc_windows_eccentricity('central', n_radial_windows,
                                                              radial_window_spacing, min_ecc)
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
        radial_full = [min_ecc*(np.exp(radial_window_spacing*(3*std_dev+n+1)) -
                                np.exp(radial_window_spacing*(-3*std_dev+n+1)))
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
    windows : torch.Tensor
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
                             std_dev=None, device=None, linear=False):
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

    Notes
    -----

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
    device : str or torch.device
        the device to create this tensor on
    linear : bool, optional
        if True, create linear windows instead of log-spaced. NOTE This is only
        for playing around with, it really is not supported or a good idea
        because the angular windows still grow in size as a function of
        eccentricity and none of the calculations will work.

    Returns
    -------
    windows : torch.Tensor
        A 3d tensor containing the (2d) log-eccentricity
        windows. Windows will be indexed along the first dimension. If
        resolution was an int, then this will be a 2d array containing
        the 1d polar angle windows

    Notes
    -----
    Equation 11 from the online methods of [1]_.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    if not linear:
        log_func = torch.log
    else:
        log_func = lambda x: x
    if std_dev is not None and std_dev != 1:
        raise Exception("Only std_dev=1 is supported (not sure if Gaussian "
                        "windows will uniformly tile image otherwise!)")
    if window_spacing is None:
        window_spacing = calc_eccentricity_window_spacing(min_ecc, max_ecc, n_windows,
                                                          std_dev=std_dev)
    n_windows = calc_eccentricity_n_windows(window_spacing, min_ecc, max_ecc*np.sqrt(2), std_dev)
    shift_arg = (log_func(torch.tensor(min_ecc, dtype=torch.float32)) + window_spacing * torch.arange(1, math.ceil(n_windows)+1, device=device)).unsqueeze(-1)
    if hasattr(resolution, '__iter__') and len(resolution) == 2:
        ecc = log_func(polar_radius(resolution, device=device) / calc_deg_to_pix(resolution, max_ecc)).unsqueeze(0)
        shift_arg = shift_arg.unsqueeze(-1)
    else:
        ecc = log_func(torch.linspace(0, max_ecc, resolution, device=device)).unsqueeze(0)
    ecc = (ecc - shift_arg) / window_spacing
    if window_type == 'gaussian':
        windows = gaussian(ecc, std_dev)
    elif window_type == 'cosine':
        windows = mother_window(ecc, transition_region_width)
    return torch.stack([w for w in windows if (w != 0).any()])


def create_pooling_windows(scaling, resolution, min_eccentricity=.5,
                           max_eccentricity=15,
                           radial_to_circumferential_ratio=2,
                           window_type='cosine', transition_region_width=.5,
                           std_dev=None, device=None):
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
    window_type : {'cosine', 'gaussian'}
        Whether to use the raised cosine function from [1]_ or a Gaussian that
        has approximately the same structure. If cosine,
        ``transition_region_width`` must be set; if gaussian, then ``std_dev``
        must be set.
    transition_region_width : `float` or None, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    std_dev : float or None, optional
        The standard deviation of the Gaussian window. WARNING -- if
        this is too small (say < 3/4), then the windows won't tile
        correctly. So we only support std_dev=1 for now.
    device : str or torch.device
        the device to create these tensors on

    Returns
    -------
    angle_windows : torch.Tensor or dict
        The 3d tensor of 2d polar angle windows. Its shape will be
        ``(n_angle_windows, *resolution)``, where the number of windows
        is inferred in this function based on the values of ``scaling``
        and ``radial_to_circumferential_width``.
    ecc_windows : torch.Tensor or dict
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
    ecc_window_spacing = calc_eccentricity_window_spacing(min_eccentricity, max_eccentricity,
                                                          scaling=scaling, std_dev=std_dev)
    n_polar_windows = calc_angular_n_windows(ecc_window_spacing / radial_to_circumferential_ratio)
    # we want to set the number of polar windows where the ratio of
    # widths is approximately what the user specified. the constraint
    # that it's an integer is more important
    n_polar_windows = int(round(n_polar_windows))
    angle_tensor = polar_angle_windows(n_polar_windows, resolution,
                                       window_type,
                                       transition_region_width=transition_region_width,
                                       std_dev=std_dev, device=device)
    ecc_tensor = log_eccentricity_windows(resolution, None, ecc_window_spacing,
                                          min_eccentricity,
                                          max_eccentricity,
                                          window_type,
                                          std_dev=std_dev,
                                          transition_region_width=transition_region_width,
                                          device=device)
    return angle_tensor, ecc_tensor


def normalize_windows(angle_windows, ecc_windows, window_eccentricity, scale=0):
    r"""normalize windows to have L1-norm of 1

    we calculate the L1-norm of single windows (that is, product of
    eccentricity and angular windows) for all angles, one middling
    eccentricity (third of the way thorugh), then average across angles
    (because of alignment with pixel grid, L1-norm will vary somewhat
    across angles).

    L1-norm scales linearly with area, which is proportional to the width in
    the angular direction times the width in the radial direction. The angular
    width grows linearly with eccentricity, while the radial width grows with
    the reciprocal of the derivative of our scaling function (that's log(ecc)
    for gaussian windows). so we use that product to scale it for the different
    windows. only eccentricity windows is normalized (don't need to divide
    both).

    Parameters
    ----------
    angle_windows : dict
        dictionary containing the angular windows
    ecc_windows : dict
        dictionary containing the eccentricity windows
    window_eccentricity : array_like
        array containing the eccentricity for each window that defines
        their location relative to each other (and so can be in either
        pixels or degrees). this is used to determine how to scale the
        L1-norm. It should probably be the central eccentricity, but it
        should not contain any zeros.
    scale : int, optional
        which scale to calculate norm for and modify

    Returns
    -------
    ecc_windows : dict
        the normalized ecc_windows. only ``scale`` is modified
    scale_factor : torch.Tensor
        the scale_factor used to normalize eccentricity windows at this
        scale (as a 3d tensor, number of eccentricity windows by 1 by
        1). stored by ``PoolingWindows`` object so we can undo it for
        ``project()`` or plotting purposes

    """
    # pick some window with a middling eccentricity
    n = ecc_windows[scale].shape[0] // 3
    # get the l1 norm of a single window
    w = torch.einsum('ahw,hw->ahw', angle_windows[scale], ecc_windows[scale][n])
    l1 = torch.norm(w, 1, (-1, -2))
    l1 = l1.mean(0)
    # the l1 norm grows with the area of the windows; the radial
    # direction width grows with the reciprocal of the derivative of
    # log(ecc), which is ecc, and the angular direction width grows
    # with the eccentricity as well. so l1 norm grows with the
    # eccentricity squared
    deriv = torch.tensor(window_eccentricity**2, dtype=torch.float32)
    deriv_scaled = deriv / deriv[n]
    scale_factor = 1 / (deriv_scaled * l1).to(torch.float32)
    while scale_factor.ndim < 3:
        scale_factor = scale_factor.unsqueeze(-1)
    # there's a chance we'll have more windows accounted for in
    # scale factor then we actually made (because we calculate
    # details for windows that go out farther, just in case). if
    # that's so, drop the extra scale factor
    if len(scale_factor) > len(ecc_windows[scale]):
        scale_factor = scale_factor[:len(ecc_windows[scale])]
    ecc_windows[scale] = ecc_windows[scale] * scale_factor
    return ecc_windows, scale_factor
