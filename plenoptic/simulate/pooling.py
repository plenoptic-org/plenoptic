"""functions to perform spatial pooling, as seen in Freeman and Simoncelli, 2011

"""
import math
import torch
import numpy as np


def calc_polar_window_width(n_windows):
    r"""calculate and return the window width for the polar windows

    this is the :math:`w_{\theta }` term in equation 10 of the paper's online methods

    Parameters
    ----------
    n_windows : `float`
        The number of windows to pack into 2 pi. Note that we don't require it to be an integer
        here, but the code that makes use of this does.

    Returns
    -------
    window_width : `float`
        The width of the polar angle windows.
    """
    return (2*np.pi) / n_windows


def calc_polar_n_windows(window_width):
    r"""calculate and return the number of polar windows

    this is the :math:`N_{\theta }` term in equation 10 of the paper's online method, which we've
    rearranged in order to get this.

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

    this is the :math:`w_e` term in equation 11 of the paper's online methods. Note that we take
    exactly one of ``n_windows`` or ``scaling`` in order to determine this value.

    Parameters
    ----------
    min_ecc : `float`
        The minimum eccentricity, the eccentricity below which we do not compute pooling windows
        (in degrees). Parameter :math:`e_0` in equation 11 of the online methods.
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in degrees). Parameter :math:`e_r`
        in equation 11 of the online methods.
    n_windows : `float` or `None`
        The number of log-eccentricity windows we create. ``n_windows`` xor ``scaling`` must be
        set.
    scaling : `float` or `None`.
        The ratio of the eccentricity window's radial full-width at half-maximum to
        eccentricity (see the ``calc_scaling`` function). ``n_windows`` xor ``scaling`` must be
        set.

    Returns
    -------
    window_width : `float`
        The width of the log-eccentricity windows.

    Notes
    -----
    No equation was given in the paper to calculate the window width, :math:`w_e` from the scaling,
    :math:`s`, so we derived it ourselves. We start with the final equation for the scaling, given
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

    The window width is strictly positive, so we only return the positive quadratic root (the one
    with plus in the numerator).

    """
    if scaling is not None:
        return np.log((scaling**2+2+scaling * np.sqrt(scaling**2+4))/2)
    elif n_windows is not None:
        return (np.log(max_ecc) - np.log(min_ecc)) / n_windows
    else:
        raise Exception("Exactly one of n_windows or scaling must be set!")


def calc_eccentricity_n_windows(window_width, min_ecc=.5, max_ecc=15):
    r"""calculate and return the number of eccentricity windows

    this is the :math:`N_e` term in equation 11 of the paper's online method, which we've
    rearranged in order to get this.

    Parameters
    ----------
    window_width : `float`
        The width of the log-eccentricity windows.
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not compute pooling windows
        (in degrees). Parameter :math:`e_0` in equation 11 of the online methods.
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in degrees). Parameter :math:`e_r`
        in equation 11 of the online methods.

    Returns
    -------
    n_windows : `float`
        The number of log-eccentricity windows we create.
    """
    return (np.log(max_ecc) - np.log(min_ecc)) / window_width


def calc_scaling(n_windows, min_ecc=.5, max_ecc=15):
    r"""calculate and return the scaling value, as reported in the paper

    Scaling is the ratio of the eccentricity window's radial full-width at half-maximum to
    eccentricity. For eccentricity, we use the window's "central eccentricity", the one where the
    input to the mother window (:math:`x` in equation 9 in the online methods) is 0.

    Parameters
    ----------
    n_windows : `float`
        The number of log-eccentricity windows we create.
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not compute pooling windows
        (in degrees). Parameter :math:`e_0` in equation 11 of the online methods.
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in degrees). Parameter :math:`e_r`
        in equation 11 of the online methods.

    Returns
    -------
    scaling : `float`.
        The ratio of the eccentricity window's radial full-width at half-maximum to
        eccentricity

    Notes
    -----
    No equation for the scaling, :math:`s`, was included in the paper, so we derived this
    ourselves. To start, we note that the mother window equation (equation 9) reaches its half-max
    (.5) at :math:`x=\pm .5`, and that, as above, we treat :math:`x=0` as the
    central eccentricity of the window. Then we must solve for these, using the values given within
    the parenthese in equation 11 as the value for :math:`x`, and take their ratios.

    Full-width half-maximum, :math:`W`, the difference between the two values of :math:`e_h`:

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

    """
    window_width = (np.log(max_ecc) - np.log(min_ecc)) / n_windows
    return np.exp(.5*window_width) - np.exp(-.5*window_width)


def mother_window(x, transition_region_width=.5):
    r"""Raised cosine 'mother' window function

    Used to give the weighting in each direction for the spatial pooling performed during the
    construction of visual metamers

    Notes
    -----
    For ``x`` values outside the function's domain, we return ``np.nan`` (not 0)

    Equation 9 from the online methods of [1]_.

    Parameters
    ----------
    x : `float` or `array_like`
        The distance in a direction
    transition_region_width : `float`
        The width of the transition region, parameter :math:`t` in equation 9 from the online
        methods.

    Returns
    -------
    `float`
        The value of the window at each value of ``x``.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature
       Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
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
        The width of the transition region, parameter :math:`t` in equation 9 from the online
        methods.

    Returns
    -------
    theta : `np.array`
        A 1d array containing the samples of the polar angle: ``np.linspace(0, 2*np.pi,
        theta_n_steps)``
    windows : `np.array`
        A 2d array containing the (1d) polar angle windows. Windows will be indexed along the
        first dimension.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature
       Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889

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
    window_width = calc_polar_window_width(n_windows)
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

    Note that exactly one of ``n_windows`` or ``window_width`` must be set.

    Notes
    -----
    Equation 11 from the online methods of [1]_.

    Parameters
    ----------
    n_windows : `float` or `None`
        The number of log-eccentricity windows we create. ``n_windows`` xor ``window_width`` must
        be set.
    window_width : `float` or `None`
        The width of the log-eccentricity windows. ``n_windows`` xor ``window_width`` must be set.
    min_ecc : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not compute pooling windows
        (in degrees). Parameter :math:`e_0` in equation 11 of the online methods.
    max_ecc : `float`, optional
        The maximum eccentricity, the outer radius of the image (in degrees). Parameter :math:`e_r`
        in equation 11 of the online methods.
    ecc_n_steps : `int`, optional
        The number of times we sample the eccentricity.
    transition_region_width : `float`
        The width of the transition region, parameter :math:`t` in equation 9 from the online
        methods.

    Returns
    -------
    eccentricity : `np.array`
        A 1d array containing the samples of eccentricity: ``np.linspace(0, max_ecc, ecc_n_steps)``
        (note that the windows start having non-NaN values at ``min_ecc`` degrees, but are sampled
        all the way down to 0 degrees)
    windows : `np.array`
        A 2d array containing the (1d) log-eccentricity windows. Windows will be indexed along the
        first dimension.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature
       Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889
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
                           theta_n_steps=1000, ecc_n_steps=1000):
    """Create 2d pooling windows (log-eccentricity by polar angle) that span the visual field

    This creates the pooling windows that we use to average image statistics for metamer generation
    as done in [1]_. This is returned as a 3d torch tensor for further use with a model, and will
    also return the theta and eccentricity grids necessary for plotting, if desired.

    Note that this is returned in polar coordinates and so if you wish to apply it to an image in
    rectangular coordinates, you'll need to make use of pyrtools's ``project_polar_to_cartesian``
    function (see Examples section)

    Parameters
    ----------
    scaling : `float` or `None`.
        The ratio of the eccentricity window's radial full-width at half-maximum to
        eccentricity (see the `calc_scaling` function).
    min_eccentricity : `float`, optional
        The minimum eccentricity, the eccentricity below which we do not compute pooling windows
        (in degrees). Parameter :math:`e_0` in equation 11 of the online methods.
    max_eccentricity : `float`, optional
        The maximum eccentricity, the outer radius of the image (in degrees). Parameter :math:`e_r`
        in equation 11 of the online methods.
    radial_to_circumferential_ratio : `float`, optional
        ``scaling`` determines the number of log-eccentricity windows we can create; this ratio
        gives us the number of polar angle ones. Based on `scaling`, we calculate the width of the
        windows in log-eccentricity, and then divide that by this number to get their width in
        polar angle. Because we require an integer number of polar angle windows, we round the
        resulting number of polar angle windows to the nearest integer, so the ratio in the
        generated windows approximate this. 2 (the default) is the value used in the paper [1]_.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in equation 9 from the online
        methods. 0.5 (the default) is the value used in the paper [1]_.
    theta_n_steps : `int`, optional
        The number of times we sample theta.
    ecc_n_steps : `int`, optional
        The number of times we sample the eccentricity.

    Returns
    -------
    windows : `torch.tensor`
        The 3d array of 2d windows (in polar angle and eccentricity). Separate windows will be
        alonged the first dimension and the tensor has the shape
        ``(n_polar_windows*n_ecc_windows, ecc_n_steps, theta_n_steps)`` (where the number of
        windows is inferred in this function based on the values of ``scaling`` and
        ``radial_to_circumferential_width``)
    theta_grid : `torch.tensor`
        The 2d array of polar angle values, as returned by meshgrid. Will have the shape
        ``(ecc_n_steps, theta_n_steps)``
    eccentricity_grid : `torch.tensor`
        The 2d array of eccentricity values, as returned by meshgrid. Will have the shape
        ``(ecc_n_steps, theta_n_steps)``

    Examples
    --------
    To use, simply call with the desired scaling (for the version seen in the paper, don't change
    any of the default arguments; compare this image to the right side of Supplementary Figure 1C)

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

    If you wish to convert this to the Cartesian coordinates, in order to apply to an image, for
    example, you must use pyrtools's ``project_polar_to_cartesian`` function

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
    These create the pooling windows, as seen in Supplementary Figure 1C, in [1]_.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature
       Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ecc_window_width = calc_eccentricity_window_width(max_eccentricity, min_eccentricity,
                                                      scaling=scaling)
    n_polar_windows = calc_polar_n_windows(ecc_window_width / radial_to_circumferential_ratio)
    # we want to set the number of polar windows where the ratio of widths is approximately what
    # the user specified. the constraint that it's an integer is more important
    theta, angle_tensor = polar_angle_windows(round(n_polar_windows), theta_n_steps,
                                              transition_region_width)
    angle_tensor = torch.tensor(angle_tensor, dtype=torch.float32, device=device)
    ecc, ecc_tensor = log_eccentricity_windows(None, ecc_window_width, min_eccentricity,
                                               max_eccentricity, ecc_n_steps,
                                               transition_region_width=transition_region_width)
    ecc_tensor = torch.tensor(ecc_tensor, dtype=torch.float32, device=device)
    windows_tensor = torch.einsum('ik,jl->ijkl', [ecc_tensor, angle_tensor])
    windows_tensor = windows_tensor.reshape((windows_tensor.shape[0] * windows_tensor.shape[1],
                                             *windows_tensor.shape[2:]))
    theta_grid, ecc_grid = np.meshgrid(theta, ecc)
    theta_grid = torch.tensor(theta_grid, dtype=torch.float32, device=device)
    ecc_grid = torch.tensor(ecc_grid, dtype=torch.float32, device=device)
    return windows_tensor, theta_grid, ecc_grid
