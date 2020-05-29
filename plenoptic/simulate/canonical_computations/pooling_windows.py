"""contains the PoolingWindows class

this is the module you should use to get the pooling used in the
VentralStream models, like those found in Freeman and Simoncelli, 2011

pooling.py contains a lot of necessary functions

"""
import torch
from torch import nn
import itertools
import warnings
import numpy as np
import pyrtools as pt
import matplotlib.pyplot as plt
import os.path as op
from ...tools.data import to_numpy, polar_radius
from . import pooling


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
    corrected_center_surround_ratio : dict
        dict of 2d tensors (eccentricity by angular windows) that gives
        the actual center/surround ratio that gets used for DoG windows
        (the different amount of stretching requires some
        correction). If ``window_type!='dog'``, this is empty. Each key
        is a different scale.
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
        windows in which the model parameters are averaged. Each key
        in the dict corresponds to a different scale and thus is a
        different size. If you have called ``parallel()``, this will be
        structured in a slightly different way (see that method for
        details)
    norm_factor : dict
        a dict of 3d tensors containing the values used to normalize
        ecc_windows. Each key corresponds to a different scale. This is
        stored to undo that normalization for plotting and projection.
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
        self.norm_factor = {}
        self.corrected_center_surround_ratio = {}
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
            self.window_max_amplitude = (1 / (std_dev * pooling.GAUSSIAN_SUM)) ** 2
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
            warnings.warn("DoG windows will only work well with windows that are pixel-wise "
                          "relatively large (to avoid aliasing near the fovea) and degree-wise "
                          "relatively small (so that the Taylor approximation of our warping "
                          "function is pretty good). Thus, it will not work well for small (in "
                          "pixels) images and only for some scaling values. Use the method"
                          "`plot_window_checks()` for help checking")
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
            self.center_max_amplitude = ((1 / (std_dev * pooling.GAUSSIAN_SUM))) ** 2
            self.center_intersecting_amplitude = self.center_max_amplitude * np.exp(-.25/2)
            self.surround_max_amplitude = ((1 / (surround_std_dev * pooling.GAUSSIAN_SUM))) ** 2
            self.surround_intersecting_amplitude = self.surround_max_amplitude * np.exp(-.25/2)
            self.window_max_amplitude = ((center_surround_ratio * self.center_max_amplitude) -
                                         (1 - center_surround_ratio) * self.surround_max_amplitude)
            self.window_intersecting_amplitude = ((center_surround_ratio * self.center_max_amplitude) * np.exp(-.25/(2*std_dev**2)) -
                                                  ((1 - center_surround_ratio) * self.surround_max_amplitude) * np.exp(-.25/(2*surround_std_dev**2)))
            self.min_ecc_mask = {}
            # we have separate center and surround dictionaries:
            self.angle_windows = {'center': {}, 'surround': {}}
            self.ecc_windows = {'center': {}, 'surround': {}}
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
            min_ecc, min_ecc_pix = pooling.calc_min_eccentricity(scaling, scaled_img_res,
                                                                 max_eccentricity)
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
            # TEMPORARY
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
                angle_windows, ecc_windows = pooling.create_pooling_windows(
                    scaling, scaled_img_res, min_ecc, max_eccentricity, std_dev=self.std_dev,
                    transition_region_width=self.transition_region_width, window_type=window_type,
                    surround_std_dev=self.surround_std_dev, transition_x=transition_x)

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
            # if we have the eccentricity one std dev away from center, use that.
            try:
                ecc = self.one_std_dev_eccentricity_degrees
            # otherwise, use the central one.
            except AttributeError:
                ecc = self.central_eccentricity_degrees
            self.ecc_windows, norm_factor, new_ratio = pooling.normalize_windows(
                self.angle_windows, self.ecc_windows, ecc, i, self.center_surround_ratio)
            self.norm_factor[i] = norm_factor
            if window_type == 'dog':
                self.corrected_center_surround_ratio[i] = new_ratio

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
        ecc_window_width = pooling.calc_eccentricity_window_spacing(scaling=self.scaling,
                                                                    std_dev=self.std_dev)
        n_polar_windows = int(round(pooling.calc_angular_n_windows(ecc_window_width / 2)))
        self.n_polar_windows = n_polar_windows
        angular_window_width = pooling.calc_angular_window_spacing(self.n_polar_windows)
        # we multiply max_eccentricity by sqrt(2) here because we want
        # to go out to the corner of the image
        window_widths = pooling.calc_window_widths_actual(angular_window_width, ecc_window_width,
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
        self.central_eccentricity_degrees = pooling.calc_windows_eccentricity(
            'central', self.n_eccentricity_bands, ecc_window_width, self.min_eccentricity,
            transition_x=self.transition_x)
        if window_type == 'gaussian':
            self.one_std_dev_eccentricity_degrees = pooling.calc_windows_eccentricity(
                '1std', self.n_eccentricity_bands, ecc_window_width, self.min_eccentricity,
                transition_x=self.transition_x, std_dev=self.std_dev)
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
            deg_to_pix = pooling.calc_deg_to_pix([j/2**i for j in self.img_res], self.max_eccentricity)
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
            for k, v in self.norm_factor.items():
                self.norm_factor[k] = v.to(*args, **kwargs)
            for k, v in self.corrected_center_surround_ratio.items():
                self.corrected_center_surround_ratio[k] = v.to(*args, **kwargs)
            for s in ['center', 'surround']:
                for k, v in self.angle_windows[s].items():
                    self.angle_windows[s][k] = v.to(*args, **kwargs)
                for k, v in self.ecc_windows[s].items():
                    self.ecc_windows[s][k] = v.to(*args, **kwargs)
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
        for k, v in other_PoolingWindows.corrected_center_surround_ratio.items():
            self.corrected_center_surround_ratio[k+scale_offset] = v

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
                r = self.corrected_center_surround_ratio[idx].flatten(-2, -1)
                return r * ctr - (1 - r) * sur
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
            norm_factor = self.norm_factor
        else:
            angle_windows = self.angle_windows[windows_key]
            ecc_windows = self.ecc_windows[windows_key]
            norm_factor = self.norm_factor
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
                                           ecc_windows[window_key] / norm_factor[window_key]])
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
                        a = angle_windows[(window_key, i)] / norm_factor[window_key]
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
                                                           angle_windows[idx], ecc_windows[idx] /
                                                           norm_factor[idx]])
            else:
                pooled_x = pooled_x.reshape((*pooled_x.shape[:2], ecc_windows[(idx, 0)].shape[0],
                                             self.n_polar_windows))
                tmp = []
                num = int(np.ceil(self.n_polar_windows / self.num_devices))
                for i in range(self.num_devices):
                    a = angle_windows[(idx, i)]
                    e = ecc_windows[(idx, i)] / norm_factor[idx]
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
        # I think this should be with the target center_surround_ratio
        # rather than the corrected one.
        return self.center_surround_ratio * ctr - (1 - self.center_surround_ratio) * sur

    def plot_windows(self, ax=None, contour_levels=None, colors='r',
                     subset=True, windows_scale=0, dog_key='center', **kwargs):
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
                if dog_key == 'center':
                    contour_levels = [self.center_intersecting_amplitude]
                elif dog_key == 'surround':
                    contour_levels = [self.surround_intersecting_amplitude]
            else:
                contour_levels = [self.window_intersecting_amplitude]
        if self.num_devices == 1:
            # attempt to not have all the windows in memory at once...
            try:
                angle_windows = self.angle_windows[windows_scale]
                ecc_windows = self.ecc_windows[windows_scale] / self.norm_factor[windows_scale]
            except KeyError:
                # then this is the DoG windows and so we grab the center
                angle_windows = self.angle_windows[dog_key][windows_scale]
                ecc_windows = (self.ecc_windows[dog_key][windows_scale] /
                               self.norm_factor[windows_scale])
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
                    windows = torch.einsum('hw,ehw->ehw', [a, self.ecc_windows[(windows_scale, device)] / self.norm_factor[windows_scale]])
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

    def plot_window_checks(self, angle_n=0, scale=0, plot_transition_x=True):
        r"""Make some plots to check whether DoG windows have been normalized properly

        Getting the DoG windows to be properly normalized (so that the
        center and surround have the proper balance) has been difficult,
        so this plot function can be used to double-check that
        everything worked out correctly

        It will create a figure with two sets of plots: the first row
        shows the L1-norm of the windows, the second shows the sum.

        For DoG windows, each row contains three plots, showing the plot
        for the center windows, the surround windows, and their
        difference.

        The center and surround plots, in both rows, should be 0 up
        until about `transition_x` (if `plot_transition_x=True`, this
        will be plotted as a dotted line), at which poitn they should
        rapidly rise and then saturate at a constant value, where
        they'll remain until the periphery, at whcih point they'll ramp
        back down to 0.

        The different plot will look like this for the L1-norm plot, but
        it should have a value of 1 when it saturates. For the sum plot,
        it will be at some much smaller value, equal to
        2*`center_surround_ratio`-1.

        For non-DoG windows, each row will have one plot and they should
        each look like a sigmoid function that runs from 1 for small
        eccentricities to 0 for high eccentricities

        You can plot multiple angle slices, and each should look more or
        less the same

        Parameters
        ------
        angle_n : int or list, optional
            Which angle slice(s) to show. Can be a single int or a list
            of ints, in which case we plot each as a separate color
        scale : int, optional
            we plot this for one scale at a time. this specifies the
            scale.
        plot_transition_x : bool, optional
            whether to plot a dotted lin showing the x-value of
            transition_x

        Returns
        -------
        fig : plt.Figure
            the figure containing the plot

        """
        if not hasattr(angle_n, '__iter__'):
            angle_n = [angle_n]
        einsum_str = 'ahw,ehw->eahw'
        legend = True
        funcs = [lambda x: torch.norm(x, 1, (-1, -2)), lambda x: torch.sum(x, (-1, -2))]
        if self.window_type == 'dog':
            angle_all = self.angle_windows['center'][scale].shape[0]
            r = self.corrected_center_surround_ratio[scale][:, angle_n].unsqueeze(-1).unsqueeze(-1)
            ctr = torch.einsum(einsum_str, self.angle_windows['center'][scale][angle_n],
                               self.ecc_windows['center'][scale])
            sur = torch.einsum(einsum_str, self.angle_windows['surround'][scale][angle_n],
                               self.ecc_windows['surround'][scale])
            windows = r * ctr - (1 - r) * sur
            data = [ctr, sur, windows]
            titles = ['Center', 'Surround', 'Difference']
        else:
            angle_all = self.angle_windows[scale].shape[0]
            windows = torch.einsum(einsum_str, self.angle_windows[scale][angle_n],
                                   self.ecc_windows[scale])
            data = [windows]
            titles = ['Windows']
        fig, axes = plt.subplots(2, len(data), figsize=(5*len(data), 10),
                                 gridspec_kw={'hspace': .4})
        if len(data) == 1:
            axes = [[axes[0]], [axes[1]]]
        for i, (f, name) in enumerate(zip(funcs, ['L1-norm', 'Sum'])):
            for j, (ax, d, t) in enumerate(zip(axes[i], data, titles)):
                d = f(d)
                # most of the time, self.central_eccentricity_degrees
                # and d will be same size, but sometimes they will not
                # not. this happens because central_eccentricity_degrees
                # contains all windows that we constructed, but the
                # ecc_windows dictionary throws away any windows that
                # have all zero (or close to zero) values. this will be
                # those at the end, because they're off the image
                ecc = self.central_eccentricity_degrees[:d.shape[0]]
                ax.semilogx(ecc, d)
                for k, dk in enumerate(d.transpose(0, 1)):
                    if i == 0 and j == 0:
                        label = angle_n[k]
                    else:
                        label = None
                    ax.scatter(ecc, dk, label=label)
                ax.set(title=t, xlabel='Window central eccentricity (deg)', ylabel=name)
                if plot_transition_x:
                    ax.vlines(self.transition_x, 0, d.max(), linestyles='--')
            fig.text(.5, [.91, .47][i], ha='center', fontsize=1.5*plt.rcParams['font.size'],
                     s=f'{name} of windows in some angle slices out of {angle_all}')
        if legend:
            fig.legend(loc='center right', title='Angle slices')
        return fig
