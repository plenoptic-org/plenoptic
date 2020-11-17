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
from ...tools.data import to_numpy
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
    window_type : {'cosine', 'gaussian'}
        Whether to use the raised cosine function from [1]_ or a Gaussian that
        has approximately the same structure. If cosine,
        ``transition_region_width`` must be set; if gaussian, then ``std_dev``
        must be set.
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
    angle_windows : dict
        A dict of 3d tensors containing the angular pooling windows in
        which the model parameters are averaged. Each key corresponds to
        a different scale and thus is a different size.
    ecc_windows : dict
        A dict of 3d tensors containing the log-eccentricity pooling
        windows in which the model parameters are averaged. Each key
        in the dict corresponds to a different scale and thus is a
        different size.
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
                 cache_dir=None, window_type='cosine', transition_region_width=.5, std_dev=None):
        super().__init__()
        if len(img_res) != 2:
            raise Exception("img_res must be 2d!")
        self.scaling = scaling
        self.min_eccentricity = float(min_eccentricity)
        self.max_eccentricity = float(max_eccentricity)
        self.img_res = img_res
        self.num_scales = num_scales
        self.window_type = window_type
        self.angle_windows = {}
        self.ecc_windows = {}
        self.norm_factor = {}
        if window_type == 'cosine':
            assert transition_region_width is not None, "cosine windows need transition region widths!"
            self.transition_region_width = float(transition_region_width)
            self.std_dev = None
            window_width_for_saving = self.transition_region_width
            self.window_max_amplitude = 1
            self.window_intersecting_amplitude = .5
        elif window_type == 'gaussian':
            assert std_dev is not None, "gaussian windows need standard deviations!"
            self.std_dev = float(std_dev)
            self.transition_region_width = None
            if std_dev != 1:
                raise Exception("Only std_dev=1 allowed for Gaussian windows!")
            window_width_for_saving = self.std_dev
            # 1 / (std_dev * GAUSSIAN_SUM) is the max in a single
            # direction (radial or angular), so the max for a single
            # window is its square
            self.window_max_amplitude = (1 / (std_dev * pooling.GAUSSIAN_SUM)) ** 2
            self.window_intersecting_amplitude = self.window_max_amplitude * np.exp(-.25/2)
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
                                   'std_dev': self.std_dev}
        for i in range(self.num_scales):
            scaled_img_res = [np.ceil(j / 2**i) for j in img_res]
            min_ecc, min_ecc_pix = pooling.calc_min_eccentricity(scaling, scaled_img_res,
                                                                 max_eccentricity)
            self.calculated_min_eccentricity_degrees.append(min_ecc)
            self.calculated_min_eccentricity_pixels.append(min_ecc_pix)
            if self.min_eccentricity is not None and min_ecc > self.min_eccentricity:
                warnings.warn(f"Creating windows for scale {i} with min_ecc "
                              f"{self.min_eccentricity}, but calculated min_ecc is {min_ecc}, so"
                              " be aware some are smaller than a pixel!")
            angle_windows = None
            ecc_windows = None
            if cache_dir is not None:
                format_kwargs = dict(scaling=scaling, max_eccentricity=self.max_eccentricity,
                                     img_res=','.join([str(int(i)) for i in scaled_img_res]),
                                     window_width=window_width_for_saving,
                                     window_type=window_type,
                                     min_eccentricity=self.min_eccentricity)
                self.cache_paths.append(cache_path_template.format(**format_kwargs))
                if op.exists(self.cache_paths[-1]):
                    warnings.warn("Loading windows from cache: %s" % self.cache_paths[-1])
                    windows = torch.load(self.cache_paths[-1])
                    angle_windows = windows['angle']
                    ecc_windows = windows['ecc']
            if angle_windows is None or ecc_windows is None:
                angle_windows, ecc_windows = pooling.create_pooling_windows(
                    scaling, scaled_img_res, self.min_eccentricity,
                    self.max_eccentricity, std_dev=self.std_dev,
                    transition_region_width=self.transition_region_width,
                    window_type=window_type)

                if cache_dir is not None:
                    warnings.warn("Saving windows to cache: %s" % self.cache_paths[-1])
                    torch.save({'angle': angle_windows, 'ecc': ecc_windows}, self.cache_paths[-1])
            self.angle_windows[i] = angle_windows
            self.ecc_windows[i] = ecc_windows
            # if we have the eccentricity one std dev away from center, use
            # that.
            try:
                ecc = self.one_std_dev_eccentricity_degrees
            # otherwise, use the central one.
            except AttributeError:
                ecc = self.central_eccentricity_degrees
            self.ecc_windows, norm_factor = pooling.normalize_windows(
                self.angle_windows, self.ecc_windows, ecc, i)
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
                                                          self.window_type, self.transition_region_width,
                                                          self.std_dev)
        self.window_width_degrees = dict(zip(['radial_top', 'radial_full', 'angular_top',
                                              'angular_full'], window_widths))
        self.n_eccentricity_bands = len(self.window_width_degrees['radial_top'])
        # transition width and std dev don't matter for central
        # eccentricity, just min and max
        self.central_eccentricity_degrees = pooling.calc_windows_eccentricity(
            'central', self.n_eccentricity_bands, ecc_window_width, self.min_eccentricity)
        if self.window_type == 'gaussian':
            self.one_std_dev_eccentricity_degrees = pooling.calc_windows_eccentricity(
                '1std', self.n_eccentricity_bands, ecc_window_width, self.min_eccentricity,
                std_dev=self.std_dev)
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
        for k, v in self.angle_windows.items():
            self.angle_windows[k] = v.to(*args, **kwargs)
        for k, v in self.ecc_windows.items():
            self.ecc_windows[k] = v.to(*args, **kwargs)
        for k, v in self.norm_factor.items():
            self.norm_factor[k] = v.to(*args, **kwargs)
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
        if isinstance(x, dict):
            pooled_x = dict((k, torch.einsum('bchw,ahw,ehw->bcea',
                                             [v.to(self.angle_windows[0].device),
                                              self.angle_windows[k[0]],
                                              self.ecc_windows[k[0]]]).flatten(2, 3))
                            for k, v in x.items())
        else:
            pooled_x = (torch.einsum('bchw,ahw,ehw->bcea', [x.to(self.angle_windows[0].device),
                                                            self.angle_windows[idx],
                                                            self.ecc_windows[idx]]).flatten(2, 3))
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
            return dict((k, torch.einsum('bchw,ahw,ehw->bceahw',
                                         [v.to(self.angle_windows[0].device),
                                          self.angle_windows[k[0]],
                                          self.ecc_windows[k[0]]]).flatten(2, 3))
                        for k, v in x.items())
        else:
            if x.ndimension() != 4:
                raise Exception("PoolingWindows input must be 4d tensors or a dict of 4d tensors!"
                                " Unsqueeze until this is true!")
            return torch.einsum('bchw,ahw,ehw->bceahw', [x.to(self.angle_windows[0].device),
                                                         self.angle_windows[idx],
                                                         self.ecc_windows[idx]]).flatten(2, 3)

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
            ``windowed_x`` is a tensor

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
            # one way to make this more general: figure out the size
            # of the tensors in x and in self.angle_windows, and
            # intelligently lookup which should be used.
            return dict((k, v.sum((-1, -2)) ) for k, v in windowed_x.items())
        else:
            return windowed_x.sum((-1, -2))

    def project(self, pooled_x, idx=0):
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

        Returns
        -------
        x : dict or torch.Tensor
            4d tensor or dictionary of 4d tensors

        See also
        --------
        forward : the opposite of this, going from image to pooled
            values

        """
        if isinstance(pooled_x, dict):
            if list(pooled_x.values())[0].ndimension() != 3:
                raise Exception("PoolingWindows input must be 3d tensors or a dict of 3d tensors!"
                                " Squeeze until this is true!")
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
                v = v.reshape((*v.shape[:2], self.ecc_windows[window_key].shape[0],
                               self.angle_windows[window_key].shape[0]))
                tmp[k] = torch.einsum('bcea,ahw,ehw->bchw',
                                      [v.to(self.angle_windows[0].device),
                                       self.angle_windows[window_key],
                                       self.ecc_windows[window_key] / self.norm_factor[window_key]])
            return tmp
        else:
            if pooled_x.ndimension() != 3:
                raise Exception("PoolingWindows input must be 3d tensors or a dict of 3d tensors!"
                                " Squeeze until this is true!")
            pooled_x = pooled_x.reshape((*pooled_x.shape[:2], self.ecc_windows[idx].shape[0],
                                         self.n_polar_windows))
            return torch.einsum('bcea,ahw,ehw->bchw', [pooled_x.to(self.angle_windows[0].device),
                                                       self.angle_windows[idx], self.ecc_windows[idx] /
                                                       self.norm_factor[idx]])

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
            contour_levels = [self.window_intersecting_amplitude]
        # attempt to not have all the windows in memory at once...
        angle_windows = self.angle_windows[windows_scale]
        ecc_windows = self.ecc_windows[windows_scale] / self.norm_factor[windows_scale]
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

    def summarize_window_sizes(self):
        r"""Summarize window sizes

        This function returns a dictionary summarizing the window sizes
        at the minimum and maximum eccentricity. Let ``min_window`` be
        the window whose center is closest to ``self.min_eccentricity``
        and ``max_window`` the one whose center is closest to
        ``self.max_eccentricity``. We find its center, FWHM (in the
        radial direction), and approximate area (at half-max) in
        degrees. We do the same in pixels, for each scale.

        Returns
        -------
        sizes : dict
            dictionary with the keys described above, summarizing window
            sizes. all values are scalar floats

        """
        min_idx = np.abs(self.central_eccentricity_degrees - self.min_eccentricity).argmin()
        max_idx = np.abs(self.central_eccentricity_degrees - self.max_eccentricity).argmin()
        sizes = {}
        central_ecc = self.central_eccentricity_degrees
        widths = self.window_width_degrees
        areas = self.window_approx_area_degrees
        for extrem, idx in zip(['min', 'max'], [min_idx, max_idx]):
            sizes[f"{extrem}_window_center_degrees"] = central_ecc[idx]
            sizes[f"{extrem}_window_fwhm_degrees"] = widths['radial_half'][idx]
            sizes[f"{extrem}_window_area_degrees"] = areas['half'][idx]
        central_ecc = self.central_eccentricity_pixels
        widths = self.window_width_pixels
        areas = self.window_approx_area_pixels
        for i in range(len(central_ecc)):
            for extrem, idx in zip(['min', 'max'], [min_idx, max_idx]):
                sizes[f"{extrem}_window_scale_{i}_center_pixels"] = central_ecc[i][idx]
                sizes[f"{extrem}_window_scale_{i}_fwhm_pixels"] = widths[i]['radial_half'][idx]
                sizes[f"{extrem}_window_scale_{i}_area_pixels"] = areas[i]['half'][idx]
        return sizes

    def plot_window_checks(self, angle_n=0, scale=0):
        r"""Make some plots to check whether windows have been normalized properly

        This creates a figure with two sets of plots: the first row shows the
        L1-norm of the windows, the second shows the sum. Each row will have
        one plot and, if everything worked correctly, they should each look
        like a sigmoid function that runs from 1 for small eccentricities to 0
        for high eccentricities

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
        angle_all = self.angle_windows[scale].shape[0]
        windows = torch.einsum(einsum_str, self.angle_windows[scale][angle_n],
                               self.ecc_windows[scale])
        fig, axes = plt.subplots(2, 1, figsize=(5, 10),
                                 gridspec_kw={'hspace': .4})
        for i, (f, name) in enumerate(zip(funcs, ['L1-norm', 'Sum'])):
            d = f(windows)
            # most of the time, self.central_eccentricity_degrees
            # and d will be same size, but sometimes they will not
            # not. this happens because central_eccentricity_degrees
            # contains all windows that we constructed, but the
            # ecc_windows dictionary throws away any windows that
            # have all zero (or close to zero) values. this will be
            # those at the end, because they're off the image
            ecc = self.central_eccentricity_degrees[:d.shape[0]]
            axes[i].semilogx(ecc, d)
            for j, dj in enumerate(d.transpose(0, 1)):
                if i == 0:
                    label = angle_n[j]
                else:
                    label = None
                axes[i].scatter(ecc, dj, label=label)
            axes[i].set(title='Windows', xlabel='Window central eccentricity (deg)', ylabel=name)
            fig.text(.5, [.91, .47][i], ha='center', fontsize=1.5*plt.rcParams['font.size'],
                     s=f'{name} of windows in some angle slices out of {angle_all}')
        if legend:
            fig.legend(loc='center right', title='Angle slices')
        return fig
