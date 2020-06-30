"""functions for ventral stream perceptual models, as seen in Freeman and Simoncelli, 2011

"""
import torch
import warnings
import numpy as np
import pyrtools as pt
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch import nn
from ..canonical_computations.non_linearities import cone
from ...tools.display import clean_up_axes, update_stem, clean_stem_plot
from ..canonical_computations.pooling_windows import PoolingWindows
from ...tools.optim import zscore_stats
from ..canonical_computations.steerable_pyramid_freq import Steerable_Pyramid_Freq
from ...tools.data import to_numpy


class VentralModel(nn.Module):
    r"""Generic class that everyone inherits. Sets up the scaling windows

    Note that we will calculate the minimum eccentricity at which the
    area of the windows at half-max exceeds one pixel (based on
    ``scaling``, ``img_res`` and ``max_eccentricity``) and, if
    ``min_eccentricity`` is below that, will throw an Exception.

    This just generates the pooling windows necessary for these models,
    given a small number of parameters. One tricky thing we do is
    generate a set of scaling windows for each scale (appropriately)
    sized. For example, the V1 model will have 4 scales, so for a 256 x
    256 image, the coefficients will have shape (256, 256), (128, 128),
    (64, 64), and (32, 32). Therefore, we need windows of the same size
    (could also up-sample the coefficient tensors, but since that would
    need to happen each iteration of the metamer synthesis,
    pre-generating appropriately sized windows is more efficient).

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

    NOTE: that we're assuming the input to this model contains values
    proportional to photon counts; thus, it should be a raw image or
    other linearized / "de-gamma-ed" image (all images meant to be
    displayed on a standard display will have been gamma-corrected,
    which involves raising their values to a power, typically 1/2.2).

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
    transition_region_width : float, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. 0.5 (the default) is the
        value used in the paper [1]_.
    cone_power : float, optional
        The first step of the model, before calculating any of the
        statistics to pool, is to raise the image to this value, which
        represents the non-linear response of the cones to photons. The
        physiological value is approximately 1/3. The default is 1.0
        (linear) because that works fine for gamma-corrected images (the
        gamma correction is typically 1/2.2, which is not too different
        from 1/3) and works much better for optimization. For synthesis
        methods, it's recommended that you pre-process your input images
        in order to get the effect of cone_power (a concave cone_power
        like 1/3 leads to difficulties converging); if you only want the
        output of this model, then 1/3 will work fine.
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
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    PoolingWindows : plenoptic.simulate.PoolingWindows
        A pooling windows object which contains the windows we use to
        pool our model's summary statistics across the image.
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
    cone_power : float
        The first step of the model, before calculating any of the
        statistics to pool, is to raise the image to this value, which
        represents the non-linear response of the cones to photons.

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5, max_eccentricity=15, num_scales=1,
                 transition_region_width=.5, cone_power=1.0, cache_dir=None, window_type='cosine',
                 std_dev=None, center_surround_ratio=.53, surround_std_dev=3, transition_x=None):
        super().__init__()
        self.PoolingWindows = PoolingWindows(scaling, img_res, min_eccentricity, max_eccentricity,
                                             num_scales, cache_dir, window_type,
                                             transition_region_width, std_dev,
                                             center_surround_ratio, surround_std_dev, transition_x)
        for attr in ['n_polar_windows', 'n_eccentricity_bands', 'scaling', 'state_dict_reduced',
                     'transition_region_width', 'window_width_pixels', 'window_width_degrees',
                     'min_eccentricity', 'max_eccentricity', 'cache_dir', 'deg_to_pix',
                     'window_approx_area_degrees', 'window_approx_area_pixels', 'cache_paths',
                     'calculated_min_eccentricity_degrees', 'calculated_min_eccentricity_pixels',
                     'central_eccentricity_pixels', 'central_eccentricity_degrees', 'img_res',
                     'window_type', 'std_dev', 'surround_std_dev', 'center_surround_ratio',
                     'transition_x']:
            setattr(self, attr, getattr(self.PoolingWindows, attr))
        self.state_dict_reduced['cone_power'] = cone_power
        self.cone_power = cone_power
        self.num_scales = 1
        self._spatial_masks = {}

    def _gen_spatial_masks(self, n_angles=4):
        r"""Generate spatial masks

        Create and return masks that allow us to specifically select
        values from ``self.representation`` that correspond to different
        regions of the image. See ``summarize_representation()`` for an
        example of how to use them

        Parameters
        ----------
        n_angles : int, optional
            The number of angular regions to subdivide the image
            into. By default, splits it into quadrants

        Returns
        -------
        masks : dict
            dictionary with a key for each (scale, angle_i) that
            contains a tensor, same shape as the representations at that
            scale, which is a boolean mask selecting the values that
            correspond to that angular region

        """
        masks = {}
        for i in range(self.num_scales):
            ecc = torch.ones_like(self.PoolingWindows.ecc_windows[i], dtype=int)
            angles = torch.zeros_like(self.PoolingWindows.angle_windows[i], dtype=int)
            for j in range(n_angles):
                angles[j*angles.shape[0]//4:(j+1)*angles.shape[0]//4] = j
            windows = torch.einsum('ahw,ehw->ea', angles, ecc)
            for j, val in enumerate(sorted(windows.unique())):
                masks[(i, f'region_{j}')] = (windows == val).flatten()
        return masks

    def to(self, *args, do_windows=True, **kwargs):
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
        if do_windows:
            self.PoolingWindows.to(*args, **kwargs)
        for k, v in self._spatial_masks.items():
            self._spatial_masks[k] = v.to(*args, **kwargs)
        nn.Module.to(self, *args, **kwargs)
        return self

    def parallel(self, devices, num_batches=1):
        r"""Parallelize the model acros multiple GPUs

        The PoolingWindows these models use can get very large -- so
        large, that it's impossible to put them all on one GPU during a
        forward call. In order to solve that issue, we can spread them
        across multiple GPUs (CPU will still work, but then things get
        very slow for synthesis). Unfortunately we can't use
        ``torch.nn.DataParallel`` for this because that only spreads the
        input/output across multiple devices, not components of a
        module. Because each window acts independently, we can split the
        different windows across devices.

        For the user, they should notice no difference between the
        parallelized and normal versions of these models *EXCEPT* if
        they try to access ``self.PoolingWindows.windows`` directly. See
        the docstring for PoolingWindows.parallel for more details on
        this.

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
        self.PoolingWindows = self.PoolingWindows.parallel(devices, num_batches)
        return self

    def unparallel(self, device=torch.device('cpu')):
        r"""Unparallelize this model, bringing everything onto one device

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
        self.PoolingWindows.unparallel(device)
        return self

    def plot_windows(self, ax=None, contour_levels=None, colors='r', subset=True, **kwargs):
        r"""plot the pooling windows on an image.

        This is just a simple little helper to plot the pooling windows
        on an existing axis. The use case is overlaying this on top of
        the image we're pooling (as returned by ``pyrtools.imshow``),
        and so we require an axis to be passed

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

        Returns
        -------
        ax : matplotlib.pyplot.axis
            The axis with the windows

        """
        return self.PoolingWindows.plot_windows(ax, contour_levels, colors, subset, **kwargs)

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
        return self.PoolingWindows.summarize_window_sizes()

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
        return self.PoolingWindows.plot_window_widths(units, scale_num, figsize, jitter)

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
        return self.PoolingWindows.plot_window_areas(units, scale_num, figsize)

    def save_reduced(self, file_path):
        r"""save the relevant parameters to make saving/loading more efficient

        This saves self.state_dict_reduced, which contains the
        attributes necessary to initialize the model plus a 'model_name'
        key, which the ``load_reduced`` method uses to determine which
        model constructor to call

        Parameters
        ----------
        file_path : str
            The path to save the model object to

        """
        torch.save(self.state_dict_reduced, file_path)

    @classmethod
    def load_reduced(cls, file_path):
        r"""load from the dictionary saved by ``save_reduced``

        Parameters
        ----------
        file_path : str
            The path to load the model object from
        """
        state_dict_reduced = torch.load(file_path)
        return cls.from_state_dict_reduced(state_dict_reduced)

    @classmethod
    def from_state_dict_reduced(cls, state_dict_reduced):
        r"""initialize model from ``state_dict_reduced``

        Parameters
        ----------
        state_dict_reduced : dict
            The reduced state dict to load
        """
        state_dict_reduced = state_dict_reduced.copy()
        model_name = state_dict_reduced.pop('model_name')
        # want to remove class if it's here
        state_dict_reduced.pop('class', None)
        if model_name == 'RGC':
            return RetinalGanglionCells(**state_dict_reduced)
        elif model_name == 'V1':
            return PrimaryVisualCortex(**state_dict_reduced)
        else:
            raise Exception("Don't know how to handle model_name %s!" % model_name)

    def summarize_representation(self, data=None, summary_func='mse', by_angle=False):
        r"""summarize representation by key and (optionally) quadrant

        This takes data and summarizes it within each key of the
        dictionary. The intended use case is to get the mean-squared
        error (by passing ``data=metamer.representation_error()``)
        within each orientation and scale. With ``by_angle=True``, also
        breaks it down by quadrant.

        Parameters
        ----------
        data : torch.Tensor, np.array, dict or None, optional
            The data to convert. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation`` or the output, with the exact same
            structure
        summary_func : {'mse', 'l2', callable}, optional
            the function to use to for summarizing the
            representation. If 'mse', we'll square and average; if 'l2',
            we'll use the L2-norm; else, should be a callable that can
            take a tensor and returns a scalar
        by_angle : bool, optional
            whether to further breakdown representation by angle. If
            False, will just have a single value per key in
            representation. If True, keys will be (k, 'region_{i}'),
            where goes from 0 to 3 and represents quadrants, starting
            from bottom right

        Returns
        -------
        summarized : dict
            dictionary containing keys from representation (or
            representation and region, see above) with values giving the
            corresponding summarized representation

        """
        if not self._spatial_masks:
            self._spatial_masks = self._gen_spatial_masks()
        if data is not None:
            if not isinstance(data, dict):
                data = self.output_to_representation(data)
        else:
            data = self.representation
        if not isinstance(data, dict):
            data = {'representation': data}
        summarized = {}
        if summary_func == 'mse':
            summary_func = lambda x: torch.pow(x, 2).mean()
        elif summary_func == 'l2':
            summary_func = lambda x: torch.norm(x, 2)
        for k, v in data.items():
            if by_angle:
                if isinstance(k, str):
                    mask_k = 0
                else:
                    mask_k = k[0]
                for i in range(4):
                    mask = self._spatial_masks[(mask_k, f'region_{i}')]
                    summarized[(k, f'region_{i}')] = summary_func(v[..., mask]).item()
            else:
                summarized[k] = summary_func(v).item()
        return summarized

    def _representation_for_plotting(self, batch_idx=0, data=None):
        r"""Get the representation in the form required for plotting

        VentralStream objects' representation has a lot of structure:
        each consists of some number of different representation types,
        each averaged per window. And the windows themselves are
        structured: we have several different eccentricity bands, each
        of which contains the same number of angular windows. We want to
        use this structure when plotting the representation, as it makes
        it easier to see what's goin on.

        The representation is either a 3d tensor, with (batch, channel,
        windows) or a dictionary of tensors with that structure, with
        each key corresponding to a different type of representation. We
        transform those 3d tensors into 1d tensors for ease of plotting,
        picking one of the batches (we only ever have 1 channel)

        We allow an optional ``data`` argument. If set, we use this data
        instead of ``self.representation``.

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension (the first one)
        data : torch.Tensor, np.array, dict or None, optional
            The data to get in shape. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_error()`` or
            another instance of this class).

        Returns
        -------
        representation_copy : np.array
            The expanded copy of the representation, which is either a
            1d tensor (if ``data``/``self.representation`` was a tensor)
            or a dict full of 1d tensors

        """
        if data is None:
            data = self.representation

        if isinstance(data, dict):
            rep_copy = {}
            for k, v in data.items():
                # if that batch_idx is included. otherwise, assume it's
                # been flattened already
                if len(v.shape) > 1:
                    v = v[batch_idx]
                rep_copy[k] = to_numpy(v).flatten()
        else:
            rep_copy = to_numpy(data[batch_idx]).flatten()
        return rep_copy

    def representation_to_output(self, data=None):
        r"""convert representation to output

        For this model, the two are the same, so this doesn't do
        anything

        Plenoptic models have two 'modes' for their representation: the
        one returned by the ``forward()`` call ("output"), which must be
        a 3d or 4d tensor, and the one stored as ``self.representation``
        ("representation"), which can be structured however you want,
        and probably represents the structure of the data. For example,
        a dictionary of tensors, where keys naming the different types
        of statistics. We want functions to convert between the two of
        them. This converts representation to output.

        Parameters
        ----------
        data : torch.Tensor, np.array, dict or None, optional
            The data to convert. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure

        """
        if data is None:
            data = self.representation
        return data

    def output_to_representation(self, data):
        r"""convert output to representation

        For this model, the two are the same, so this doesn't do
        anything

        Plenoptic models have two 'modes' for their representation: the
        one returned by the ``forward()`` call ("output"), which must be
        a 3d or 4d tensor, and the one stored as ``self.representation``
        ("representation"), which can be structured however you want,
        and probably represents the structure of the data. For example,
        a dictionary of tensors, where keys naming the different types
        of statistics. We want functions to convert between the two of
        them. This converts output to representation.

        Parameters
        ----------
        data : torch.Tensor, np.array, dict, optional
            The data to convert. Should look like the value returned by
            this model's forward call (i.e., a 3d tensor).

        """
        return data

    @classmethod
    def _get_title(cls, title_list, idx, default_title):
        r"""helper function for dealing with the default way we handle title

        We have a couple possible ways of handling title in these
        plotting functions, so this helper function consolidates
        that.

        When picking a title, we know we'll either have a list of titles
        or a single None.

        - If it's None, we want to just use the default title.

        - If it's a list, pick the appropriate element of the list

          - If that includes '|' (the pipe character), then append the
            default title on the other side of the pipe

        Then return

        Parameters
        ----------
        title_list : list or None
            A list of strs or Non
        idx : int
            An index into title_list. Can be positive or negative.
        default_title : str
            The title to use if title_list is None or to include if
            title_list[idx] includes a pipe

        Returns
        -------
        title : str
            The title to use
        """
        try:
            title = title_list[idx]
            if '|' in title:
                if title.index('|') == 0:
                    # then assume it's at the beginning
                    title = default_title + ' ' + title
                else:
                    # then assume it's at the end
                    title += ' ' + default_title
        except TypeError:
            # in this case, title is None
            title = default_title
        return title

    def update_plot(self, axes, batch_idx=0, data=None):
        r"""Update the information in our representation plot

        This is used for creating an animation of the representation
        over time. In order to create the animation, we need to know how
        to update the matplotlib Artists, and this provides a simple way
        of doing that. It relies on the fact that we've used
        ``plot_representation`` to create the plots we want to update
        and so know that they're stem plots.

        We take the axes containing the representation information (note
        that this is probably a subset of the total number of axes in
        the figure, if we're showing other information, as done by
        ``Metamer.animate``), grab the representation from plotting and,
        since these are both lists, iterate through them, updating as we
        go.

        We can optionally accept a data argument, in which case it
        should look just like the representation of this model (or be
        able to transformed into that form, see
        ``PrimaryVisualCortex._representation_for_plotting`).

        In order for this to be used by ``FuncAnimation``, we need to
        return Artists, so we return a list of the relevant artists, the
        ``markerline`` and ``stemlines`` from the ``StemContainer``.

        Parameters
        ----------
        axes : list
            A list of axes to update. We assume that these are the axes
            created by ``plot_representation`` and so contain stem plots
            in the correct order.
        batch_idx : int, optional
            Which index to take from the batch dimension (the first one)
        data : torch.Tensor, np.array, dict or None, optional
            The data to show on the plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_error()`` or
            another instance of this class).

        Returns
        -------
        stem_artists : list
            A list of the artists used to update the information on the
            stem plots

        """
        stem_artists = []
        axes = [ax for ax in axes if len(ax.containers) == 1]
        data = self._representation_for_plotting(batch_idx, data)
        if not isinstance(data, dict):
            data = {'rep': data}
        for ax, d in zip(axes, data.values()):
            sc = update_stem(ax.containers[0], d)
            stem_artists.extend([sc.markerline, sc.stemlines])
        return stem_artists


class RetinalGanglionCells(VentralModel):
    r"""A wildly simplistic model of retinal ganglion cells (RGCs)

    This model averages together the pixel intensities in each of its
    pooling windows to generate a super simple
    representation. Currently, does not do anything to model the optics
    of the eye (no lens point-spread function), the photoreceptors (no
    cone lattice), or the center-surround nature of retinal ganglion
    cells' receptive fields.

    Note that we will calculate the minimum eccentricity at which the
    area of the windows at half-max exceeds one pixel (based on
    ``scaling``, ``img_res`` and ``max_eccentricity``) and, if
    ``min_eccentricity`` is below that, will throw an Exception.

    We can optionally cache the windows tensor we create, if
    ``cache_dir`` is not None. In that case, we'll also check to see if
    appropriate cached windows exist before creating them and load them
    if they do. The path we'll use is
    ``{cache_dir}/scaling-{scaling}_size-{img_res}_e0-{min_eccentricity}_
    em-{max_eccentricity}_t-{transition_region_width}.pt``.

    NOTE: that we're assuming the input to this model contains values
    proportional to photon counts; thus, it should be a raw image or
    other linearized / "de-gamma-ed" image (all images meant to be
    displayed on a standard display will have been gamma-corrected,
    which involves raising their values to a power, typically 1/2.2).

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
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. 0.5 (the default) is the
        value used in the paper [1]_.
    cone_power : float, optional
        The first step of the model, before calculating any of the
        statistics to pool, is to raise the image to this value, which
        represents the non-linear response of the cones to photons. The
        physiological value is approximately 1/3. The default is 1.0
        (linear) because that works fine for gamma-corrected images (the
        gamma correction is typically 1/2.2, which is not too different
        from 1/3) and works much better for optimization. For synthesis
        methods, it's recommended that you pre-process your input images
        in order to get the effect of cone_power (a concave cone_power
        like 1/3 leads to difficulties converging); if you only want the
        output of this model, then 1/3 will work fine.
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
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    windows : torch.tensor
        A list of 3d tensors containing the pooling windows in which the
        pixel intensities are averaged. Each entry in the list
        corresponds to a different scale and thus is a different size
        (though they should all have the same number of windows)
    image : torch.tensor
        A 2d containing the image most recently analyzed.
    cone_responses : torch.tensor
        A 2d tensor containing the cone responses to the most recent
        image analyzed. That is, ``po.non_linearities.cone(image,
        self.cone_power)``
    representation : torch.tensor
        A tensor containing the averages of the pixel intensities within
        each pooling window for ``self.image``. This will be 3d: (batch,
        channel, windows).
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
        degrees. There are four keys: 'radial_top', 'radial_full',
        'angular_top', and 'angular_full', corresponding to a 2x2 for
        the widths in the radial and angular directions by the 'top' and
        'full' widths (top is the width of the flat-top region of each
        window, where the window's value is 1; full is the width of the
        entire window). Each value is a list containing the widths for
        the windows in different eccentricity bands. To visualize these,
        see the ``plot_window_sizes`` method.
    window_width_pixels : list
        List of dictionaries containing the widths of the windows in
        pixels; each entry in the list corresponds to the widths for a
        different scale, as in ``windows``. See above for explanation of
        the dictionaries. To visualize these, see the
        ``plot_window_sizes`` method.
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
        List of strings, one per scale, that we either saved or loaded
        the cached windows tensors from
    cone_power : float
        The first step of the model, before calculating any of the
        statistics to pool, is to raise the image to this value, which
        represents the non-linear response of the cones to photons.
    center_representation : torch.tensor or None
        the representation of the RGC centers. None if window_type is
        'gaussian' or 'cosine', tensor if window_type is 'dog'. this is
        not necessary for synthesis or computing the actual
        representation (that's all handled within PoolingWindows), so
        may remove this if it ends up taking too much memory
    surround_representation : torch.tensor or None
        the representation of the RGC surrounds. None if window_type is
        'gaussian' or 'cosine', tensor if window_type is 'dog'. this is
        not necessary for synthesis or computing the actual
        representation (that's all handled within PoolingWindows), so
        may remove this if it ends up taking too much memory

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5, max_eccentricity=15,
                 transition_region_width=.5, cone_power=1.0, cache_dir=None, window_type='cosine',
                 std_dev=None, center_surround_ratio=.53, surround_std_dev=3, transition_x=None):
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity,
                         transition_region_width=transition_region_width, cone_power=cone_power,
                         cache_dir=cache_dir, window_type=window_type, std_dev=std_dev,
                         center_surround_ratio=center_surround_ratio, transition_x=transition_x,
                         surround_std_dev=surround_std_dev)
        self.state_dict_reduced.update({'model_name': 'RGC'})
        self.image = None
        self.representation = None
        self.center_representation = None
        self.surround_representation = None

    def forward(self, image):
        r"""Generate the RGC representation of an image

        Parameters
        ----------
        image : torch.tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width). If it has fewer than 4 dimensions,
            we will unsqueeze it until its 4d

        Returns
        -------
        representation : torch.tensor
            A 3d tensor containing the averages of the pixel intensities
            within each pooling window for ``image``

        """
        while image.ndimension() < 4:
            image = image.unsqueeze(0)
        self.image = image.detach().clone()
        cone_responses = cone(image, self.cone_power)
        self.cone_responses = cone_responses.detach().clone()
        self.representation = self.PoolingWindows(cone_responses)
        if self.window_type == 'dog':
            self.center_representation = self.PoolingWindows.forward(cone_responses,
                                                                     windows_key='center')
            self.surround_representation = self.PoolingWindows.forward(cone_responses,
                                                                       windows_key='surround')
        return self.representation

    def _plot_helper(self, n_cols=1, figsize=(10, 5), ax=None, title=None, batch_idx=0, data=None):
        r"""helper function for plotting that takes care of a lot of the standard stuff

        If n_cols is 1, we assume this RGC model has ``window_type``
        'gaussian' or 'cosine'. Only other use case we expect is
        n_cols=3, for ``window_type='dog'``. Should still work if these
        expectations are violated, but things might not be quite correct

        Parameters
        ----------
        n_cols : int
            The number oc columns in the (sub-)figure we're creating (we
            always have a single row)
        figsize : tuple, optional
            The size of the figure to create
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        title : str or None, optional
            The title to put above this axis. If you want no title, pass
            the empty string (``''``). If None, will use the default,
            'Mean pixel intensity in each window'
        batch_idx : int, optional
            Which index to take from the batch dimension (the first one)
        data : torch.Tensor, np.array, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_error()`` or
            another instance of this class).

        Returns
        -------
        ax : matplotlib.pyplot.axis
            The axis we've set up
        data : np.array
            The output of self._representation_for_plotting(batch_idx, data)
        title : str
            The title to use

        """
        if not isinstance(title, list):
            title = [title] * n_cols
        if ax is None:
            fig, ax = plt.subplots(1, n_cols, figsize=figsize)
        else:
            warnings.warn("ax is not None, so we're ignoring figsize...")
            if n_cols > 1:
                ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
                gs = ax.get_subplotspec().subgridspec(1, n_cols)
                fig = ax.figure
                ax = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
        if n_cols > 1:
            title = [self._get_title(title, i, t) for i, t in enumerate(['difference', 'center',
                                                                         'surround'])]
        else:
            title = self._get_title(title, 0, 'mean pixel intensity')
        data = self._representation_for_plotting(batch_idx, data)
        return ax, data, title

    def plot_representation(self, figsize=(10, 5), ylim=None, ax=None, title=None, batch_idx=0,
                            data=None):
        r"""plot the representation of the RGC model

        There are two types of RGC models: average pixel intensity per
        window (with ``window_type`` 'gaussian' or 'cosine'), and
        center-surround receptive fields (with ``window_type='dog'``).

        1. Because our model just takes the average pixel intensities in
           each window, our representation plot is just a simple stem
           plot showing each of these average intensities (different
           positions on the x axis correspond to different windows). We
           have a small break in the data to show where we've moved out
           to the next eccentricity ring.

        2. Two possibilities here. If ``data`` is set, then we default
           to the above plot, because we can't back out the center and
           surround components. If ``data=None``, then we use the stored
           attributes ``representation, center_representation,
           surround_representation`` to plot the center, surround, and
           their weighted difference as separate stem plots. Note that
           the weighted difference is the only actual representation
           (others are just provided so you can better understand the
           difference). In this case, you should probably set
           ``ylim=False`` and make the figure wider, something like
           ``figsize=(15, 5)``.

        In either case, this looks better when it's wider than it is
        tall (like the default figsize suggests)

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create
        ylim : tuple, False, or None
            If a tuple, the y-limits to use for this plot. If None, we use
            the default, slightly adjusted so that the minimum is 0. If
            False, we do nothing.
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        title : str or None, optional
            The title to put above this axis. If you want no title, pass
            the empty string (``''``). If None, will use the default,
            'mean pixel intensity'. If it includes a '|' (pipe), then
            we'll append the default to the other side of the pipe.
        batch_idx : int, optional
            Which index to take from the batch dimension (the first one)
        data : torch.Tensor, np.array, dict or None, optional
            The data to plot. If None, we use
            ``self.representation`` (if ``self.window_type=='dog'``,
            things are slightly different, see above for more
            details). Else, should look like ``self.representation``,
            with the exact same structure (e.g., as returned by
            ``metamer.representation_error()`` or another instance of
            this class).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        axes : list
            A list of axes that contain the plots we've created

        """
        if self.window_type != 'dog' or data is not None:
            ax, data, title = self._plot_helper(1, figsize, ax, title, batch_idx, data)
            clean_stem_plot(data, ax, title, ylim)
            axes = [ax]
            fig = ax.figure
        else:
            data = {'difference': self.representation, 'center': self.center_representation,
                    'surround': self.surround_representation}
            axes, data, title = self._plot_helper(3, figsize, ax, title, batch_idx, data)
            for ax, d, t in zip(axes, data.values(), title):
                clean_stem_plot(d, ax, t, ylim)
            fig = axes[0].figure
        # fig won't always be defined, but this will return the figure belonging to our axis
        return fig, axes

    def plot_representation_image(self, figsize=(5, 5), ax=None, title=None, batch_idx=0,
                                  data=None, vrange='indep1', zoom=1):
        r"""Plot representation as an image, using the weights from PoolingWindows

        There are two types of RGC models: average pixel intensity per
        window (with ``window_type`` 'gaussian' or 'cosine'), and
        center-surround receptive fields (with ``window_type='dog'``).

        1. Our representation has a single value for each pooling
           window, so we take that value and multiply it by the pooling
           window, and then sum across all windows. Thus the value at a
           single pixel shows a weighted sum of the representation.

           By setting ``data``, you can use this to visualize any vector
           with the same length as the number of windows. For example,
           you can view metamer synthesis error by setting
           ``data=metamer.representation_error()`` (then you'd probably
           want to set ``vrange='auto0'`` in order to change the
           colormap to a diverging one cenetered at 0).

        2. We cannot invert the representation directly (we can't unmix
           the center and surround), and so the input must be image-like
           (4d), instead of representation-like (3d). If ``data=None``,
           then we use the stored ``self.image``. Otherwise, we follow
           the same logic as above. See the docstring of
           ``PoolingWindows.project_dog()`` for more details.

           In this case, we create three axes (similar to
           ``plot_representation``), one each for the center, surround,
           and their weighted difference. Note that the weighted
           difference is the only actual representation (others are just
           provided so you can better understand the difference). Also,
           you should probably make the figsize wider, something like
           ``figsize=(16, 5)``.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        title : str or None, optional
            The title to put above this axis. If you want no title, pass
            the empty string (``''``). If None, will use the default,
            'mean pixel intensity'. If it includes a '|' (pipe), then
            we'll append the default to the other side of the pipe.
        batch_idx : int, optional
            Which index to take from the batch dimension (the first one)
        data : torch.Tensor, np.array, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_error()`` or
            another instance of this class).
        vrange : str or tuple, optional
            The vrange value to pass to pyrtools.imshow
        zoom : float or int, optional
            If a float, must be castable to an int or an inverse power
            of 2. The amount to zoom the images in/out by.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        axes : list
            A list of axes that contain the plots we've created

        """
        if self.window_type != 'dog':
            ax, data, title = self._plot_helper(1, figsize, ax, title, batch_idx, data)
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
            # project expects a 3d tensor
            data = self.PoolingWindows.project(torch.tensor(data).unsqueeze(0).unsqueeze(0))
            pt.imshow(to_numpy(data.squeeze()), vrange=vrange, ax=ax, title=title, zoom=zoom)
            fig = ax.figure
            axes = [ax]
        else:
            if data is not None:
                if data.ndim != 4:
                    raise Exception("For RGC with DoG windows, data must be image-like (not the"
                                    " representation)! We can't invert the representation and so"
                                    " must do it from scratch (see PoolingWindows.project_dog() "
                                    "docstring for more details)")
            else:
                data = self.image
            axes, _, title = self._plot_helper(3, figsize, ax, title, batch_idx, None)
            data = {'difference': self.PoolingWindows.project_dog(data),
                    'center': self.PoolingWindows.project(self.center_representation,
                                                          windows_key='center'),
                    'surround': self.PoolingWindows.project(self.surround_representation,
                                                            windows_key='surround')}
            for ax, d, t in zip(axes, data.values(), title):
                ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
                pt.imshow(to_numpy(d.squeeze()), vrange=vrange, ax=ax, title=t, zoom=zoom)
            fig = axes[0].figure
        return fig, axes


class PrimaryVisualCortex(VentralModel):
    r"""Model V1 using the Steerable Pyramid

    This just models V1 as containing complex cells and a representation
    of the mean luminance. For the complex cells, we take the outputs of
    the complex steerable pyramid and takes the complex modulus of them
    (that is, squares, sums, and takes the square root across the real
    and imaginary parts; this is a phase-invariant measure of the local
    magnitude). The mean luminance representation is the same as that
    computed by the RetinalGanglionCell model.

    Note that we will calculate the minimum eccentricity at which the
    area of the windows at half-max exceeds one pixel (based on
    ``scaling``, ``img_res`` and ``max_eccentricity``) and, if
    ``min_eccentricity`` is below that, will throw an Exception.

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

    NOTE: that we're assuming the input to this model contains values
    proportional to photon counts; thus, it should be a raw image or
    other linearized / "de-gamma-ed" image (all images meant to be
    displayed on a standard display will have been gamma-corrected,
    which involves raising their values to a power, typically 1/2.2).

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
    num_scales : int, optional
        The number of scales (spatial frequency bands) in the steerable
        pyramid we use to build the V1 representation
    order : int, optional
        The Gaussian derivative order used for the steerable
        filters. Default value is 3.  Note that to achieve steerability
        the minimum number of orientation is ``order`` + 1, and is used
        here (that's currently all we support, though could extend
        fairly simply)
    min_eccentricity : float, optional
        The eccentricity at which the pooling windows start.
    max_eccentricity : float, optional
        The eccentricity at which the pooling windows end.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. 0.5 (the default) is the
        value used in the paper [1]_.
    normalize_dict : dict, optional
        Dict containing the statistics to normalize, as generated by
        ``po.optim.generate_norm_stats``. If this is an empty dict, we
        don't normalize the model. If it's non-empty, we expect it to
        have only two keys: "complex_cell_responses" and
        "cone_responses"
    cone_power : float, optional
        The first step of the model, before calculating any of the
        statistics to pool, is to raise the image to this value, which
        represents the non-linear response of the cones to photons. The
        physiological value is approximately 1/3. The default is 1.0
        (linear) because that works fine for gamma-corrected images (the
        gamma correction is typically 1/2.2, which is not too different
        from 1/3) and works much better for optimization. For synthesis
        methods, it's recommended that you pre-process your input images
        in order to get the effect of cone_power (a concave cone_power
        like 1/3 leads to difficulties converging); if you only want the
        output of this model, then 1/3 will work fine.
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    half_octave_pyramid : bool, optional
        Whether to include the half octaves in the model's
        representation. If False (the default), we only use the
        steerable pyramid constructed on the original image. If True, we
        include a second pyramid, constructed on a version of the image
        that's been down-sampled by a factor of sqrt(2) (using bicubic
        interpolation), in order to include the frequencies centered at
        half-octave steps and thus have a more complete representation
        of frequency space
    include_highpass : bool, optional
        Whether to include the high-pass residual in the model or
        not.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    num_scales : int, optional
        The number of scales (spatial frequency bands) in the steerable
        pyramid we use to build the V1 representation
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    order : int, optional
        The Gaussian derivative order used for the steerable
        filters. Default value is 3.  Note that to achieve steerability
        the minimum number of orientation is ``order`` + 1, and is used
        here (that's currently all we support, though could extend
        fairly simply)
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    windows : torch.tensor
        A list of 3d tensors containing the pooling windows in which the
        complex cell responses are averaged. Each entry in the list
        corresponds to a different scale and thus is a different size
        (though they should all have the same number of windows)
    image : torch.tensor
        A 2d tensor containing the most recent image analyzed.
    cone_responses : torch.tensor
        A 2d tensor containing the cone responses to the most recent
        image analyzed. That is, ``po.non_linearities.cone(image,
        self.cone_power)``
    pyr_coeffs : dict
        The dictionary containing the (complex-valued) coefficients of
        the steerable pyramid built on ``self.image``. Each of these is
        5d: ``(1, 1, *img_res, 2)``. The first two dimensions are for
        batch and channel, the last dimension contains the real and
        imaginary components of the complex number; channel is
        unnecessary for us but we might be able to get batch working.
    complex_cell_responses : dict
        Dictionary containing the complex cell responses, the squared
        and summed (i.e., the squared complex modulus) of
        ``self.pyr_coeffs``. Does not include the residual high- and
        low-pass bands. Each of these is now 4d: ``(1, 1, *img_res)``.
    mean_luminance : torch.tensor
        A 1d tensor representing the mean luminance of the image, found
        by averaging the pixel values of the image using the windows at
        the lowest scale. This is identical to the RetinalGanglionCell
        representation of the image with the same ``scaling`` value.
    representation : torch.tensor
        A dictionary containing the 'complex cell responses' (that is,
        the squared, summed, and square-rooted outputs of the complex
        steerable pyramid) and the mean luminance of the image in the
        pooling windows. Each of these is a 3d tensor: (batch, channel,
        windows)
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
        degrees. There are four keys: 'radial_top', 'radial_full',
        'angular_top', and 'angular_full', corresponding to a 2x2 for
        the widths in the radial and angular directions by the 'top' and
        'full' widths (top is the width of the flat-top region of each
        window, where the window's value is 1; full is the width of the
        entire window). Each value is a list containing the widths for
        the windows in different eccentricity bands. To visualize these,
        see the ``plot_window_sizes`` method.
    window_width_pixels : list
        List of dictionaries containing the widths of the windows in
        pixels; each entry in the list corresponds to the widths for a
        different scale, as in ``windows``. See above for explanation of
        the dictionaries. To visualize these, see the
        ``plot_window_sizes`` method.
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
        List of strings, one per scale, that we either saved or loaded
        the cached windows tensors from
    normalize_dict : dict
        Dict containing the statistics to normalize, as generated by
        ``po.optim.generate_norm_stats``. If this is an empty dict, we
        don't normalize the model. If it's non-empty, we expect it to
        have only two keys: "complex_cell_responses" and
        "cone_responses"
    to_normalize : list
        List of attributes that we want to normalize by whitening (for
        PrimaryVisualCortex, that's just "complex_cell_responses")
    scales : list
        List of the scales in the model, from fine to coarse. Used for
        synthesizing in coarse-to-fine order
    cone_power : float
        The first step of the model, before calculating any of the
        statistics to pool, is to raise the image to this value, which
        represents the non-linear response of the cones to photons.
    include_highpass : bool, optional
        Whether the high-pass residual is included in the model or not.

    """
    def __init__(self, scaling, img_res, num_scales=4, order=3, min_eccentricity=.5,
                 max_eccentricity=15, transition_region_width=.5, normalize_dict={},
                 cone_power=1.0, cache_dir=None, half_octave_pyramid=False,
                 include_highpass=False, window_type='cosine', std_dev=None):
        if window_type == 'dog':
            raise Exception('DoG windows not supported for V1')
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity, num_scales,
                         transition_region_width=transition_region_width, cone_power=cone_power,
                         cache_dir=cache_dir, window_type=window_type, std_dev=std_dev)
        self.state_dict_reduced.update({'order': order, 'model_name': 'V1',
                                        'num_scales': num_scales,
                                        'normalize_dict': normalize_dict,
                                        'include_highpass': include_highpass})
        # these are DoG-associated keys and so aren't supported
        # here. they end up in this dict because they come from the
        # PoolingWindows' dict
        for k in ['transition_x', 'center_surround_ratio', 'surround_std_dev']:
            self.state_dict_reduced.pop(k)
        self.num_scales = num_scales
        self.order = order
        self.complex_steerable_pyramid = Steerable_Pyramid_Freq(img_res, self.num_scales,
                                                                self.order, is_complex=True)
        self.scales = ['mean_luminance']
        if half_octave_pyramid:
            self.half_octave_img_res = [int(round(i / np.sqrt(2))) for i in img_res]
            # want this to be even. for plotting purposes, the more
            # dividible by 2 this number is, the easier our lives will
            # be
            for i, r in enumerate(self.half_octave_img_res):
                if r % 2 == 1:
                    self.half_octave_img_res[i] += 1
            second_PoolingWindows = PoolingWindows(scaling, self.half_octave_img_res,
                                                   min_eccentricity, max_eccentricity,
                                                   num_scales-1, cache_dir, window_type,
                                                   transition_region_width, std_dev)
            self.PoolingWindows.merge(second_PoolingWindows)
            self.half_octave_pyramid = Steerable_Pyramid_Freq(self.half_octave_img_res,
                                                              num_scales-1, order,
                                                              is_complex=True)
            for i in range(num_scales)[::-1]:
                self.scales.extend([i, i-.5])
            self.scales = self.scales[:-1]
        else:
            self.half_octave_pyramid = None
            self.scales += list(range(num_scales))[::-1]
        self.image = None
        self.pyr_coeffs = None
        self.complex_cell_responses = None
        self.mean_luminance = None
        self.representation = None
        self.include_highpass = include_highpass
        self.to_normalize = ['complex_cell_responses', 'cone_responses']
        if self.include_highpass:
            self.scales += ['residual_highpass']
            self.to_normalize += ['residual_highpass']
        self.normalize_dict = normalize_dict

    def to(self, *args, do_windows=True, **kwargs):
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
        self.complex_steerable_pyramid.to(*args, **kwargs)
        if self.half_octave_pyramid:
            self.half_octave_pyramid.to(*args, **kwargs)
        for k, v in self.normalize_dict.items():
            if isinstance(v, dict):
                for l, w in v.items():
                    self.normalize_dict[k][l] = w.to(*args, **kwargs)
            else:
                self.normalize_dict[k] = v.to(*args, **kwargs)
        super(self.__class__, self).to(do_windows=do_windows, *args, **kwargs)
        return self

    def forward(self, image, scales=[]):
        r"""Generate the V1 representation of an image

        Parameters
        ----------
        image : torch.tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width). If it has fewer than 4 dimensions,
            we will unsqueeze it until its 4d
        scales : list, optional
            Which scales to include in the returned representation. If
            an empty list (the default), we include all
            scales. Otherwise, can contain subset of values present in
            this model's ``scales`` attribute (ints up to
n            self.num_scales-1, the str 'mean_luminance', or, if
            ``half_octave_pyramid`` was set to True during
            initialization, the floats that lie between the integer
            scales: e.g., .5, 1.5, 2.5). Can contain a single value or
            multiple values. If it's an int or float, we include all
            orientations from that scale.

        Returns
        -------
        representation : torch.tensor
            A 3d tensor containing the averages of the
            'complex cell responses', that is, the squared and summed
            outputs of the complex steerable pyramid.

        """
        while image.ndimension() < 4:
            image = image.unsqueeze(0)
        if not scales:
            scales = self.scales
        # this is a little weird here: the image that we detach and
        # clone here is just a copy that we keep around for later
        # examination.
        self.image = image.detach().clone()
        cone_responses = cone(image, self.cone_power)
        # we save this here so that it can be normalized. At this point,
        # it's not normalized, but it will be during the
        # zscore_stats(self.normalize_dict, self) call below. We also
        # zscore cone_responses there because we want the values that go
        # into the mean pixel intensity to be normalized but not the
        # values that go into the steerable pyramid.
        self.cone_responses = cone_responses
        self.pyr_coeffs = {}
        if any([i in self.complex_steerable_pyramid.scales for i in scales]):
            self.pyr_coeffs.update(self.complex_steerable_pyramid(cone_responses, scales))
        half_octave_scales = [i-.5 for i in scales if not isinstance(i, str)]
        if (self.half_octave_pyramid and
            any([i in self.half_octave_pyramid.scales for i in half_octave_scales])):
            half_cones = nn.functional.interpolate(cone_responses, self.half_octave_img_res,
                                                   mode='bicubic')
            half_octave_pyr_coeffs = self.half_octave_pyramid(half_cones, half_octave_scales)
            self.pyr_coeffs.update(dict(((k[0]+.5, k[1]), v)
                                        for k, v in half_octave_pyr_coeffs.items()
                                        if not isinstance(k, str)))
        # to get the energy, we just square and sum across the real and
        # imaginary parts (because there are complex tensors yet, this
        # is the final dimension). the if statement avoids the residuals
        self.complex_cell_responses = dict((k, torch.pow(v, 2).sum(-1))
                                           for k, v in self.pyr_coeffs.items()
                                           if not isinstance(k, str))
        if self.include_highpass and 'residual_highpass' in scales:
            self.residual_highpass = self.pyr_coeffs['residual_highpass']
        if self.normalize_dict:
            self = zscore_stats(self.normalize_dict, self)
        if self.complex_cell_responses:
            self.mean_complex_cell_responses = self.PoolingWindows(self.complex_cell_responses)
        else:
            self.mean_complex_cell_responses = {}
        self.representation = self.mean_complex_cell_responses
        if 'mean_luminance' in scales:
            self.mean_luminance = self.PoolingWindows(self.cone_responses)
            self.representation['mean_luminance'] = self.mean_luminance
        if self.include_highpass and 'residual_highpass' in scales:
            self.mean_residual_highpass = self.PoolingWindows(self.residual_highpass)
            self.representation['residual_highpass'] = self.mean_residual_highpass
        return self.representation_to_output()

    def _representation_for_plotting(self, batch_idx=0, data=None):
        r"""Get data into the form required for plotting

        PrimaryVisualCortex objects' representation has a lot of
        structure: each consists of some number of different
        representation types, each averaged per window. And the windows
        themselves are structured: we have several different
        eccentricity bands, each of which contains the same number of
        angular windows. We want to use this structure when plotting the
        representation, as it makes it easier to see what's goin on.

        The representation is a dictionary of tensors with that
        structure, with each key corresponding to a different type of
        representation. We transform those 4d tensors into 1d tensors
        for ease of plotting, picking one of the batches (we only ever
        have 1 channel) and collapsing the different windows onto one
        dimension, putting a NaN between each eccentricity band.

        We allow an optional ``data`` argument. If set, we use this data
        instead of ``self.representation``. In addition to being
        structured like the ``self.representation`` dictionary, it can
        also be an array or tensor, like this object's ``forward``
        method returns (and thus is stored by the various synthesis
        objects). This function then transforms that array into the
        dictionary we expect (taking advantage of the fact that, from
        that dictionary, we know the number of representation types and
        the number of each type of window) and passes it on to the
        parent class's ``_representation_for_plotting`` to finish it up.

        We expect this to be plotted using
        ``plenoptic.tools.display.clean_stem_plot``, and return a tuple
        ``xvals`` for use with that function to replace the base line
        (by default, ``plt.stem`` doesn't insert a break in the baseline
        if there's a NaN in the data, but we want that for ease of
        visualization)

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension (the first one)
        data : torch.Tensor, np.array, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should be a dictionary of 4d
            tensors like ``self.representation``, or a 3d tensor, like
            the value returned by ``self.forward()`` (e.g., as returned
            by ``metamer.representation_error()``).

        Returns
        -------
        representation_copy : np.array
            The expanded copy of the representation, which is a dict
            full of 1d tensors, with np.nan inserted between each
            eccentricity band
        xvals : tuple
            A 2-tuple of lists, containing the start (``xvals[0]``) and
            stop (``xvals[1]``) x values for plotting. For use with
            plt.hlines, like so: ``plt.hlines(len(xvals[0])*[0],
            xvals[0], xvals[1])``

        """
        if data is not None and not isinstance(data, dict):
            data = self.output_to_representation(data)
        return super()._representation_for_plotting(batch_idx, data)

    def representation_to_output(self, data=None):
        r"""convert representation to output

        This takes data that looks like ``self.representation`` (i.e., a
        dictionary whose keys are ``"mean_luminance"`` and ``(scale,
        orientation)`` tuples) and flattens it to an output that looks
        like what we get from the ``forward()`` call.

        Plenoptic models have two 'modes' for their representation: the
        one returned by the ``forward()`` call ("output"), which must be
        a 3d or 4d tensor, and the one stored as ``self.representation``
        ("representation"), which can be structured however you want,
        and probably represents the structure of the data. For example,
        a dictionary of tensors, where keys naming the different types
        of statistics. We want functions to convert between the two of
        them. This converts representation to output.

        Parameters
        ----------
        data : torch.Tensor, np.array, dict or None, optional
            The data to convert. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure

        """
        if data is None:
            data = self.representation
        return torch.cat(list(data.values()), dim=2)

    def output_to_representation(self, data):
        r"""convert output to representation

        This takes data that looks like the output from the
        ``forward()`` call (a big 3d tensor) and returns something that
        looks like ``self.representation`` (i.e., a dictionary whose
        keys are ``"mean_luminance"`` and ``(scale, orientation)``
        tuples).

        NOTE: for this to work, ``forward()`` must have been called
        before

        Plenoptic models have two 'modes' for their representation: the
        one returned by the ``forward()`` call ("output"), which must be
        a 3d or 4d tensor, and the one stored as ``self.representation``
        ("representation"), which can be structured however you want,
        and probably represents the structure of the data. For example,
        a dictionary of tensors, where keys naming the different types
        of statistics. We want functions to convert between the two of
        them. This converts output to representation.

        Parameters
        ----------
        data : torch.Tensor, np.array, dict, optional
            The data to convert. Should look like the value returned by
            this model's forward call (i.e., a 3d tensor).

        """
        data_dict = {}
        idx = 0
        for k, v in self.representation.items():
            data_dict[k] = data[:, :, idx:idx+v.shape[-1]].reshape(v.shape)
            idx += v.shape[-1]
        data = data_dict
        return data

    def _plot_helper(self, n_rows, n_cols, figsize=(25, 15), ax=None, title=None, batch_idx=0,
                     data=None):
        r"""helper function for plotting that takes care of a lot of the standard stuff

        Parameters
        ----------
        n_rows : int
            The number of rows in the (sub-)figure we're creating
        n_cols : int
            The number oc columns in the (sub-)figure we're creating
        figsize : tuple, optional
            The size of the figure to create (ignored if ``ax`` is not
            None)
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on (in
            which case we ignore ``figsize``). If None, we create our
            own figure to hold it
        title : str, list, or None, optional
            Titles to use. If list or None, this does nothing to it. If
            a str, we turn it into a list with ``(n_rows*n_cols)``
            entries, all identical and the same as the user-pecified
            title
        batch_idx : int, optional
            Which index to take from the batch dimension (the first one)
        data : torch.Tensor, np.array, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure, or
            the structure returned by ``self.forward`` (e.g., as
            returned by ``metamer.representation_error()`` or another
            instance of this class).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing our subplots
        gs : matplotlib.gridspec.GridSpec
            The GridSpec object to use for creating subplots. You should
            use it with ``fig`` to add subplots by indexing into it,
            like so: ``fig.add_subplot(gs[0, 1])``
        data : np.array
            The output of self._representation_for_plotting(batch_idx, data)
        title : list or None
            If title was None or a list, we did nothing to it. If it was
            a str, we made sure its length is (n_rows * n_cols)

        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            gs = mpl.gridspec.GridSpec(n_rows, n_cols, fig)
        else:
            warnings.warn("ax is not None, so we're ignoring figsize...")
            # want to make sure the axis we're taking over is basically invisible.
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
            gs = ax.get_subplotspec().subgridspec(n_rows, n_cols)
            fig = ax.figure
        data = self._representation_for_plotting(batch_idx, data)
        if isinstance(title, str):
            # then this is a single str, so we'll make it the same on
            # every subplot
            title = (n_rows * n_cols) * [title]
        return fig, gs, data, title

    def plot_representation(self, figsize=(25, 15), ylim=None, ax=None, title=None, batch_idx=0,
                            data=None):
        r"""plot the representation of the V1 model

        Since our PrimaryVisualCortex model has more statistics than the
        RetinalGanglionCell model, this is a much more complicated
        plot. We end up creating a grid, showing each band and scale
        separately, and then a separate plot, off to the side, for the
        mean pixel intensity.

        Despite this complication, we can still take an ``ax`` argument
        to plot on some portion of a figure. We make use of matplotlib's
        powerful ``GridSpec`` to arrange things to our liking.

        Each plot has a small break in the data to show where we've
        moved out to the next eccentricity ring.

        Note that this looks better when it's wider than it is tall
        (like the default figsize suggests)

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create (ignored if ``ax`` is not
            None)
        ylim : tuple, False, or None
            If a tuple, the y-limits to use for this plot. If None, we use
            the default, slightly adjusted so that the minimum is 0. If
            False, we do nothing.
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on (in
            which case we ignore ``figsize``). If None, we create our
            own figure to hold it
        title : str, list, or None, optional
            Titles to use. If a list of strings, must be the same length
            as ``data`` (or, if ``data`` is None, of
            ``self.representation``), and each value will be put above
            the subplots. If a str, the same title will be put above
            each subplot. If None, we use the default choice, which
            specifies the scale and orientation of each plot (and the
            mean intensity). If it includes a '|' (pipe), then we'll
            append the default to the other side of the pipe.
        batch_idx : int, optional Which
            index to take from the batch dimension (the first one)
        data : torch.Tensor, np.array, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure, or
            the structure returned by ``self.forward`` (e.g., as
            returned by ``metamer.representation_error()`` or another
            instance of this class).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        axes : list
            A list of axes that contain the plots we've created

        """
        # order is number of orientations - 1 and we want to have
        # columns equal to number of orientations + 1
        if self.half_octave_pyramid is not None:
            n_cols = 2 * self.num_scales
            n_rows = self.order + 1
            col_multiplier = 4
            col_offset = .5
        else:
            n_cols = self.num_scales + 1
            n_rows = self.order + 1
            col_multiplier = 2
            col_offset = 1
        fig, gs, data, title_list = self._plot_helper(2*n_rows, 2*n_cols, figsize,
                                                      ax, title, batch_idx, data)
        axes = []
        for i, (k, v) in enumerate(data.items()):
            if isinstance(k, tuple):
                t = self._get_title(title_list, i, "scale %s, band %s" % k)
                ax = fig.add_subplot(gs[int(2*k[1]):int(2*(k[1]+1)), int(col_multiplier*k[0]):
                                        int(col_multiplier*(k[0]+col_offset))])
                ax = clean_stem_plot(v, ax, t, ylim)
                axes.append(ax)
            elif k == 'mean_luminance':
                t = self._get_title(title_list, -1, "mean pixel intensity")
                if n_rows != 1:
                    ax = fig.add_subplot(gs[n_rows-2:n_rows, 2*(n_cols-1):])
                else:
                    ax = fig.add_subplot(gs[n_rows-1:n_rows, 2*(n_cols-1):])
                ax = clean_stem_plot(v, ax, t, ylim)
                axes.append(ax)
            elif k == 'residual_highpass':
                t = self._get_title(title_list, -1, "residual highpass")
                ax = fig.add_subplot(gs[n_rows:n_rows+2, 2*(n_cols-1):])
                ax = clean_stem_plot(v, ax, t, ylim)
                axes.append(ax)
        return fig, axes

    def plot_representation_image(self, figsize=(27, 5), ax=None, title=None, batch_idx=0,
                                  data=None, vrange='auto1', zoom=1):
        r"""Plot representation as an image, using the weights from PoolingWindows

        Our representation is composed of pooled energy at several
        different scales and orientations, plus the pooled mean pixel
        intensity. In order to visualize these as images, we take each
        statistic, multiply it by the pooling windows, then sum across
        windows, as in
        ``RetinalGanglionCells.plot_representation_image``. We also sum
        across orientations at the same scale, so that we end up with an
        image for each scale (each will be a different size), plus one
        for mean pixel intensity, which we attempt to zoom so that they
        are all shown at approximately the same size.

        Similar to ``self.plot_representation``, you can set ``data`` to
        visualize something else. As in that function, ``data`` must
        have the same structure as ``self.representation`` (i.e., a
        dictionary of 3d tensors/arrays) or as the value returned by
        ``self.forward()`` (i.e., a large 3d tensor/array). For example,
        you can view metamer synthesis error by setting
        ``data=metamer.representation_error()`` (then you'd probably
        want to set ``vrange='auto0'`` in order to change the colormap
        to a diverging one cenetered at 0).

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create (ignored if ``ax`` is not
            None)
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on (in
            which case we ignore ``figsize``). If None, we create our
            own figure to hold it
        title : str, list, or None, optional
            Titles to use. If a list of strings, must be the same length
            as ``data`` (or, if ``data`` is None, of
            ``self.representation``), and each value will be put above
            the subplots. If a str, the same title will be put above
            each subplot. If None, we use the default choice, which
            specifies the scale and orientation of each plot (and the
            mean intensity). If it includes a '|' (pipe), then we'll
            append the default to the other side of the pipe.
        batch_idx : int, optional Which
            index to take from the batch dimension (the first one)
        data : torch.Tensor, np.array, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure, or
            the structure returned by ``self.forward`` (e.g., as
            returned by ``metamer.representation_error()`` or another
            instance of this class).
        vrange : str or tuple, optional
            The vrange value to pass to pyrtools.imshow
        zoom : float or int, optional
            If a float, must be castable to an int or an inverse power
            of 2. The amount to zoom the images in/out by.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        axes : list
            A list of axes that contain the plots we've created

        """
        if self.half_octave_pyramid is not None:
            n_cols = 2 * self.num_scales
            ax_multiplier = 2
        else:
            n_cols = self.num_scales + 1
            ax_multiplier = 1
        if self.include_highpass:
            n_cols += 1
        fig, gs, data, title_list = self._plot_helper(1, n_cols, figsize, ax, title,
                                                      batch_idx, data)
        titles = []
        axes = []
        imgs = []
        zooms = []
        # project expects a dictionary of 3d tensors
        data = self.PoolingWindows.project(dict((k, torch.Tensor(v).unsqueeze(0).unsqueeze(0))
                                                for k, v in data.items()))
        for i in self.scales:
            if isinstance(i, str):
                continue
            titles.append(self._get_title(title_list, i, "scale %s" % i))
            img = np.zeros(data[(i, 0)].shape).squeeze()
            for j in range(self.order+1):
                d = data[(i, j)].squeeze()
                img += to_numpy(d)
            ax = fig.add_subplot(gs[int(ax_multiplier * i)])
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
            imgs.append(img)
            axes.append(ax)
            if isinstance(i, int):
                zooms.append(zoom * round(data[(0, 0)].shape[-1] / img.shape[-1]))
            elif isinstance(i, float):
                zooms.append(zoom * round(data[(0.5, 0)].shape[-1] / img.shape[-1]))
        if self.include_highpass:
            ax = fig.add_subplot(gs[-2])
        else:
            ax = fig.add_subplot(gs[-1])
        ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
        axes.append(ax)
        titles.append(self._get_title(title_list, -1, "mean pixel intensity"))
        imgs.append(to_numpy(data['mean_luminance'].squeeze()))
        zooms.append(zoom)
        if self.include_highpass:
            ax = fig.add_subplot(gs[-1])
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
            axes.append(ax)
            titles.append(self._get_title(title_list, -1, "residual highpass"))
            imgs.append(to_numpy(data['residual_highpass'].squeeze()))
            zooms.append(zoom)
        vrange, cmap = pt.tools.display.colormap_range(imgs, vrange)
        for ax, img, t, vr, z in zip(axes, imgs, titles, vrange, zooms):
            pt.imshow(img, ax=ax, vrange=vr, cmap=cmap, title=t, zoom=z)
        return fig, axes


class MomentsModel(VentralModel):
    r"""pool the image moments (mean, variance, etc)

    this is a null model, used to show that the identity of the moments
    matter, not just their number

    NOTE: this implementation is not finished

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
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. 0.5 (the default) is the
        value used in the paper [1]_.
    cone_power : float, optional
        The first step of the model, before calculating any of the
        statistics to pool, is to raise the image to this value, which
        represents the non-linear response of the cones to photons. The
        physiological value is approximately 1/3. The default is 1.0
        (linear) because that works fine for gamma-corrected images (the
        gamma correction is typically 1/2.2, which is not too different
        from 1/3) and works much better for optimization. For synthesis
        methods, it's recommended that you pre-process your input images
        in order to get the effect of cone_power (a concave cone_power
        like 1/3 leads to difficulties converging); if you only want the
        output of this model, then 1/3 will work fine.
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.


    """
    def __init__(self, scaling, img_res, n_moments, min_eccentricity=.5, max_eccentricity=15,
                 transition_region_width=.5, cone_power=1.0, cache_dir=None, window_type='cosine',
                 std_dev=None):
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity,
                         transition_region_width=transition_region_width, cone_power=cone_power,
                         cache_dir=cache_dir, window_type=window_type, std_dev=std_dev)
        self.state_dict_reduced.update({'model_name': 'Moments'})
        self.image = None
        self.representation = None
        self.n_moments = n_moments
        self.powers = torch.arange(1, self.n_moments+1).reshape((1, self.n_moments, 1, 1))

    def forward(self, image):
        r"""Generate the Moments representation of an image

        Parameters
        ----------
        image : torch.tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width). If it has fewer than 4 dimensions,
            we will unsqueeze it until its 4d

        Returns
        -------
        representation : torch.tensor
            A 3d tensor containing the averages of the pixel intensities
            within each pooling window for ``image``

        """
        self.image = image.detach().clone()
        cone_responses = cone(image, self.cone_power)
        self.cone_responses = cone_responses.detach().clone()
        self.moments = torch.pow(cone_responses, self.powers)
        self.representation = self.PoolingWindows(self.moments)
        return self.representation
