"""functions for ventral stream perceptual models, as seen in Freeman and Simoncelli, 2011

"""
import torch
import itertools
import warnings
from torch import nn
import matplotlib as mpl
import numpy as np
from ..canonical_computations.non_linearities import rectangular_to_polar_dict
from ...tools.display import clean_up_axes, update_stem, clean_stem_plot
from ..canonical_computations.pooling import PoolingWindows
from ..canonical_computations.steerable_pyramid_freq import Steerable_Pyramid_Freq
import matplotlib.pyplot as plt


class VentralModel(nn.Module):
    r"""Generic class that everyone inherits. Sets up the scaling windows

    This just generates the pooling windows necessary for these models,
    given a small number of parameters. One tricky thing we do is
    generate a set of scaling windows for each scale (appropriately)
    sized. For example, the V1 model will have 4 scales, so for a 256 x
    256 image, the coefficients will have shape (256, 256), (128, 128),
    (64, 64), and (32, 32). Therefore, we need windows of the same size
    (could also up-sample the coefficient tensors, but since that would
    need to happen each iteration of the metamer synthesis,
    pre-generating appropriately sized windows is more efficient).

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

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5, max_eccentricity=15, num_scales=1,
                 transition_region_width=.5):
        super().__init__()
        self.PoolingWindows = PoolingWindows(scaling, img_res, min_eccentricity, max_eccentricity,
                                             num_scales, transition_region_width,
                                             flatten_windows=False)
        for attr in ['n_polar_windows', 'n_eccentricity_bands', 'scaling', 'state_dict_reduced',
                     'transition_region_width', 'window_width_pixels', 'window_width_degrees',
                     'min_eccentricity', 'max_eccentricity']:
            setattr(self, attr, getattr(self.PoolingWindows, attr))

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
        self.PoolingWindows.plot_windows(ax, contour_levels, colors, **kwargs)

    def plot_window_sizes(self, units='degrees', scale_num=0, figsize=(5, 5), jitter=.25):
        r"""plot the size of the windows, in degrees or pixels

        We plot the size of the window in both angular and radial
        direction, as well as showing both the 'top' and 'full' width
        (top is the width of the flat-top region of each window, where
        the window's value is 1; full is the width of the entire window)

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
        self.PoolingWindows.plot_window_sizes(units, scale_num, figsize, jitter)

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

    def _representation_for_plotting(self, batch_idx=0, data=None):
        r"""Get the representation in the form required for plotting

        VentralStream objects' representation has a lot of structure:
        each consists of some number of different representation types,
        each averaged per window. And the windows themselves are
        structured: we have several different eccentricity bands, each
        of which contains the same number of angular windows. We want to
        use this structure when plotting the representation, as it makes
        it easier to see what's goin on.

        The representation is either a 4d tensor, with (batch, channel,
        polar angle windows, eccentricity bands) or a dictionary of
        tensors with that structure, with each key corresponding to a
        different type of representation. We transform those 4d tensors
        into 1d tensors for ease of plotting, picking one of the batches
        (we only ever have 1 channel) and collapsing the different
        windows onto one dimension, putting a NaN between each
        eccentricity band.

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
            The data to get in shape. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_ratio()`` or
            another instance of this class).

        Returns
        -------
        representation_copy : np.array
            The expanded copy of the representation, which is either a
            1d tensor (if ``data``/``self.representation`` was a tensor)
            or a dict full of 1d tensors, with np.nan inserted between
            each eccentricity band
        xvals : tuple
            A 2-tuple of lists, containing the start (``xvals[0]``) and
            stop (``xvals[1]``) x values for plotting. For use with
            plt.hlines, like so: ``plt.hlines(len(xvals[0])*[0],
            xvals[0], xvals[1])``

        """
        if data is None:
            data = self.representation

        # this helper functions takes our 4d tensor and converts it to
        # 1d, with nans inserted between the eccentricity bands. we want
        # to plot this data as a stem plot, so we want something 1d
        def add_nans(x):
            nans = np.nan * np.ones((*x.shape[:-1], 1))
            return np.concatenate([x, nans], -1).flatten()
        if isinstance(data, dict):
            rep_copy = dict((k, add_nans(v[batch_idx])) for k, v in data.items())
            rep_example = list(rep_copy.values())[0]
        else:
            rep_copy = add_nans(data[batch_idx])
            rep_example = rep_copy
        xvals = ([], [])
        for i in np.where(np.isnan(rep_example))[0]:
            try:
                xvals[0].append(xvals[1][-1]+2)
            except IndexError:
                # this will happen on the first time through
                xvals[0].append(0)
            xvals[1].append(i-1)
        return rep_copy, xvals

    def _update_plot(self, axes, batch_idx=0, data=None):
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
            (e.g., as returned by ``metamer.representation_ratio()`` or
            another instance of this class).

        Returns
        -------
        stem_artists : list
            A list of the artists used to update the information on the
            stem plots

        """
        stem_artists = []
        axes = [ax for ax in axes if len(ax.containers) == 1]
        data, _ = self._representation_for_plotting(batch_idx, data)
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
    windowed_image : torch.tensor
        A 3d tensor containing windowed views of ``self.image``
    representation : torch.tensor
        A tensor containing the averages of the pixel intensities within
        each pooling window for ``self.image``. This will be 4d: (batch,
        channel, polar angle, eccentricity).
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

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5, max_eccentricity=15,
                 transition_region_width=.5):
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity,
                         transition_region_width=transition_region_width)
        self.state_dict_reduced.update({'model_name': 'RGC'})
        self.image = None
        self.windowed_image = None
        self.representation = None

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
        self.image = image.clone().detach()
        self.representation = self.PoolingWindows(image)
        # we keep the batch and channel indices, only flattening the
        # ones after that
        return self.representation.flatten(2)

    def plot_representation(self, figsize=(10, 5), ylim=None, ax=None, title=None, batch_idx=0,
                            data=None):
        r"""plot the representation of the RGC model

        Because our model just takes the average pixel intensities in
        each window, our representation plot is just a simple stem plot
        showing each of these average intensities (different positions
        on the x axis correspond to different windows). We have a small
        break in the data to show where we've moved out to the next
        eccentricity ring.

        Note that this looks better when it's wider than it is tall
        (like the default figsize suggests)

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create
        ylim : tuple or None, optional
            If not None, the y-limits to use for this plot. If None, we
            use the default, slightly adjusted so that the minimum is 0
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
            (e.g., as returned by ``metamer.representation_ratio()`` or
            another instance of this class).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        axes : list
            A list of axes (with one element) that contain the plots
            we've created

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            warnings.warn("ax is not None, so we're ignoring figsize...")
        rep_copy, xvals = self._representation_for_plotting(batch_idx, data)
        if title is None:
            title = 'Mean pixel intensity in each window'
        clean_stem_plot(rep_copy, ax, title, ylim, xvals)
        # fig won't always be defined, but this will return the figure belonging to our axis
        return ax.figure, [ax]


class PrimaryVisualCortex(VentralModel):
    r"""Model V1 using the Steerable Pyramid

    This just models V1 as containing complex cells and a representation
    of the mean luminance. For the complex cells, we take the outputs of
    the complex steerable pyramid and takes the complex modulus of them
    (that is, squares, sums, and takes the square root across the real
    and imaginary parts; this is a phase-invariant measure of the local
    magnitude). The mean luminance representation is the same as that
    computed by the RetinalGanglionCell model.

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
        A 2d containing the most recent image analyzed.
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
    windowed_complex_cell_responses : dict
        Dictionary containing the windowed complex cell responses. Each
        of these is 5d: ``(1, 1, W, *img_res)``, where ``W`` is the
        number of windows (which depends on the ``scaling`` parameter).
    mean_luminance : torch.tensor
        A 1d tensor representing the mean luminance of the image, found
        by averaging the pixel values of the image using the windows at
        the lowest scale. This is identical to the RetinalGanglionCell
        representation of the image with the same ``scaling`` value.
    representation : torch.tensor
        A dictionary containing the 'complex cell responses' (that is,
        the squared, summed, and square-rooted outputs of the complex
        steerable pyramid) and the mean luminance of the image in the
        pooling windows. Each of these is a 4d tensor: (batch, channel,
        polar angle, eccentricity)
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

    """
    def __init__(self, scaling, img_res, num_scales=4, order=3, min_eccentricity=.5,
                 max_eccentricity=15, transition_region_width=.5):
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity, num_scales,
                         transition_region_width=transition_region_width)
        self.state_dict_reduced.update({'order': order, 'model_name': 'V1',
                                        'num_scales': num_scales})
        self.num_scales = num_scales
        self.order = order
        self.complex_steerable_pyramid = Steerable_Pyramid_Freq(img_res, self.num_scales,
                                                                self.order, is_complex=True)
        self.image = None
        self.pyr_coeffs = None
        self.complex_cell_responses = None
        self.windowed_complex_cell_responses = None
        self.mean_luminance = None
        self.representation = None

    def forward(self, image):
        r"""Generate the V1 representation of an image

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
            A 3d tensor containing the averages of the
            'complex cell responses', that is, the squared and summed
            outputs of the complex steerable pyramid.

        """
        while image.ndimension() < 4:
            image = image.unsqueeze(0)
        self.image = image.clone().detach()
        self.pyr_coeffs = self.complex_steerable_pyramid(image)
        self.complex_cell_responses = rectangular_to_polar_dict(self.pyr_coeffs)[0]
        self.mean_complex_cell_responses = self.PoolingWindows(self.complex_cell_responses)
        self.mean_luminance = self.PoolingWindows(image)
        self.representation = self.mean_complex_cell_responses
        self.representation['mean_luminance'] = self.mean_luminance
        return torch.cat(list(self.representation.values()), dim=1).unsqueeze(0).flatten(2)

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
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_ratio()`` or
            another instance of this class).

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
            data_dict = {}
            idx = 0
            for k, v in self.representation.items():
                numel = np.multiply(*v.shape[2:])
                data_dict[k] = data[:, :, idx:idx+numel].reshape(v.shape)
                idx += numel
            data = data_dict
        return super()._representation_for_plotting(batch_idx, data)

    def plot_representation(self, figsize=(25, 15), ylim=None, ax=None, titles=None, batch_idx=0,
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
        ylim : tuple or None, optional
            If not None, the y-limits to use for this plot. If None, we
            use the default, slightly adjusted so that the minimum is 0
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on (in
            which case we ignore ``figsize``). If None, we create our
            own figure to hold it
        titles : list or None, optional
            A list of strings, each of which is the title to put above
            the subplots. If None, we use the default choice, which
            specifies the scale and orientation of each plot (and the
            mean intensity). If a list, must have the right number of
            titles: ``self.num_scales*(self.order+1)+1`` (the last one
            is ``self.mean_luminance``)
        batch_idx : int, optional
            Which index to take from the batch dimension (the first one)
        data : torch.Tensor, np.array, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure, or
            the structure returned by ``self.forward`` (e.g., as
            returned by ``metamer.representation_ratio()`` or another
            instance of this class).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        axes : list
            A list of axes that contain the plots we've created

        """
        axes = []
        if ax is None:
            # we add 2 to order because we're adding one to get the
            # number of orientations and then another one to add an
            # extra column for the mean luminance plot
            fig = plt.figure(figsize=figsize)
            gs = mpl.gridspec.GridSpec(2*self.num_scales, 2*(self.order+2), fig)
        else:
            warnings.warn("ax is not None, so we're ignoring figsize...")
            # want to make sure the axis we're taking over is basically invisible.
            ax = clean_up_axes(ax, spines_to_remove=['top', 'right', 'bottom', 'left'])
            ax.yaxis.set_visible(False)
            gs = ax.get_subplotspec().subgridspec(2*self.num_scales, 2*(self.order+2))
            fig = ax.figure
        rep_copy, xvals = self._representation_for_plotting(batch_idx, data)
        for k, v in rep_copy.items():
            if isinstance(k, tuple):
                title = "scale %02d, band%02d" % k
                ax = fig.add_subplot(gs[2*k[0]:2*(k[0]+1), 2*k[1]:2*(k[1]+1)])
                ax = clean_stem_plot(v, ax, title, ylim, xvals)
                axes.append(ax)
            else:
                ax = fig.add_subplot(gs[self.num_scales-1:self.num_scales+1, 2*(self.order+1):])
                ax = clean_stem_plot(v, ax, "Mean pixel intensity", ylim, xvals)
                axes.append(ax)
        return fig, axes
