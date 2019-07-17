"""functions for ventral stream perceptual models, as seen in Freeman and Simoncelli, 2011

"""
import torch
import itertools
import warnings
from torch import nn
import matplotlib as mpl
import numpy as np
import pyrtools as pt
from ..tools.fit import complex_modulus
from ..tools.display import clean_up_axes, update_stem
from .pooling import (create_pooling_windows, calc_window_widths_actual, calc_angular_n_windows,
                      calc_eccentricity_window_width, calc_angular_window_width,
                      calc_windows_central_eccentricity)
from .steerable_pyramid_freq import Steerable_Pyramid_Freq
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
    zero_thresh : float, optional
        The "cut-off value" below which we consider numbers to be
        zero. We want to determine the number of non-zero elements in
        each window (in order to properly average them), but after
        projecting (and interpolating) the windows from polar into
        rectangular coordinates, we end up with some values very near
        zero (on the order 1e-40 to 1e-30). These are so small that they
        don't matter for actually computing the values within the
        windows but they will mess up our calculation of the number of
        non-zero elements in each window, so we treat all numbers below
        ``zero_thresh`` as being zero for the purpose of computing
        ``window_num_pixels``.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    windows : list
        A list of 3d tensors containing the pooling windows in which the
        model parameters are averaged. Each entry in the list
        corresponds to a different scale and thus is a different size
        (though they should all have the same number of windows)
    window_num_pixels : list
        A list of 1d tensors containing the number of non-zero elements
        in each window; we use this to correctly average within each
        window. Each entry in the list corresponds to a different scale
        (they should all have the same number of elements).
    state_dict_sparse : dict
        A dictionary containing those attributes necessary to initialize
        the model, plus a 'model_name' field. This is used for
        saving/loading the models, since we don't want to keep the (very
        large) representation and intermediate steps around. To save,
        use ``self.save_sparse(filename)``, and then load from that same
        file using the class method ``po.simul.VentralModel(filename)``
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
        different scale, as in ``windows`` and
        ``window_num_pixels``. See above for explanation of the
        dictionaries. To visualize these, see the ``plot_window_sizes``
        method.
    n_polar_windows : int
        The number of windows we have in the polar angle dimension
        (within each eccentricity band)
    n_eccentricity_bands : int
        The number of eccentricity bands in our model

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5, max_eccentricity=15, num_scales=1,
                 zero_thresh=1e-20):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if img_res[0] != img_res[1]:
            raise Exception("For now, we only support square images!")
        self.scaling = scaling
        self.min_eccentricity = min_eccentricity
        self.max_eccentricity = max_eccentricity
        self.windows = []
        self.window_num_pixels = []
        self.window_width_pixels = []
        ecc_window_width = calc_eccentricity_window_width(min_eccentricity, max_eccentricity,
                                                          scaling=scaling)
        self.n_polar_windows = round(calc_angular_n_windows(ecc_window_width / 2))
        angular_window_width = calc_angular_window_width(self.n_polar_windows)
        window_widths = calc_window_widths_actual(angular_window_width, ecc_window_width,
                                                  min_eccentricity, max_eccentricity)
        self.window_width_degrees = dict(zip(['radial_top', 'radial_full', 'angular_top',
                                              'angular_full'], window_widths))
        self.state_dict_sparse = {'scaling': scaling, 'img_res': img_res,
                                  'min_eccentricity': min_eccentricity, 'zero_thresh': zero_thresh,
                                  'max_eccentricity': max_eccentricity}
        for i in range(num_scales):
            windows, theta, ecc = create_pooling_windows(scaling, min_eccentricity,
                                                         max_eccentricity,
                                                         ecc_n_steps=img_res[0] // 2**i,
                                                         theta_n_steps=img_res[1] // 2**i)

            windows = torch.tensor([pt.project_polar_to_cartesian(w) for w in windows],
                                   dtype=torch.float32, device=self.device)
            # need this to be float32 so we can divide the representation by it.
            self.window_num_pixels.append((windows > zero_thresh).sum((1, 2), dtype=torch.float32))
            self.windows.append(windows)
            # we convert from degrees to pixels here, by multiplying the
            # width in degrees by (radius in pixels) / (radius in degrees)
            deg_to_pix = (img_res[0] / (2**(i+1))) / max_eccentricity
            # each value is a list, so we need to use list comprehension
            # to scale them all appropriately
            self.window_width_pixels.append(dict((k, [i*deg_to_pix for i in v]) for k, v in
                                                 self.window_width_degrees.copy().items()))
        self.n_eccentricity_bands = int(self.windows[0].shape[0] // self.n_polar_windows)

    def save_sparse(self, file_path):
        r"""save the relevant parameters to make saving/loading more efficient

        This saves self.state_dict_sparse, which, by default, just
        contains scaling, img_res, min_eccentricity, max_eccentricity,
        zero_thresh

        Parameters
        ----------
        file_path : str
            The path to save the model object to

        """
        torch.save(self.state_dict_sparse, file_path)

    @classmethod
    def load_sparse(cls, file_path):
        r"""load from the dictionary put together by ``save_sparse``

        Parameters
        ----------
        file_path : str
            The path to load the model object from
        """
        state_dict_sparse = torch.load(file_path)
        return cls.from_state_dict_sparse(state_dict_sparse)

    @classmethod
    def from_state_dict_sparse(cls, state_dict_sparse):
        r"""load from the dictionary put together by ``save_sparse``

        Parameters
        ----------
        state_dict_sparse : dict
            The sparse state dict to load
        """
        state_dict_sparse = state_dict_sparse.copy()
        model_name = state_dict_sparse.pop('model_name')
        # want to remove class if it's here
        state_dict_sparse.pop('class', None)
        if model_name == 'RGC':
            return RetinalGanglionCells(**state_dict_sparse)
        elif model_name == 'V1':
            return PrimaryVisualCortex(**state_dict_sparse)
        else:
            raise Exception("Don't know how to handle model_name %s!" % model_name)

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
        if units == 'degrees':
            data = self.window_width_degrees
        elif units == 'pixels':
            data = self.window_width_pixels[scale_num]
        else:
            raise Exception("units must be one of {'pixels', 'degrees'}, not %s!" % units)
        ecc_window_width = calc_eccentricity_window_width(self.min_eccentricity,
                                                          self.max_eccentricity,
                                                          scaling=self.scaling)
        central_ecc = calc_windows_central_eccentricity(len(data['radial_top']), ecc_window_width,
                                                        self.min_eccentricity)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if jitter is not None:
            jitter_vals = {'radial': -jitter, 'angular': jitter}
        else:
            jitter_vals = {'radial': 0, 'angular': 0}
        keys = ['radial_top', 'radial_full', 'angular_top', 'angular_full']
        marker_styles = ['C0o', 'C0.', 'C1o', 'C1.']
        line_styles = ['C0-', 'C0-', 'C1-', 'C1-']
        for k, m, l in zip(keys, marker_styles, line_styles):
            ax.stem(np.array(central_ecc)+jitter_vals[k.split('_')[0]], data[k], l, m, label=k,
                    use_line_collection=True)
        ax.set_ylabel('Window size (%s)' % units)
        ax.set_xlabel('Window central eccentricity (%s)' % units)
        ax.legend(loc='upper left')
        return fig

    def _representation_for_plotting(self):
        r"""Get the representation in the form required for plotting

        VentralStream objects' representation is a straightforward 1d
        tensor. However, that hides a lot of structure to the
        representation: each consists of some number of different
        representation types, each averaged per window. And the windows
        themselves are structured: we have several different
        eccentricity bands, each of which contains the same number of
        angular windows. We want to use this structure when plotting the
        representation, as it makes it easier to see what's goin on. So
        we take a copy of the representation and make it 2d, separating
        out each representation type. We then take each representation
        type and add ``np.nan`` between each eccentricity band. This way
        we can make a separate plot for each representation type and, at
        a glance, see the eccentricity bands separated out.

        We expect this to be plotted using ``plt.stem``, and return a
        tuple ``xvals`` for use with ``plt.hlines`` to replace the base
        line (by default, ``plt.stem`` doesn't insert a break in the
        baseline if there's a NaN in the data, but we want that for ease
        of visualization)

        Returns
        -------
        representation_copy : np.array
            The expanded copy of the representation, which is now 2d,
            ``(num_representation_types, num_windows+x)`` (where ``x``
            is the number of NaNs we've inserted), and contains np.nan
            between each eccentricity band
        xvals : tuple
            A 2-tuple of lists, containing the start (``xvals[0]``) and
            stop (``xvals[1]``) x values for plotting. For use with
            plt.hlines, like so: ``plt.hlines(len(xvals[0])*[0],
            xvals[0], xvals[1])``

        """
        representation_len = int(self.n_polar_windows * self.n_eccentricity_bands)
        # we can't compute this during initialization, but could move it
        # to the forward pass if it looks useful...
        num_representation_types = int(self.representation.shape[0] / representation_len)
        rep_copy = np.nan*np.empty((num_representation_types,
                                    representation_len+self.n_eccentricity_bands))
        xvals = ([], [])
        for i in range(self.n_eccentricity_bands):
            new_idx = (int(i*self.n_polar_windows), int((i+1)*self.n_polar_windows))
            xvals[0].append(new_idx[0]+i)
            xvals[1].append(new_idx[1]+(i-1))
            for j in range(num_representation_types):
                old_idx = [num + j*representation_len for num in new_idx]
                rep_copy[j, new_idx[0]+i:new_idx[1]+i] = self.representation[old_idx[0]:
                                                                             old_idx[1]]
        return rep_copy, xvals

    @staticmethod
    def _plot_representation(ax, data, xvals, title, ylim):
        r"""convenience wrapper for plotting representation

        This plots the data, baseline, cleans up the axis, and sets the
        title

        Should not be called by users directly, but helper function for
        the various plot_representation() functions

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            The axis to plot the data on
        data : np.array
            The data to plot (as a stem plot)
        xvals : tuple
            A 2-tuple of lists, containing the start (``xvals[0]``) and
            stop (``xvals[1]``) x values for plotting.
        title : str
            The title to put on the axis
        ylim : tuple or None, optional
            If not None, the y-limits to use for this plot. If None, we
            use the default, slightly adjusted so that the minimum is 0

        Returns
        -------
        ax : matplotlib.pyplot.axis
            The axis with the plot

        """
        ax.stem(data, basefmt=' ', use_line_collection=True)
        ax.hlines(len(xvals[0])*[0], xvals[0], xvals[1], colors='C3', zorder=10)
        ax = clean_up_axes(ax, ylim, ['top', 'right', 'bottom'])
        ax.set_title(title)
        return ax

    def _update_plot(self, axes):
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

        Note that this means we DO NOT accept the data to update on the
        plot; we grab it from the model's representation. This means
        you'll probably need to do a fake update of the representation
        by setting the value fo the ``representation`` attribute
        directly. This is a little indirect, but means that we can rely
        on our ``_representation_for_plotting`` function to get the data
        in the right shape

        In order for this to be used by ``FuncAnimation``, we need to
        return Artists, so we return a list of the relevant artists, the
        ``markerline`` and ``stemlines`` from the ``StemContainer``.

        Parameters
        ----------
        axes : list
            A list of axes to update. We assume that these are the axes
            created by ``plot_representation`` and so contain stem plots
            in the correct order.

        Returns
        -------
        stem_artists : list
            A list of the artists used to update the information on the
            stem plots

        """
        stem_artists = []
        axes = [ax for ax in axes if len(ax.containers) == 1]
        data, _ = self._representation_for_plotting()
        for ax, d in zip(axes, data):
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
    zero_thresh : float, optional
        The "cut-off value" below which we consider numbers to be
        zero. We want to determine the number of non-zero elements in
        each window (in order to properly average them), but after
        projecting (and interpolating) the windows from polar into
        rectangular coordinates, we end up with some values very near
        zero (on the order 1e-40 to 1e-30). These are so small that they
        don't matter for actually computing the values within the
        windows but they will mess up our calculation of the number of
        non-zero elements in each window, so we treat all numbers below
        ``zero_thresh`` as being zero for the purpose of computing
        ``window_num_pixels``.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    windows : torch.tensor
        A list of 3d tensors containing the pooling windows in which the
        pixel intensities are averaged. Each entry in the list
        corresponds to a different scale and thus is a different size
        (though they should all have the same number of windows)
    window_num_pixels : list
        A list of 1d tensors containing the number of non-zero elements
        in each window; we use this to correctly average within each
        window. Each entry in the list corresponds to a different scale
        (they should all have the same number of elements).
    image : torch.tensor
        A 2d containing the image most recently analyzed.
    windowed_image : torch.tensor
        A 3d tensor containing windowed views of ``self.image``
    representation : torch.tensor
        A flattened (ergo 1d) tensor containing the averages of the
        pixel intensities within each pooling window for ``self.image``
    state_dict_sparse : dict
        A dictionary containing those attributes necessary to initialize
        the model, plus a 'model_name' field. This is used for
        saving/loading the models, since we don't want to keep the (very
        large) representation and intermediate steps around. To save,
        use ``self.save_sparse(filename)``, and then load from that same
        file using the class method ``po.simul.VentralModel(filename)``
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
        different scale, as in ``windows`` and
        ``window_num_pixels``. See above for explanation of the
        dictionaries. To visualize these, see the ``plot_window_sizes``
        method.
    n_polar_windows : int
        The number of windows we have in the polar angle dimension
        (within each eccentricity band)
    n_eccentricity_bands : int
        The number of eccentricity bands in our model

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5, max_eccentricity=15,
                 zero_thresh=1e-20):
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity,
                         zero_thresh=zero_thresh)
        self.state_dict_sparse.update({'model_name': 'RGC'})
        self.image = None
        self.windowed_image = None
        self.representation = None

    def forward(self, image):
        r"""Generate the RGC representation of an image

        Parameters
        ----------
        image : torch.tensor
            A 2d tensor containing the image to analyze.

        Returns
        -------
        representation : torch.tensor
            A flattened (ergo 1d) tensor containing the averages of the
            pixel intensities within each pooling window for ``image``

        """
        self.image = image.clone().detach()
        self.windowed_image = torch.einsum('jk,ijk->ijk', [image, self.windows[0]])
        # we want to normalize by the size of each window
        representation = self.windowed_image.sum((1, 2))
        self.representation = representation / self.window_num_pixels[0]
        return self.representation

    def plot_representation(self, figsize=(10, 5), ylim=None, ax=None, title=None):
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
        rep_copy, xvals = self._representation_for_plotting()
        if title is None:
            title = 'Mean pixel intensity in each window'
        self._plot_representation(ax, rep_copy[0], xvals, title, ylim)
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
    zero_thresh : float, optional
        The "cut-off value" below which we consider numbers to be
        zero. We want to determine the number of non-zero elements in
        each window (in order to properly average them), but after
        projecting (and interpolating) the windows from polar into
        rectangular coordinates, we end up with some values very near
        zero (on the order 1e-40 to 1e-30). These are so small that they
        don't matter for actually computing the values within the
        windows but they will mess up our calculation of the number of
        non-zero elements in each window, so we treat all numbers below
        ``zero_thresh`` as being zero for the purpose of computing
        ``window_num_pixels``.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    num_scales : int, optional
        The number of scales (spatial frequency bands) in the steerable
        pyramid we use to build the V1 representation
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
    window_num_pixels : list
        A list of 1d tensors containing the number of non-zero elements
        in each window; we use this to correctly average within each
        window. Each entry in the list corresponds to a different scale
        (they should all have the same number of elements).
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
        A flattened (ergo 1d) tensor containing the averages of the
        'complex cell responses' (that is, the squared, summed, and
        square-rooted outputs of the complex steerable pyramid) and the
        mean luminance of the image in the pooling windows.
    state_dict_sparse : dict
        A dictionary containing those attributes necessary to initialize
        the model, plus a 'model_name' field. This is used for
        saving/loading the models, since we don't want to keep the (very
        large) representation and intermediate steps around. To save,
        use ``self.save_sparse(filename)``, and then load from that same
        file using the class method ``po.simul.VentralModel(filename)``
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
        different scale, as in ``windows`` and
        ``window_num_pixels``. See above for explanation of the
        dictionaries. To visualize these, see the ``plot_window_sizes``
        method.
    n_polar_windows : int
        The number of windows we have in the polar angle dimension
        (within each eccentricity band)
    n_eccentricity_bands : int
        The number of eccentricity bands in our model

    """
    def __init__(self, scaling, img_res, num_scales=4, order=3, min_eccentricity=.5,
                 max_eccentricity=15, zero_thresh=1e-20):
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity, num_scales,
                         zero_thresh)
        self.state_dict_sparse.update({'order': order, 'model_name': 'V1',
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
            A 2d tensor containing the image to analyze.

        Returns
        -------
        representation : torch.tensor
            A flattened (ergo 1d) tensor containing the averages of the
            'complex cell responses', that is, the squared and summed
            outputs of the complex steerable pyramid.

        """
        while image.ndimension() < 4:
            image = image.unsqueeze(0)
        self.image = image.clone().detach()
        self.pyr_coeffs = self.complex_steerable_pyramid(image)
        self.complex_cell_responses = dict((k, complex_modulus(v)) for k, v in
                                           self.pyr_coeffs.items() if not isinstance(k, str))
        self.windowed_complex_cell_responses = dict(
            (k, torch.einsum('ijkl,wkl->ijwkl', [v, self.windows[k[0]]]))
            for k, v in self.complex_cell_responses.items())
        windowed_image = torch.einsum('ijkl,wkl->ijwkl', [image, self.windows[0]])
        # we want to normalize by the size of each window
        mean_luminance = windowed_image.sum((-1, -2))
        self.mean_luminance = (mean_luminance / self.window_num_pixels[0]).flatten()
        mean_complex_cells = torch.cat([(v.sum((-1, -2)) / self.window_num_pixels[k[0]]).flatten()
                                        for k, v in self.windowed_complex_cell_responses.items()])
        self.representation = torch.cat([mean_complex_cells, self.mean_luminance])
        return self.representation

    def plot_representation(self, figsize=(25, 15), ylim=None, ax=None, titles=None):
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
        rep_copy, xvals = self._representation_for_plotting()
        if titles is None:
            titles = ["scale %02d, band %02d" % (h, b) for h, b in
                      itertools.product(range(self.num_scales), range(self.order+1))]
            titles += ['Mean pixel intensity']
        for i in range(self.num_scales):
            for j in range(self.order+1):
                ax = fig.add_subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = self._plot_representation(ax, rep_copy[i*self.num_scales+j], xvals,
                                               titles[i*self.num_scales+j], ylim)
                axes.append(ax)
        ax = fig.add_subplot(gs[self.num_scales-1:self.num_scales+1, 2*(self.order+1):])
        ax = self._plot_representation(ax, rep_copy[-1], xvals, titles[-1], ylim)
        axes.append(ax)
        return fig, axes
