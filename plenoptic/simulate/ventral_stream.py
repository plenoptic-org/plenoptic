"""functions for ventral stream perceptual models, as seen in Freeman and Simoncelli, 2011

"""
import torch
from torch import nn
import numpy as np
import pyrtools as pt
from ..tools.fit import complex_modulus
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
        n_polar_windows = calc_angular_n_windows(ecc_window_width / 2)
        angular_window_width = calc_angular_window_width(round(n_polar_windows))
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
            ax.stem(np.array(central_ecc)+jitter_vals[k.split('_')[0]], data[k], l, m, label=k)
        ax.set_ylabel('Window size (%s)' % units)
        ax.set_xlabel('Window central eccentricity (%s)' % units)
        ax.legend(loc='upper left')
        return fig


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
