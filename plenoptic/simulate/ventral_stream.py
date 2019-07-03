"""functions for ventral stream perceptual models, as seen in Freeman and Simoncelli, 2011

"""
import torch
from torch import nn
import numpy as np
import pyrtools as pt
from ..tools.fit import complex_modulus
from .pooling import create_pooling_windows
from .steerable_pyramid_freq import Steerable_Pyramid_Freq


class VentralModel(nn.Module):
    """Generic class that everyone inherits. Sets up the scaling windows

    This just generates the pooling windows necessary for these models, given a small number of
    parameters. One tricky thing we do is generate a set of scaling windows for each scale
    (appropriately) sized. For example, the V1 model will have 4 scales, so for a 256 x 256 image,
    the coefficients will have shape (256, 256), (128, 128), (64, 64), and (32, 32). Therefore, we
    need windows of the same size (could also up-sample the coefficient tensors, but since that
    would need to happen each iteration of the metamer synthesis, pre-generating appropriately
    sized windows is more efficient).

    Parameters
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows. Other pooling windows
        parameters (``radial_to_circumferential_ratio``, ``transition_region_width``) cannot be set
        here. If that ends up being of interest, will change that.
    img_res : tuple
        The resolution of our image (should therefore contains integers). Will use this to generate
        appropriately sized pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    num_scales : int
        The number of scales to generate masks for. For the RGC model, this should be 1, otherwise
        should match the number of scales in the steerable pyramid.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    windows : list
        A list of 3d tensors containing the pooling windows in which the model parameters are
        averaged. Each entry in the list corresponds to a different scale and thus is a different
        size (though they should all have the same number of windows)
    window_sizes : list
        A list of 1d tensors containing the number of non-zero elements in each window; we use this
        to correctly average within each window. Each entry in the list corresponds to a different
        scale (they should all have the same number of elements).

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5, max_eccentricity=15, num_scales=1,
                 zero_thresh=1e-20):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.scaling = scaling
        self.min_eccentricity = min_eccentricity
        self.max_eccentricity = max_eccentricity
        self.windows = []
        self.window_sizes = []
        for i in range(num_scales):
            windows, theta, ecc = create_pooling_windows(scaling, min_eccentricity,
                                                         max_eccentricity,
                                                         ecc_n_steps=img_res[0] // 2**i,
                                                         theta_n_steps=img_res[1] // 2**i)

            windows = torch.tensor([pt.project_polar_to_cartesian(w) for w in windows],
                                   dtype=torch.float32, device=self.device)
            # need this to be float32 so we can divide the representation by it.
            self.window_sizes.append((windows > zero_thresh).sum((1, 2), dtype=torch.float32))
            self.windows.append(windows)


class RetinalGanglionCells(VentralModel):
    """A wildly simplistic model of retinal ganglion cells (RGCs)

    This model averages together the pixel intensities in each of its pooling windows to generate a
    super simple representation. Currently, does not do anything to model the optics of the eye (no
    lens point-spread function), the photoreceptors (no cone lattice), or the center-surround
    nature of retinal ganglion cells' receptive fields.

    Parameters
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows. Other pooling windows
        parameters (``radial_to_circumferential_ratio``, ``transition_region_width``) cannot be set
        here. If that ends up being of interest, will change that.
    img_res : tuple
        The resolution of our image (should therefore contains integers). Will use this to generate
        appropriately sized pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    windows : torch.tensor
        A list of 3d tensors containing the pooling windows in which the pixel intensities are
        averaged. Each entry in the list corresponds to a different scale and thus is a different
        size (though they should all have the same number of windows)
    window_sizes : list
        A list of 1d tensors containing the number of non-zero elements in each window; we use this
        to correctly average within each window. Each entry in the list corresponds to a different
        scale (they should all have the same number of elements).
    image : torch.tensor
        A 2d containing the image most recently analyzed.
    windowed_image : torch.tensor
        A 3d tensor containing windowed views of ``self.image``
    representation : torch.tensor
        A flattened (ergo 1d) tensor containing the averages of the pixel intensities within each
        pooling window for ``self.image``

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5, max_eccentricity=15,
                 zero_thresh=1e-20):
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity,
                         zero_thresh=zero_thresh)
        self.image = None
        self.windowed_image = None
        self.representation = None

    def forward(self, image):
        """Generate the RGC representation of an image

        Parameters
        ----------
        image : torch.tensor
            A 2d tensor containing the image to analyze.

        Returns
        -------
        representation : torch.tensor
            A flattened (ergo 1d) tensor containing the averages of the pixel intensities within
            each pooling window for ``image``

        """
        self.image = image
        self.windowed_image = torch.einsum('jk,ijk->ijk', [self.image, self.windows[0]])
        # we want to normalize by the size of each window
        representation = self.windowed_image.sum((1, 2))
        self.representation = representation / self.window_sizes[0]
        return self.representation


class PrimaryVisualCortex(VentralModel):
    """Model V1 using the Steerable Pyramid

    This just models V1 as containing complex cells: we take the outputs of the steerable pyramid
    and square and sum them.

    Parameters
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows. Other pooling windows
        parameters (``radial_to_circumferential_ratio``, ``transition_region_width``) cannot be set
        here. If that ends up being of interest, will change that.
    img_res : tuple
        The resolution of our image (should therefore contains integers). Will use this to generate
        appropriately sized pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    windows : torch.tensor
        A list of 3d tensors containing the pooling windows in which the complex cell responses are
        averaged. Each entry in the list corresponds to a different scale and thus is a different
        size (though they should all have the same number of windows)
    window_sizes : list
        A list of 1d tensors containing the number of non-zero elements in each window; we use this
        to correctly average within each window. Each entry in the list corresponds to a different
        scale (they should all have the same number of elements).
    image : torch.tensor
        A 2d containing the most recent image analyzed.
    pyr_coeffs : dict
        The dictionary containing the (complex-valued) coefficients of the steerable pyramid built
        on ``self.image``. Each of these is 5d: ``(1, 1, *img_res, 2)``. The first two dimensions
        are for batch and channel, the last dimension contains the real and imaginary components of
        the complex number; channel is unnecessary for us but we might be able to get batch
        working.
    complex_cell_responses : dict
        Dictionary containing the complex cell responses, the squared and summed (i.e., the squared
        complex modulus) of ``self.pyr_coeffs``. Does not include the residual high- and low-pass
        bands. Each of these is now 4d: ``(1, 1, *img_res)``.
    windowed_complex_cell_responses : dict
        Dictionary containing the windowed complex cell responses. Each of these is 5d: ``(1, 1, W,
        *img_res)``, where ``W`` is the number of windows (which depends on the ``scaling``
        parameter).
    representation : torch.tensor
        A flattened (ergo 1d) tensor containing the averages of the 'complex cell responses', that
        is, the squared and summed outputs of the complex steerable pyramid.

    """
    def __init__(self, scaling, img_res, num_scales=4, order=3, min_eccentricity=.5,
                 max_eccentricity=15, zero_thresh=1e-20):
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity, num_scales,
                         zero_thresh)
        self.num_scales = num_scales
        self.order = order
        self.complex_steerable_pyramid = Steerable_Pyramid_Freq(img_res, self.num_scales,
                                                                self.order, is_complex=True)
        self.image = None
        self.pyr_coeffs = None
        self.complex_cell_responses = None
        self.windowed_complex_cell_responses = None
        self.representation = None

    def forward(self, image):
        """Generate the V1 representation of an image

        Parameters
        ----------
        image : torch.tensor
            A 2d tensor containing the image to analyze.

        Returns
        -------
        representation : torch.tensor
            A flattened (ergo 1d) tensor containing the averages of the 'complex cell responses',
            that is, the squared and summed outputs of the complex steerable pyramid.

        """
        self.image = image
        while image.ndimension() < 4:
            image = image.unsqueeze(0)
        self.pyr_coeffs = self.complex_steerable_pyramid(image)
        # SHOULD THIS BE COMPLEX MODULUS (sqrt) OR SQUARED? (in which case we've just squared and
        # summed); paper seems to describe both
        self.complex_cell_responses = dict((k, complex_modulus(v)**2) for k, v in
                                           self.pyr_coeffs.items() if not isinstance(k, str))
        self.windowed_complex_cell_responses = dict(
            (k, torch.einsum('ijkl,wkl->ijwkl', [v, self.windows[k[0]]]))
            for k, v in self.complex_cell_responses.items())
        self.representation = torch.cat([(v.sum((-1, -2)) / self.window_sizes[k[0]]).flatten()
                                         for k, v in self.windowed_complex_cell_responses.items()])
        return self.representation
