"""
Portilla-Simoncelli texture statistics.

The Portilla-Simoncelli (PS) texture statistics are a set of image
statistics, first described in [1]_, that are proposed as a sufficient set
of measurements for describing visual textures. That is, if two texture
images have the same values for all PS texture stats, humans should
consider them as members of the same family of textures.

References
----------
.. [1] J Portilla and E P Simoncelli. A Parametric Texture Model based on
   Joint Statistics of Complex Wavelet Coefficients. Int'l Journal of
   Computer Vision. 40(1):49-71, October, 2000.
   https://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
   https://www.cns.nyu.edu/~lcv/texture/
"""

from collections import OrderedDict
from typing import Literal

import einops
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft
import torch.nn as nn
from torch import Tensor

from ...tools import signal, stats
from ...tools.data import to_numpy
from ...tools.display import clean_stem_plot, clean_up_axes, update_stem
from ...tools.validate import validate_input
from ..canonical_computations.steerable_pyramid_freq import (
    SCALES_TYPE as PYR_SCALES_TYPE,
)
from ..canonical_computations.steerable_pyramid_freq import (
    SteerablePyramidFreq,
)

SCALES_TYPE = Literal["pixel_statistics"] | PYR_SCALES_TYPE


class PortillaSimoncelli(nn.Module):
    r"""
    Portila-Simoncelli texture statistics.

    The Portilla-Simoncelli (PS) texture statistics are a set of image
    statistics, first described in [1]_, that are proposed as a sufficient set
    of measurements for describing visual textures. That is, if two texture
    images have the same values for all PS texture stats, humans should
    consider them as members of the same family of textures.

    The PS stats are computed based on the
    :class:`~plenoptic.simulate.canonical_computations.steerable_pyramid_freq.SteerablePyramidFreq`
    [2]_. They consist of the local auto-correlations, cross-scale
    (within-orientation) correlations, and cross-orientation (within-scale)
    correlations of both the pyramid coefficients and the local energy (as
    computed by those coefficients). Additionally, they include the first four
    global moments (mean, variance, skew, and kurtosis) of the image and
    down-sampled versions of that image. See the paper and notebook for more
    description.

    Parameters
    ----------
    image_shape
        Shape of input image.
    n_scales
        The number of pyramid scales used to measure the statistics.
    n_orientations
        The number of orientations used to measure the statistics.
    spatial_corr_width
        The width of the spatial cross- and auto-correlation statistics.

    Attributes
    ----------
    scales: list
        The names of the unique scales of coefficients in the pyramid, used for
        coarse-to-fine metamer synthesis.

    Raises
    ------
    ValueError
        If the height or width of ``image`` cannot be divided by 2 ``n_scales``
        times. This is necessary because of how the model handles multiscale
        representations.

    References
    ----------
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model based on
       Joint Statistics of Complex Wavelet Coefficients. Int'l Journal of
       Computer Vision. 40(1):49-71, October, 2000.
       https://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
       https://www.cns.nyu.edu/~lcv/texture/
    .. [2] E P Simoncelli and W T Freeman, "The Steerable Pyramid: A Flexible
       Architecture for Multi-Scale Derivative Computation," Second Int'l Conf
       on Image Processing, Washington, DC, Oct 1995.
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        n_scales: int = 4,
        n_orientations: int = 4,
        spatial_corr_width: int = 9,
    ):
        super().__init__()

        self.image_shape = image_shape
        if any([(image_shape[-1] / 2**i) % 2 for i in range(n_scales)]) or any(
            [(image_shape[-2] / 2**i) % 2 for i in range(n_scales)]
        ):
            raise ValueError(
                "Because of how the Portilla-Simoncelli model handles "
                "multiscale representations, it only works with images"
                " whose shape can be divided by 2 `n_scales` times."
            )
        self.spatial_corr_width = spatial_corr_width
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self._pyr = SteerablePyramidFreq(
            self.image_shape,
            height=self.n_scales,
            order=self.n_orientations - 1,
            is_complex=True,
            tight_frame=False,
        )

        self.scales = (
            ["pixel_statistics", "residual_lowpass"]
            + [ii for ii in range(n_scales - 1, -1, -1)]
            + ["residual_highpass"]
        )

        # Dictionary defining shape of the statistics and which scale they're
        # associated with
        scales_shape_dict = self._create_scales_shape_dict()

        # Dictionary defining necessary statistics, that is, those that are not
        # redundant
        self._necessary_stats_dict = self._create_necessary_stats_dict(
            scales_shape_dict
        )
        # turn this into tensor we can use in forward pass. first into a
        # boolean mask...
        _necessary_stats_mask = einops.pack(
            list(self._necessary_stats_dict.values()), "*"
        )[0]
        # then into a tensor of indices
        _necessary_stats_mask = torch.where(_necessary_stats_mask)[0]
        self.register_buffer("_necessary_stats_mask", _necessary_stats_mask)

        # This array is composed of the following values: 'pixel_statistics',
        # 'residual_lowpass', 'residual_highpass' and integer values from 0 to
        # self.n_scales-1. It is the same size as the representation tensor
        # returned by this object's forward method. It must be a numpy array so
        # we can have a mixture of ints and strs (and so we can use np.isin
        # later)
        self._representation_scales = einops.pack(
            list(scales_shape_dict.values()), "*"
        )[0]
        # just select the scales of the necessary stats.
        self._representation_scales = self._representation_scales[
            self._necessary_stats_mask
        ]
        # This model has no trainable parameters, so it's always in eval mode
        self.eval()

    def _create_scales_shape_dict(self) -> OrderedDict:
        """
        Create dictionary defining scales and shape of each stat.

        This dictionary functions as metadata which is used for two main
        purposes:

        - Scale assignment. In order for optimization to work well, we proceed
          in a "coarse-to-fine" manner. That is, we start optimization by only
          considering the statistics related to the lowest frequencies, and
          gradually add in those related to higher and higher frequencies. This
          is similar to blurring the objective function and then gradually
          adding in finer and finer details. The numbers in this dictionary map
          the computed statistics to their corresponding scales, which we use
          in remove_scales to throw away some stats as needed.

        - Redundant stat identification. As described at the bottom of the
          notebook, the model incidentally computes a whole bunch of redundant
          stats, because auto- and cross-correlation matrices have certain
          symmetries. the _create_necessary_stats_dict method accepts the
          dictionary created here as input and uses the values to get the
          shapes of these and insert True/False as necessary.

        Returns
        -------
        scales_shape_dict
           Dictionary defining shape and associated scales of each computed
           statistic. The keys name each statistic, with dummy arrays as
           values. These arrays have the same shape as the stat (excluding
           batch and channel), with values defining which scale they correspond
           to.
        """
        shape_dict = OrderedDict()
        # There are 6 pixel statistics
        shape_dict["pixel_statistics"] = np.array(6 * ["pixel_statistics"])

        # These are the basic building blocks of the scale assignments for many
        # of the statistics calculated by the PortillaSimoncelli model.
        scales = np.arange(self.n_scales)
        # the cross-scale correlations exclude the coarsest scale
        scales_without_coarsest = np.arange(self.n_scales - 1)
        # the statistics computed on the reconstructed bandpass images have an
        # extra scale corresponding to the lowpass residual
        scales_with_lowpass = np.array(
            scales.tolist() + ["residual_lowpass"], dtype=object
        )

        # now we go through each statistic in order and create a dummy array
        # full of 1s with the same shape as the actual statistic (excluding the
        # batch and channel dimensions, as each stat is computed independently
        # across those dimensions). We then multiply it by one of the scales
        # arrays above to turn those 1s into values describing the
        # corresponding scale.

        auto_corr_mag = np.ones(
            (
                self.spatial_corr_width,
                self.spatial_corr_width,
                self.n_orientations,
                self.n_scales,
            ),
            dtype=int,
        )
        # this rearrange call is turning scales from 1d with shape (n_scales, )
        # to 4d with shape (1, 1, n_scales, 1), so that it matches
        # auto_corr_mag. the following rearrange calls do similar.
        auto_corr_mag *= einops.rearrange(scales, "s -> 1 1 1 s")
        shape_dict["auto_correlation_magnitude"] = auto_corr_mag

        shape_dict["skew_reconstructed"] = scales_with_lowpass

        shape_dict["kurtosis_reconstructed"] = scales_with_lowpass

        auto_corr = np.ones(
            (
                self.spatial_corr_width,
                self.spatial_corr_width,
                self.n_scales + 1,
            ),
            dtype=object,
        )
        auto_corr *= einops.rearrange(scales_with_lowpass, "s -> 1 1 s")
        shape_dict["auto_correlation_reconstructed"] = auto_corr

        shape_dict["std_reconstructed"] = scales_with_lowpass

        cross_orientation_corr_mag = np.ones(
            (self.n_orientations, self.n_orientations, self.n_scales),
            dtype=int,
        )
        cross_orientation_corr_mag *= einops.rearrange(scales, "s -> 1 1 s")
        shape_dict["cross_orientation_correlation_magnitude"] = (
            cross_orientation_corr_mag
        )

        mags_std = np.ones((self.n_orientations, self.n_scales), dtype=int)
        mags_std *= einops.rearrange(scales, "s -> 1 s")
        shape_dict["magnitude_std"] = mags_std

        cross_scale_corr_mag = np.ones(
            (self.n_orientations, self.n_orientations, self.n_scales - 1),
            dtype=int,
        )
        cross_scale_corr_mag *= einops.rearrange(scales_without_coarsest, "s -> 1 1 s")
        shape_dict["cross_scale_correlation_magnitude"] = cross_scale_corr_mag

        cross_scale_corr_real = np.ones(
            (self.n_orientations, 2 * self.n_orientations, self.n_scales - 1),
            dtype=int,
        )
        cross_scale_corr_real *= einops.rearrange(scales_without_coarsest, "s -> 1 1 s")
        shape_dict["cross_scale_correlation_real"] = cross_scale_corr_real

        shape_dict["var_highpass_residual"] = np.array(["residual_highpass"])

        return shape_dict

    def _create_necessary_stats_dict(
        self, scales_shape_dict: OrderedDict
    ) -> OrderedDict:
        """
        Create mask specifying the necessary statistics.

        Some of the statistics computed by the model are redundant, due to
        symmetries. For example, about half of the values in the
        autocorrelation matrices are duplicates. See the Portilla-Simoncelli
        notebook for more details.

        Parameters
        ----------
        scales_shape_dict
            Dictionary defining shape and associated scales of each computed
            statistic.

        Returns
        -------
        necessary_stats_dict
            Dictionary defining which statistics are necessary (i.e., not
            redundant). Will have the same keys as scales_shape_dict, with the
            values being boolean tensors of the same shape as
            scales_shape_dict's corresponding values. True denotes the
            statistics that will be included in the model's output, while False
            denotes the redundant ones we will toss.
        """
        mask_dict = scales_shape_dict.copy()
        # Pre-compute some necessary indices.
        # Lower triangular indices (including diagonal), for auto correlations
        tril_inds = torch.tril_indices(self.spatial_corr_width, self.spatial_corr_width)
        # Get the second half of the diagonal, i.e., everything from the center
        # element on. These are all repeated for the auto correlations. (As
        # these are autocorrelations (rather than auto-covariance) matrices,
        # they've been normalized by the variance and so the center element is
        # always 1, and thus uninformative)
        diag_repeated = torch.arange(
            start=self.spatial_corr_width // 2, end=self.spatial_corr_width
        )
        # Upper triangle indices, including diagonal. These are redundant stats
        # for cross_orientation_correlation_magnitude (because we've normalized
        # this matrix to be true cross-correlations, the diagonals are all 1,
        # like for the auto-correlations)
        triu_inds = torch.triu_indices(self.n_orientations, self.n_orientations)
        for k, v in mask_dict.items():
            if k in [
                "auto_correlation_magnitude",
                "auto_correlation_reconstructed",
            ]:
                # Symmetry M_{i,j} = M_{n-i+1, n-j+1}
                # Start with all False, then place True in necessary stats.
                mask = torch.zeros(v.shape, dtype=torch.bool)
                mask[tril_inds[0], tril_inds[1]] = True
                # if spatial_corr_width is even, then the first row is not
                # redundant with anything either
                if np.mod(self.spatial_corr_width, 2) == 0:
                    mask[0] = True
                mask[diag_repeated, diag_repeated] = False
            elif k == "cross_orientation_correlation_magnitude":
                # Symmetry M_{i,j} = M_{j,i}.
                # Start with all True, then place False in redundant stats.
                mask = torch.ones(v.shape, dtype=torch.bool)
                mask[triu_inds[0], triu_inds[1]] = False
            else:
                # all of the other stats have no redundancies
                mask = torch.ones(v.shape, dtype=torch.bool)
            mask_dict[k] = mask
        return mask_dict

    def forward(self, image: Tensor, scales: list[SCALES_TYPE] | None = None) -> Tensor:
        r"""
        Generate Texture Statistics representation of an image.

        Note that separate batches and channels are analyzed in parallel.

        For any representation that contains info across scales, the scales always run
        from fine to coarse, representing all orientations at a given scale before
        moving on.

        Parameters
        ----------
        image
            A 4d tensor (batch, channel, height, width) containing the image(s) to
            analyze.
        scales
            Which scales to include in the returned representation. If None, we
            include all scales. Otherwise, can contain subset of values present
            in this model's ``scales`` attribute, and the returned tensor will
            then contain the subset corresponding to those scales.

        Returns
        -------
        representation_tensor
            3d tensor of shape (batch, channel, stats) containing the measured
            texture statistics.

        Raises
        ------
        ValueError
            If ``image`` is not 4d or has a dtype other than float or complex.

        Examples
        --------
        >>> import plenoptic as po
        >>> img = po.data.curie()
        >>> portilla_simoncelli_model = po.simul.PortillaSimoncelli(img.shape[2:])
        >>> representation_tensor = portilla_simoncelli_model(img)
        >>> representation_tensor.shape
        torch.Size([1, 1, 1046])
        """
        validate_input(image)

        # pyr_dict is the dictionary of complex-valued tensors returned by the
        # steerable pyramid. pyr_coeffs is a list (length n_scales) of 5d
        # tensors, each of shape (batch, channel, scales, n_orientations,
        # height, width) containing the complex-valued oriented bands, while
        # highpass is a real-valued 4d tensor of shape (batch, channel, height,
        # width). Note that the residual lowpass in pyr_dict has been demeaned.
        # We keep both the dict and list of pyramid coefficients because we
        # need the dictionary for reconstructing the image done later on.
        pyr_dict, pyr_coeffs, highpass, _ = self._compute_pyr_coeffs(image)

        # Now, we create several intermediate representations that we'll use to
        # compute the texture statistics later.

        # First, two intermediate dictionaries: magnitude_pyr_coeffs and
        # real_pyr_coeffs, which contain the demeaned magnitude of the pyramid
        # coefficients and the real part of the pyramid coefficients
        # respectively.
        (
            mag_pyr_coeffs,
            real_pyr_coeffs,
        ) = self._compute_intermediate_representations(pyr_coeffs)

        # Then, the reconstructed lowpass image at each scale. (this is a list
        # of length n_scales+1 containing tensors of shape (batch, channel,
        # height, width))
        reconstructed_images = self._reconstruct_lowpass_at_each_scale(pyr_dict)
        # the reconstructed_images list goes from coarse-to-fine, but we want
        # each of the stats computed from it to go from fine-to-coarse, so we
        # reverse its direction.
        reconstructed_images = reconstructed_images[::-1]

        # Now, start calculating the PS texture stats.

        # Calculate pixel statistics (mean, variance, skew, kurtosis, min,
        # max).
        pixel_stats = self._compute_pixel_stats(image)

        # Compute the central autocorrelation of the coefficient magnitudes. This is a
        # tensor of shape: (batch, channel, spatial_corr_width, spatial_corr_width,
        # n_orientations, n_scales). var_mags is a tensor of shape (batch, channel,
        # n_orientations, n_scales)
        autocorr_mags, mags_var = self._compute_autocorr(mag_pyr_coeffs)
        # mags_var is the variance of the magnitude coefficients at each scale (it's an
        # intermediary of the computation of the auto-correlations). We take the square
        # root to get the standard deviation.
        mags_std = mags_var.sqrt()

        # Compute the central autocorrelation of the reconstructed lowpass
        # images at each scale (and their variances). autocorr_recon is a
        # tensor of shape (batch, channel, spatial_corr_width,
        # spatial_corr_width, n_scales+1), and var_recon is a tensor of shape
        # (batch, channel, n_scales+1)
        autocorr_recon, var_recon = self._compute_autocorr(reconstructed_images)
        # Compute the standard deviation, skew, and kurtosis of each
        # reconstructed lowpass image. std_recon, skew_recon, and
        # kurtosis_recon will all end up as tensors of shape (batch, channel,
        # n_scales+1)
        std_recon = var_recon.sqrt()
        skew_recon, kurtosis_recon = self._compute_skew_kurtosis_recon(
            reconstructed_images, var_recon, pixel_stats[..., 1]
        )

        # Compute the cross-orientation correlations between the magnitude
        # coefficients at each scale. this will be a tensor of shape (batch,
        # channel, n_orientations, n_orientations, n_scales)
        cross_ori_corr_mags = self._compute_cross_correlation(
            mag_pyr_coeffs, mag_pyr_coeffs, mags_var, mags_var
        )

        # If we have more than one scale, compute the cross-scale correlations
        if self.n_scales != 1:
            # First, double the phase the coefficients, so we can correctly
            # compute correlations across scales.
            (
                phase_doubled_mags,
                phase_doubled_sep,
            ) = self._double_phase_pyr_coeffs(pyr_coeffs)
            # Compute the cross-scale correlations between the magnitude
            # coefficients. For each coefficient, we're correlating it with the
            # coefficients at the next-coarsest scale. this will be a tensor of
            # shape (batch, channel, n_orientations, n_orientations,
            # n_scales-1)
            cross_scale_corr_mags = self._compute_cross_correlation(
                mag_pyr_coeffs[:-1], phase_doubled_mags, mags_var[..., :-1]
            )
            # Compute the cross-scale correlations between the real
            # coefficients and the real and imaginary coefficients at the next
            # coarsest scale. this will be a tensor of shape (batch, channel,
            # n_orientations, 2*n_orientations, n_scales-1)
            cross_scale_corr_real = self._compute_cross_correlation(
                real_pyr_coeffs[:-1], phase_doubled_sep
            )

        # Compute the variance of the highpass residual
        var_highpass_residual = highpass.pow(2).mean(dim=(-2, -1))

        # Now, combine all these stats together, first into a list
        all_stats = [
            pixel_stats,
            autocorr_mags,
            skew_recon,
            kurtosis_recon,
            autocorr_recon,
            std_recon,
            cross_ori_corr_mags,
            mags_std,
        ]
        if self.n_scales != 1:
            all_stats += [cross_scale_corr_mags, cross_scale_corr_real]
        all_stats += [var_highpass_residual]
        # And then pack them into a 3d tensor
        representation_tensor, pack_info = einops.pack(all_stats, "b c *")

        # the only time when this is None is during testing, when we make sure
        # that our assumptions are all valid.
        if self._necessary_stats_mask is None:
            # store this so we can unpack this info (only possible when we've
            # discarded no stats)
            self._pack_info = pack_info
        else:
            # Throw away all redundant statistics
            representation_tensor = representation_tensor.index_select(
                -1, self._necessary_stats_mask
            )

        # Return the subset of stats corresponding to the specified scale.
        if scales is not None:
            representation_tensor = self.remove_scales(representation_tensor, scales)

        return representation_tensor

    def remove_scales(
        self, representation_tensor: Tensor, scales_to_keep: list[SCALES_TYPE]
    ) -> Tensor:
        """
        Remove statistics not associated with scales.

        For a given representation_tensor and a list of scales_to_keep, this
        attribute removes all statistics *not* associated with those scales.

        Note that calling this method will always remove statistics.

        Parameters
        ----------
        representation_tensor
            3d tensor containing the measured representation statistics.
        scales_to_keep
            Which scales to include in the returned representation. Can contain
            subset of values present in this model's ``scales`` attribute, and
            the returned tensor will then contain the subset of the full
            representation corresponding to those scales.

        Returns
        -------
        limited_representation_tensor
            Representation tensor with some statistics removed.

        Examples
        --------
        >>> import plenoptic as po
        >>> img = po.data.curie()
        >>> portilla_simoncelli_model = po.simul.PortillaSimoncelli(img.shape[2:])
        >>> representation_tensor = portilla_simoncelli_model(img)
        >>> representation_tensor.shape
        torch.Size([1, 1, 1046])
        >>> limited_representation_tensor = portilla_simoncelli_model.remove_scales(
        ...     representation_tensor, scales_to_keep=[0]
        ... )
        >>> limited_representation_tensor.shape
        torch.Size([1, 1, 261])
        """
        # this is necessary because object is the dtype of
        # self._representation_scales
        scales_to_keep = np.array(scales_to_keep, dtype=object)
        # np.isin returns a 1d boolean array of the same shape as
        # self._representation_scales with True at each location where that
        # value appears in scales_to_keep. where then converts this boolean
        # array into indices
        ind = np.where(np.isin(self._representation_scales, scales_to_keep))[0]
        ind = torch.from_numpy(ind).to(representation_tensor.device)
        return representation_tensor.index_select(-1, ind)

    def convert_to_tensor(self, representation_dict: OrderedDict) -> Tensor:
        r"""
        Convert dictionary of statistics to a tensor.

        The output has shape (batch, channel, n_statistics), flattening and
        concatenating across all statistic classes. The dictionary representation
        may be easier to make sense of.

        Parameters
        ----------
        representation_dict
             Dictionary of representation.

        Returns
        -------
        rep
            3d tensor of statistics.

        See Also
        --------
        convert_to_dict
            Convert tensor representation to dictionary.

        Examples
        --------
        >>> import plenoptic as po
        >>> import torch
        >>> img = po.data.curie()
        >>> portilla_simoncelli_model = po.simul.PortillaSimoncelli(img.shape[2:])
        >>> representation_tensor = portilla_simoncelli_model(img)
        >>> representation_dict = portilla_simoncelli_model.convert_to_dict(
        ...     representation_tensor
        ... )
        >>> representation_tensor_new = portilla_simoncelli_model.convert_to_tensor(
        ...     representation_dict
        ... )
        >>> torch.equal(representation_tensor, representation_tensor_new)
        True
        """
        rep = einops.pack(list(representation_dict.values()), "b c *")[0]
        # then get rid of all the nans / unnecessary stats
        return rep.index_select(-1, self._necessary_stats_mask)

    def convert_to_dict(self, representation_tensor: Tensor) -> OrderedDict:
        """
        Convert tensor of statistics to a dictionary.

        While the tensor representation is required by plenoptic's synthesis
        objects, the dictionary representation is easier to manually inspect.

        This dictionary will contain NaNs in its values: these are placeholders
        for the redundant statistics.

        Parameters
        ----------
        representation_tensor
            3d tensor of statistics.

        Returns
        -------
        rep
            Dictionary of representation, with informative keys.

        Raises
        ------
        ValueError
            If ``representation_tensor`` has an unexpected number of elements. This can
            happen if some elements were manually removed from
            ``representation_tensor``, if a non-``None`` value was passed to ``forward``
            when computing it, or if it was computed using a different instantiation of
            the model.

        See Also
        --------
        convert_to_tensor:
            Convert dictionary representation to tensor.

        Examples
        --------
        >>> import plenoptic as po
        >>> img = po.data.curie()
        >>> portilla_simoncelli_model = po.simul.PortillaSimoncelli(
        ...     img.shape[2:], n_scales=3
        ... )
        >>> representation_tensor = portilla_simoncelli_model(img)
        >>> representation_dict = portilla_simoncelli_model.convert_to_dict(
        ...     representation_tensor
        ... )
        >>> # We will go through and examine each of these keys individually
        >>> # Shape is (batch, channel, 6): first four moments plus min and max
        >>> # of input image
        >>> representation_dict["pixel_statistics"].shape
        torch.Size([1, 1, 6])
        >>> # Shape is (batch, channel, spatial_corr_width, spatial_corr_width,
        >>> # n_orientations, n_scales)
        >>> representation_dict["auto_correlation_magnitude"].shape
        torch.Size([1, 1, 9, 9, 4, 3])
        >>> # Shape is (batch, channel, n_scales+1)
        >>> representation_dict["skew_reconstructed"].shape
        torch.Size([1, 1, 4])
        >>> # Shape is (batch, channel, n_scales+1)
        >>> representation_dict["kurtosis_reconstructed"].shape
        torch.Size([1, 1, 4])
        >>> # Shape is (batch, channel, spatial_corr_width, spatial_corr_width,
        >>> # n_scales+1)
        >>> representation_dict["auto_correlation_reconstructed"].shape
        torch.Size([1, 1, 9, 9, 4])
        >>> # Shape is (batch, channel, n_scales+1)
        >>> representation_dict["std_reconstructed"].shape
        torch.Size([1, 1, 4])
        >>> # Shape is (batch, channel, n_orientations, n_orientations, n_scales)
        >>> representation_dict["cross_orientation_correlation_magnitude"].shape
        torch.Size([1, 1, 4, 4, 3])
        >>> # Shape is (batch, channel, n_orientations, n_scales)
        >>> representation_dict["magnitude_std"].shape
        torch.Size([1, 1, 4, 3])
        >>> # Shape is (batch, channel, n_orientations, n_orientations, n_scales-1)
        >>> representation_dict["cross_scale_correlation_magnitude"].shape
        torch.Size([1, 1, 4, 4, 2])
        >>> # Shape is (batch, channel, n_orientations, 2*n_orientations, n_scales-1)
        >>> representation_dict["cross_scale_correlation_real"].shape
        torch.Size([1, 1, 4, 8, 2])
        >>> # Shape is (batch, channel, 1)
        >>> representation_dict["var_highpass_residual"].shape
        torch.Size([1, 1, 1])
        """
        if representation_tensor.shape[-1] != len(self._representation_scales):
            raise ValueError(
                "representation tensor is the wrong length (expected"
                f" {len(self._representation_scales)} but got"
                f" {representation_tensor.shape[-1]})! Did you remove some of"
                " the scales? (i.e., by setting scales in the forward pass)?"
                " convert_to_dict does not support such tensors."
            )

        rep = self._necessary_stats_dict.copy()
        n_filled = 0
        for k, v in rep.items():
            # each statistic is a tensor with batch and channel dimensions as
            # found in representation_tensor and all the other dimensions
            # determined by the values in necessary_stats_dict.
            shape = (*representation_tensor.shape[:2], *v.shape)
            new_v = torch.nan * torch.ones(
                shape,
                dtype=representation_tensor.dtype,
                device=representation_tensor.device,
            )
            # v.sum() gives the number of necessary elements from this stat
            this_stat_vec = representation_tensor[..., n_filled : n_filled + v.sum()]
            # use boolean indexing to put the values from new_stat_vec in the
            # appropriate place
            new_v[..., v] = this_stat_vec
            rep[k] = new_v
            n_filled += v.sum()
        return rep

    def _compute_pyr_coeffs(
        self, image: Tensor
    ) -> tuple[OrderedDict, list[Tensor], Tensor, Tensor]:
        """
        Compute pyramid coefficients of image.

        Note that the residual lowpass has been demeaned independently for each
        batch and channel (and this is true of the lowpass returned separately
        as well as the one included in pyr_coeffs_dict).

        Parameters
        ----------
        image
            4d tensor of shape (batch, channel, height, width) containing the
            image.

        Returns
        -------
        pyr_coeffs_dict
            OrderedDict of containing all pyramid coefficients.
        pyr_coeffs
            List of length n_scales, containing 5d tensors of shape (batch,
            channel, n_orientations, height, width) containing the complex-valued
            oriented bands (note that height and width shrink by half on each
            scale). This excludes the residual highpass and lowpass bands.
        highpass
            The residual highpass as a real-valued 4d tensor (batch, channel,
            height, width).
        lowpass
            The residual lowpass as a real-valued 4d tensor (batch, channel,
            height, width). This tensor has been demeaned (independently for
            each batch and channel).
        """
        pyr_coeffs = self._pyr.forward(image)
        # separate out the residuals and demean the residual lowpass
        lowpass = pyr_coeffs["residual_lowpass"]
        lowpass = lowpass - lowpass.mean(dim=(-2, -1), keepdim=True)
        pyr_coeffs["residual_lowpass"] = lowpass
        highpass = pyr_coeffs["residual_highpass"]

        # This is a list of tensors, one for each scale, where each tensor is
        # of shape (batch, channel, n_orientations, height, width) (note that
        # height and width halves on each scale)
        coeffs_list = [
            torch.stack([pyr_coeffs[(i, j)] for j in range(self.n_orientations)], 2)
            for i in range(self.n_scales)
        ]
        return pyr_coeffs, coeffs_list, highpass, lowpass

    @staticmethod
    def _compute_pixel_stats(image: Tensor) -> Tensor:
        """
        Compute the PS pixel stats: first four moments, min, and max.

        Parameters
        ----------
        image
            4d tensor of shape (batch, channel, height, width) containing input
            image. Stats are computed independently for each batch and channel.

        Returns
        -------
        pixel_stats
            3d tensor of shape (batch, channel, 6) containing the mean,
            variance, skew, kurtosis, minimum pixel value, and maximum pixel
            value (in that order).
        """  # numpydoc ignore=ES01
        mean = torch.mean(image, dim=(-2, -1), keepdim=True)
        # we use torch.var instead of plenoptic.tools.variance, because our
        # variance is the uncorrected (or sample) variance and we want the
        # corrected one here.
        var = torch.var(image, dim=(-2, -1))
        skew = stats.skew(image, mean=mean, var=var, dim=[-2, -1])
        kurtosis = stats.kurtosis(image, mean=mean, var=var, dim=[-2, -1])
        # can't compute min/max over two dims simultaneously with
        # torch.min/max, so use einops
        img_min = einops.reduce(image, "b c h w -> b c", "min")
        img_max = einops.reduce(image, "b c h w -> b c", "max")
        # mean needed to be unflattened to be used by skew and kurtosis
        # correctly, but we'll want it to be flattened like this in the final
        # representation tensor
        return einops.pack([mean, var, skew, kurtosis, img_min, img_max], "b c *")[0]

    @staticmethod
    def _compute_intermediate_representations(
        pyr_coeffs: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """
        Compute useful intermediate representations.

        These representations are:
          1) demeaned magnitude of the pyramid coefficients,
          2) real part of the pyramid coefficients

        These two are used in computing some of the texture representation.

        Parameters
        ----------
        pyr_coeffs
            Complex steerable pyramid coefficients (without residuals), as list
            of length n_scales, containing 5d tensors of shape (batch, channel,
            n_orientations, height, width).

        Returns
        -------
        magnitude_pyr_coeffs
           List of length n_scales, containing 5d tensors of shape (batch,
           channel, n_orientations, height, width) (same as ``pyr_coeffs``),
           containing the demeaned magnitude of the steerable pyramid
           coefficients (i.e., ``coeffs.abs() - coeffs.abs().mean((-2, -1))``).
        real_pyr_coeffs :
           List of length n_scales, containing 5d tensors of shape (batch,
           channel, n_orientations, height, width) (same as ``pyr_coeffs``),
           containing the real components of the coefficients (i.e.
           ``coeffs.real``).
        """
        magnitude_pyr_coeffs = [coeff.abs() for coeff in pyr_coeffs]
        magnitude_means = [
            mag.mean((-2, -1), keepdim=True) for mag in magnitude_pyr_coeffs
        ]
        magnitude_pyr_coeffs = [
            mag - mn for mag, mn in zip(magnitude_pyr_coeffs, magnitude_means)
        ]
        real_pyr_coeffs = [coeff.real for coeff in pyr_coeffs]
        return magnitude_pyr_coeffs, real_pyr_coeffs

    def _reconstruct_lowpass_at_each_scale(
        self, pyr_coeffs_dict: OrderedDict
    ) -> list[Tensor]:
        """
        Reconstruct the lowpass unoriented image at each scale.

        The autocorrelation, standard deviation, skew, and kurtosis of each of
        these images is part of the texture representation.

        Parameters
        ----------
        pyr_coeffs_dict
            Dictionary containing the steerable pyramid coefficients, with the
            lowpass residual demeaned.

        Returns
        -------
        reconstructed_images
            List of length n_scales+1 containing the reconstructed unoriented
            image at each scale, from fine to coarse. The final image is
            reconstructed just from the residual lowpass image. Each is a 4d
            tensor, this is a list because they are all different heights and
            widths.
        """
        reconstructed_images = [
            self._pyr.recon_pyr(pyr_coeffs_dict, levels=["residual_lowpass"])
        ]
        # go through scales backwards
        for lev in range(self.n_scales - 1, -1, -1):
            recon = self._pyr.recon_pyr(pyr_coeffs_dict, levels=[lev])
            reconstructed_images.append(recon + reconstructed_images[-1])
        # now downsample as necessary, so that these end up the same size as
        # their corresponding coefficients. We multiply by the factor of 4 here
        # in order to approximately equalize the steerable pyramid coefficient
        # values across scales. This could also be handled by making the
        # pyramid tight frame
        reconstructed_images[:-1] = [
            signal.shrink(r, 2 ** (self.n_scales - i)) * 4 ** (self.n_scales - i)
            for i, r in enumerate(reconstructed_images[:-1])
        ]
        return reconstructed_images

    def _compute_autocorr(self, coeffs_list: list[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Compute the autocorrelation of some statistics.

        Parameters
        ----------
        coeffs_list
            List (of length s) of tensors of shape (batch, channel, *, height,
            width), where * is zero or one additional dimensions. Intended use
            case: ``magnitude_pyr_coeffs`` (which is list of length ``n_scales`` of 5d
            tensors, with * containing ``n_orientations``) or ``reconstructed_images``
            (which is a list of length ``n_scales+1`` of 4d tensors).

        Returns
        -------
        autocorrs
            Tensor of shape (batch, channel, spatial_corr_width,
            spatial_corr_width, *, s) containing the autocorrelation (up to
            distance ``spatial_corr_width//2``) of each element in
            ``coeffs_list``, computed independently over all but the final two
            dimensions.
        vars
            3d Tensor of shape (batch, channel, *, s) containing the variance
            of each element in ``coeffs_list``, computed independently over all
            but the final two dimensions.

        Raises
        ------
        ValueError
            If ``coeffs_list`` contains tensors that have other than 4 or 5 dimensions.
        """  # numpydoc ignore=ES01
        if coeffs_list[0].ndim == 5:
            dims = "o"
        elif coeffs_list[0].ndim == 4:
            dims = ""
        else:
            raise ValueError(
                "coeffs_list must contain tensors of either 4 or 5 dimensions!"
            )
        acs = [
            signal.center_crop(signal.autocorrelation(coeff), self.spatial_corr_width)
            for coeff in coeffs_list
        ]
        acs = torch.stack(acs, 2)
        var = signal.center_crop(acs, 1)
        acs = acs / var
        var = einops.rearrange(var, f"b c s {dims} 1 1 -> b c {dims} s")
        return einops.rearrange(acs, f"b c s {dims} a1 a2 -> b c a1 a2 {dims} s"), var

    @staticmethod
    def _compute_skew_kurtosis_recon(
        reconstructed_images: list[Tensor], var_recon: Tensor, img_var: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the skew and kurtosis of each lowpass reconstructed image.

        For each scale, if the ratio of its variance to the original image's
        pixel variance is below a threshold of
        ``torch.finfo(img_var.dtype).resolution`` (``1e-6`` for ``float32``,
        ``1e-15`` for ``float64``), skew and kurtosis are assigned default
        values of ``0`` or ``3``, respectively.

        Parameters
        ----------
        reconstructed_images
            List of length ``n_scales+1`` containing the reconstructed unoriented
            image at each scale, from fine to coarse. The final image is
            reconstructed just from the residual lowpass image.
        var_recon
            Tensor of shape (batch, channel, n_scales+1) containing the
            variance of each tensor in reconstruced_images.
        img_var
            Tensor of shape (batch, channel) containing the pixel variance
            (from ``pixel_stats`` tensor).

        Returns
        -------
        skew_recon, kurtosis_recon
            Tensors of shape (batch, channel, n_scales+1) containing the skew
            and kurtosis, respectively, of each tensor in
            ``reconstructed_images``.
        """
        skew_recon = [
            stats.skew(im, mean=0, var=var_recon[..., i], dim=[-2, -1])
            for i, im in enumerate(reconstructed_images)
        ]
        skew_recon = torch.stack(skew_recon, -1)
        kurtosis_recon = [
            stats.kurtosis(im, mean=0, var=var_recon[..., i], dim=[-2, -1])
            for i, im in enumerate(reconstructed_images)
        ]
        kurtosis_recon = torch.stack(kurtosis_recon, -1)
        skew_default = torch.zeros_like(skew_recon)
        kurtosis_default = 3 * torch.ones_like(kurtosis_recon)
        # if this variance ratio is too small, then use the default values
        # instead. unsqueeze is used here because var_recon is shape (batch,
        # channel, scales+1), whereas img_var is just (batch, channel)
        res = torch.finfo(img_var.dtype).resolution
        unstable_locs = var_recon / img_var.unsqueeze(-1) < res
        skew_recon = torch.where(unstable_locs, skew_default, skew_recon)
        kurtosis_recon = torch.where(unstable_locs, kurtosis_default, kurtosis_recon)
        return skew_recon, kurtosis_recon

    def _compute_cross_correlation(
        self,
        coeffs_tensor: list[Tensor],
        coeffs_tensor_other: list[Tensor],
        coeffs_var: None | Tensor = None,
        coeffs_other_var: None | Tensor = None,
    ) -> Tensor:
        """
        Compute cross-correlations.

        Parameters
        ----------
        coeffs_tensor, coeffs_tensor_other
            The two lists of length scales, each containing 5d tensors of shape
            (batch, channel, n_orientations, height, width) to be correlated.
        coeffs_var, coeffs_other_var
            Two optional tensors containing the variances of coeffs_tensor and
            coeffs_tensor_other, respectively, in case they've already been computed.
            Should be of shape (batch, channel, n_orientations, n_scales). Used to
            normalize the covariances into cross-correlations.

        Returns
        -------
        cross_corrs
            Tensor of shape (batch, channel, n_orientations, n_orientations,
            scales) containing the cross-correlations at each
            scale.
        """  # numpydoc ignore=ES01
        covars = []
        for i, (coeff, coeff_other) in enumerate(
            zip(coeffs_tensor, coeffs_tensor_other)
        ):
            # precompute this, which we'll use for normalization
            numel = torch.mul(*coeff.shape[-2:])
            # compute the covariance
            covar = einops.einsum(
                coeff, coeff_other, "b c o1 h w, b c o2 h w -> b c o1 o2"
            )
            covar = covar / numel
            # Then normalize it to get the Pearson product-moment correlation
            # coefficient, see
            # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html.
            if coeffs_var is None:
                # First, compute the variances of each coeff
                coeff_var = einops.einsum(
                    coeff, coeff, "b c o1 h w, b c o1 h w -> b c o1"
                )
                coeff_var = coeff_var / numel
            else:
                coeff_var = coeffs_var[..., i]
            if coeffs_other_var is None:
                # First, compute the variances of each coeff
                coeff_other_var = einops.einsum(
                    coeff_other, coeff_other, "b c o1 h w, b c o1 h w -> b c o1"
                )
                coeff_other_var = coeff_other_var / numel
            else:
                coeff_other_var = coeffs_other_var[..., i]
            # Then compute the outer product of those variances.
            var_outer_prod = einops.einsum(
                coeff_var, coeff_other_var, "b c o1, b c o2 -> b c o1 o2"
            )
            # And the sqrt of this is what we use to normalize the covariance
            # into the cross-correlation
            covars.append(covar / var_outer_prod.sqrt())
        return torch.stack(covars, -1)

    @staticmethod
    def _double_phase_pyr_coeffs(
        pyr_coeffs: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """
        Upsample and double the phase of pyramid coefficients.

        This is trick is key to correctly computing the correlation between
        coefficients at different spatial scales.

        Parameters
        ----------
        pyr_coeffs
            Complex steerable pyramid coefficients (without residuals), as list
            of length n_scales, containing 5d tensors of shape (batch, channel,
            n_orientations, height, width).

        Returns
        -------
        doubled_phase_mags
            The demeaned magnitude (i.e., pyr_coeffs.abs()) of each upsampled
            double-phased coefficient. List of length n_scales-1 containing
            tensors of same shape the input (the finest scale has been
            removed).
        doubled_phase_separate
            The real and imaginary parts of each double-phased coefficient.
            List of length n_scales-1, containing tensors of shape (batch,
            channel, 2*n_orientations, height, width), with the real component
            found at the same orientation index as the input, and the imaginary
            at orientation+self.n_orientations. (The finest scale has been
            removed).
        """
        doubled_phase_mags = []
        doubled_phase_sep = []
        # don't do this for the finest scale
        for coeff in pyr_coeffs[1:]:
            # We divide by the factor of 4 here in order to approximately
            # equalize the steerable pyramid coefficient values across scales.
            # This could also be handled by making the pyramid tight frame
            doubled_phase = signal.expand(coeff, 2) / 4.0
            doubled_phase = signal.modulate_phase(doubled_phase, 2)
            doubled_phase_mag = doubled_phase.abs()
            doubled_phase_mag = doubled_phase_mag - doubled_phase_mag.mean(
                (-2, -1), keepdim=True
            )
            doubled_phase_mags.append(doubled_phase_mag)
            doubled_phase_sep.append(
                einops.pack([doubled_phase.real, doubled_phase.imag], "b c * h w")[0]
            )
        return doubled_phase_mags, doubled_phase_sep

    def plot_representation(
        self,
        data: Tensor,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        ylim: tuple[float, float] | Literal[False] | None = None,
        batch_idx: int = 0,
        title: str | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        r"""
        Plot the representation in a human viewable format.

        We plot the representation as stem plots with data separated out by
        statistic type.

        This plots the representation of a single batch and averages over all
        channels in the representation.

        We create the following axes:

        - pixels+var_highpass: marginal pixel statistics (first four moments,
          min, max) and variance of the residual highpass.

        - std+skew+kurtosis recon: the standard deviation, skew, and kurtosis
          of the reconstructed lowpass image at each scale

        - magnitude_std: the standard deviation of the steerable pyramid
          coefficient magnitudes at each orientation and scale.

        - auto_correlation_reconstructed: the auto-correlation of the
          reconstructed lowpass image at each scale (summarized using Euclidean
          norm).

        - auto_correlation_magnitude: the auto-correlation of the pyramid
          coefficient magnitudes at each scale and orientation (summarized
          using Euclidean norm).

        - cross_orientation_correlation_magnitude: the cross-correlations
          between each orientation at each scale (summarized using Euclidean
          norm)

        If ``self.n_scales > 1``, we also have combination of the following, where
        all cross-correlations are summarized using Euclidean norm over the
        channel dimension:

        - cross_scale_correlation_magnitude: the cross-correlations between the
          pyramid coefficient magnitude at one scale and the same orientation
          at the next-coarsest scale.

        - cross_scale_correlation_real: the cross-correlations between the real
          component of the pyramid coefficients and the real and imaginary
          components (at the same orientation) at the next-coarsest scale.

        Parameters
        ----------
        data
            The data to show on the plot. Else, should look like the output of
            ``self.forward(img)``, with the exact same structure (e.g., as
            returned by ``metamer.representation_error()`` or another instance
            of this class).
        ax
            Axes where we will plot the data. If a ``plt.Axes`` instance, will
            subdivide into 6 or 8 new axes (depending on self.n_scales). If
            None, we create a new figure.
        figsize
            The size of the figure to create. Must be ``None`` if ax is not ``None``. If
            both figsize and ax are ``None``, then we set ``figsize=(15, 15)``.
        ylim
            If not None, the y-limits to use for this plot. If None, we use the
            default, slightly adjusted so that the minimum is 0. If False, do not
            change y-limits.
        batch_idx
            Which index to take from the batch dimension (the first one).
        title
            Title for the plot.

        Returns
        -------
        fig
            Figure containing the plot.
        axes
            List of 6 or 8 axes containing the plot (depending on ``self.n_scales``).

        Raises
        ------
        ValueError
            If both ``figsize`` and ``ax`` are not ``None``.

        Examples
        --------
        .. plot::

          >>> import plenoptic as po
          >>> img = po.data.curie()
          >>> portilla_simoncelli_model = po.simul.PortillaSimoncelli(img.shape[2:])
          >>> representation_tensor = portilla_simoncelli_model(img)
          >>> fig, axes = portilla_simoncelli_model.plot_representation(
          ...     representation_tensor, figsize=(13, 6)
          ... )
        """
        if ax is None and figsize is None:
            figsize = (15, 15)
        elif ax is not None and figsize is not None:
            raise ValueError("figsize can't be set if ax is not None")
        # pick the batch_idx we want (but keep the data 3d), and average over
        # channels (but keep the data 3d). We keep data 3d because
        # convert_to_dict relies on it.
        data = data[batch_idx].unsqueeze(0).mean(1, keepdim=True)
        # each of these values should now be a 3d tensor with 1 element in each
        # of the first two dims
        rep = {k: v[0, 0] for k, v in self.convert_to_dict(data).items()}
        data = self._representation_for_plotting(rep)

        # Determine plot grid layout
        if self.n_scales != 1:
            n_rows = 3
            n_cols = int(np.ceil(len(data) / n_rows))
        else:
            # then we don't have any cross-scale correlations, so fewer axes.
            n_rows = 2
            n_cols = int(np.ceil(len(data) / n_rows))

        # Set up grid spec
        if ax is None:
            # we add 2 to order because we're adding one to get the
            # number of orientations and then another one to add an
            # extra column for the mean luminance plot
            fig = plt.figure(figsize=figsize)
            gs = mpl.gridspec.GridSpec(n_rows, n_cols, fig)
        else:
            # want to make sure the axis we're taking over is basically invisible.
            ax = clean_up_axes(
                ax, False, ["top", "right", "bottom", "left"], ["x", "y"]
            )
            gs = ax.get_subplotspec().subgridspec(n_rows, n_cols)
            fig = ax.figure

        # plot data
        axes = []
        for i, (k, v) in enumerate(data.items()):
            ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
            ax = clean_stem_plot(to_numpy(v).flatten(), ax, k, ylim=ylim)
            axes.append(ax)

        if title is not None:
            fig.suptitle(title)

        return fig, axes

    def _representation_for_plotting(self, rep: OrderedDict) -> OrderedDict:
        r"""
        Convert into a representation that is more convenient for plotting.

        Intended as a helper function for plot_representation.

        Parameters
        ----------
        rep
            Dictionary of representation, with informative keys.

        Returns
        -------
        plot_rep
            Dictionary of representation summarized for plotting.

        Raises
        ------
        ValueError
            If the tensors in ``rep`` looks like they have more than one batch
            or channel. Should select or average over those dimensions.
        ValueError
            If ``rep`` contains unexpected keys.
        """
        if rep["skew_reconstructed"].ndim > 1:
            raise ValueError(
                "Currently, only know how to plot single batch and channel at"
                " a time! Select and/or average over those dimensions"
            )
        data = OrderedDict()
        data["pixels+var_highpass"] = torch.cat(
            [rep.pop("pixel_statistics"), rep.pop("var_highpass_residual")]
        )
        data["std+skew+kurtosis recon"] = torch.cat(
            (
                rep.pop("std_reconstructed"),
                rep.pop("skew_reconstructed"),
                rep.pop("kurtosis_reconstructed"),
            )
        )

        data["magnitude_std"] = rep.pop("magnitude_std")

        # want to plot these in a specific order
        all_keys = [
            "auto_correlation_reconstructed",
            "auto_correlation_magnitude",
            "cross_orientation_correlation_magnitude",
            "cross_scale_correlation_magnitude",
            "cross_scale_correlation_real",
        ]
        if set(rep.keys()) != set(all_keys):
            raise ValueError("representation has unexpected keys!")
        for k in all_keys:
            # if we only have one scale, no cross-scale stats
            if k.startswith("cross_scale") and self.n_scales == 1:
                continue
            # we compute L2 norm manually, since there are NaNs (marking
            # redundant stats)
            data[k] = rep[k].pow(2).nansum((0, 1)).sqrt().flatten()

        return data

    def update_plot(
        self,
        axes: list[plt.Axes],
        data: Tensor,
        batch_idx: int = 0,
    ) -> list[plt.Artist]:
        r"""
        Update the information in our representation plot.

        This is used for creating an animation of the representation
        over time. In order to create the animation, we need to know how
        to update the matplotlib Artists, and this provides a simple way
        of doing that. It relies on the fact that we've used
        ``plot_representation`` to create the plots we want to update
        and so know that they're stem plots.

        We take the axes containing the representation information (note that
        this is probably a subset of the total number of axes in the figure, if
        we're showing other information, as done by ``Metamer.animate``), grab
        the representation from plotting and, since these are both lists,
        iterate through them, updating them to the values in ``data`` as we go.

        In order for this to be used by ``FuncAnimation``, we need to
        return Artists, so we return a list of the relevant artists, the
        ``markerline`` and ``stemlines`` from the ``StemContainer``.

        Currently, this averages over all channels in the representation.

        Parameters
        ----------
        axes
            A list of axes to update. We assume that these are the axes
            created by ``plot_representation`` and so contain stem plots
            in the correct order.
        data
            The data to show on the plot. Else, should look like the output of
            ``self.forward(img)``, with the exact same structure (e.g., as
            returned by ``metamer.representation_error()`` or another instance
            of this class).
        batch_idx
            Which index to take from the batch dimension (the first one).

        Returns
        -------
        stem_artists
            A list of the artists used to update the information on the
            stem plots.

        Examples
        --------
        This method is meant to be used by animation functions, so users won't
        typically use this directly.

        >>> import plenoptic as po
        >>> img = po.data.curie()
        >>> portilla_simoncelli_model = po.simul.PortillaSimoncelli(img.shape[2:])
        >>> representation_tensor = portilla_simoncelli_model.forward(img)
        >>> fig, axes = portilla_simoncelli_model.plot_representation(
        ...     representation_tensor
        ... )
        >>> new_img = po.data.einstein()
        >>> new_representation_tensor = portilla_simoncelli_model.forward(new_img)
        >>> stem_artists = portilla_simoncelli_model.update_plot(
        ...     axes, new_representation_tensor
        ... )
        """
        stem_artists = []
        axes = [ax for ax in axes if len(ax.containers) == 1]
        # pick the batch_idx we want (but keep the data 3d), and average over
        # channels (but keep the data 3d). We keep data 3d because
        # convert_to_dict relies on it.
        data = data[batch_idx].unsqueeze(0).mean(1, keepdim=True)
        # each of these values should now be a 3d tensor with 1 element in each
        # of the first two dims
        rep = {k: v[0, 0] for k, v in self.convert_to_dict(data).items()}
        rep = self._representation_for_plotting(rep)
        for ax, d in zip(axes, rep.values()):
            if isinstance(d, dict):
                vals = np.array([to_numpy(dd) for dd in d.values()])
            else:
                vals = to_numpy(d.flatten())

            sc = update_stem(ax.containers[0], vals)
            stem_artists.extend([sc.markerline, sc.stemlines])
        return stem_artists
