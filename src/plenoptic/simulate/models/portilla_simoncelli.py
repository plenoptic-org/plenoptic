import torch
import einops
from torch import Tensor
import torch.fft
import torch.nn as nn
from ..canonical_computations.steerable_pyramid_freq import SteerablePyramidFreq
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
from ...tools.display import clean_up_axes, update_stem, clean_stem_plot
from ...tools.data import to_numpy
from ...tools import stats, signal
from ...tools.validate import validate_input
from typing import Tuple, List, Literal, Union, Optional, Dict

SCALES_TYPE = Union[
    int, Literal["pixel_statistics", "residual_lowpass", "residual_highpass"]
]


class PortillaSimoncelli(nn.Module):
    r"""Portila-Simoncelli texture statistics.

    The Portilla-Simoncelli (PS) texture statistics are a set of image
    statistics, first described in [1]_, that are proposed as a sufficient set
    of measurements for describing visual textures. That is, if two texture
    images have the same values for all PS texture stats, humans should
    consider them as belonging to the same family of texture.

    The PS stats are computed based on the steerable pyramid [2]_. They consist
    of the local auto-correlations, cross-scale (within-orientation)
    correlations, and cross-orientation (within-scale) correlations of both the
    pyramid coefficients and the local energy (as computed by those
    coefficients). Additionally, they include the first four global moments
    (mean, variance, skew, and kurtosis) of the image and down-sampled versions
    of that image. See the paper and notebook for more description.

    Parameters
    ----------
    image_shape:
        Shape of input image.
    n_scales:
        The number of pyramid scales used to measure the statistics (default=4)
    n_orientations:
        The number of orientations used to measure the statistics (default=4)
    spatial_corr_width:
        The width of the spatial cross- and auto-correlation statistics in the representation

    Attributes
    ----------
    scales: list
        The names of the unique scales of coefficients in the pyramid.

    References
    ----------
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model based on
       Joint Statistics of Complex Wavelet Coefficients. Int'l Journal of
       Computer Vision. 40(1):49-71, October, 2000.
       http://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
       http://www.cns.nyu.edu/~lcv/texture/
    .. [2] E P Simoncelli and W T Freeman, "The Steerable Pyramid: A Flexible
       Architecture for Multi-Scale Derivative Computation," Second Int'l Conf
       on Image Processing, Washington, DC, Oct 1995.

    """

    def __init__(
        self,
        image_shape: Tuple[int, int],
        n_scales: int = 4,
        n_orientations: int = 4,
        spatial_corr_width: int = 9,
    ):
        super().__init__()

        self.image_shape = image_shape
        if (any([(image_shape[-1] / 2**i) % 2 for i in range(n_scales)]) or
            any([(image_shape[-2] / 2**i) % 2 for i in range(n_scales)])):
            raise ValueError("Because of how the Portilla-Simoncelli model handles "
                             "multiscale representations, it only works with images"
                             " whose shape can be divided by 2 `n_scales` times.")
        self.spatial_corr_width = spatial_corr_width
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.pyr = SteerablePyramidFreq(
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
        self._necessary_stats_dict = self._create_necessary_stats_dict(scales_shape_dict)
        # turn this into vector we can use in forward pass. first into a
        # boolean mask...
        self._necessary_stats_mask = einops.pack(list(self._necessary_stats_dict.values()), '*')[0]
        # then into a tensor of indices
        self._necessary_stats_mask = torch.where(self._necessary_stats_mask)[0]

        # This vector is composed of the following values: 'pixel_statistics',
        # 'residual_lowpass', 'residual_highpass' and integer values from 0 to
        # self.n_scales-1. It is the same size as the representation vector
        # returned by this object's forward method. It must be a numpy array so
        # we can have a mixture of ints and strs (and so we can use np.in1d
        # later)
        self._representation_scales = einops.pack(list(scales_shape_dict.values()), '*')[0]
        # just select the scales of the necessary stats.
        self._representation_scales = self._representation_scales[self._necessary_stats_mask]

    def _create_scales_shape_dict(self) -> OrderedDict:
        """Create dictionary defining scales and shape of each stat.

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
        shape_dict['pixel_statistics'] = np.array(6*['pixel_statistics'])

        # These are the basic building blocks of the scale assignments for many
        # of the statistics calculated by the PortillaSimoncelli model.
        scales = np.arange(self.n_scales)
        # the cross-scale correlations exclude the coarsest scale
        scales_without_coarsest = np.arange(self.n_scales-1)
        # the statistics computed on the reconstructed bandpass images have an
        # extra scale corresponding to the lowpass residual
        scales_with_lowpass = np.concatenate([scales, ["residual_lowpass"]],
                                             dtype=object)

        # now we go through each statistic in order and create a dummy array
        # full of 1s with the same shape as the actual statistic (excluding the
        # batch and channel dimensions, as each stat is computed independently
        # across those dimensions). We then multiply it by one of the scales
        # arrays above to turn those 1s into values describing the
        # corresponding scale.

        auto_corr_mag = np.ones((self.spatial_corr_width, self.spatial_corr_width,
                                 self.n_scales, self.n_orientations), dtype=int)
        auto_corr_mag *= einops.rearrange(scales, 's -> 1 1 s 1')
        shape_dict['auto_correlation_magnitude'] = auto_corr_mag

        shape_dict['skew_reconstructed'] = scales_with_lowpass

        shape_dict['kurtosis_reconstructed'] = scales_with_lowpass

        auto_corr = np.ones((self.spatial_corr_width, self.spatial_corr_width,
                             self.n_scales+1), dtype=object)
        auto_corr *= einops.rearrange(scales_with_lowpass, 's -> 1 1 s')
        shape_dict['auto_correlation_reconstructed'] = auto_corr

        shape_dict['std_reconstructed'] = scales_with_lowpass

        cross_orientation_corr_mag = np.ones((self.n_orientations, self.n_orientations,
                                              self.n_scales), dtype=int)
        cross_orientation_corr_mag *= einops.rearrange(scales, 's -> 1 1 s')
        shape_dict['cross_orientation_correlation_magnitude'] = cross_orientation_corr_mag

        cross_scale_corr_mag = np.ones((self.n_orientations, self.n_orientations,
                                        self.n_scales-1), dtype=int)
        cross_scale_corr_mag *= einops.rearrange(scales_without_coarsest, 's -> 1 1 s')
        shape_dict['cross_scale_correlation_magnitude'] = cross_scale_corr_mag

        cross_scale_corr_real = np.ones((self.n_orientations, 2*self.n_orientations,
                                         self.n_scales-1), dtype=int)
        cross_scale_corr_real *= einops.rearrange(scales_without_coarsest, 's -> 1 1 s')
        shape_dict['cross_scale_correlation_real'] = cross_scale_corr_real

        shape_dict['var_highpass_residual'] = np.array(["residual_highpass"])

        return shape_dict

    def _create_necessary_stats_dict(self, scales_shape_dict: OrderedDict) -> OrderedDict:
        """Create mask specifying the necessary statistics.

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
        # Pre compute some necessary indices.
        # Lower triangular indices (including diagonal), for auto correlations
        tril_inds = torch.tril_indices(self.spatial_corr_width,
                                       self.spatial_corr_width)
        # Get the second half of the diagonal, i.e., everything from the center
        # element on. These are all repeated for the auto correlations. (As
        # these are autocorrelations (rather than auto-covariance) matrices,
        # they've been normalized by the variance and so the center element is
        # always 1, and thus uninformative)
        diag_repeated = torch.arange(start=self.spatial_corr_width//2,
                                     end=self.spatial_corr_width)
        # Upper triangle indices, excluding diagonal. These are redundant stats
        # for cross_orientation_correlation_magnitude. Setting offset=1 means
        # this will not include the diagonals, which are not redundant (if we
        # were not normalizing the autocorrelation matrices, the diagonals of
        # the cross-orientation correlations would be redundant with the
        # central elements of auto_correlation_magnitude)
        triu_inds = torch.triu_indices(self.n_orientations,
                                       self.n_orientations, offset=1)
        for k, v in mask_dict.items():
            if k in ["auto_correlation_magnitude", "auto_correlation_reconstructed"]:
                # Symmetry M_{i,j} = M_{n-i+1, n_j+1}
                # Start with all False, then place True in necessary stats.
                mask = torch.zeros(v.shape, dtype=bool)
                mask[tril_inds[0], tril_inds[1]] = True
                # if spatial_corr_width is even, then the first row is not
                # redundant with anything either
                if np.mod(self.spatial_corr_width, 2) == 0:
                    mask[0] = True
                mask[diag_repeated, diag_repeated] = False
            elif k == 'cross_orientation_correlation_magnitude':
                # Symmetry M_{i,j} = M_{j,i}.
                # Start with all True, then place False in redundant stats.
                mask = torch.ones(v.shape, dtype=bool)
                mask[triu_inds[0], triu_inds[1]] = False
            else:
                # all of the other stats have no redundancies
                mask = torch.ones(v.shape, dtype=bool)
            mask_dict[k] = mask
        return mask_dict

    def forward(
        self, image: Tensor, scales: Optional[List[SCALES_TYPE]] = None
    ) -> Tensor:
        r"""Generate Texture Statistics representation of an image.

        Note that separate batches and channels are analyzed in parallel.

        Parameters
        ----------
        image :
            A 4d tensor (batch, channel, height, width) containing the image(s) to
            analyze.
        scales :
            Which scales to include in the returned representation. If None, we
            include all scales. Otherwise, can contain subset of values present
            in this model's ``scales`` attribute, and the returned vector will
            then contain the subset of the full representation corresponding to
            those scales.

        Returns
        -------
        representation_vector:
            3d tensor of shape (B,C,S) containing the measured texture
            statistics.

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

        ### SECTION 1 (STATISTIC: pixel_statistics) ##################
        #  Calculate pixel statistics (mean, variance, skew, kurtosis, min, max).
        pixel_stats = self._compute_pixel_stats(image)

        ### SECTION 2 (STATISTIC: mean_magnitude) ####################
        # Calculate the mean of the magnitude of each band of pyramid
        # coefficients.  Additionally, this section creates two
        # other dictionaries of coefficients: magnitude_pyr_coeffs
        # and real_pyr_coeffs, which contain the demeaned magnitude of the
        # pyramid coefficients and the real part of the pyramid
        # coefficients respectively.

        # calculate two intermediate representations:
        #   1) demeaned magnitude of the pyramid coefficients,
        #   2) real part of the pyramid coefficients
        mag_pyr_coeffs, real_pyr_coeffs = self._compute_intermediate_representations(pyr_coeffs)

        ### SECTION 3 (STATISTICS: auto_correlation_magnitude,
        #                          skew_reconstructed,
        #                          kurtosis_reconstructed,
        #                          auto_correlation_reconstructed) #####
        #
        # Calculates:
        # 1) the central auto-correlation of the magnitude of each
        # orientation/scale band.
        #
        # 2) the central auto-correlation of the low-pass residuals
        # for each scale of the pyramid (auto_correlation_reconstructed),
        # where the residual at each scale is reconstructed from the
        # previous scale.  (Note: the lowpass residual of the pyramid
        # is low-pass filtered before this reconstruction process begins,
        # see below).
        #
        # 3) the skew and the kurtosis of the reconstructed
        # low-pass residuals (skew_reconstructed, kurtosis_reconstructed).
        # The skew and kurtosis are calculated with the auto-correlation
        # statistics because like #2 (above) they rely on the reconstructed
        # low-pass residuals, making it more efficient (in terms of memory
        # and/or compute time) to calculate it at the same time.

        # list of length n_scales+1 containing tensors of shape (batch,
        # channel, height, width)
        reconstructed_images = self._reconstruct_lowpass_at_each_scale(pyr_dict)
        # the reconstructed_images list goes from coarse-to-fine, but we want
        # each of the stats computed from it to go from fine-to-coarse, so we
        # reverse its direction.
        reconstructed_images = reconstructed_images[::-1]

        # tensor of shape: (batch, channel, spatial_corr_width,
        # spatial_corr_width, n_scales, n_orientations)
        autocorr_mags, _ = self._compute_autocorr(mag_pyr_coeffs)

        # autocorr_recon is a tensor of shape (batch, channel,
        # spatial_corr_width, spatial_corr_width, n_scales+1), and var_recon is
        # a tensor of shape (batch, channel, n_scales+1)
        autocorr_recon, var_recon = self._compute_autocorr(reconstructed_images)
        # std_recon, skew_recon, and kurtosis_recon will all end up as shape
        # (batch, channel, n_scales+1)
        std_recon = var_recon**0.5
        skew_recon, kurtosis_recon = self._compute_skew_kurtosis_recon(reconstructed_images, var_recon, pixel_stats[..., 1])

        ### SECTION 4 (STATISTICS: cross_orientation_correlation_magnitude,
        #                          cross_scale_correlation_magnitude,
        #                          cross_scale_correlation_real) ###########
        # Calculates cross-orientation and cross-scale correlations for the
        # real parts and the magnitude of the pyramid coefficients.
        #

        # this will be a tensor of shape (batch, channel, n_orientations,
        # n_orientations, n_scales) containing the cross-orientation
        # correlations between the magnitudes (within scale)
        cross_ori_corr_mags = self._compute_cross_correlation(mag_pyr_coeffs, mag_pyr_coeffs)

        if self.n_scales != 1:
            # double the phase the coefficients, so we can correctly compute
            # correlations across scales.
            phase_doubled_mags, phase_doubled_sep = self._double_phase_pyr_coeffs(pyr_coeffs)
            # this will be a tensor of shape (batch, channel, n_orientations,
            # n_orientations, n_scales-1) containing the cross-scale
            # correlations between the magnitudes.
            cross_scale_corr_mags = self._compute_cross_correlation(mag_pyr_coeffs[:-1], phase_doubled_mags)
            # this will be a tensor of shape (batch, channel, n_orientations,
            # 2*n_orientations, n_scales-1) containing the cross-scale
            # correlations between the real components of the coefficients.
            cross_scale_corr_real = self._compute_cross_correlation(real_pyr_coeffs[:-1], phase_doubled_sep)

        # SECTION 5: the variance of the high-pass residual, of shape (batch, channel)
        var_highpass_residual = highpass.pow(2).mean(dim=(-2, -1))

        all_stats = [pixel_stats, autocorr_mags, skew_recon,
                     kurtosis_recon, autocorr_recon, std_recon,
                     cross_ori_corr_mags]
        if self.n_scales != 1:
            all_stats += [cross_scale_corr_mags, cross_scale_corr_real]
        all_stats += [var_highpass_residual]
        representation_vector = einops.pack(all_stats, 'b c *')[0]

        # throw away all redundant statistics
        representation_vector = representation_vector.index_select(-1, self._necessary_stats_mask)

        if scales is not None:
            representation_vector = self.remove_scales(representation_vector, scales)

        return representation_vector

    def remove_scales(
            self, representation_vector: Tensor, scales_to_keep: List[SCALES_TYPE]
    ) -> Tensor:
        """Remove statistics not associated with scales

        For a given representation_vector and a list of scales_to_keep, this
        attribute removes all statistics *not* associated with those scales.

        Note that calling this method will always remove statistics.

        Parameters
        ----------
        representation_vector:
            3d tensor containing the measured representation statistics.
        scales_to_keep:
            Which scales to include in the returned representation. Can contain
            subset of values present in this model's ``scales`` attribute, and
            the returned vector will then contain the subset of the full
            representation corresponding to those scales.

        Returns
        ------
        limited_representation_vector :
            Representation vector with some statistics removed.

        """
        # this is necessary because object is the dtype of
        # self._representation_scales
        scales_to_keep = np.array(scales_to_keep, dtype=object)
        # np.in1d returns a 1d boolean array of the same shape as
        # self._representation_scales with True at each location where that
        # value appears in scales_to_keep. where then converts this boolean
        # array into indices
        ind = np.where(np.in1d(self._representation_scales, scales_to_keep))[0]
        ind = torch.from_numpy(ind).to(representation_vector.device)
        return representation_vector.index_select(-1, ind)

    def convert_to_vector(self, representation_dict: OrderedDict) -> Tensor:
        r"""Converts dictionary of statistics to a vector.

        While the dictionary representation is easier to manually inspect, the
        vector representation is required by plenoptic's synthesis objects.

        Parameters
        ----------
        representation_dict :
             Dictionary of representation.

        Returns
        -------
        3d vector of statistics.

        See also
        --------
        convert_to_dict:
            Convert vector representation to dictionary.

        """
        rep = einops.pack(list(representation_dict.values()), 'b c *')[0]
        # then get rid of all the nans / unnecessary stats
        return rep.index_select(-1, self._necessary_stats_mask)

    def convert_to_dict(self, representation_vector: Tensor) -> OrderedDict:
        """Converts vector of statistics to a dictionary.

        While the vector representation is required by plenoptic's synthesis
        objects, the dictionary representation is easier to manually inspect.

        This dictionary will contain NaNs in its values: these are placeholders
        for the redundant statistics.

        Parameters
        ----------
        representation_vector
            3d vector of statistics.

        Returns
        -------
        Dictionary of representation, with informative keys.

        See also
        --------
        convert_to_vector:
            Convert dictionary representation to vector.

        """
        if representation_vector.shape[-1] != len(self._representation_scales):
            raise ValueError(
                "representation vector is the wrong length (expected "
                f"{len(self._representation_scales)} but got {representation_vector.shape[-1]})!"
                " Did you remove some of the scales? (i.e., by setting "
                "scales in the forward pass)? convert_to_dict does not "
                "support such vectors."
            )

        rep = self._necessary_stats_dict.copy()
        n_filled = 0
        for k, v in rep.items():
            # each statistic should be a tensor with batch and channel
            # dimensions as found in representation_vector and all the other
            # dimensions determined by the values in necessary_stats_dict.
            shape = (*representation_vector.shape[:2], *v.shape)
            new_v = torch.nan * torch.ones(shape, dtype=representation_vector.dtype,
                                           device=representation_vector.device)
            # v.sum() gives the number of necessary elements from this stat
            this_stat_vec = representation_vector[..., n_filled:n_filled+v.sum()]
            # use boolean indexing to put the values from new_stat_vec in the
            # appropriate place
            new_v[..., v] = this_stat_vec
            rep[k] = new_v
            n_filled += v.sum()
        return rep

    def _compute_pyr_coeffs(self, image: Tensor) -> Tuple[OrderedDict, List[Tensor], Tensor, Tensor]:
        """Compute pyramid coefficients of image.

        Note that the residual lowpass hsa been demeaned independently for each
        batch and channel (and this is true of the lowpass returned separately
        as well as the one included in pyr_coeffs_dict)

        Parameters
        ----------
        image :
            4d tensor of shape (batch, channel, height, width) containing the
            image

        Returns
        -------
        pyr_coeffs_dict :
            OrderedDict of containing all complex-valued pyramid coefficients
        pyr_coeffs :
            List of length n_scales, containing 5d tensors of shape (batch,
            channel, n_orientations, height, width) containing the complex-valued
            oriented bands (note that height and width shrink by half on each
            scale)
        highpass :
            The residual highpass as a real-valued 4d tensor (batch, channel,
            height, width)
        lowpass :
            The residual lowpass as a real-valued 4d tensor (batch, channel,
            height, width). This tensor has been demeaned (independently for
            each batch and channel).

        """
        pyr_coeffs = self.pyr.forward(image)
        # separate out the residuals and demean the residual lowpass
        lowpass = pyr_coeffs['residual_lowpass']
        lowpass = lowpass - lowpass.mean(dim=(-2, -1), keepdim=True)
        pyr_coeffs['residual_lowpass'] = lowpass
        highpass = pyr_coeffs['residual_highpass']

        # This is a list of tensors, one for each scale, where each tensor is
        # of shape (batch, channel, n_orientations, height, width) (note that
        # height and width halves on each scale)
        coeffs_list = [torch.stack([pyr_coeffs[(i, j)] for j in range(self.n_orientations)], 2)
                       for i in range(self.n_scales)]
        return pyr_coeffs, coeffs_list, highpass, lowpass

    @staticmethod
    def _compute_pixel_stats(image: Tensor) -> Tensor:
        """Compute the pixel stats: first four moments, min, and max.

        Parameters
        ----------
        image :
            4d tensor of shape (batch, channel, height, width) containing input
            image. Stats are computed indepently for each batch and channel.

        Returns
        -------
        pixel_stats :
            3d tensor of shape (batch, channel, 6) containing the mean,
            variance, skew, kurtosis, minimum pixel value, and maximum pixel
            value (in that order)

        """
        mean = torch.mean(image, dim=(-2, -1), keepdim=True)
        var = torch.var(image, dim=(-2, -1))
        skew = stats.skew(image, mean=mean, var=var, dim=[-2, -1])
        kurtosis = stats.kurtosis(image, mean=mean, var=var, dim=[-2, -1])
        # can't compute min/max over two dims simultaneously with
        # torch.min/max, so use einops
        img_min = einops.reduce(image, 'b c h w -> b c', 'min')
        img_max = einops.reduce(image, 'b c h w -> b c', 'max')
        # mean needed to be unflattened to be used by skew and kurtosis
        # correctly, but we'll want it to be flattened like this in the final
        # representation vector
        return einops.pack([mean, var, skew, kurtosis, img_min, img_max], 'b c *')[0]

    def _compute_intermediate_representations(self, pyr_coeffs: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Compute useful intermediate representations.

        These representations are:
          1) demeaned magnitude of the pyramid coefficients,
          2) real part of the pyramid coefficients

        These two are used in computing some of the texture representation.

        Parameters
        ----------
        pyr_coeffs :
            Complex steerable pyramid coefficients (without residuals), as list
            of length n_scales, containing 5d tensors of shape (batch, channel,
            n_orientations, height, width)

        Returns
        -------
        magnitude_pyr_coeffs :
           List of length n_scales, containing 5d tensors of shape (batch,
           channel, n_orientations, height, width) (same as ``pyr_coeffs``),
           containing the demeaned magnitude of the steerable pyramid
           coefficients (i.e., coeffs.abs() - coeffs.abs().mean((-2, -1)))
        real_pyr_coeffs :
           List of length n_scales, containing 5d tensors of shape (batch,
           channel, n_orientations, height, width) (same as ``pyr_coeffs``),
           containing the real components of the coefficients (i.e.
           coeffs.real)

        """
        magnitude_pyr_coeffs = [coeff.abs() for coeff in pyr_coeffs]
        magnitude_means = [mag.mean((-2, -1), keepdim=True) for mag in magnitude_pyr_coeffs]
        magnitude_pyr_coeffs = [mag - mn for mag, mn in zip(magnitude_pyr_coeffs, magnitude_means)]
        real_pyr_coeffs = [coeff.real for coeff in pyr_coeffs]
        return magnitude_pyr_coeffs, real_pyr_coeffs

    def _reconstruct_lowpass_at_each_scale(self, pyr_coeffs_dict: OrderedDict) -> List[Tensor]:
        """Reconstruct the lowpass unoriented image at each scale.

        The autocorrelation, skew, and kurtosis of each of these images is part
        of the texture representation.

        Parameters
        ----------
        pyr_coeffs_dict :
            Dictionary containing the steerable pyramid coefficients, with the
            lowpass residual demeaned.

        Returns
        -------
        reconstructed_images :
            List of length n_scales+1 containing the reconstructed unoriented
            image at each scale, from fine to coarse. The final image is
            reconstructed just from the residual lowpass image. Each is a 4d
            tensor, this is a list because they are all different heights and
            widths.

        """
        reconstructed_images = [self.pyr.recon_pyr(pyr_coeffs_dict, levels=['residual_lowpass'])]
        # go through scales backwards
        for lev in range(self.n_scales-1, -1, -1):
            recon = self.pyr.recon_pyr(pyr_coeffs_dict, levels=lev)
            reconstructed_images.append(recon + reconstructed_images[-1])
        # now downsample as necessary
        reconstructed_images[:-1] = [signal.shrink(r, 2**(self.n_scales-i)) * 4**(self.n_scales-i)
                                     for i, r in enumerate(reconstructed_images[:-1])]
        return reconstructed_images

    def _compute_autocorr(self, coeffs_list: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Compute the autocorrelation of some statistics.

        Parameters
        ----------
        coeffs_list :
            List (of length s) of tensors of shape (batch, channel, *, height,
            width), where * is zero or one additional dimensions. Intended use
            case: magnitude_pyr_coeffs (which is list of length n_scales of 5d
            tensors, with * containing n_orientations) or reconstructed_images
            (which is a list of length n_scales+1 of 4d tensors)

        Returns
        -------
        autocorrs :
            Tensor of shape (batch, channel, spatial_corr_width,
            spatial_corr_width, s, *) containing the autocorrelation (up to
            distance ``spatial_corr_width//2``) of each element in
            ``coeffs_list``, computed independently over all but the final two
            dimensions.
        vars :
            3d Tensor of shape (batch, channel, s, *) containing the variance
            of each element in ``coeffs_list``, computed independently over all
            but the final two dimensions.

        """
        if coeffs_list[0].ndim == 5:
            dims = 's o'
        elif coeffs_list[0].ndim == 4:
            dims = 's'
        else:
            raise ValueError("coeffs_list must contain tensors of either 4 or 5 dimensions!")
        acs = [signal.autocorrelation(coeff) for coeff in coeffs_list]
        var = [signal.center_crop(ac, 1) for ac in acs]
        acs = [ac/v for ac, v in zip(acs, var)]
        var = einops.pack(var, 'b c *')[0]
        acs = [signal.center_crop(ac, self.spatial_corr_width) for ac in acs]
        acs = torch.stack(acs, 2)
        return einops.rearrange(acs, f'b c {dims} a1 a2 -> b c a1 a2 {dims}'), var

    def _compute_skew_kurtosis_recon(self, reconstructed_images: List[Tensor], var_recon: Tensor,
                                     img_var: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the skew and kurtosis of each lowpass reconstructed image.

        For each scale, if the ratio of its variance to the pixel variance of
        the original image are below a threshold of 1e-6, skew and kurtosis are
        assigned default values of 0 or 3, respectively.

        Parameters
        ----------
        reconstructed_images :
            List of length n_scales+1 containing the reconstructed unoriented
            image at each scale, from fine to coarse. The final image is
            reconstructed just from the residual lowpass image.
        var_recon :
            Tensor of shape (batch, channel, n_scales+1) containing the
            variance of each tensor in reconstruced_images
        img_var :
            Tensor of shape (batch, channel) containing the pixel variance
            (from pixel_stats tensor)

        Returns
        -------
        skew_recon, kurtosis_recon :
            Tensors of shape (batch, channel, n_scales+1) containing the skew
            and kurtosis, respectively, of each tensor in
            ``reconstructed_images``.

        """
        skew_recon = [stats.skew(r, mean=0, var=var_recon[..., i], dim=[-2, -1])
                      for i, r in enumerate(reconstructed_images)]
        skew_recon = torch.stack(skew_recon, -1)
        kurtosis_recon = [stats.kurtosis(r, mean=0, var=var_recon[..., i], dim=[-2, -1])
                          for i, r in enumerate(reconstructed_images)]
        kurtosis_recon = torch.stack(kurtosis_recon, -1)
        skew_default = torch.zeros_like(skew_recon)
        kurtosis_default = 3 * torch.ones_like(kurtosis_recon)
        # if this variance ratio is too small, then use the default values
        # instead. unsqueeze is used here because var_recon is shape (batch,
        # channel, scales+1), whereas img_var is just (batch, channel)
        unstable_locs = var_recon / img_var.unsqueeze(-1) > 1e-6
        skew_recon = torch.where(unstable_locs, skew_recon, skew_default)
        kurtosis_recon = torch.where(unstable_locs, kurtosis_recon, kurtosis_default)
        return skew_recon, kurtosis_recon

    def _compute_cross_correlation(self, coeffs_tensor: List[Tensor], coeffs_tensor_other: List[Tensor]) -> Tensor:
        """Compute cross-correlations.

        Parameters
        ----------
        coeffs_tensor, coeffs_tensor_other :
            The two lists of length scales, each containing 5d tensors of shape
            (batch, channel, n_orientations, height, width) to be correlated.

        Returns
        -------
        cross_corrs :
            Tensor of shape (batch, channel, n_orientations, n_orientations,
            scales) containing the cross-correlations at each
            scale.

        """
        cross_corrs = []
        for coeff, coeff_other in zip(coeffs_tensor, coeffs_tensor_other):
            cross_corr = einops.einsum(coeff, coeff_other,
                                           'b c o1 h w, b c o2 h w -> b c o1 o2')
            cross_corr = cross_corr / torch.mul(*coeff.shape[-2:])
            std = torch.std(coeff, (-3, -2, -1), keepdim=True).squeeze(-1)
            std_other = torch.std(coeff_other, (-3, -2, -1), keepdim=True).squeeze(-1)
            cross_corr = cross_corr / (std * std_other)
            cross_corrs.append(cross_corr)
        cross_corrs = torch.stack(cross_corrs, -1)
        return cross_corrs

    def _double_phase_pyr_coeffs(self, pyr_coeffs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Upsample and double the phase of pyramid coefficients.

        Parameters
        ----------
        pyr_coeffs :
            Complex steerable pyramid coefficients (without residuals), as list
            of length n_scales, containing 5d tensors of shape (batch, channel,
            n_orientations, height, width)

        Returns
        -------
        doubled_phase_mags :
            The demeaned magnitude (i.e., pyr_coeffs.abs()) of each upsampled
            double-phased coefficient. List of length n_scales-1 containing
            tensors of same shape the input (the finest scale has been
            removed).
        doubled_phase_separate :
            The real and imaginary parts of each double-phased coefficient.
            List of length n_scales-1, containing tensors of shape (batch,
            channel, 2*n_orientations, height, width), with the real component
            found at the same orientation index as the input, and the imaginary
            at orientation+self.n_orientations. (The finest scale has been
            removed.)

        """
        doubled_phase_mags = []
        doubled_phase_sep = []
        # don't do this for the finest scale
        for coeff in pyr_coeffs[1:]:
            doubled_phase = signal.expand(coeff, 2) / 4.0
            doubled_phase = signal.modulate_phase(doubled_phase, 2)
            doubled_phase_mag = doubled_phase.abs()
            doubled_phase_mag = doubled_phase_mag - doubled_phase_mag.mean((-2, -1), keepdim=True)
            doubled_phase_mags.append(doubled_phase_mag)
            ## this minus here SHOULD BE REMOVED, it's just because they computed
            ## the negative real component by accident
            doubled_phase_sep.append(einops.pack([-doubled_phase.real, doubled_phase.imag],
                                                 'b c * h w')[0])
        return doubled_phase_mags, doubled_phase_sep

    def plot_representation(
            self,
            data: Tensor,
            ax: Optional[mpl.axes.Axes] = None,
            figsize: Tuple[float, float] = (15, 15),
            ylim: Optional[Tuple[float, float]] = None,
            batch_idx: int = 0,
            title: Optional[str] = None,
    ) -> Tuple[mpl.figure.Figure, List[mpl.axes.Axes]]:

        r"""Plot the representation in a human viewable format -- stem
        plots with data separated out by statistic type.

        Currently, this averages over all channels in the representation.

        Parameters
        ----------
        data :
            The data to show on the plot. Else, should look like the output of
            ``self.forward(img)``, with the exact same structure (e.g., as
            returned by ``metamer.representation_error()`` or another instance
            of this class).
        ax :
            Axes where we will plot the data. If an ``mpl.axes.Axes``, will
            subdivide into 7 new axes. If None, we create a new figure.
        figsize :
            The size of the figure. Ignored if ax is not None.
        ylim :
            The ylimits of the plot.
        batch_idx :
            Which index to take from the batch dimension (the first one)
        title : string
            Title for the plot

        Returns
        -------
        fig:
            Figure containing the plot
        axes:
            List of 7 axes containing the plot

        """
        n_rows = 3
        n_cols = 3

        # pick the batch_idx we want (but keep the data 3d), and average over
        # channels (but keep the data 3d). We keep data 3d because
        # convert_to_dict relies on it.
        data = data[batch_idx].unsqueeze(0).mean(1, keepdim=True)
        # each of these values should now be a 3d tensor with 1 element in each
        # of the first two dims
        rep = {k: v[0, 0] for k, v in self.convert_to_dict(data).items()}
        data = self._representation_for_plotting(rep)

        # Set up grid spec
        if ax is None:
            # we add 2 to order because we're adding one to get the
            # number of orientations and then another one to add an
            # extra column for the mean luminance plot
            fig = plt.figure(figsize=figsize)
            gs = mpl.gridspec.GridSpec(n_rows, n_cols, fig)
        else:
            # warnings.warn("ax is not None, so we're ignoring figsize...")
            # want to make sure the axis we're taking over is basically invisible.
            ax = clean_up_axes(
                ax, False, ["top", "right", "bottom", "left"], ["x", "y"]
            )
            gs = ax.get_subplotspec().subgridspec(n_rows, n_cols)
            fig = ax.figure

        # plot data
        axes = []
        for i, (k, v) in enumerate(data.items()):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax = clean_stem_plot(to_numpy(v).flatten(), ax, k, ylim=ylim)
            axes.append(ax)

        if title is not None:
            fig.suptitle(title)

        return fig, axes

    def _representation_for_plotting(self, rep: OrderedDict) -> OrderedDict:
        r"""Converts the data into a dictionary representation that is more convenient for plotting.

        Intended as a helper function for plot_representation.

        """
        if rep['skew_reconstructed'].ndim > 1:
            raise ValueError("Currently, only know how to plot single batch and channel at a time! "
                             "Select and/or average over those dimensions")
        data = OrderedDict()
        data["pixels+var_highpass"] = torch.cat([rep.pop("pixel_statistics"),
                                                 rep.pop("var_highpass_residual")])
        data["std+skew+kurtosis recon"] = torch.cat(
            (
                rep.pop("std_reconstructed"),
                rep.pop("skew_reconstructed"),
                rep.pop("kurtosis_reconstructed"),
            )
        )

        for k, v in rep.items():
            # we compute L2 norm manually, since there are NaNs (marking
            # redundant stats)
            data[k] = v.pow(2).nansum((0, 1)).sqrt().flatten()

        return data

    def update_plot(
            self,
            axes: List[mpl.axes.Axes],
            data: Tensor,
            batch_idx: int = 0,
    ) -> List[mpl.artist.Artist]:
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
        should look just like the representation of this model.

        In order for this to be used by ``FuncAnimation``, we need to
        return Artists, so we return a list of the relevant artists, the
        ``markerline`` and ``stemlines`` from the ``StemContainer``.

        Currently, this averages over all channels in the representation.

        Parameters
        ----------
        axes :
            A list of axes to update. We assume that these are the axes
            created by ``plot_representation`` and so contain stem plots
            in the correct order.
        batch_idx :
            Which index to take from the batch dimension (the first one)
        data :
            The data to show on the plot. Else, should look like the output of
            ``self.forward(img)``, with the exact same structure (e.g., as
            returned by ``metamer.representation_error()`` or another instance
            of this class).

        Returns
        -------
        stem_artists :
            A list of the artists used to update the information on the
            stem plots

        """
        stem_artists = []
        axes = [ax for ax in axes if len(ax.containers) == 1]
        data = self.convert_to_dict(data[batch_idx].mean(0))
        rep = self._representation_for_plotting(data)
        for ax, d in zip(axes, rep.values()):
            if isinstance(d, dict):
                vals = np.array([dd.detach() for dd in d.values()])
            else:
                vals = d.flatten().detach().numpy()

            sc = update_stem(ax.containers[0], vals)
            stem_artists.extend([sc.markerline, sc.stemlines])
        return stem_artists

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
        self.pyr = self.pyr.to(*args, **kwargs)
        self._necessary_stats_mask = self._necessary_stats_mask.to(*args, **kwargs)
        return self
