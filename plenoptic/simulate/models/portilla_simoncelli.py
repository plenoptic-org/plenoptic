import torch
import torch.fft
import torch.nn as nn
from ..canonical_computations.steerable_pyramid_freq import Steerable_Pyramid_Freq
from ...tools.conv import blur_downsample
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
from ...tools.display import clean_up_axes, update_stem, clean_stem_plot
from ...tools.data import to_numpy


class PortillaSimoncelli(nn.Module):
    r"""Model for measuring texture statistics originally proposed in [1] for the purpose of 
    synthesizing texture metamers. These statistics are proposed in [1] as a sufficient set
    measurements for describing and synthesizing a given visual texture.

    Currently we do not support batch measurement of images.

    Parameters
    ----------
    n_scales: int, optional
        The number of pyramid scales used to measure the statistics (default=4)
    n_orientations: int, optional
        The number of orientations used to measure the statistics (default=4)
    spatial_corr_width: int, optional
        The width of the spatial cross- and auto-correlation statistics in the representation
    use_true_correlations: bool
        In the original Portilla-Simoncelli model the statistics in the representation
        that are labelled correlations were actually covariance matrices (i.e. not properly
        scaled).  In order to match the original statistics use_true_correlations must be
        set to false. But in order to synthesize metamers from this model use_true_correlations
        must be set to true (default).

    Attributes
    ----------
    pyr: Steerable_Pyramid_Freq
        The complex steerable pyramid object used to calculate the portilla-simoncelli representation
    pyr_coeffs: OrderedDict
        The coefficients of the complex steerable pyramid.
    mag_pyr_coeffs: OrderedDict
        The magnitude of the pyramid coefficients.
    real_pyr_coeffs: OrderedDict
        The real parts of the pyramid coefficients.
    scales: list
        The names of the unique scales of coefficients in the pyramid.
    representation_scales: list
        The scale for each coefficient in its vector form
    representation: dictionary
        A dictionary containing the Portilla-Simoncelli statistics

    References
    -----
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model based on
       Joint Statistics of Complex Wavelet Coefficients. Int'l Journal of
       Computer Vision. 40(1):49-71, October, 2000.
       http://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
       http://www.cns.nyu.edu/~lcv/texture/

    """

    def __init__(
        self,
        im_shape,
        n_scales=4,
        n_orientations=4,
        spatial_corr_width=9,
        use_true_correlations=True,
    ):
        super().__init__()

        self.image_shape = im_shape
        self.spatial_corr_width = spatial_corr_width
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.pyr = Steerable_Pyramid_Freq(
            self.image_shape,
            height=self.n_scales,
            order=self.n_orientations - 1,
            is_complex=True,
            tight_frame=False,
        )
        self.filterPyr = Steerable_Pyramid_Freq(
            self.pyr._lomasks[-1].shape[-2:], height=0, order=1,
            tight_frame=False
        )
        self.unoriented_band_pyrs = [
            Steerable_Pyramid_Freq(
                himask.shape[-2:],
                height=1,
                order=self.n_orientations - 1,
                is_complex=False,
                tight_frame=False,
            )
            # want to go through these masks backwards
            for himask in self.pyr._himasks
        ]

        self.use_true_correlations = use_true_correlations
        self.scales = (
            ["pixel_statistics", "residual_lowpass"]
            + [ii for ii in range(n_scales - 1, -1, -1)]
            + ["residual_highpass"]
        )
        self.representation_scales = self._get_representation_scales()

    def _get_representation_scales(self):
        r"""returns a vector that indicates the scale of each value in the representation (Portilla-Simoncelli statistics)"""
        pixel_statistics = ["pixel_statistics"] * 6

        # magnitude_means
        magnitude_means = (
            ["residual_highpass"]
            + [
                s
                for s in range(0, self.n_scales)
                for i in range(0, self.n_orientations)
            ]
            + ["residual_lowpass"]
        )

        sc = [s for s in range(0, self.n_scales)]
        sc_lowpass = sc + ["residual_lowpass"]

        # skew_reconstructed
        skew_reconstructed = sc_lowpass

        # kurtosis_reconstructed
        kurtosis_reconstructed = sc_lowpass

        # variance_reconstructed
        std_reconstructed = sc_lowpass

        auto_correlation = (
            self.spatial_corr_width * self.spatial_corr_width
        ) * sc_lowpass
        auto_correlation_magnitude = (
            self.spatial_corr_width * self.spatial_corr_width
        ) * [s for s in sc for i in range(0, self.n_orientations)]

        cross_orientation_correlation_magnitude = (
            self.n_orientations * self.n_orientations
        ) * sc_lowpass
        cross_orientation_correlation_real = (
            4 * self.n_orientations * self.n_orientations
        ) * sc_lowpass

        cross_scale_correlation_magnitude = (
            self.n_orientations * self.n_orientations
        ) * sc
        cross_scale_correlation_real = (
            2 * self.n_orientations * max(2 * self.n_orientations, 5)
        ) * sc
        var_highpass_residual = ["residual_highpass"]

        if self.use_true_correlations:
            scales = (
                pixel_statistics
                + magnitude_means
                + auto_correlation_magnitude
                + skew_reconstructed
                + kurtosis_reconstructed
                + auto_correlation
                + std_reconstructed
                + cross_orientation_correlation_magnitude
                + cross_scale_correlation_magnitude
                + cross_orientation_correlation_real
                + cross_scale_correlation_real
                + var_highpass_residual
            )
        else:
            scales = (
                pixel_statistics
                + magnitude_means
                + auto_correlation_magnitude
                + skew_reconstructed
                + kurtosis_reconstructed
                + auto_correlation
                + cross_orientation_correlation_magnitude
                + cross_scale_correlation_magnitude
                + cross_orientation_correlation_real
                + cross_scale_correlation_real
                + var_highpass_residual
            )

        return scales

    def forward(self, image, scales=None):
        r"""Generate Texture Statistics representation of an image (see reference [1]_)

        Parameters
        ----------
        image : torch.Tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width). If it has fewer than 4 dimensions,
            we will unsqueeze it until its 4d
        scales : list, optional
            Which scales to include in the returned representation. If an empty
            list (the default), we include all scales. Otherwise, can contain
            subset of values present in this model's ``scales`` attribute.

        Returns
        -------
        representation_vector: torch.Tensor
            A flattened tensor (1d) containing the measured representation statistics.

        """

        if image.shape[0]>1:
            raise ValueError("Batch size should be 1. Portilla Simoncelli doesn't support batch operations.")


        device = image.device
        while image.ndimension() < 4:
            image = image.unsqueeze(0)

        self.pyr_coeffs = self.pyr.forward(image)
        self.representation = OrderedDict()

        ### SECTION 1 (STATISTIC: pixel_statistics) ##################
        #  Calculate pixel statistics (mean, variance, skew, kurtosis, min, max).
        self.representation["pixel_statistics"] = OrderedDict()
        self.representation["pixel_statistics"]["mean"] = torch.mean(image)
        self.representation["pixel_statistics"]["var"] = torch.var(image)
        self.representation["pixel_statistics"]["skew"] = PortillaSimoncelli.skew(
            image
        )
        self.representation["pixel_statistics"][
            "kurtosis"
        ] = PortillaSimoncelli.kurtosis(image)
        self.representation["pixel_statistics"]["min"] = torch.min(image)
        self.representation["pixel_statistics"]["max"] = torch.max(image)

        ### SECTION 2 (STATISTIC: mean_magnitude) ####################
        # Calculate the mean of the magnitude of each band of pyramid
        # coefficients.  Additionally, this section creates two
        # other dictionaries of coefficients: magnitude_pyr_coeffs
        # and real_pyr_coeffs, which contain the magnitude of the
        # pyramid coefficients and the real part of the pyramid
        # coefficients respectively.
        self.representation["magnitude_means"] = self._calculate_magnitude_means()

        ### SECTION 3 (STATISTICS: auto_correlation_magnitude,
        #                          skew_reconstructed,
        #                          kurtosis_reconstructed,
        #                          auto_correlation_reconstructed) #####
        #
        # Calculates the central auto-correlation of the magnitude of each
        # orientation/scale band.
        #
        # Calculates the skew and the kurtosis of the reconstructed
        # low-pass residuals (skew_reconstructed, kurtosis_reconstructed).
        #
        # Calculates the central auto-correlation of the low-pass residuals
        # for each scale of the pyramid (auto_correlation_reconstructed),
        # where the residual at each scale is reconstructed from the
        # previous scale.  (Note: the lowpass residual of the pyramid
        # is low-pass filtered before this reconstruction process begins,
        # see below).

        # Initialize statistics
        # let's remove the normalization from the auto_correlation statistics
        self.representation["auto_correlation_magnitude"] = torch.zeros(
            [
                self.spatial_corr_width,
                self.spatial_corr_width,
                self.n_scales,
                self.n_orientations,
            ],
            device=image.device
        )
        self.representation["skew_reconstructed"] = torch.empty((self.n_scales + 1, 1),
                                                                device=image.device)
        self.representation["kurtosis_reconstructed"] = torch.empty(
            (self.n_scales + 1, 1), device=image.device
        )
        self.representation["auto_correlation_reconstructed"] = torch.zeros(
            [self.spatial_corr_width, self.spatial_corr_width, self.n_scales + 1],
            device=image.device
        )

        if self.use_true_correlations:
            self.representation["std_reconstructed"] = torch.empty(self.n_scales + 1, 1,
                                                                   device=image.device)

        self._calculate_autocorrelation_skew_kurtosis()

        ### SECTION 4 (STATISTICS: cross_orientation_correlation_magnitude,
        #                          cross_scale_correlation_magnitude,
        #                          cross_orientation_correlation_real,
        #                          cross_scale_correlation_real) ###########
        # Calculates cross-orientation and cross-scale correlations for the
        # real parts and the magnitude of the pyramid coefficients.
        #

        # Initialize statistics
        self.representation["cross_orientation_correlation_magnitude"] = torch.zeros(
            self.n_orientations, self.n_orientations, self.n_scales + 1,
            device=image.device
        )
        self.representation["cross_scale_correlation_magnitude"] = torch.zeros(
            self.n_orientations, self.n_orientations, self.n_scales,
            device=image.device
        )
        self.representation["cross_orientation_correlation_real"] = torch.zeros(
            max(2 * self.n_orientations, 5),
            max(2 * self.n_orientations, 5),
            self.n_scales + 1,
            device=image.device
        )
        self.representation["cross_scale_correlation_real"] = torch.zeros(
            2 * self.n_orientations, max(2 * self.n_orientations, 5), self.n_scales,
            device=image.device
        )

        self._calculate_crosscorrelations()

        # STATISTIC: var_highpass_residual or the variance of the high-pass residual
        self.representation["var_highpass_residual"] = (
            self.pyr_coeffs["residual_highpass"].pow(2).mean().unsqueeze(0)
        )

        representation_vector = self.convert_to_vector()

        if scales is not None:
            ind = torch.LongTensor(
                [
                    i
                    for i, s in enumerate(self.representation_scales)
                    if s in scales
                ]
            ).to(device)
            return representation_vector.index_select(0, ind)

        return representation_vector.unsqueeze(0).unsqueeze(0)

    def convert_to_vector(self):
        r"""Converts dictionary of statistics to a vector (for synthesis).

        Returns
        -------
         -- : torch.Tensor
            Flattened 1d vector of statistics.

        """
        list_of_stats = [
            torch.cat([vv.flatten() for vv in val.values()])
            if isinstance(val, OrderedDict)
            else val.flatten()
            for (key, val) in self.representation.items()
        ]
        return torch.cat(list_of_stats)

    def convert_to_dict(self, vec):
        vec = vec.squeeze()
        rep = OrderedDict()
        rep["pixel_statistics"] = OrderedDict()
        rep["pixel_statistics"]["mean"] = vec[0]
        rep["pixel_statistics"]["var"] = vec[1]
        rep["pixel_statistics"]["skew"] = vec[2]
        rep["pixel_statistics"]["kurtosis"] = vec[3]
        rep["pixel_statistics"]["min"] = vec[4]
        rep["pixel_statistics"]["max"] = vec[5]

        n_filled = 6

        # magnitude_means
        rep["magnitude_means"] = OrderedDict()
        for ii, (k, v) in enumerate(self.representation["magnitude_means"].items()):
            rep["magnitude_means"][k] = vec[n_filled + ii]
        n_filled += ii + 1

        # auto_correlation_magnitude
        nn = (
            self.spatial_corr_width
            * self.spatial_corr_width
            * self.n_scales
            * self.n_orientations
        )
        rep["auto_correlation_magnitude"] = vec[n_filled : (n_filled + nn)].unflatten(
            0,
            (
                self.spatial_corr_width,
                self.spatial_corr_width,
                self.n_scales,
                self.n_orientations,
            ),
        )
        n_filled += nn

        # skew_reconstructed & kurtosis_reconstructed
        nn = self.n_scales + 1
        rep["skew_reconstructed"] = vec[n_filled : (n_filled + nn)]
        n_filled += nn

        rep["kurtosis_reconstructed"] = vec[n_filled : (n_filled + nn)]
        n_filled += nn

        # auto_correlation_reconstructed
        nn = self.spatial_corr_width * self.spatial_corr_width * (self.n_scales + 1)
        rep["auto_correlation_reconstructed"] = vec[
            n_filled : (n_filled + nn)
        ].unflatten(
            0, (self.spatial_corr_width, self.spatial_corr_width, self.n_scales + 1)
        )
        n_filled += nn

        if self.use_true_correlations:
            nn = self.n_scales + 1
            rep["std_reconstructed"] = vec[n_filled : (n_filled + nn)]
            n_filled += nn

        # cross_orientation_correlation_magnitude
        nn = self.n_orientations * self.n_orientations * (self.n_scales + 1)
        rep["cross_orientation_correlation_magnitude"] = vec[
            n_filled : (n_filled + nn)
        ].unflatten(0, (self.n_orientations, self.n_orientations, self.n_scales + 1))
        n_filled += nn

        # cross_scale_correlation_magnitude
        nn = self.n_orientations * self.n_orientations * self.n_scales
        rep["cross_scale_correlation_magnitude"] = vec[
            n_filled : (n_filled + nn)
        ].unflatten(0, (self.n_orientations, self.n_orientations, self.n_scales))
        n_filled += nn

        # cross_orientation_correlation_real
        nn = (
            max(2 * self.n_orientations, 5)
            * max(2 * self.n_orientations, 5)
            * (self.n_scales + 1)
        )
        rep["cross_orientation_correlation_real"] = vec[
            n_filled : (n_filled + nn)
        ].unflatten(
            0,
            (
                max(2 * self.n_orientations, 5),
                max(2 * self.n_orientations, 5),
                self.n_scales + 1,
            ),
        )
        n_filled += nn

        # cross_scale_correlation_real
        nn = 2 * self.n_orientations * max(2 * self.n_orientations, 5) * self.n_scales
        rep["cross_scale_correlation_real"] = vec[n_filled : (n_filled + nn)].unflatten(
            0, (2 * self.n_orientations, max(2 * self.n_orientations, 5), self.n_scales)
        )
        n_filled += nn

        # var_highpass_residual
        rep["var_highpass_residual"] = vec[n_filled]
        n_filled += 1

        return rep

    def _calculate_magnitude_means(self):
        r"""Calculates the mean of the pyramid coefficient magnitudes.  Also
        stores two dictionaries, one containing the magnitudes of the pyramid
        coefficient and the other containing the real parts.

        Returns
        -------
        magnitude_means: OrderedDict
            The mean of the pyramid coefficient magnitudes.

        """

        # subtract mean from lowest scale band
        self.pyr_coeffs["residual_lowpass"] = self.pyr_coeffs[
            "residual_lowpass"
        ] - torch.mean(self.pyr_coeffs["residual_lowpass"])

        # calculate two new sets of coefficients: 1) magnitude of the pyramid coefficients, 2) real part of the pyramid coefficients
        self.magnitude_pyr_coeffs = OrderedDict()
        self.real_pyr_coeffs = OrderedDict()
        for key, val in self.pyr_coeffs.items():
            if key in ["residual_lowpass", "residual_highpass"]:  # not complex
                self.magnitude_pyr_coeffs[key] = torch.abs(val).squeeze()
                self.real_pyr_coeffs[key] = val.squeeze()
            else:  # complex
                self.magnitude_pyr_coeffs[key] = val.abs().squeeze()
                self.real_pyr_coeffs[key] = val.real.squeeze()

        # STATISTIC: magnitude_means or the mean magnitude of each pyramid band
        magnitude_means = OrderedDict()
        for (key, val) in self.magnitude_pyr_coeffs.items():
            magnitude_means[key] = torch.mean(val)
            self.magnitude_pyr_coeffs[key] = (
                self.magnitude_pyr_coeffs[key] - magnitude_means[key]
            )  # subtract mean of magnitude

        return magnitude_means

    def expand(im, mult):
        r"""Resize an image (im) by a multiplier (mult).

        Parameters
        ----------
        im: torch.Tensor
            An image for expansion.
        mult: int
            Multiplier by which to resize image.

        Returns
        -------
        im_large: torch.Tensor
            resized image

        """
        im = im.squeeze()

        mx = im.shape[0]
        my = im.shape[1]
        my = mult * my
        mx = mult * mx

        fourier = mult ** 2 * torch.fft.fftshift(torch.fft.fftn(im))
        fourier_large = torch.zeros(my, mx, device=fourier.device,
                                    dtype=fourier.dtype)

        y1 = int(my / 2 + 1 - my / (2 * mult))
        y2 = int(my / 2 + my / (2 * mult))
        x1 = int(mx / 2 + 1 - mx / (2 * mult))
        x2 = int(mx / 2 + mx / (2 * mult))

        fourier_large[y1:y2, x1:x2] = fourier[1 : int(my / mult), 1 : int(mx / mult)]
        fourier_large[y1 - 1, x1:x2] = fourier[0, 1 : int(mx / mult)] / 2
        fourier_large[y2, x1:x2] = fourier[0, 1 : int(mx / mult)].flip(0) / 2
        fourier_large[y1:y2, x1 - 1] = fourier[1 : int(my / mult), 0] / 2
        fourier_large[y1:y2, x2] = fourier[1 : int(my / mult), 0].flip(0) / 2
        esq = fourier[0, 0] / 4
        fourier_large[y1 - 1, x1 - 1] = esq
        fourier_large[y1 - 1, x2] = esq
        fourier_large[y2, x1 - 1] = esq
        fourier_large[y2, x2] = esq

        fourier_large = torch.fft.fftshift(fourier_large)

        # finish this
        im_large = torch.fft.ifft2(fourier_large)

        return im_large.type(im.dtype)

    def _calculate_autocorrelation_skew_kurtosis(self):
        r"""Calculate the autocorrelation for the real parts and magnitudes of the
        coefficients. Calculate the skew and kurtosis at each scale.

        """

        # low-pass filter the low-pass residual.  We're still not sure why the original matlab code does this...
        lowpass = self.pyr_coeffs["residual_lowpass"]
        filter_pyr_coeffs = self.filterPyr.forward(lowpass)
        reconstructed_image = filter_pyr_coeffs["residual_lowpass"].squeeze()

        # Find the auto-correlation of the low-pass residual
        channel_size = torch.min(torch.tensor(lowpass.shape[-2:])).to(float)
        center = int(np.floor([(self.spatial_corr_width - 1) / 2]))
        le = int(np.min((channel_size / 2 - 1, center)))
        (
            self.representation["auto_correlation_reconstructed"][
                center - le : center + le + 1,
                center - le : center + le + 1,
                self.n_scales,
            ],
            vari,
        ) = self.compute_autocorrelation(reconstructed_image)
        (
            self.representation["skew_reconstructed"][self.n_scales],
            self.representation["kurtosis_reconstructed"][self.n_scales],
        ) = self.compute_skew_kurtosis(reconstructed_image, vari)

        if self.use_true_correlations:
            self.representation["std_reconstructed"][self.n_scales] = vari ** 0.5

        for this_scale in range(self.n_scales - 1, -1, -1):
            for nor in range(0, self.n_orientations):
                ch = self.magnitude_pyr_coeffs[(this_scale, nor)]
                channel_size = np.min((ch.shape[-1], ch.shape[-2]))
                le = int(np.min((channel_size / 2.0 - 1, center)))
                # Find the auto-correlation of the magnitude band
                (
                    self.representation["auto_correlation_magnitude"][
                        center - le : center + le + 1,
                        center - le : center + le + 1,
                        this_scale,
                        nor,
                    ],
                    vari,
                ) = self.compute_autocorrelation(ch)

            reconstructed_image = (
                PortillaSimoncelli.expand(reconstructed_image, 2) / 4.0
            )
            reconstructed_image = reconstructed_image.unsqueeze(0).unsqueeze(0)

            # reconstruct the unoriented band for this scale
            unoriented_band_pyr = self.unoriented_band_pyrs[this_scale]
            unoriented_pyr_coeffs = unoriented_band_pyr.forward(reconstructed_image)
            for ii in range(0, self.n_orientations):
                unoriented_pyr_coeffs[(0, ii)] = (
                    self.real_pyr_coeffs[(this_scale, ii)].unsqueeze(0).unsqueeze(0)
                )
            unoriented_band = unoriented_band_pyr.recon_pyr(unoriented_pyr_coeffs,levels=[0])

            # Add the unoriented band to the image reconstruction
            reconstructed_image = reconstructed_image + unoriented_band

            # Find auto-correlation of the reconstructed image
            (
                self.representation["auto_correlation_reconstructed"][
                    center - le : center + le + 1,
                    center - le : center + le + 1,
                    this_scale,
                ],
                vari,
            ) = self.compute_autocorrelation(reconstructed_image)
            if self.use_true_correlations:
                self.representation["std_reconstructed"][this_scale] = vari ** 0.5
            # Find skew and kurtosis of the reconstructed image
            (
                self.representation["skew_reconstructed"][this_scale],
                self.representation["kurtosis_reconstructed"][this_scale],
            ) = self.compute_skew_kurtosis(reconstructed_image, vari)

    def _calculate_crosscorrelations(self):
        r"""Calculate the cross-orientation and cross-scale correlations for the real parts
        and the magnitudes of the pyramid coefficients.

        """

        for this_scale in range(0, self.n_scales):
            band_num_el = self.real_pyr_coeffs[(this_scale, 0)].numel()
            if this_scale < self.n_scales - 1:
                next_scale_mag = torch.empty((band_num_el, self.n_orientations),
                                             device=self.pyr.hi0mask.device)
                next_scale_real = torch.empty((band_num_el, self.n_orientations * 2),
                                              device=self.pyr.hi0mask.device)

                for nor in range(0, self.n_orientations):
                    
                    upsampled = (
                        PortillaSimoncelli.expand(
                            self.pyr_coeffs[(this_scale + 1, nor)].squeeze(), 2
                        )
                        / 4.0
                    )

                    # Here we double the phase of the upsampled band.  This trick
                    # allows us to find the correlation between content in two adjacent
                    # spatial scales.
                    X = upsampled.abs() * torch.cos(
                        2 * torch.atan2(upsampled.real, upsampled.imag)
                    )
                    Y = upsampled.abs() * torch.sin(
                        2 * torch.atan2(upsampled.real, upsampled.imag)
                    )

                    # Save the components
                    next_scale_real[:, nor] = X.t().flatten()
                    next_scale_real[:, nor + self.n_orientations] = Y.t().flatten()

                    # Save the magnitude
                    mag = (X ** 2 + Y ** 2) ** 0.5
                    next_scale_mag[:, nor] = (mag - mag.mean()).t().flatten()

            else:
                upsampled = (
                    PortillaSimoncelli.expand(
                        self.real_pyr_coeffs["residual_lowpass"].squeeze(), 2
                    )
                    / 4.0
                )
                upsampled = upsampled.t()
                next_scale_real = torch.stack(
                    (
                        upsampled.flatten(),
                        upsampled.roll(1, 0).flatten(),
                        upsampled.roll(-1, 0).flatten(),
                        upsampled.roll(1, 1).flatten(),
                        upsampled.roll(-1, 1).flatten(),
                    ),
                    1,
                )
                next_scale_mag = torch.empty((0), device=upsampled.device)

            orientation_bands_mag = (
                torch.stack(
                    tuple(
                        [
                            aa.t()
                            for aa in [
                                self.magnitude_pyr_coeffs[(this_scale, ii)]
                                for ii in range(0, self.n_orientations)
                            ]
                        ]
                    )
                )
                .view((self.n_orientations, band_num_el))
                .t()
            )

            if next_scale_mag.shape[0] > 0:
                np0 = next_scale_mag.shape[1]
            else:
                np0 = 0

            self.representation["cross_orientation_correlation_magnitude"][
                0 : self.n_orientations, 0 : self.n_orientations, this_scale
            ] = self.compute_crosscorrelation( orientation_bands_mag.t(), orientation_bands_mag, band_num_el)

            if np0 > 0:
                self.representation["cross_scale_correlation_magnitude"][
                    0 : self.n_orientations, 0:np0, this_scale
                ] = self.compute_crosscorrelation(
                    orientation_bands_mag.t(), next_scale_mag, band_num_el
                )

                # correlations on the low-pass residuals
                if this_scale == self.n_scales - 1:
                    self.representation["cross_orientation_correlation_magnitude"][
                        0:np0, 0:np0, this_scale + 1
                    ] = self.compute_crosscorrelation(
                        next_scale_mag.t(), next_scale_mag, band_num_el / 4.0
                    )

            orientation_bands_real = (
                torch.stack(
                    tuple(
                        [
                            aa.t()
                            for aa in [
                                self.real_pyr_coeffs[(this_scale, ii)].squeeze()
                                for ii in range(0, self.n_orientations)
                            ]
                        ]
                    )
                )
                .view((self.n_orientations, band_num_el))
                .t()
            )

            if next_scale_real.shape[0] > 0:
                nrp = next_scale_real.shape[1]
            else:
                nrp = 0
            self.representation["cross_orientation_correlation_real"][
                0 : self.n_orientations, 0 : self.n_orientations, this_scale
            ] = self.compute_crosscorrelation(
                orientation_bands_real.t(), orientation_bands_real, band_num_el
            )
            if nrp > 0:
                self.representation["cross_scale_correlation_real"][
                    0 : self.n_orientations, 0:nrp, this_scale
                ] = self.compute_crosscorrelation(
                    orientation_bands_real.t(), next_scale_real, band_num_el
                )
                if (
                    this_scale == self.n_scales - 1
                ):  # correlations on the low-pass residuals
                    self.representation["cross_orientation_correlation_real"][
                        0:nrp, 0:nrp, this_scale + 1
                    ] = self.compute_crosscorrelation(
                        next_scale_real.t(), next_scale_real, (band_num_el / 4.0)
                    )

    def compute_crosscorrelation(self, ch1, ch2, band_num_el):
        r"""Computes either the covariance of the two matrices or the cross-correlation
        depending on the value self.use_true_correlations.

        Parameters
        ----------
        ch1: torch.Tensor
            First matrix for cross correlation.
        ch2: torch.Tensor
            Second matrix for cross correlation.
        band_num_el: int
            Number of elements for bands in the scale

        Returns
        -------
        torch.Tensor
            cross-correlation.

        """

        if self.use_true_correlations:
            return ch1 @ ch2 / (band_num_el * ch1.std() * ch2.std())
        else:
            return ch1 @ ch2 / (band_num_el)

    def compute_autocorrelation(self, ch):
        r"""Computes the autocorrelation and variance of a given matrix (ch)

        Parameters
        ----------
        ch: torch.Tensor

        Returns
        -------
        ac: torch.Tensor
            Autocorrelation of matrix (ch).
        vari: torch.Tensor
            Variance of matrix (ch).

        """

        channel_size = torch.min(torch.tensor(ch.shape[-2:])).to(float)

        # Calculate the edges of the central auto-correlation
        center = int(np.floor([(self.spatial_corr_width - 1) / 2]))
        le = int(np.min((channel_size / 2.0 - 1, center)))  # center of the image ???

        # Find the center of the channel
        cy = int(ch.shape[-1] / 2)
        cx = int(ch.shape[-2] / 2)

        # Calculate the auto-correlation
        ac = torch.fft.fft2(ch.squeeze())
        ac = ac.real.pow(2) + ac.imag.pow(2)
        ac = torch.fft.ifft2(ac)
        ac = torch.fft.fftshift(ac.unsqueeze(0)).squeeze() / torch.numel(ch)

        # Return only the central auto-correlation
        ac = ac.real[cx - le : cx + le + 1, cy - le : cy + le + 1]
        vari = ac[le, le]

        if self.use_true_correlations:
            ac = ac / vari

        return ac, vari

    def compute_skew_kurtosis(self, ch, vari):
        r"""Computes the skew and kurtosis of ch.

        Skew and kurtosis of ch are computed.  If the ratio of its variance (vari)
        and the pixel variance of the original image are below a certain
        threshold (1e-6) skew and kurtosis are assigned the default values (0,3). 

        Parameters
        ----------
        ch: torch.Tensor
        vari: torch.Tensor
            variance of ch

        Returns
        -------
        skew: torch.Tensor
            skew of ch or default value (0)
        kurtosis: torch.Tensor
            kurtosis of ch or default value (3)

        """

        # Find the skew and the kurtosis of the low-pass residual
        if vari / self.representation["pixel_statistics"]["var"] > 1e-6:
            skew = PortillaSimoncelli.skew(ch, mu=0, var=vari)
            kurtosis = PortillaSimoncelli.kurtosis(ch, mu=0, var=vari)
        else:
            skew = 0
            kurtosis = 3

        return skew, kurtosis

    def skew(X, mu=None, var=None):
        r"""Computes the skew of a matrix X.

        Parameters
        ----------
        X: torch.Tensor
            matrix to compute the skew of.
        mu: torch.Tensor or None, optional
            pre-computed mean. If None, we compute it.
        var: torch.Tensor or None, optional
            pre-computed variance. If None, we compute it.

        Returns
        -------
        skew: torch.Tensor
            skew of the matrix X

        """
        if mu is None:
            mu = X.mean()
        if var is None:
            var = X.var()
        return torch.mean((X - mu).pow(3)) / (var.pow(1.5))

    def kurtosis(X, mu=None, var=None):
        r"""Computes the kurtosis of a matrix X.

        Parameters
        ----------
        X: torch.Tensor
            matrix to compute the kurtosis of.
        mu: torch.Tensor
            pre-computed mean. If None, we compute it.
        var: torch.Tensor
            pre-computed variance. If None, we compute it.

        Returns
        -------
        kurtosis: torch.Tensor
            kurtosis of the matrix X

        """
        # implementation is only for real components
        if mu is None:
            mu = X.mean()
        if var is None:
            var = X.var()
        return torch.mean(torch.abs(X - mu).pow(4)) / (var.pow(2))




    def plot_representation(
        self, data=None, ax=None, figsize=(15, 15), ylim=None, batch_idx=0, title=None
    ):

        r""" Plot the representation in a human viewable format -- stem
        plots with data separated out by statistic type.

        
        Parameters
        ----------
        data : torch.Tensor, dict, or None, optional
            The data to show on the plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_error()`` or
            another instance of this class).
        ax : 
            axis where we will plot the data
        figsize : (int, int), optional
            the size of the figure
        ylim : (int,int) or None, optional
        batch_idx : int, optional
            Which index to take from the batch dimension (the first one)
        title : string
            title for the plot

        Returns
        -------
        data : torch.Tensor, dict, or None, optional
            The data that was plotted. 
            

        """

        n_rows = 3
        n_cols = 3

        if data is None:
            rep = self.representation
        else:
            rep = self.convert_to_dict(data)

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
            if isinstance(v, OrderedDict):
                # need to make sure these are not tensors when we call the
                # plotting function
                ax = clean_stem_plot([to_numpy(v_) for v_ in v.values()], ax, k,
                                     ylim=ylim)
            else:
                ax = clean_stem_plot(to_numpy(v).flatten(), ax, k, ylim=ylim)

            axes.append(ax)

        return fig, axes



    def _representation_for_plotting(self, rep, batch_idx=0):
        r""" Converts the data into a dictionary representation that is more convenient for plotting.  Intended
        as a helper function for plot_representation.

        """
        data = OrderedDict()
        data["pixels+var_highpass"] = rep["pixel_statistics"]
        data["pixels+var_highpass"]["var_highpass_residual"] = rep[
            "var_highpass_residual"
        ]
        if self.use_true_correlations:
            data["var+skew+kurtosis"] = torch.stack(
                (
                    rep["std_reconstructed"],
                    rep["skew_reconstructed"],
                    rep["kurtosis_reconstructed"],
                )
            )
        else:
            data["skew+kurtosis"] = torch.stack(
                (rep["skew_reconstructed"], rep["kurtosis_reconstructed"])
            )

        for (k, v) in rep.items():
            if k not in [
                "pixel_statistics",
                "var_highpass_residual",
                "kurtosis_reconstructed",
                "skew_reconstructed",
                "std_reconstructed",
            ]:

                if not isinstance(v, dict) and v.squeeze().dim() >= 3:
                    vals = OrderedDict()
                    for ss in range(0, v.shape[2]):
                        tmp = torch.norm(v[:, :, ss, ...], p=2, dim=[0, 1])
                        if len(tmp.shape) == 0:
                            tmp = tmp.unsqueeze(0)
                        vals[ss] = tmp
                    dk = torch.cat(list(vals.values()))
                    data[k] = dk

                else:
                    data[k] = v

        return data

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
        should look just like the representation of this model.

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
        data : torch.Tensor, dict, or None, optional
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
        if not isinstance(data, dict):
            data = self.convert_to_dict(data)
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
        self.filterPyr = self.filterPyr.to(*args, **kwargs)
        self.unoriented_band_pyrs = [pyr.to(*args, **kwargs) for pyr in
                                     self.unoriented_band_pyrs]
        return self
