import torch
import torch.nn as nn
from einops import rearrange, reduce

from ...tools.signal import maximum, minimum, rectangular_to_polar, autocorr
# from ...tools.conv import correlate_downsample
from ...tools.stats import kurtosis, skew, variance
from ..canonical_computations.steerable_pyramid_freq import \
    Steerable_Pyramid_Freq


class Texture_Statistics(nn.Module):
    """ Developping a texture analysis model inspired by [1]_ and [2]_

    References
    ----------
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model based
        on Joint Statistics of Complex Wavelet Coefficients.
        Int'l Journal of Computer Vision. 40(1):49-71, October, 2000.
        http://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
        http://www.cns.nyu.edu/~lcv/texture/
    .. [2] Mallat, S., Zhang, S. and Rochette, G.
        Phase harmonic correlations and convolutional neural networks.
        2020. Information and Inference: A Journal of the IMA

    TODO
    ----
    - generalize phase harmonics
    - also include non phase corrected cross scale correlations
    - optimize in pyr coeff space
    - auto_corr should operate on different scales with corresponding
    length scale (to downsample, or not to downsample? that is the question)
    - explore cross orientation phase correction, with multiscale orientation
    ie. derivative order as ori bandwidth

    """

    def __init__(self, image_size, n_ori=4, n_scale=4, n_shifts=7):
        super().__init__()

        self.n_scale = n_scale
        self.n_ori = n_ori
        self.n_shifts = n_shifts

        self.pyr = Steerable_Pyramid_Freq(image_size, height=n_scale,
                                          is_complex=True, order=n_ori-1,
                                          downsample=False)
        # self.pyr = Steerable_Pyramid_Freq(, height=1, order=0)

    def forward(self, x, y=None):
        assert x.ndim == 4
        n_batch, n_channels, h, w = x.shape
        n_scale = self.n_scale
        n_ori = self.n_ori
        n_shifts = self.n_shifts

        # in the not downsampled version of the pyramid, this wont be a concern
        # nth = np.log2(min(h, w)/n_shifts)
        # if nth <= n_scale+1:
        #     print('Warning: Na will be cut off for levels above #%d !\n',
        #            floor(nth+1));
        # la = floor((Na-1)/2);

        # TODO
        # need extra channel rotation for color images

        # 1) Marginal Statistics
        # Pixel statistics
        dims = (1, 2, 3)
        x_mean = torch.mean(x, dim=dims, keepdim=True)
        x_vari = variance(x, mean=x_mean, dim=dims, keepdim=True)
        x_skew = skew(x, mean=x_mean, var=x_vari, dim=dims, keepdim=True)
        x_kurt = kurtosis(x, mean=x_mean, var=x_vari, dim=dims, keepdim=True)
        x_mini = minimum(x, dim=dims, keepdim=True)
        x_maxi = maximum(x, dim=dims, keepdim=True)
        marginal_stats = torch.cat((x_vari, x_skew, x_kurt, x_mini, x_maxi),
                                   dim=1)
        marginal_stats = rearrange(marginal_stats, 'b k () () -> b k')

        # adding  little bit of noise
        # x = x + (x_maxi - x_mini) / 1000 * torch.randn_like(x)

        # Steerable pyramid
        if y is None:
            y = self.pyr(x)

        # Subtract mean of lowBand
        # NOTE: so that autocorr acts on centered signal
        y[:, -1] -= reduce(y[:, -1], 'b h w -> b () ()', 'mean')

        # TODO: when complex number is stable in pytorch
        # z = y[:, 1:-1:2] + 1j * y[:, 2:-1:2]
        # phase = torch.angle(z)
        # energy = torch.abs(z)

        # backward relies on torch.sign
        # which is not yet supported for complex, so for now:
        energy, phase = rectangular_to_polar(y[:, 1:-1:2],
                                             y[:, 2:-1:2])
        # real_pyr_coeff = y[:, ::2]
        abs_pyr_coeff = torch.cat([torch.sqrt(y[:, 0:1]**2),
                                   energy,
                                   torch.sqrt(y[:, -1:]**2)],
                                  dim=1)

        # Subtract mean of magnitude
        abs_pyr_coeff_mean = reduce(abs_pyr_coeff,
                                    'b c h w -> b c () ()', 'mean')
        abs_pyr_coeff -= abs_pyr_coeff_mean
        abs_pyr_coeff_mean = rearrange(abs_pyr_coeff_mean,
                                       'b c () () -> b c')

        # 2) skew and kurt of residual lowpass at each scale
        # do step by step reconstruction
        # Laplacian_pyr
        # pixelLPStats

        # 3) raw Coefficient Correlation
        # Compute central autoCorr of lowband
        auto_corr_real = torch.zeros(n_batch, n_scale+1, n_shifts, n_shifts)

        # TODO: useless?
        # low_pass = correlate_downsample(y[:, -1], 'lo0filt')

        # auto_corr_real[:,
        #               l-e:l+e+1,
        #               l-e:l+e+1,
        #               n_scale+1] = autocorr(low_pass, n_shifts)

        # Compute autoCorr of the combined (non-oriented) real band

        skew_scale = torch.zeros(n_batch, n_scale+1)
        kurt_scale = torch.zeros(n_batch, n_scale+1)

        # 4) Coefficient magnitude statistics
        # Compute  central autoCorr of each Mag band
        auto_corr_abs = autocorr(abs_pyr_coeff[:, 1:])

        # Compute the cross-correlation matrices of the coefficient magnitudes
        # pyramid at the different levels and orientations
        # cousinMagCorr
        # parentMagCorr
        # cousinRealCorr
        # parentRealCorr

        # 5) Cross-Scale Phase Statistics.
        # rearrange
        cousin_corr_abs = torch.zeros((n_batch, n_ori, n_ori, n_scale+1))
        parent_corr_abs = torch.zeros((n_batch, n_ori, n_ori, n_scale))
        cousin_corr_real = torch.zeros((n_batch, 2*n_ori, 2*n_ori, n_scale+1))
        parent_corr_real = torch.zeros((n_batch, 2*n_ori, 2*n_ori, n_scale+1))

        # Ce = torch.einsum('bchw,bdhw->bcd', energy, energy).view(
        # x.shape[0], -1) / energy.shape[-1]**2
        # Cp = torch.einsum('bchw,bdhw->bcd', phase, phase).view(
        # x.shape[0], -1) / phase.shape[-1]**2
        # Cp = torch.einsum('bchw,bdhw->bc', phase, phase).view(
        # x.shape[0], -1) / phase.shape[-1]**2
        # , Ce.view(x.shape[0], -1), Cp.view(x.shape[0], -1)

        #  NEW
        #  generalization by Mallat
        #  Phase harmonic covariance

        # Calculate the mean, range and variance of the LF and HF
        # residuals' energy.
        # varianceHPR
        variance_highpass = reduce(y[:, 0] ** 2, 'b h w -> b ()', 'mean')

        # 'pixelStats', statg0,
        # 'pixelLPStats', statsLPim,
        # 'autoCorrReal', acr,
        # 'autoCorrMag', ace,
        # 'magMeans', magMeans0,
        # 'cousinMagCorr', C0,
        # 'parentMagCorr', Cx0,
        # 'cousinRealCorr', Cr0,
        # 'parentRealCorr', Crx0,
        # 'varianceHPR', vHPR0

        stats = torch.cat(
            (marginal_stats,
             variance_highpass,
             skew_scale,
             kurt_scale,
             abs_pyr_coeff_mean,
             rearrange(auto_corr_abs, 'b c n1 n2 -> b (c n1 n2)'),
             rearrange(auto_corr_real, 'b c n1 n2 -> b (c n1 n2)'),
             rearrange(cousin_corr_abs, 'b o1 o2 s -> b (o1 o2 s)'),
             rearrange(parent_corr_abs, 'b o1 o2 s -> b (o1 o2 s)'),
             rearrange(cousin_corr_real, 'b o1 o2 s -> b (o1 o2 s)'),
             rearrange(parent_corr_real, 'b o1 o2 s -> b (o1 o2 s)')
             ),
            dim=1)
        return stats
