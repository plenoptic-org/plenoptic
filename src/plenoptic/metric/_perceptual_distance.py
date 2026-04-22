"""
Metrics designed to model human perceptual distance.

Metrics that model human perceptual distance seek to answer the question "how different
do humans find these two images?".
"""  # numpydoc ignore=EX01

import warnings
from typing import Literal

import torch
import torch.nn.functional as F

from ..process import normalized_laplacian_pyramid
from ..process.metric import _ssim_parts

__all__ = [
    "ms_ssim",
    "nlpd",
    "ssim",
]


def __dir__() -> list[str]:
    return __all__


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    weighted: bool = False,
    pad: Literal[False, "constant", "reflect", "replicate", "circular"] = False,
) -> torch.Tensor:
    r"""
    Compute the structural similarity index.

    As described in Wang et al., 2004 [1]_, the structural similarity index (SSIM) is a
    perceptual distance metric, giving the distance between two images. SSIM is based on
    three comparison measurements between the two images: luminance, contrast, and
    structure. All of these are computed convolutionally across the images. See the
    references for more information.

    This implementation follows the original implementation, as found online [2]_, as
    well as providing the option to use the weighted version used in Wang and
    Simoncelli, 2008 [4]_ (which was shown to consistently improve the image quality
    prediction on the LIVE database). More info can be found online [3]_.

    Note that this is a similarity metric (not a distance), and so 1 means the
    two images are identical and 0 means they're very different. When the two
    images are negatively correlated, SSIM can be negative. SSIM is bounded
    between -1 and 1.

    This function returns the mean SSIM, a scalar-valued metric giving the
    average over the whole image. For the SSIM map (showing the computed value
    across the image), call :func:`~plenoptic.process.ssim_map`.

    Parameters
    ----------
    img1
        The first image or batch of images, of shape (batch, channel, height, width).
    img2
        The second image or batch of images, of shape (batch, channel, height, width).
        The heights and widths of ``img1`` and ``img2`` must be the same. The numbers of
        batches and channels of ``img1`` and ``img2`` need to be broadcastable: either
        they are the same or one of them is 1. The output will be computed separately
        for each channel (so channels are treated in the same way as batches). Both
        images should have values between 0 and 1. Otherwise, the result may be
        inaccurate, and we will raise a warning (but will still compute it).
    weighted
        Whether to use the original, unweighted SSIM version (``False``) as used
        in [1]_ or the weighted version (``True``) as used in [4]_. See Notes
        section for the weight.
    pad :
        If not ``False``, how to pad the image for the convolutions computing the
        local average of each image. See :func:`torch.nn.functional.pad` for how
        these work.

    Returns
    -------
    mssim
        2d tensor of shape (batch, channel) containing the mean SSIM for each
        image, averaged over the whole image.

    Raises
    ------
    ValueError
        If either ``img1`` or ``img2`` is not 4d.
    ValueError
        If ``img1`` and ``img2`` have different height or width.
    ValueError
        If ``img1`` and ``img2`` have different batch or channel, unless one of them has
        a 1 there, so they can be broadcast.
    ValueError
        If ``img1`` and ``img2`` have different dtypes.

    Warns
    -----
    UserWarning
        If either ``img1`` or ``img2`` has multiple channels, as SSIM was designed for
        grayscale images.
    UserWarning
        If at least one scale from either ``img1`` or ``img2`` has height or width of
        less than 11, since SSIM uses an 11x11 convolutional kernel.

    Notes
    -----
    The weight used when ``weighted=True`` is:

    .. math::
       \log((1+\frac{\sigma_1^2}{C_2})(1+\frac{\sigma_2^2}{C_2}))

    where :math:`\sigma_1^2` and :math:`\sigma_2^2` are the variances of ``img1``
    and ``img2``, respectively, and :math:`C_2` is a constant. See [4]_ for more
    details.

    References
    ----------
    .. [1] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
       quality assessment: From error measurement to structural similarity"
       IEEE Transactions on Image Processing, vol. 13, no. 1, Jan. 2004.
    .. [2] matlab code: `<https://www.cns.nyu.edu/~lcv/ssim/ssim_index.m>`_
    .. [3] project page: `<https://www.cns.nyu.edu/~lcv/ssim/>`_
    .. [4] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
       competition: A methodology for comparing computational models of
       perceptual discriminability. Journal of Vision, 8(12), 1–13.
       https://dx.doi.org/10.1167/8.12.8

    Examples
    --------
    >>> import plenoptic as po
    >>> import torch
    >>> po.set_seed(0)
    >>> img = po.data.einstein()
    >>> po.metric.ssim(img, img + torch.rand_like(img))
    tensor([[0.0519]])
    """
    # these are named map_ssim instead of the perhaps more natural ssim_map
    # because that's the name of a function
    map_ssim, _, weight = _ssim_parts(img1, img2, pad)
    if not weighted:
        mssim = map_ssim.mean((-1, -2))
    else:
        mssim = (map_ssim * weight).sum((-1, -2)) / weight.sum((-1, -2))

    if min(img1.shape[2], img1.shape[3]) < 11:
        warnings.warn(
            "SSIM uses 11x11 convolutional kernel, but the height and/or "
            "the width of the input image is smaller than 11, so the "
            "kernel size is set to be the minimum of these two numbers."
        )
    return mssim


def ms_ssim(
    img1: torch.Tensor, img2: torch.Tensor, power_factors: torch.Tensor | None = None
) -> torch.Tensor:
    r"""
    Multiscale structural similarity index (MS-SSIM).

    As described in Wang et al., 2003 [9]_, multiscale structural similarity index
    (MS-SSIM) is an improvement upon structural similarity index (SSIM) that takes into
    account the perceptual distance between two images on different scales.

    SSIM is based on three comparison measurements between the two images:
    luminance, contrast, and structure. All of these are computed convolutionally
    across the images, producing three maps instead of scalars. The SSIM map is
    the elementwise product of these three maps. See :func:`ssim` and
    :func:`~plenoptic.process.ssim_map` for a full description of SSIM.

    To get images of different scales, average pooling operations with kernel
    size 2 are performed recursively on the input images. The product of
    contrast map and structure map (the "contrast-structure map") is computed
    for all but the coarsest scales, and the overall SSIM map is only computed
    for the coarsest scale. Their mean values are raised to exponents and
    multiplied to produce MS-SSIM:

    .. math::
        MSSSIM = {SSIM}_M^{a_M} \prod_{i=1}^{M-1} ({CS}_i)^{a_i}

    Here :math:`M` is the number of scales, :math:`{CS}_i` is the mean value
    of the contrast-structure map for the i'th finest scale, and :math:`{SSIM}_M`
    is the mean value of the SSIM map for the coarsest scale. If at least one
    of these terms are negative, the value of MS-SSIM is zero. The values of
    :math:`a_i, i=1,...,M` are taken from the argument ``power_factors``.

    Parameters
    ----------
    img1
        The first image or batch of images, of shape (batch, channel, height, width).
    img2
        The second image or batch of images, of shape (batch, channel, height, width).
        The heights and widths of ``img1`` and ``img2`` must be the same. The numbers of
        batches and channels of ``img1`` and ``img2`` need to be broadcastable: either
        they are the same or one of them is 1. The output will be computed separately
        for each channel (so channels are treated in the same way as batches). Both
        images should have values between 0 and 1. Otherwise, the result may be
        inaccurate, and we will raise a warning (but will still compute it).
    power_factors
        Power exponents for the mean values of maps, for different scales (from
        fine to coarse). The length of this array determines the number of scales.
        If ``None``, set to ``[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]``, which is what
        psychophysical experiments in Wang et al., 2003 [9]_ found.

    Returns
    -------
    msssim
        2d tensor of shape (batch, channel) containing the MS-SSIM for each image.

    Raises
    ------
    ValueError
        If either ``img1`` or ``img2`` is not 4d.
    ValueError
        If ``img1`` and ``img2`` have different height or width.
    ValueError
        If ``img1`` and ``img2`` have different batch or channel, unless one of them has
        a 1 there, so they can be broadcast.
    ValueError
        If ``img1`` and ``img2`` have different dtypes.

    Warns
    -----
    UserWarning
        If either ``img1`` or ``img2`` has multiple channels, as MS-SSIM was designed
        for grayscale images.
    UserWarning
        If at least one scale from either ``img1`` or ``img2`` has height or width of
        less than 11, since SSIM uses an 11x11 convolutional kernel.

    References
    ----------
    .. [9] Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
       structural similarity for image quality assessment." The Thrity-Seventh
       Asilomar Conference on Signals, Systems & Computers, 2003. Vol. 2. IEEE, 2003.

    Examples
    --------
    >>> import plenoptic as po
    >>> import torch
    >>> po.set_seed(0)
    >>> img = po.data.einstein()
    >>> po.metric.ms_ssim(img, img + torch.rand_like(img))
    tensor([[0.4684]])
    """
    if power_factors is None:
        power_factors = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def downsample(img: torch.Tensor) -> torch.Tensor:  # numpydoc ignore=GL08
        img = F.pad(img, (0, img.shape[3] % 2, 0, img.shape[2] % 2), mode="replicate")
        img = F.avg_pool2d(img, kernel_size=2)
        return img

    msssim = 1
    for i in range(len(power_factors) - 1):
        _, contrast_structure_map, _ = _ssim_parts(img1, img2, func_name="MS-SSIM")
        msssim *= F.relu(contrast_structure_map.mean((-1, -2))).pow(power_factors[i])
        img1 = downsample(img1)
        img2 = downsample(img2)
    map_ssim, _, _ = _ssim_parts(img1, img2, func_name="MS-SSIM")
    msssim *= F.relu(map_ssim.mean((-1, -2))).pow(power_factors[-1])

    if min(img1.shape[2], img1.shape[3]) < 11:
        warnings.warn(
            "SSIM uses 11x11 convolutional kernel, but for some scales "
            "of the input image, the height and/or the width is smaller "
            "than 11, so the kernel size in SSIM is set to be the "
            "minimum of these two numbers for these scales."
        )
    return msssim


def nlpd(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized Laplacian Pyramid Distance.

    As described in Laparra et al., 2016 [12]_, this is an image quality metric based on
    the transformations associated with the early visual system: local luminance
    subtraction and local contrast gain control.

    A laplacian pyramid subtracts a local estimate of the mean luminance at six scales.
    Then a local gain control divides these centered coefficients by a weighted sum of
    absolute values in spatial neighborhood.

    These weights parameters were optimized for redundancy reduction over an training
    database of (undistorted) natural images, as described in the paper. Parameters were
    copied from matlab code used for the paper, found online [13]_.

    Note that we compute root mean squared error for each scale, and then average over
    these, effectively giving larger weight to the lower frequency coefficients
    (which are fewer in number, due to subsampling).

    Parameters
    ----------
    img1
        The first image or batch of images of shape (batch, channel, height, width).
    img2
        The second image or batch of images of shape (batch, channel, height, width).
        The heights and widths of ``img1`` and ``img2`` must be the same. The numbers of
        batches and channels of ``img1`` and ``img2`` need to be broadcastable: either
        they are the same or one of them is 1. The output will be computed separately
        for each channel (so channels are treated in the same way as batches). Both
        images should have values between 0 and 1. Otherwise, the result may be
        inaccurate, and we will raise a warning (but will still compute it).

    Returns
    -------
    distance
        The normalized Laplacian Pyramid distance, with shape (batch, channel).

    Raises
    ------
    ValueError
        If either ``img1`` or ``img2`` is not 4d.
    ValueError
        If ``img1`` and ``img2`` have different height or width.
    ValueError
        If ``img1`` and ``img2`` have different batch or channel, unless one of them has
        a 1 there, so they can be broadcast.
    ValueError
        If ``img1`` and ``img2`` have different dtypes.

    Warns
    -----
    UserWarning
        If either ``img1`` or ``img2`` has multiple channels, as SSIM was designed for
        grayscale images.
    UserWarning
        If either ``img1`` or ``img2`` has a value outside of range ``[0, 1]``.

    References
    ----------
    .. [12] Laparra, V., Ballé, J., Berardino, A. and Simoncelli, E.P., 2016. Perceptual
        image quality assessment using a normalized Laplacian pyramid. Electronic
        Imaging, 2016(16), pp.1-6.
    .. [13] matlab code: `<https://www.cns.nyu.edu/~lcv/NLPyr/NLP_dist.m>`_

    Examples
    --------
    >>> import plenoptic as po
    >>> einstein_img = po.data.einstein()
    >>> curie_img = po.data.curie()
    >>> po.metric.nlpd(einstein_img, curie_img)
    tensor([[1.3507]])
    """
    if not img1.ndim == img2.ndim == 4:
        raise ValueError(
            "Input images should have four dimensions: (batch, channel, height, width)"
        )
    if img1.shape[-2:] != img2.shape[-2:]:
        raise ValueError("img1 and img2 must have the same height and width!")
    for i in range(2):
        if img1.shape[i] != img2.shape[i] and img1.shape[i] != 1 and img2.shape[i] != 1:
            raise ValueError(
                "Either img1 and img2 should have the same number of "
                "elements in the batch and channel dimensions, or one of "
                "them should be 1! But got shapes "
                f"{img1.shape}, {img2.shape} instead"
            )
    if img1.shape[1] > 1 or img2.shape[1] > 1:
        warnings.warn(
            "NLPD was designed for grayscale images and here it will be"
            " computed separately for each channel (so channels are treated in"
            " the same way as batches)."
        )

    img_ranges = torch.stack([img1.min(), img1.max(), img2.min(), img2.max()])
    if (img_ranges > 1).any() or (img_ranges < 0).any():
        warnings.warn(
            "Image range falls outside [0, 1]. NLPD output may not make sense."
        )

    y1 = normalized_laplacian_pyramid(img1)
    y2 = normalized_laplacian_pyramid(img2)

    epsilon = 1e-10  # for optimization purpose (stabilizing the gradient around zero)
    dist = []
    for i in range(6):
        dist.append(torch.sqrt(torch.mean((y1[i] - y2[i]) ** 2, dim=(2, 3)) + epsilon))

    return torch.stack(dist).mean(dim=0)
