import numpy as np
import torch
import torch.nn.functional as F
import warnings

from ..simulate.canonical_computations import Laplacian_Pyramid, Steerable_Pyramid_Freq
from ..simulate.canonical_computations import local_gain_control_dict, rectangular_to_polar_dict
from ..simulate.canonical_computations.filters import circular_gaussian2d

import os

dirname = os.path.dirname(__file__)


def _ssim_parts(img1, img2, dynamic_range):
    """Calcluates the various components used to compute SSIM

    This should not be called by users directly, but is meant to assist for
    calculating SSIM and MS-SSIM.

    Parameters
    ----------
    img1 : torch.Tensor
        4d tensor with first image to compare
    img2 : torch.Tensor
        4d tensor with second image to compare. Must have the same height and
        width (last two dimensions) as `img1`
    dynamic_range : int, optional.
        dynamic range of the images. Note we assume that both images have the
        same dynamic range. 1, the default, is appropriate for float images
        between 0 and 1, as is common in synthesis. 2 is appropriate for float
        images between -1 and 1, and 255 is appropriate for standard 8-bit
        integer images. We'll raise a warning if it looks like your value is
        not appropriate for `img1` or `img2`, but will calculate it anyway.

    """
    img_ranges = torch.tensor([[img1.min(), img1.max()], [img2.min(), img2.max()]])
    if dynamic_range == 1:
        if (img_ranges > 1).any() or (img_ranges < 0).any():
            warnings.warn("dynamic_range is 1 but image range falls outside [0, 1]"
                          f" img1: {img_ranges[0]}, img2: {img_ranges[1]}. "
                          "Continuing anyway...")
    elif dynamic_range == 2:
        if (img_ranges > 1).any() or (img_ranges < -1).any():
            warnings.warn("dynamic_range is 2 but image range falls outside [-1, 1]"
                          f" img1: {img_ranges[0]}, img2: {img_ranges[1]}. "
                          "Continuing anyway...")
    elif dynamic_range == 255:
        if (img_ranges > 255).any() or (img_ranges < 0).any():
            warnings.warn("dynamic_range is 255 but image range falls outside [0, 255]"
                          f" img1: {img_ranges[0]}, img2: {img_ranges[1]}. "
                          "Continuing anyway...")
    padd = 0
    (n_batches, n_channels, height, width) = img1.shape
    if n_channels > 1:
        warnings.warn("SSIM was developed on grayscale images, no guarantee "
                      "it will make sense for more than one channel!")
    if img2.shape[-2:] != (height, width):
        raise Exception("img1 and img2 must have the same height and width!")
    if n_batches != img2.shape[0]:
        if n_batches != 1 and img2.shape[0] != 1:
            raise Exception("Either img1 and img2 should have the same of "
                            "elements in the batch dimension, or one of "
                            "them should be 1! But got shapes "
                            f"{img1.shape}, {img2.shape} instead")

    real_size = min(11, height, width)
    std = torch.tensor(1.5).to(img1.device)
    window = circular_gaussian2d(real_size, std=std, out_channels=n_channels)

    # these two checks are guaranteed with our above bits, but if we add
    # ability for users to set own window, they'll be necessary
    if (window.sum((-1, -2)) > 1).any():
        warnings.warn("window should have sum of 1! normalizing...")
        window = window / window.sum((-1, -2), keepdim=True)
    if window.ndim != 4:
        raise Exception("window must have 4 dimensions!")

    mu1 = F.conv2d(img1, window, padding=padd, groups=n_channels)
    mu2 = F.conv2d(img2, window, padding=padd, groups=n_channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=n_channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=n_channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=n_channels) - mu1_mu2

    C1 = (0.01 * dynamic_range) ** 2
    C2 = (0.03 * dynamic_range) ** 2

    # SSIM is the product of a luminance component, a contrast component, and a
    # structure component. The contrast-structure component has to be separated
    # when computing MS-SSIM.
    luminance_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast_structure_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    map_ssim = luminance_map * contrast_structure_map

    # the weight used for stability
    weight = torch.log((1 + sigma1_sq/C2) * (1 + sigma2_sq/C2))
    return map_ssim, contrast_structure_map, weight


def ssim(img1, img2, weighted=False, dynamic_range=1):
    r"""Structural similarity index

    As described in [1]_, the structural similarity index (SSIM) is a
    perceptual distance metric, giving the distance between two images. SSIM is
    based on three comparison measurements between the two images: luminance,
    contrast, and structure. All of these are computed convolutionally across the
    images. See the references for more information.

    This implementation follows the original implementation, as found at [2]_,
    as well as providing the option to use the weighted version used in [4]_
    (which was shown to consistently improve the image quality prediction on
    the LIVE database).

    Note that this is a similarity metric (not a distance), and so 1 means the
    two images are identical and 0 means they're very different. It is bounded
    between 0 and 1.

    This function returns the mean SSIM, a scalar-valued metric giving the
    average over the whole image. For the SSIM map (showing the computed value
    across the image), call `ssim_map`.

    Parameters
    ----------
    img1 : torch.Tensor
        4d tensor with first image to compare
    img2 : torch.Tensor
        4d tensor with second image to compare. Must have the same height and
        width (last two dimensions) as `img1`
    weighted : bool, optional
        whether to use the original, unweighted SSIM version (`False`) as used
        in [1]_ or the weighted version (`True`) as used in [4]_. See Notes
        section for the weight
    dynamic_range : int, optional.
        dynamic range of the images. Note we assume that both images have the
        same dynamic range. 1, the default, is appropriate for float images
        between 0 and 1, as is common in synthesis. 2 is appropriate for float
        images between -1 and 1, and 255 is appropriate for standard 8-bit
        integer images. We'll raise a warning if it looks like your value is
        not appropriate for `img1` or `img2`, but will calculate it anyway.

    Returns
    ------
    mssim : torch.Tensor
        2d tensor of shape (batch, channel) containing the mean SSIM for each
        image, averaged over the whole image

    Notes
    -----
    The weight used when `weighted=True` is:

    .. math::
       \log((1+\frac{\sigma_1^2}{C_2})(1+\frac{\sigma_2^2}{C_2}))

    where :math:`sigma_1^2` and :math:`sigma_2^2` are the variances of `img1`
    and `img2`, respectively, and :math:`C_2` is a constant which depends on
    `dynamic_range`. See [4]_ for more details.

    References
    ----------
    .. [1] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
       quality assessment: From error measurement to structural similarity"
       IEEE Transactions on Image Processing, vol. 13, no. 1, Jan. 2004.
    .. [2] [matlab code](https://www.cns.nyu.edu/~lcv/ssim/ssim_index.m)
    .. [3] [project page](https://www.cns.nyu.edu/~lcv/ssim/)
    .. [4] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
       competition: A methodology for comparing computational models of
       perceptual discriminability. Journal of Vision, 8(12), 1–13.
       http://dx.doi.org/10.1167/8.12.8

    """
    # these are named map_ssim instead of the perhaps more natural ssim_map
    # because that's the name of a function
    map_ssim, _, weight = _ssim_parts(img1, img2, dynamic_range)
    if not weighted:
        mssim = map_ssim.mean((-1, -2))
    else:
        mssim = (map_ssim*weight).sum((-1, -2)) / weight.sum((-1, -2))

    if min(img1.shape[2], img1.shape[3]) < 11:
        warnings.warn("SSIM uses 11x11 convolutional kernel, but the height and/or "
                      "the width of the input image is smaller than 11, so the "
                      "kernel size is set to be the minimum of these two numbers.")
    return mssim


def ssim_map(img1, img2, dynamic_range=1):
    """Structural similarity index map

    As described in [1]_, the structural similarity index (SSIM) is a
    perceptual distance metric, giving the distance between two images. SSIM is
    based on three comparison measurements between the two images: luminance,
    contrast, and structure. All of these are computed convolutionally across the
    images. See the references for more information.

    This implementation follows the original implementation, as found at [2]_,
    as well as providing the option to use the weighted version used in [4]_
    (which was shown to consistently improve the image quality prediction on
    the LIVE database).

    Note that this is a similarity metric (not a distance), and so 1 means the
    two images are identical and 0 means they're very different. It is bounded
    between 0 and 1.

    This function returns the SSIM map, showing the SSIM values across the
    image. For the mean SSIM (a single value metric), call `ssim`.

    Parameters
    ----------
    img1 : torch.Tensor
        4d tensor with first image to compare
    img2 : torch.Tensor
        4d tensor with second image to compare. Must have the same height and
        width (last two dimensions) as `img1`
    weighted : bool, optional
        whether to use the original, unweighted SSIM version (`False`) as used
        in [1]_ or the weighted version (`True`) as used in [4]_. See Notes
        section for the weight
    dynamic_range : int, optional.
        dynamic range of the images. Note we assume that both images have the
        same dynamic range. 1, the default, is appropriate for float images
        between 0 and 1, as is common in synthesis. 2 is appropriate for float
        images between -1 and 1, and 255 is appropriate for standard 8-bit
        integer images. We'll raise a warning if it looks like your value is
        not appropriate for `img1` or `img2`, but will calculate it anyway.

    Returns
    ------
    ssim_map : torch.Tensor
        4d tensor containing the map of SSIM values.

    References
    ----------
    .. [1] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
       quality assessment: From error measurement to structural similarity"
       IEEE Transactions on Image Processing, vol. 13, no. 1, Jan. 2004.
    .. [2] [matlab code](https://www.cns.nyu.edu/~lcv/ssim/ssim_index.m)
    .. [3] [project page](https://www.cns.nyu.edu/~lcv/ssim/)
    .. [4] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
       competition: A methodology for comparing computational models of
       perceptual discriminability. Journal of Vision, 8(12), 1–13.
       http://dx.doi.org/10.1167/8.12.8

    """
    if min(img1.shape[2], img1.shape[3]) < 11:
        warnings.warn("SSIM uses 11x11 convolutional kernel, but the height and/or "
                      "the width of the input image is smaller than 11, so the "
                      "kernel size is set to be the minimum of these two numbers.")
    return _ssim_parts(img1, img2, dynamic_range)[0]


def ms_ssim(img1, img2, dynamic_range=1, power_factors=None):
    r"""Multiscale structural similarity index (MS-SSIM)

    As described in [1]_, multiscale structural similarity index (MS-SSIM) is
    an improvement upon structural similarity index (SSIM) that takes into
    account the perceptual distance between two images on different scales.

    SSIM is based on three comparison measurements between the two images:
    luminance, contrast, and structure. All of these are computed convolutionally
    across the images, producing three maps instead of scalars. The SSIM map is
    the elementwise product of these three maps. See `metric.ssim` and
    `metric.ssim_map` for a full description of SSIM.

    To get images of different scales, average pooling operations with kernel
    size 2 are performed recursively on the input images. The product of
    contrast map and structure map (the "contrast-structure map") is computed
    for all but the coarsest scales, and the overall SSIM map is only computed
    for the coarsest scale. Their mean values are raised to exponents and
    multiplied to produce MS-SSIM:
    .. math::
        MSSSIM = {SSIM}_M^{a_M} \prod_{i=1}^{M-1} ({CS}_i)^{a_i}
    Here :math: `M` is the number of scales, :math: `{CS}_i` is the mean value
    of the contrast-structure map for the i'th finest scale, and :math: `{SSIM}_M`
    is the mean value of the SSIM map for the coarsest scale. If at least one
    of these terms are negative, the value of MS-SSIM is zero. The values of
    :math: `a_i, i=1,...,M` are taken from the argument `power_factors`.

    Parameters
    ----------
    img1 : torch.Tensor
        4d tensor with first image to compare
    img2 : torch.Tensor
        4d tensor with second image to compare. Must have the same height and
        width (last two dimensions) as `img1`
    dynamic_range : int, optional.
        dynamic range of the images. Note we assume that both images have the
        same dynamic range. 1, the default, is appropriate for float images
        between 0 and 1, as is common in synthesis. 2 is appropriate for float
        images between -1 and 1, and 255 is appropriate for standard 8-bit
        integer images. We'll raise a warning if it looks like your value is
        not appropriate for `img1` or `img2`, but will calculate it anyway.
    power_factors : 1D array, optional.
        power exponents for the mean values of maps, for different scales (from
        fine to coarse). The length of this array determines the number of scales.
        By default, this is set to [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        which is what psychophysical experiments in [1]_ found.

    Returns
    ------
    msssim : torch.Tensor
        2d tensor of shape (batch, channel) containing the MS-SSIM for each image

    References
    ----------
    .. [1] Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
       structural similarity for image quality assessment." The Thrity-Seventh
       Asilomar Conference on Signals, Systems & Computers, 2003. Vol. 2. IEEE, 2003.

    """
    if power_factors is None:
        power_factors = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def downsample(img):
        img = F.pad(img, (0, img.shape[3] % 2, 0, img.shape[2] % 2), mode="replicate")
        img = F.avg_pool2d(img, kernel_size=2)
        return img

    msssim = 1
    for i in range(len(power_factors) - 1):
        _, contrast_structure_map, _ = _ssim_parts(img1, img2, dynamic_range)
        msssim *= F.relu(contrast_structure_map.mean((-1, -2))).pow(power_factors[i])
        img1 = downsample(img1)
        img2 = downsample(img2)
    map_ssim, _, _ = _ssim_parts(img1, img2, dynamic_range)
    msssim *= F.relu(map_ssim.mean((-1, -2))).pow(power_factors[-1])

    if min(img1.shape[2], img1.shape[3]) < 11:
        warnings.warn("SSIM uses 11x11 convolutional kernel, but for some scales "
                      "of the input image, the height and/or the width is smaller "
                      "than 11, so the kernel size in SSIM is set to be the "
                      "minimum of these two numbers for these scales.")
    return msssim


def normalized_laplacian_pyramid(im):
    """computes the normalized Laplacian Pyramid using pre-optimized parameters

    Arguments
    --------
    im: torch.Tensor
    Returns
    -------
    normalized_laplacian_activations: list of torch.Tensor
    """

    (_, channel, height, width) = im.size()

    N_scales = 6
    spatialpooling_filters = np.load(dirname + '/DN_filts.npy')
    sigmas = np.load(dirname + '/DN_sigmas.npy')

    L = Laplacian_Pyramid(n_scales=N_scales)
    laplacian_activations = L.analysis(im)

    padd = 2
    normalized_laplacian_activations = []
    for N_b in range(0, N_scales):
        filt = torch.tensor(spatialpooling_filters[N_b], dtype=torch.float32,
                            device=im.device).unsqueeze(0).unsqueeze(0)
        filtered_activations = F.conv2d(torch.abs(laplacian_activations[N_b]), filt, padding=padd, groups=channel)
        normalized_laplacian_activations.append(laplacian_activations[N_b] / (sigmas[N_b] + filtered_activations))

    return normalized_laplacian_activations


def nlpd(IM_1, IM_2):
    """Normalized Laplacian Pyramid Distance

    As described in  [1]_, this is an image quality metric based on the transformations associated with the early
    visual system: local luminance subtraction and local contrast gain control

    A laplacian pyramid subtracts a local estimate of the mean luminance at six scales.
    Then a local gain control divides these centered coefficients by a weighted sum of absolute values
    in spatial neighborhood.

    These weights parameters were optimized for redundancy reduction over an training
    database of (undistorted) natural images.

    Note that we compute root mean squared error for each scale, and then average over these,
    effectively giving larger weight to the lower frequency coefficients
    (which are fewer in number, due to subsampling).

    Parameters
    ----------
    IM_1: torch.Tensor
        image, (1 x 1 x H x W)
    IM_2: torch.Tensor
        image, (1 x 1 x H x W)

    Returns
    -------
    distance: float

    Note
    ----
    only accepts single channel images

    References
    ----------
    .. [1] Laparra, V., Ballé, J., Berardino, A. and Simoncelli, E.P., 2016. Perceptual image quality
       assessment using a normalized Laplacian pyramid. Electronic Imaging, 2016(16), pp.1-6.
    """

    y = normalized_laplacian_pyramid(torch.cat((IM_1, IM_2), 0))

    # for optimization purpose (stabilizing the gradient around zero)
    epsilon = 1e-10
    dist = []
    for i in range(6):
        dist.append(torch.sqrt(torch.mean((y[i][0] - y[i][1]) ** 2) + epsilon))

    return torch.stack(dist).mean()


def nspd(IM_1, IM_2, O=1, S=5, complex=True):
    """Normalized steerable pyramid distance

    spatially local normalization pool

    under construction
    """

    if complex:
        linear = Steerable_Pyramid_Freq(IM_1.shape[-2:], order=O, height=S, is_complex=True)
        non_linear = rectangular_to_polar_dict
    else:
        linear = Steerable_Pyramid_Freq(IM_1.shape[-2:], order=O, height=S)
        non_linear = local_gain_control_dict

    linear.to(IM_1.device)
    pyr = linear(torch.cat((IM_1, IM_2), 0))

    norm, state = non_linear(pyr)

    # for optimization purpose (stabilizing the gradient around zero)
    epsilon = 1e-10
    dist = []
    for key in state.keys():
        # TODO learn weights on TID2013
        dist.append(torch.sqrt(torch.mean((norm[key][0] - norm[key][1]) ** 2) + epsilon))
        dist.append(torch.sqrt(torch.mean((state[key][0] - state[key][1]) ** 2) + epsilon))

    return torch.stack(dist).mean()
