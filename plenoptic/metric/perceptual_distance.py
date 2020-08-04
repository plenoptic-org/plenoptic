import numpy as np
import torch
import torch.nn.functional as F
import warnings

from ..simulate.canonical_computations import Laplacian_Pyramid, Steerable_Pyramid_Freq
from ..simulate.canonical_computations import local_gain_control, rectangular_to_polar_dict

import os
dirname = os.path.dirname(__file__)

# TODO
# clean up, test and document (MS)SSIM


def _gaussian(window_size=11, sigma=1.5):
    """Normalized, centered Gaussian

    1d Gaussian of size `window_size`, centered half-way, with variable std
    deviation, and sum of 1.

    With default values, this is the 1d Gaussian used to generate the windows
    for SSIM

    Parameters
    ----------
    window_size : int, optional
        size of the gaussian
    sigma : float, optional
        std dev of the gaussian

    Returns
    -------
    window : torch.tensor
        1d gaussian

    """
    x = torch.arange(window_size, dtype=torch.float32)
    mu = window_size//2
    gauss = torch.exp(-(x-mu)**2 / (2*sigma**2))
    return gauss


def create_window(window_size=11, n_channels=1):
    """Create 2d Gaussian window

    Creates 4d tensor containing a 2d Gaussian window (with 1 batch and
    `n_channels` channels), normalized so that each channel has a sum of 1.

    With default parameters, this is the Gaussian window used to compute the
    statistics for SSIM.

    Parameters
    ----------
    window_size : int, optional
        height/width of the window
    n_channels : int, optional
        number of channels

    Returns
    -------
    window : torch.tensor
        4d tensor containing the Gaussian windows

    """
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(n_channels, 1, window_size, window_size).contiguous()
    return window / window.sum((-1, -2))


def ssim(img1, img2, weighted=False, dynamic_range=1):
    """Structural similarity index

    As described in [1]_, the structural similarity index (SSIM) is a
    perceptual distance metric, giving the distance between two images. SSIM is
    based on three comparison measurements between the two images: luminance,
    contrast, and structure. All of these are computed in windows across the
    images. See the references for more information.

    This implementations follows the original implementation, as found at [2]_,
    as well as providing the option to use the weighted version used in [4]_
    (which was shown to consistently improve the image quality prediction on
    the LIVE database).

    Argument
    --------
    img1 : torch.tensor
        4d tensor with first image to compare
    img2 : torch.tensor
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
    mssim : torch.tensor
        2d tensor containing the mean SSIM for each image, averaged over the
        whole image
    ssim_map : torch.tensor
        4d tensor containing the SSIM map, giving the SSIM value at each
        location on the image

    References
    ----------
    .. [1] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
       quality assessment: From error measurement to structural similarity"
       IEEE Transactios on Image Processing, vol. 13, no. 1, Jan. 2004.
    .. [2] [matlab code](https://www.cns.nyu.edu/~lcv/ssim/ssim_index.m)
    .. [3] [project page](https://www.cns.nyu.edu/~lcv/ssim/)
    .. [4] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
       competition: A methodology for comparing computational models of
       perceptual discriminability. Journal of Vision, 8(12), 1–13.
       http://dx.doi.org/10.1167/8.12.8

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
    window = create_window(real_size, n_channels=n_channels).to(img1.device)
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

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if not weighted:
        mssim = ssim_map.mean((-1, -2))
    else:
        weight = torch.log(torch.matmul((1+(sigma1_sq/C2)), (1+(sigma2_sq/C2))))
        mssim = (ssim_map*weight).sum((-1, -2)) / weight.sum((-1, -2))

    return mssim, ssim_map


def msssim(img1, img2, window_size=11, size_average=True, dynamic_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True,
                       dynamic_range=dynamic_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, dynamic_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.dynamic_range = dynamic_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


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
        filt = torch.tensor(spatialpooling_filters[N_b], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
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
    .. [1] Laparra, V., Ballé, J., Berardino, A. and Simoncelli, E.P., 2016. Perceptual image quality assessment using a normalized Laplacian pyramid. Electronic Imaging, 2016(16), pp.1-6.
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

    TODO

    under construction
    """

    if complex:
        linear = Steerable_Pyramid_Freq(IM_1.shape[-2:], order=O, height=S, is_complex=True)
        non_linear = rectangular_to_polar_dict
    else:
        linear = Steerable_Pyramid_Freq(IM_1.shape[-2:], order=O, height=S)
        non_linear = local_gain_control

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
