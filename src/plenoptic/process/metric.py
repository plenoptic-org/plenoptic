"""Image-processing functions used by metrics."""  # numpydoc ignore=ES01

import warnings
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .convolutions import same_padding
from .filters import circular_gaussian2d
from .laplacian_pyramid import LaplacianPyramid

DIRNAME = resources.files("plenoptic.process")


__all__ = [
    "normalized_laplacian_pyramid",
    "ssim_map",
]


def __dir__() -> list[str]:
    return __all__


def _ssim_parts(
    img1: torch.Tensor,
    img2: torch.Tensor,
    pad: Literal[False, "constant", "reflect", "replicate", "circular"] = False,
    func_name: str = "SSIM",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the various components used to compute SSIM.

    This should not be called by users directly, but is meant to assist for
    calculating SSIM and MS-SSIM.

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
    pad
        If not ``False``, how to pad the image for the convolutions computing the
        local average of each image. See :func:`torch.nn.functional.pad` for how
        these work.
    func_name
        Name of the function that called this one, in order to raise more helpful error
        / warning messages.

    Returns
    -------
    map_ssim
        Map of SSIM values across the image.
    contrast_structure_map
        Map of contrast structure values.
    weight
        Weight used for stability of computation.

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
    """  # numpydoc ignore=EX01
    img_ranges = torch.stack([img1.min(), img1.max(), img2.min(), img2.max()])
    if (img_ranges > 1).any() or (img_ranges < 0).any():
        warnings.warn(
            f"Image range falls outside [0, 1]. {func_name} output may not make sense.",
        )

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
            "SSIM was designed for grayscale images and here it will be"
            " computed separately for each channel (so channels are treated in"
            " the same way as batches).",
        )
    if img1.dtype != img2.dtype:
        raise ValueError("Input images must have same dtype!")

    real_size = min(11, img1.shape[2], img1.shape[3])
    std = torch.as_tensor(1.5).to(img1.device)
    window = circular_gaussian2d(real_size, std=std).to(img1.dtype)

    # these two checks are guaranteed with our above bits, but if we add
    # ability for users to set own window, they'll be necessary
    window_sum = window.sum((-1, -2), keepdim=True)
    if not torch.allclose(window_sum, torch.ones_like(window_sum)):
        warnings.warn("window should have sum of 1! normalizing...")
        window = window / window_sum
    if window.ndim != 4:
        raise ValueError("window must have 4 dimensions!")

    if pad is not False:
        img1 = same_padding(img1, (real_size, real_size), pad_mode=pad)
        img2 = same_padding(img2, (real_size, real_size), pad_mode=pad)

    def windowed_average(img: torch.Tensor) -> torch.Tensor:  # numpydoc ignore=GL08
        padding = 0
        (n_batches, n_channels, _, _) = img.shape
        img = img.reshape(n_batches * n_channels, 1, img.shape[2], img.shape[3])
        img_average = F.conv2d(img, window, padding=padding)
        img_average = img_average.reshape(
            n_batches, n_channels, img_average.shape[2], img_average.shape[3]
        )
        return img_average

    mu1 = windowed_average(img1)
    mu2 = windowed_average(img2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = windowed_average(img1 * img1) - mu1_sq
    sigma2_sq = windowed_average(img2 * img2) - mu2_sq
    sigma12 = windowed_average(img1 * img2) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    # SSIM is the product of a luminance component, a contrast component, and a
    # structure component. The contrast-structure component has to be separated
    # when computing MS-SSIM.
    luminance_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast_structure_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    map_ssim = luminance_map * contrast_structure_map

    # the weight used for stability
    weight = torch.log((1 + sigma1_sq / C2) * (1 + sigma2_sq / C2))
    return map_ssim, contrast_structure_map, weight


def ssim_map(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Structural similarity index map.

    As described in Wang et al., 2004 [5]_, the structural similarity index (SSIM) is a
    perceptual distance metric, giving the distance between two images. SSIM is based on
    three comparison measurements between the two images: luminance, contrast, and
    structure. All of these are computed convolutionally across the images. See the
    references for more information.

    This implementation follows the original implementation, as found online [6]_, as
    well as providing the option to use the weighted version used in Wang and
    Simoncelli, 2008 [8]_ (which was shown to consistently improve the image quality
    prediction on the LIVE database). More info can be found online [7]_.

    Note that this is a similarity metric (not a distance), and so 1 means the
    two images are identical and 0 means they're very different. When the two
    images are negatively correlated, SSIM can be negative. SSIM is bounded
    between -1 and 1.

    This function returns the SSIM map, showing the SSIM values across the
    image. For the mean SSIM (a single value metric), call :func:`ssim`.

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

    Returns
    -------
    ssim_map
        4d tensor containing the map of SSIM values.

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

    References
    ----------
    .. [5] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
       quality assessment: From error measurement to structural similarity"
       IEEE Transactions on Image Processing, vol. 13, no. 1, Jan. 2004.
    .. [6] matlab code `<https://www.cns.nyu.edu/~lcv/ssim/ssim_index.m>`_
    .. [7] project page `<https://www.cns.nyu.edu/~lcv/ssim/>`_
    .. [8] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
       competition: A methodology for comparing computational models of
       perceptual discriminability. Journal of Vision, 8(12), 1–13.
       https://dx.doi.org/10.1167/8.12.8

    Examples
    --------
    >>> import plenoptic as po
    >>> import torch
    >>> po.set_seed(0)
    >>> img = po.data.einstein()
    >>> ssim_map = po.process.ssim_map(img, img + torch.rand_like(img))
    >>> ssim_map.shape
    torch.Size([1, 1, 246, 246])
    """
    if min(img1.shape[2], img1.shape[3]) < 11:
        warnings.warn(
            "SSIM uses 11x11 convolutional kernel, but the height and/or "
            "the width of the input image is smaller than 11, so the "
            "kernel size is set to be the minimum of these two numbers."
        )
    return _ssim_parts(img1, img2)[0]


def normalized_laplacian_pyramid(img: torch.Tensor) -> list[torch.Tensor]:
    """
    Compute the normalized Laplacian Pyramid using pre-optimized parameters.

    Model parameters are those used in Laparra et al., 2016 [10]_, copied from the
    matlab code used in the paper, found online [11]_.

    Parameters
    ----------
    img
        Image, or batch of images of shape (batch, channel, height, width). This
        representation is designed for grayscale images and will be computed separately
        for each channel (so channels are treated in the same way as batches).

    Returns
    -------
    normalized_laplacian_activations
        The normalized Laplacian Pyramid with six scales.

    References
    ----------
    .. [10] Laparra, V., Ballé, J., Berardino, A. and Simoncelli, E.P., 2016. Perceptual
        image quality assessment using a normalized Laplacian pyramid. Electronic
        Imaging, 2016(16), pp.1-6.
    .. [11] matlab code: `<https://www.cns.nyu.edu/~lcv/NLPyr/NLP_dist.m>`_

    Examples
    --------
    .. plot::

      >>> import plenoptic as po
      >>> img = po.data.einstein()
      >>> pyramid = po.process.normalized_laplacian_pyramid(img)
      >>> [p.shape for p in pyramid]
      [torch.Size([1, 1, 256, 256]),
       torch.Size([1, 1, 128, 128]),
       torch.Size([1, 1, 64, 64]),
       torch.Size([1, 1, 32, 32]),
       torch.Size([1, 1, 16, 16]),
       torch.Size([1, 1, 8, 8])]
      >>> po.plot.imshow(pyramid, col_wrap=3)
      <PyrFigure size ...>
    """
    (_, channel, _, _) = img.size()

    N_scales = 6
    spatialpooling_filters = np.load(Path(DIRNAME) / "DN_filts.npy")

    sigmas = np.load(Path(DIRNAME) / "DN_sigmas.npy")

    L = LaplacianPyramid(n_scales=N_scales, scale_filter=True)
    laplacian_activations = L.forward(img)

    padding = 2
    normalized_laplacian_activations = []
    for N_b in range(0, N_scales):
        filt = torch.as_tensor(
            spatialpooling_filters[N_b], dtype=img.dtype, device=img.device
        ).repeat(channel, 1, 1, 1)
        filtered_activations = F.conv2d(
            torch.abs(laplacian_activations[N_b]),
            filt,
            padding=padding,
            groups=channel,
        )
        normalized_laplacian_activations.append(
            laplacian_activations[N_b] / (sigmas[N_b] + filtered_activations)
        )

    return normalized_laplacian_activations
