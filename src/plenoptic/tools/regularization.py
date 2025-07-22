"""Tools for regularizing image synthesis."""
# numpydoc ignore=ES01

import torch
from torch import Tensor


def penalize_range(
    synth_img: Tensor,
) -> Tensor:
    r"""
    Calculate quadratic penalty on values outside of ``allowed_range``.

    Instead of clamping values to exactly fall in a range, this provides
    a 'softer' way of doing it, by imposing a quadratic penalty on any
    values outside the allowed_range. All values within the
    allowed_range have a penalty of 0.

    Parameters
    ----------
    synth_img
        The tensor to penalize. the synthesized image.
    allowed_range
        2-tuple of values giving the (min, max) allowed values.
    **kwargs
        Ignored, only present to absorb extra arguments.

    Returns
    -------
    penalty
        Penalty for values outside range.
    """
    allowed_range = (0.0, 1.0)
    # Using clip like this is equivalent to using boolean indexing (e.g.,
    # synth_img[synth_img < allowed_range[0]]) but much faster
    below_min = torch.clip(synth_img - allowed_range[0], max=0).pow(2).sum()
    above_max = torch.clip(synth_img - allowed_range[1], min=0).pow(2).sum()
    return below_min + above_max


def histogram_matching(target_img: Tensor, n_bins: int = 20, sigma: float = 0.05) -> callable:
    """
    Create a function that computes the distance between the pixel
    histogram the target image, and the pixel histogram of the
    input image. Uses a soft histogram to allow for differentiability.

    Parameters
    ----------
    target_img
        The target image to compute the distance to.
    n_bins
        Number of bins for the histogram.
    sigma
        Standard deviation for the Gaussian kernel used in the soft histogram.

    Returns
    -------
    hist_fun
        Function that computes the histogram distance.
    """
    target_hist = _soft_histogram(target_img, n_bins, sigma)

    def hist_fun(synth_img: Tensor) -> Tensor:
        """
        Compute the pixel histogram distance between the synthesized image
        and the target image.

        Parameters
        ----------
        synth_img
            The synthesized image to compare against the target image.

        Returns
        -------
        dist
            The distance between the histograms of the synthesized and target images.
        """
        synth_hist = _soft_histogram(synth_img, n_bins, sigma)
        dist = torch.sum((synth_hist - target_hist) ** 2)
        return dist

    return hist_fun


def spectral_matching(target_img: Tensor) -> callable:
    """
    Create a function that computes the distance between the
    power spectrum of the target image and the power
    spectrum of an input image.

    Parameters ----------
    target_img
        The target image to compare the distance to.

    Returns
    -------
    spec_fun
        Function that computes the power spectrum distance.
    """
    target_spec = _amplitude_spectrum(target_img)
    target_norm = torch.norm(target_spec, dim=(-1), keepdim=True)

    def spec_fun(synth_img: Tensor) -> Tensor:
        """
        Compute the power spectrum distance between the synthesized image
        and the target image.

        Parameters
        ----------
        synth_img
            The synthesized image to compare against the target image.

        Returns
        -------
        dist
            The distance between the power spectra of the synthesized and target images.
        """
        synth_spec = _amplitude_spectrum(synth_img)
        synth_norm = torch.norm(synth_spec, dim=(-1), keepdim=True)

        diff = synth_spec - target_spec
        dist = torch.sum(diff ** 2)

        return dist

    return spec_fun


def _soft_histogram(x: Tensor, n_bins: int, sigma: float):
    """
    Compute a soft and differentiable histogram for a tensor.

    Parameters
    ----------
    x
        Input tensor to compute the histogram for.
    bins
        Number of bins for the histogram.
    sigma
        Standard deviation for the Gaussian kernel used in the soft histogram.

    Returns
    -------
    hist
        Soft histogram of the input tensor.
    """
    bins = torch.linspace(0, 1, n_bins).unsqueeze(0)
    x = x.view(-1, 1)
    dist = (x - bins) ** 2
    weights = torch.exp(-dist / (2 * sigma ** 2))
    hist = weights.sum(dim=0)
    hist = hist / hist.sum()
    return hist


def _amplitude_spectrum(img: torch.Tensor):
    """
    Compute the amplitude spectrum of an image, weighting the
    amplitude by frequency to emphasize higher frequencies, and
    downsampling by average pooling.

    Parameters
    ----------
    img
        Input image tensor of shape (B, C, H, W).

    Returns
    -------
    amplitude_spec
        Amplitude spectrum of the input image, normalized and centered.
    """
    img = img - img.mean()

    fourier = torch.fft.fft2(img)
    amplitude_spec = torch.abs(fourier)

    # Normalize the amplitude spectrum
    norm_factor = (img.shape[-1] * img.shape[-2])
    amplitude_spec = amplitude_spec / norm_factor

    amplitude_spec = torch.fft.fftshift(amplitude_spec)

    # Apply whitening
    r = _radius_grid(amplitude_spec) + 1e-6  # Avoid division by zero
    whitening = r # Avoid division by zero

    amplitude_spec = amplitude_spec * whitening.view(1, 1, *amplitude_spec.shape[-2:])

    # Apply binning of the amplitude spectrum
    amplitude_spec = torch.nn.functional.avg_pool2d(
        amplitude_spec, kernel_size=8, stride=4, padding=0
    ).squeeze(0)

    return amplitude_spec


def _radius_grid(amplitude_spec: torch.Tensor):
    """
    Compute the radial distance grid for a 2D amplitude spectrum.
    This function computes the mean amplitude for each integer radius.

    Parameters
    ----------
    amplitude_spec
        Batch of 2D tensor (B, C, H, W) representing the amplitude spectrum.

    Returns:
    --------
    radius_grid
        2D tensor of radial distances from the center of the spectrum.
    """
    H, W = amplitude_spec.shape[-2:]
    cy, cx = H // 2, W // 2   # center indices

    # Coordinate grids
    y = torch.arange(H, device=amplitude_spec.device) - cy
    x = torch.arange(W, device=amplitude_spec.device) - cx
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Radial distance
    radius_grid = torch.sqrt(xx**2 + yy**2)

    return radius_grid
