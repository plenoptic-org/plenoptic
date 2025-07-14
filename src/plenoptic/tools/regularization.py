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
    Create a function that returns the distance between the pixel
    histograms of an input image with respect to a target image.

    Parameters
    ----------
    target_img
        The target image to match the histogram against.
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
