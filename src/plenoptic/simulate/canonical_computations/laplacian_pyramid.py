"""
Laplacian pyramid.

Simple class for handling the Laplacian Pyramid.
"""

import torch
import torch.nn as nn

from ...tools.conv import blur_downsample, upsample_blur


class LaplacianPyramid(nn.Module):
    """
    Laplacian Pyramid in Torch.

    The Laplacian pyramid (Burt and Adelson, 1983, [1]_) is a multiscale image
    representation. It decomposes the image by computing the local mean using Gaussian
    blurring filters and subtracting it from the image and repeating this operation on
    the local mean itself after downsampling. This representation is overcomplete and
    invertible.

    Parameters
    ----------
    n_scales
        Number of scales to compute.
    scale_filter
        If ``True``, the norm of the downsampling/upsampling filter is 1. If ``False``,
        it is 2. If the norm is 1, the image is multiplied by 4 during the upsampling
        operation; the net effect is that the :math:`n` -th scale of the pyramid is
        divided by :math:`2^n`.

    Attributes
    ----------
    n_scales : int
        Number of computed scales.
    scale_filter : bool
        Whether the filter is scaled or not.

    References
    ----------
    .. [1] Burt, P. and Adelson, E., 1983. The Laplacian pyramid as a compact
       image code. IEEE Transactions on communications, 31(4), pp.532-540.

    Examples
    --------
    >>> import plenoptic as po
    >>> lpyr = po.simul.LaplacianPyramid(n_scales=4, scale_filter=True)
    """

    def __init__(self, n_scales: int = 5, scale_filter: bool = False):
        super().__init__()
        self.n_scales = n_scales
        self.scale_filter = scale_filter
        # This model has no trainable parameters, so it's always in eval mode
        self.eval()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Build the Laplacian pyramid of an image.

        Builds a Laplacian pyramid of height ``self.n_scales``. Because the tensor at
        each scale will have a different height and width, we return a list of tensors
        instead of a single tensor.

        Parameters
        ----------
        x
            Image, or batch of images of shape (batch, channel, height, width). If there
            are multiple batches or channels, the Laplacian is computed separately for
            each of them.

        Returns
        -------
        y
            Laplacian pyramid representation, each element of the list corresponds to a
            scale, from fine to coarse.

        Examples
        --------
        .. plot::
          :context: reset

          >>> import plenoptic as po
          >>> img = po.data.einstein()
          >>> lpyr = po.simul.LaplacianPyramid()
          >>> po.imshow(lpyr(img))
          <PyrFigure ...>
        """
        y = []
        for scale in range(self.n_scales - 1):
            odd = torch.as_tensor(x.shape)[2:4] % 2
            x_down = blur_downsample(x, scale_filter=self.scale_filter)
            x_up = upsample_blur(x_down, odd, scale_filter=self.scale_filter)
            y.append(x - x_up)
            x = x_down
        y.append(x)

        return y

    def recon_pyr(self, y: list[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct the image from its Laplacian pyramid coefficients.

        The input to ``recon_pyr`` should be list of tensors similar to those returned
        by ``self.forward``.

        Parameters
        ----------
        y
            Laplacian pyramid representation, each element of the list
            corresponds to a scale, from fine to coarse. ``len(y)`` should be
            ``self.n_scales``.

        Returns
        -------
        x
            Image, or batch of images.

        Examples
        --------
        .. plot::
          :context: reset

          >>> import plenoptic as po
          >>> import torch
          >>> img = po.data.einstein()
          >>> lpyr = po.simul.LaplacianPyramid()
          >>> coeffs = lpyr(img)
          >>> recon = lpyr.recon_pyr(coeffs)
          >>> torch.allclose(img, recon)
          True
          >>> titles = ["Original", "Reconstructed", "Difference"]
          >>> po.imshow([img, recon, img - recon], title=titles)
          <PyrFigure ...>
        """
        x = y[self.n_scales - 1]
        for scale in range(self.n_scales - 1, 0, -1):
            odd = torch.as_tensor(y[scale - 1].shape)[2:4] % 2
            y_up = upsample_blur(x, odd, scale_filter=self.scale_filter)
            x = y[scale - 1] + y_up

        return x
