import torch
import torch.nn as nn
from ...tools.conv import blur_downsample, upsample_blur


class Laplacian_Pyramid(nn.Module):
    """
    The Laplacian pyramid [1]_ is a multiscale image representation.
    It decomposes the image by computing the local mean using Gaussian
    blurring filters and substracting it from the image and repeating this
    operation on the local mean itself after downsampling. This
    representation is overcomplete and invertible.

    Argument
    --------
    n_scales: int
        number of scales to compute
    scale_filter: bool, optional
        If true, the norm of the downsampling/upsampling filter is 1. If false (default), it is 2.
        If the norm is 1, the image is multiplied by 4 during the upsampling operation; the net effect
        is that the `n`th scale of the pyramid is divided by `2^n`.

    Reference
    ---------
    .. [1] Burt, P. and Adelson, E., 1983. The Laplacian pyramid as a compact
    image code. IEEE Transactions on communications, 31(4), pp.532-540.
    """

    def __init__(self, n_scales=5, scale_filter=False):
        super().__init__()
        self.n_scales = n_scales
        self.scale_filter = scale_filter

    def analysis(self, x):
        """

        Arguments
        ---------
        x: torch.Tensor of shape (B, C, H, W)
            Image, or batch of images. If there are multiple channels,
            the Laplacian is computed separately for each of them
        Returns
        -------
        y: list of torch.Tensor
            Laplacian pyramid representation, each element of the list
            corresponds to a scale, from fine to coarse
        """

        y = []
        for scale in range(self.n_scales - 1):
            odd = torch.tensor(x.shape)[2:4] % 2
            x_down = blur_downsample(x, scale_filter=self.scale_filter)
            x_up = upsample_blur(x_down, odd, scale_filter=self.scale_filter)
            y.append(x - x_up)
            x = x_down
        y.append(x)

        return y

    def synthesis(self, y):
        """

        Arguments
        ---------
        y: list of torch.Tensor
            Laplacian pyramid representation, each element of the list
            corresponds to a scale, from fine to coarse
        Returns
        -------
        x: torch.Tensor of shape (B, C, H, W)
            Image, or batch of images
        """
        x = y[self.n_scales - 1]
        for scale in range(self.n_scales - 1, 0, -1):
            odd = torch.tensor(y[scale - 1].shape)[2:4] % 2
            y_up = upsample_blur(x, odd, scale_filter=self.scale_filter)
            x = y[scale - 1] + y_up

        return x
