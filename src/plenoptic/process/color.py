"""Color-space transformations."""

import torch
from torch import Tensor

__all__ = [
    "OPC",
    "PCA",
]


def __dir__() -> list[str]:
    return __all__


def _validate_color_image(image: Tensor) -> None:
    if image.ndim not in (3, 4):
        raise ValueError(
            "Expected a 3d or 4d color image tensor."
            f" Got {image.ndim}d tensor with shape {tuple(image.shape)}."
        )
    if image.shape[-3] != 3:
        raise ValueError(
            "Expected an image with three color channels."
            f" Got {image.shape[-3]} channels."
        )


class OPC(torch.nn.Module):
    """Transform an RGB image to opponent-cone (OPC) space.

    This first maps RGB values to approximate LMS cone responses, then maps
    those responses to achromatic, red-green, and blue-yellow channels. The
    matrices are from the PooledStatisticsMetamers implementation [1]_.

    References
    ----------
    .. [1] https://github.com/ProgramofComputerGraphics/PooledStatisticsMetamers/
       blob/main/poolstatmetamer/color_utils.py
    """

    def __init__(self):
        super().__init__()
        rgb_to_lms = torch.tensor(
            [
                [0.3811, 0.5783, 0.0402],
                [0.1967, 0.7244, 0.0782],
                [0.0241, 0.1288, 0.8444],
            ]
        )
        lms_to_opponent = torch.tensor(
            [
                [0.5, 0.5, 0.0],
                [-4.0, 4.0, 0.0],
                [0.5, 0.5, -1.0],
            ]
        )
        self.register_buffer("matrix", lms_to_opponent @ rgb_to_lms)

    def forward(self, image: Tensor) -> Tensor:
        """Transform ``image`` from RGB to opponent-cone space.

        Returns
        -------
        transformed_image
            The opponent-cone image, with the same shape as ``image``.
        """
        _validate_color_image(image)
        return torch.einsum("ij,...jhw->...ihw", self.matrix, image)


class PCA(torch.nn.Module):
    """Center and whiten color channels with respect to a target image PCA.

    Parameters
    ----------
    image
        A 3d tensor with shape ``(channel, height, width)`` or a 4d tensor with
        shape ``(1, channel, height, width)``. The fitted transform is fixed
        after initialization and can be applied to later image batches.
    max_relative_scaling
        Maximum ratio between the largest and smallest scaling applied along
        the principal components by the whitening transform. Must be at least 1.
        By default, no relative scaling limit is applied.
    """

    def __init__(self, image: Tensor, max_relative_scaling: float = float("inf")):
        super().__init__()
        _validate_color_image(image)
        if not max_relative_scaling >= 1:
            raise ValueError("max_relative_scaling must be greater than or equal to 1.")
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError("PCA must be fit to a single image.")
            image = image.squeeze(0)

        mean = image.mean(dim=(-2, -1), keepdim=True)
        centered = (image - mean).flatten(1)
        covariance = centered @ centered.mT / centered.shape[-1]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalue_floor = eigenvalues.max() / max_relative_scaling**2
        eigenvalues = eigenvalues.clamp_min(eigenvalue_floor)
        eigenvalues = eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).eps)
        transform = torch.einsum("i,ji->ij", eigenvalues.rsqrt(), eigenvectors)
        self.register_buffer("mean", mean.detach())
        self.register_buffer("matrix", transform.detach())

    def forward(self, image: Tensor) -> Tensor:
        """Center and whiten ``image`` using the fitted PCA transform.

        Returns
        -------
        transformed_image
            The PCA-whitened image, with the same shape as ``image``.
        """
        _validate_color_image(image)
        centered = image - self.mean
        return torch.einsum("ij,...jhw->...ihw", self.matrix, centered)
