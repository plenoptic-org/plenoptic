"""Class versions of perceptual metric functions."""  # numpydoc ignore=ES01

import torch
from deprecated.sphinx import deprecated

from .perceptual_distance import normalized_laplacian_pyramid


@deprecated(
    "NLP will be removed soon, use perceptual_distance.normalized_laplacian_pyramid directly",  # noqa: E501
    "1.2.0",
)
class NLP(torch.nn.Module):
    r"""
    Simple class for implementing normalized laplacian pyramid.

    This class just calls
    :func:`~plenoptic.metric.perceptual_distance.normalized_laplacian_pyramid`
    on the image and returns a 3d tensor with the flattened activations.

    NOTE: synthesis using this class will not be the exact same as
    synthesis using the :func:`~plenoptic.metric.perceptual_distance.nlpd` function,
    because the ``nlpd`` function uses the root-mean square of the
    L2 distance (i.e., ``torch.sqrt(torch.mean(x-y)**2))`` as the distance metric
    between representations.

    Model parameters are those used in [1]_, copied from the matlab code used in the
    paper, found at [2]_.

    References
    ----------
    .. [1] Laparra, V., Ballé, J., Berardino, A. and Simoncelli, E.P., 2016. Perceptual
        image quality assessment using a normalized Laplacian pyramid. Electronic
        Imaging, 2016(16), pp.1-6.
    .. [2] `matlab code <https://www.cns.nyu.edu/~lcv/NLPyr/NLP_dist.m>`_
    """

    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute flattened NLP activations.

        WARNING: For now this only supports images with batch and
        channel size 1

        Parameters
        ----------
        image
            Image to pass to normalized_laplacian_pyramid.

        Returns
        -------
        representation
            3d tensor with flattened NLP activations.

        Raises
        ------
        Exception
            If ``image`` has more than one batch or channel.
        """
        if image.shape[0] > 1 or image.shape[1] > 1:
            raise Exception("For now, this only supports batch and channel size 1")
        activations = normalized_laplacian_pyramid(image)
        # activations is a list of tensors, each at a different scale
        # (down-sampled by factors of 2). To combine these into one
        # vector, we need to flatten each of them and then unsqueeze so
        # it is 3d

        return torch.cat([i.flatten() for i in activations]).unsqueeze(0).unsqueeze(0)
