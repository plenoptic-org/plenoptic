import torch
from .perceptual_distance import normalized_laplacian_pyramid


class NLP(torch.nn.Module):
    r"""simple class for implementing normalized laplacian pyramid

    This class just calls
    ``plenoptic.metric.normalized_laplacian_pyramid`` on the image and
    returns a 3d tensor with the flattened activations.

    NOTE: synthesis using this class will not be the exact same as
    synthesis using the ``plenoptic.metric.nlpd`` function (by default),
    because the synthesis methods use ``torch.norm(x - y, p=2)`` as the
    distance metric between representations, whereas ``nlpd`` uses the
    root-mean square of the distance (i.e.,
    ``torch.sqrt(torch.mean(x-y)**2))``

    """
    def __init__(self):
        super().__init__()

    def forward(self, image):
        """returns flattened NLP activations

        WARNING: For now this only supports images with batch and
        channel size 1

        Parameters
        ----------
        image : torch.Tensor
            image to pass to normalized_laplacian_pyramid

        Returns
        -------
        representation : torch.Tensor
            3d tensor with flattened NLP activations

        """
        if image.shape[0] > 1 or image.shape[1] > 1:
            raise Exception("For now, this only supports batch and channel size 1")
        activations = normalized_laplacian_pyramid(image)
        # activations is a list of tensors, each at a different scale
        # (down-sampled by factors of 2). To combine these into one
        # vector, we need to flatten each of them and then unsqueeze so
        # it is 3d
        return torch.cat([i.flatten() for i in activations]).unsqueeze(0).unsqueeze(0)
