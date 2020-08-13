import torch


class Identity(torch.nn.Module):
    r"""simple class that just returns a copy of the image

    We use this as a "dummy model" for metrics that we don't have the
    representation for. We use this as the model and then just change
    the objective function.

    """
    def __init__(self, name=None):
        super().__init__()
        if name is not None:
            self.name = name

    def forward(self, img):
        """Return a copy of the image

        Parameters
        ----------
        img : torch.Tensor
            The image to return

        Returns
        -------
        img : torch.Tensor
            a clone of the input image

        """
        return img.clone()
