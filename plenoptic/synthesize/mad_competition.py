import torch
import torch.nn as nn


class MADCompetition(nn.Module):
    """Generate maximally-differentiating images for two models

    In MAD Competition, we start with a reference image and generate two pairs of images. We
    proceed as follows:

    - Add Gaussian white noise to the reference image in order to perturb it. This gives us the
      "initial image"
    - do stuff

    And so we end up with two pairs of images, one of which contains the images which produce the
    largest and smallest responses in model 1 while keeping model 2's response as close to constant
    as possible, while the other pair of images does the inverse (differentiates model 2's
    responses as much as possible while keeping model 1's response as close to constant as
    possible).

    Warning
    -------
    There are several limitations to this implementation:

    1. Both two models both have to be functioning `torch.nn.Module`, with a `forward()` method and
       a the ability to propagate gradients backwards through them.

    2. Both models must take an arbitrary grayscale image as input (currently, we do not support
       color images, movies, or batches of images).

    3. Both models must produce a scalar output (the prediction). We do not currently support
       models that produce a vector of predictions (for example, firing rates of a population of
       neurons or BOLD signals of voxels across a brain region)

    Parameters
    ----------
    model_1, model_2 : `torch.nn.Module`
        The two models to compare.
    reference_image : `array_like`
        The 2d grayscale image to generate the maximally-differentiating stimuli from.

    Notes
    -----
    Method described in [1]_.

    References
    -----
    .. [1] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD) competition: A
           methodology for comparing computational models of perceptual discriminability. Journal
           of Vision, 8(12), 1–13. http://dx.doi.org/10.1167/8.12.8

    """

    def __init__(self, model_1, model_2, reference_image):
        super().__init__()

        self.model_1 = model_1
        self.model_2 = model_2
        self.reference_image = torch.tensor(reference_image, requires_grad=True,
                                            dtype=torch.float32)

    def synthesize(self, seed=0, initial_noise=1):
        """Synthesize two pairs of maximally-differentiation images

        Parameters
        ----------
        seed : `int`
            seed to initialize the random (pytorch) random number generator with
        initial_noise : `float`
            standard deviation of the Gaussian noise used to create the initial image from the
            reference image
        """
        torch.manual_seed(seed)

        self.initial_image = (self.reference_image +
                              initial_noise * torch.randn_like(self.reference_image))

        # then there are two loops of optimization
