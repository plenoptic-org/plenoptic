import abc


class Clamper(metaclass=abc.ABCMeta):
    """Abstract superclass for all clampers

    All clampers operate on images to clamp some properties of the
    image, ensuring that they are equal to some pre-set properties
    (typically, those same properties computed on another image).

    """

    @abc.abstractmethod
    def clamp(self):
        """Clamp the image
        """
        pass


class RangeClamper(Clamper):
    """Clamps the range between two values

    This Clamper ensures that the range of the image is fixed. This is
    the most common Clamper, since we generally want to make sure that
    our synthesized images always lie between 0 and 1.

    Parameters
    ----------
    range : tuple
        tuple of floats that specify the possible range

    """
    def __init__(self, range=(0, 1)):
        self.range = range

    def clamp(self, im):
        """Clamp ``im`` so its range lies within ``range``

        We use ``torch.clamp``, so that all values below
        ``self.range[0]`` are set to ``self.range[0]`` and all values
        above ``self.range[1]`` are set to ``self.range[1]``.

        Parameters
        ----------
        im : torch.Tensor
            The image to clamp

        """
        im = im.clamp(self.range[0], self.range[1])
        return im
