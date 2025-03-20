from importlib import resources

import torch

from ..tools.data import load_images
from .fetch import DOWNLOADABLE_FILES, fetch_data

__all__ = [
    "einstein",
    "curie",
    "parrot",
    "reptile_skin",
    "color_wheel",
    "fetch_data",
    "DOWNLOADABLE_FILES",
]


def __dir__():
    return __all__


FILES = resources.files(__name__)


def einstein(as_gray: bool = True) -> torch.Tensor:
    """An example grayscale image of Albert Einstein.

    Parameters
    ----------
    as_gray : bool
        Whether to load a single grayscale channel or 3 RGB channels

    Returns
    -------
    image : torch.Tensor
        4d tensor of the image

    Examples
    --------
    >>> import plenoptic as po
    >>> einstein = po.data.einstein()
    >>> einstein.shape
    torch.Size([1, 1, 256, 256])

    >>> po.imshow(einstein)
    <PyrFigure size 512x640 with 1 Axes>

    >>> einstein_rgb = po.data.einstein(as_gray=False)
    >>> einstein_rgb.shape
    torch.Size([1, 3, 256, 256])

    """
    return load_images(FILES / "einstein.pgm", as_gray=as_gray)


def curie(as_gray: bool = True) -> torch.Tensor:
    """An example grayscale image of Marie Curie.

    Parameters
    ----------
    as_gray : bool
        Whether to load a single grayscale channel or 3 RGB channels

    Returns
    -------
    image : torch.Tensor
        4d tensor of the image

    Examples
    --------
    >>> import plenoptic as po
    >>> curie = po.data.curie()
    >>> curie.shape
    torch.Size([1, 1, 256, 256])

    >>> po.imshow(curie)
    <PyrFigure size 512x640 with 1 Axes>

    >>> curie_rgb = po.data.curie(as_gray=False)
    >>> curie_rgb.shape
    torch.Size([1, 3, 256, 256])

    """
    return load_images(FILES / "curie.pgm", as_gray=as_gray)


def parrot(as_gray: bool = True) -> torch.Tensor:
    """An example grayscale image of a parrot.

    Parameters
    ----------
    as_gray : bool
        Whether to load a single grayscale channel or 3 RGB channels

    Returns
    -------
    image : torch.Tensor
        4d tensor of the image

    Examples
    --------
    >>> import plenoptic as po
    >>> parrot = po.data.parrot()
    >>> parrot.shape
    torch.Size([1, 1, 254, 266])

    >>> po.imshow(parrot)
    <PyrFigure size 532x634 with 1 Axes>

    >>> parrot_rgb = po.data.parrot(as_gray=False)
    >>> parrot_rgb.shape
    torch.Size([1, 3, 254, 266])

    """
    return load_images(FILES / "parrot.png", as_gray=as_gray)


def reptile_skin(as_gray: bool = True) -> torch.Tensor:
    """An example grayscale image of reptile skin.

    Parameters
    ----------
    as_gray : bool
        Whether to load a single grayscale channel or 3 RGB channels

    Returns
    -------
    image : torch.Tensor
        4d tensor of the image

    Examples
    --------
    >>> import plenoptic as po
    >>> reptile_skin = po.data.reptile_skin()
    >>> reptile_skin.shape
    torch.Size([1, 1, 256, 256])

    >>> po.imshow(reptile_skin)
    <PyrFigure size 512x640 with 1 Axes>

    >>> reptile_skin_rgb = po.data.reptile_skin(as_gray=False)
    >>> reptile_skin_rgb.shape
    torch.Size([1, 3, 256, 256])

    """
    return load_images(FILES / "reptile_skin.pgm", as_gray=as_gray)


def color_wheel(as_gray: bool = False) -> torch.Tensor:
    """An example image of a color wheel.

    Parameters
    ----------
    as_gray : bool
        Whether to load a single grayscale channel or 3 RGB channels

    Returns
    -------
    image : torch.Tensor
        4d tensor of the image

    Examples
    --------
    >>> import plenoptic as po
    >>> color_wheel = po.data.color_wheel()
    >>> color_wheel.shape
    torch.Size([1, 3, 600, 600])

    >>> po.imshow(color_wheel, as_rgb=True)
    <PyrFigure size 1200x1500 with 1 Axes>

    >>> color_wheel_gray = po.data.color_wheel(as_gray=True)
    >>> color_wheel_gray.shape
    torch.Size([1, 1, 600, 600])

    >>> po.imshow(color_wheel_gray)
    <PyrFigure size 1200x1500 with 1 Axes>

    """
    return load_images(FILES / "color_wheel.jpg", as_gray=as_gray)
