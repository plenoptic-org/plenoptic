"""Example images to use with plenoptic."""

from importlib import resources

import torch

from ..tensors import load_images

FILES = resources.files("plenoptic.data")

__all__ = [
    "einstein",
    "curie",
    "parrot",
    "reptile_skin",
    "color_wheel",
    "macaque",
]


def __dir__() -> list[str]:
    return __all__


def einstein(as_gray: bool = True) -> torch.Tensor:
    """
    Return an example grayscale image of Albert Einstein.

    Parameters
    ----------
    as_gray :
        Whether to load a single grayscale channel or 3 RGB channels.

    Returns
    -------
    image :
        4d tensor of the image.

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> einstein = po.data.einstein()
      >>> einstein.shape
      torch.Size([1, 1, 256, 256])
      >>> po.plot.imshow(einstein)
      <PyrFigure size ...>

    >>> import plenoptic as po
    >>> einstein_rgb = po.data.einstein(as_gray=False)
    >>> einstein_rgb.shape
    torch.Size([1, 3, 256, 256])
    """  # numpydoc ignore=ES01
    return load_images(FILES / "einstein.pgm", as_gray=as_gray)


def curie(as_gray: bool = True) -> torch.Tensor:
    """
    Return an example grayscale image of Marie Curie.

    Parameters
    ----------
    as_gray :
        Whether to load a single grayscale channel or 3 RGB channels.

    Returns
    -------
    image :
        4d tensor of the image.

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> curie = po.data.curie()
      >>> curie.shape
      torch.Size([1, 1, 256, 256])
      >>> po.plot.imshow(curie)
      <PyrFigure size ...>

    >>> import plenoptic as po
    >>> curie_rgb = po.data.curie(as_gray=False)
    >>> curie_rgb.shape
    torch.Size([1, 3, 256, 256])
    """  # numpydoc ignore=ES01
    return load_images(FILES / "curie.pgm", as_gray=as_gray)


def parrot(as_gray: bool = True) -> torch.Tensor:
    """
    Return an example grayscale image of a parrot.

    Parameters
    ----------
    as_gray :
        Whether to load a single grayscale channel or 3 RGB channels.

    Returns
    -------
    image :
        4d tensor of the image.

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> parrot = po.data.parrot()
      >>> parrot.shape
      torch.Size([1, 1, 254, 266])
      >>> po.plot.imshow(parrot)
      <PyrFigure size ...>

    >>> import plenoptic as po
    >>> parrot_rgb = po.data.parrot(as_gray=False)
    >>> parrot_rgb.shape
    torch.Size([1, 3, 254, 266])
    """  # numpydoc ignore=ES01
    return load_images(FILES / "parrot.png", as_gray=as_gray)


def reptile_skin(as_gray: bool = True) -> torch.Tensor:
    """
    Return an example grayscale image of reptile skin.

    Parameters
    ----------
    as_gray :
        Whether to load a single grayscale channel or 3 RGB channels.

    Returns
    -------
    image :
        4d tensor of the image.

    Examples
    --------
    .. plot::
       :context: reset

      >>> import plenoptic as po
      >>> reptile_skin = po.data.reptile_skin()
      >>> reptile_skin.shape
      torch.Size([1, 1, 256, 256])
      >>> po.plot.imshow(reptile_skin)
      <PyrFigure size ...>

    >>> import plenoptic as po
    >>> reptile_skin_rgb = po.data.reptile_skin(as_gray=False)
    >>> reptile_skin_rgb.shape
    torch.Size([1, 3, 256, 256])
    """  # numpydoc ignore=ES01
    return load_images(FILES / "reptile_skin.pgm", as_gray=as_gray)


def color_wheel(as_gray: bool = False) -> torch.Tensor:
    """
    Return an example image of a color wheel.

    Parameters
    ----------
    as_gray :
        Whether to load a single grayscale channel or 3 RGB channels.

    Returns
    -------
    image :
        4d tensor of the image.

    Examples
    --------
    .. plot::
       :context: reset

      >>> import plenoptic as po
      >>> color_wheel = po.data.color_wheel()
      >>> color_wheel.shape
      torch.Size([1, 3, 600, 600])
      >>> po.plot.imshow(color_wheel, as_rgb=True, zoom=0.5)
      <PyrFigure size ...>

    .. plot::
      :context: close-figs

      >>> import plenoptic as po
      >>> color_wheel_gray = po.data.color_wheel(as_gray=True)
      >>> color_wheel_gray.shape
      torch.Size([1, 1, 600, 600])
      >>> po.plot.imshow(color_wheel_gray, zoom=0.5)
      <PyrFigure size ...>
    """  # numpydoc ignore=ES01
    return load_images(FILES / "color_wheel.jpg", as_gray=as_gray)


def macaque(as_gray: bool = False) -> torch.Tensor:
    """
    Return an example image of a macaque.

    Parameters
    ----------
    as_gray :
        Whether to load a single grayscale channel or 3 RGB channels.

    Returns
    -------
    image :
        4d tensor of the image.

    Notes
    -----
    This is one of the `monkey selfies
    <https://en.wikipedia.org/wiki/Monkey_selfie_copyright_dispute>`_ and is in the
    public domain.

    It was originally downloaded from `wikimedia
    <https://commons.wikimedia.org/wiki/Category:Monkey_selfie>`_

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> macaque = po.data.macaque()
      >>> macaque.shape
      torch.Size([1, 3, 1500, 1085])
      >>> po.plot.imshow(macaque[..., :-1], as_rgb=True, zoom=0.25)
      <PyrFigure size ...>

    To resize this image for use with an ImageNet-trained model, centering
    the monkey's face:

    .. plot::
      :context: close-figs

      >>> macaque = po.process.blur_downsample(macaque, 2)[..., :-60, :]
      >>> macaque = po.process.center_crop(macaque, 224)
      >>> macaque.shape
      torch.Size([1, 3, 224, 224])
      >>> po.plot.imshow(macaque)
      <PyrFigure size ...>
    """  # numpydoc ignore=ES01
    return load_images(FILES / "macaca_nigra_self-portrait.jpg", as_gray=as_gray)
