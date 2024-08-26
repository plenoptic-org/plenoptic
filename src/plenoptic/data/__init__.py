from . import data_utils
from .fetch import fetch_data, DOWNLOADABLE_FILES
import torch
from importlib import resources
from ..tools.data import load_images

__all__ = ['einstein', 'curie', 'parrot', 'reptile_skin',
           'color_wheel', 'fetch_data', 'DOWNLOADABLE_FILES']
def __dir__():
    return __all__

FILES = resources.files(__name__)


def einstein(as_gray: bool = True) -> torch.Tensor:
    return load_images(FILES / 'einstein.pgm', as_gray=as_gray)


def curie(as_gray: bool = True) -> torch.Tensor:
    return load_images(FILES / 'curie.pgm', as_gray=as_gray)


def parrot(as_gray: bool = False) -> torch.Tensor:
    return load_images(FILES / 'parrot.png', as_gray=as_gray)


def reptile_skin(as_gray: bool = True) -> torch.Tensor:
    return load_images(FILES / 'reptile_skin.pgm', as_gray=as_gray)


def color_wheel(as_gray: bool = False) -> torch.Tensor:
    return load_images(FILES / 'color_wheel.jpg', as_gray=as_gray)
