from . import data_utils
from .fetch import fetch_data, DOWNLOADABLE_FILES
import torch

__all__ = ['einstein', 'curie', 'Parrot', 'reptile_skin',
           'fetch_data', 'DOWNLOADABLE_FILES']
def __dir__():
    return __all__


def einstein() -> torch.Tensor:
    return data_utils.get('einstein')


def curie() -> torch.Tensor:
    return data_utils.get('curie')


def Parrot(as_gray: bool = False) -> torch.Tensor:
    return data_utils.get('Parrot', as_gray=as_gray)


def reptile_skin() -> torch.Tensor:
    return data_utils.get('reptile_skin')
