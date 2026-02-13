"""Example images to use with plenoptic."""  # numpydoc ignore=ES01

from .fetch import DOWNLOADABLE_FILES, fetch_data
from .images import color_wheel, curie, einstein, parrot, reptile_skin

__all__ = [
    "einstein",
    "curie",
    "parrot",
    "reptile_skin",
    "color_wheel",
    "DOWNLOADABLE_FILES",
    "fetch_data",
]


def __dir__() -> list[str]:
    return __all__
