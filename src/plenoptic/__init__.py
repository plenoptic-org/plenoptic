# ruff: noqa: F401
# ruff: noqa: I001
# Import order matters here to avoid circular dependencies

from . import simulate as simul
from . import synthesize as synth
from . import data, metric, tools
from .tools.data import load_images, to_numpy
from .tools.display import animshow, imshow, pyrshow

# preface with underscore so they're not exposed to __all__
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _get_version
import contextlib as _contextlib


with _contextlib.suppress(_PackageNotFoundError):
    __version__ = _get_version("plenoptic")
