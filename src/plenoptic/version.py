"""Plenoptic version information."""

__all__ = ["__version__"]


import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("plenoptic")
