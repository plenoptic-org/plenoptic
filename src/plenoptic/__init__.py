"""
Plenoptic is a python library for model-based synthesis of perceptual stimuli.

For plenoptic, models are those of visual information processing: they accept an image
as input, perform some computations, and return some output, which can be mapped to
neuronal firing rate, fMRI BOLD response, behavior on some task, image category, etc.
The intended audience is researchers in neuroscience, psychology, and machine learning.
The generated stimuli enable interpretation of model properties through examination of
features that are enhanced, suppressed, or discarded. More importantly, they can
facilitate the scientific process, through use in further perceptual or neural
experiments aimed at validating or falsifying model predictions.
"""

# preface with underscore so they're not exposed to __all__
import contextlib as _contextlib
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _get_version

with _contextlib.suppress(_PackageNotFoundError):
    __version__ = _get_version("plenoptic")


import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
