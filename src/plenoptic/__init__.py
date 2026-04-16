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

import lazy_loader as lazy

__default_getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)


def __getattr__(attr):  # noqa: ANN202, ANN001
    if attr.startswith("synth"):
        raise AttributeError(
            f"`plenoptic.{attr.split('.')[0]}` not available from "
            "plenoptic 2.0 onwards. All synthesis objects now live in "
            "the main namespace (e.g., `plenoptic.Metamer`). See "
            "Migration Guide in documentation for details."
        )
    elif attr.startswith("simul"):
        raise AttributeError(
            f"`plenoptic.{attr.split('.')[0]}` not available from "
            "plenoptic 2.0 onwards. All model objects now live in "
            "the `plenoptic.models` namespace, and all "
            "canonical_computations live in the `plenoptic.model_components`"
            " namespace. See Migration Guide in documentation for details."
        )
    elif attr.startswith("tools"):
        raise AttributeError(
            f"`plenoptic.{attr.split('.')[0]}` not available from "
            "plenoptic 2.0 onwards. The corresponding functions now live in other "
            "relevant modules. See Migration Guide in documentation for details."
        )
    elif attr in ["imshow", "animshow", "pyrshow"]:
        raise AttributeError(
            f"`plenoptic.{attr}` was moved in "
            "plenoptic 2.0. It can be found in the `plenoptic.plot` namespace"
            f" (i.e., `plenoptic.plot.{attr}`). See Migration Guide in "
            "documentation for details."
        )
    else:
        return __default_getattr__(attr)
