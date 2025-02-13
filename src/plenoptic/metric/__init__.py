# ruff: noqa: F401

from .classes import NLP

# the following will be used in plenoptic/__init__.py to mark them as safe for
# torch.load, see https://github.com/plenoptic-org/plenoptic/issues/313 for discussion
from .model_metric import _SAFE_FUNCS as _mm_safe_funcs
from .model_metric import model_metric
from .naive import _SAFE_FUNCS as _naive_safe_funcs
from .naive import mse
from .perceptual_distance import _SAFE_FUNCS as _perc_safe_funcs
from .perceptual_distance import ms_ssim, nlpd, ssim, ssim_map

_SAFE_FUNCS = [*_mm_safe_funcs, *_naive_safe_funcs, *_perc_safe_funcs]
__all__ = ["NLP", "model_metric", "mse", "ms_ssim", "nlpd", "ssim", "ssim_map"]


def __dir__() -> list[str]:
    return __all__
