# ruff: noqa: F401
# ruff: noqa: I001
# Import order matters here to avoid circular dependencies

from . import simulate as simul
from . import synthesize as synth
from . import data, metric, tools
from .tools.data import load_images, to_numpy
from .tools.display import animshow, imshow, pyrshow
from .version import version as __version__

# Now mark the following functions and objects as safe for torch.load, see
# https://github.com/plenoptic-org/plenoptic/issues/313 for discussion

import torch
import collections

# mark the loss functions and metrics from plenoptic as safe
from .tools.optim import _SAFE_FUNCS as _safe_optim
from .metric import _SAFE_FUNCS as _safe_metric

# mark the optimizers and learning rate schedulers from torch as safe
_optims = [
    opt
    for opt in torch.optim.Optimizer.__subclasses__()
    if "torch.optim" in opt.__module__
]
_scheds = [
    sch
    for sch in torch.optim.lr_scheduler.LRScheduler.__subclasses__()
    if "torch.optim.lr_scheduler" in sch.__module__
]

_plenoptic_funcs = [eval(f"tools.optim.{f}") for f in _safe_optim]
_plenoptic_funcs += [eval(f"metric.{f}") for f in _safe_metric]
torch.serialization.add_safe_globals(
    [dict, collections.defaultdict, *_optims, *_scheds, *_plenoptic_funcs]
)

__all__ = [
    "simul",
    "synth",
    "data",
    "metric",
    "tools",
    "load_images",
    "to_numpy",
    "animshow",
    "imshow",
    "pyrshow",
    "__version__",
]


def __dir__() -> list[str]:
    return __all__
