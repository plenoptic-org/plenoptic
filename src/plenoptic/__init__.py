# ruff: noqa: F401
# ruff: noqa: I001
# Import order matters here to avoid circular dependencies

from . import simulate as simul
from . import synthesize as synth
from . import data, metric, tools
from .tools.data import load_images, to_numpy
from .tools.display import animshow, imshow, pyrshow
from .version import version as __version__

import torch
import collections

# mark the optimizers and learning rate schedulers from torch as safe
optims = [
    opt
    for opt in torch.optim.Optimizer.__subclasses__()
    if "torch.optim" in opt.__module__
]
scheds = [
    sch
    for sch in torch.optim.lr_scheduler.LRScheduler.__subclasses__()
    if "torch.optim.lr_scheduler" in sch.__module__
]
# mark the loss functions from plenoptic as safe
plenoptic_funcs = [eval(f"tools.optim.{f}") for f in tools.optim.OPTIM_FUNCS]
torch.serialization.add_safe_globals(
    [dict, collections.defaultdict, *optims, *scheds, *plenoptic_funcs]
)
