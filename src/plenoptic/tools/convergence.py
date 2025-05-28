"""
Functions that check for optimization convergence/stabilization.

The functions herein generally differ in what they are checking for
convergence: loss, pixel change, etc.

They should probably be able to accept the following arguments, in this order
(they can accept more):

- ``synth``: an OptimizedSynthesis object to check.

- ``stop_criterion``: the value used as criterion / tolerance that our
  convergence target is compared against.

- ``stop_iters_to_check``: how many iterations back to check for convergence.

They must return a single ``bool``: ``True`` if we've reached convergence,
``False`` if not.
"""

# to avoid circular import error:
# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..synthesize.metamer import MetamerCTF
    from ..synthesize.synthesis import OptimizedSynthesis


def loss_convergence(
    synth: OptimizedSynthesis,
    stop_criterion: float,
    stop_iters_to_check: int,
) -> bool:
    r"""
    Check whether the loss has stabilized and, if so, return True.

    We check whether:

    - We have been synthesizing for ``stop_iters_to_check`` iterations,
      i.e. ``len(synth.losses) > stop_iters_to_check``.

    - Loss has decreased by less than ``stop_criterion`` over the past
      ``stop_iters_to_check`` iterations.

    If both conditions are met, we return ``True``. Else, we return ``False``.

    Parameters
    ----------
    synth
        The OptimizedSynthesis object to check.
    stop_criterion
        If the loss over the past ``stop_iters_to_check`` has changed
        less than ``stop_criterion``, we terminate synthesis.
    stop_iters_to_check
        How many iterations back to check in order to see if the
        loss has stopped decreasing (for ``stop_criterion``).

    Returns
    -------
    loss_stabilized :
        Whether the loss has stabilized or not.
    """
    return (
        len(synth.losses) > stop_iters_to_check
        and abs(synth.losses[-stop_iters_to_check] - synth.losses[-1]) < stop_criterion
    )


def coarse_to_fine_enough(synth: MetamerCTF, i: int, ctf_iters_to_check: int) -> bool:
    r"""
    Check whether we've been synthesized all scales for long enough.

    This is meant to be paired with another convergence check, such as
    ``loss_convergence``.

    We check whether:

    - We have finished synthesizing each individual scale, i.e. ``synth.scales[0] ==
      "all"``.

    - We have been synthesizing all scales for more than ``ctf_iters_to_check``
      iterations, i.e. ``i - synth.scales_timing["all"][0]) > ctf_iters_to_check``.

    If both conditions are met, we return ``True``. Else, we return ``False``.

    Parameters
    ----------
    synth
        The MetamerCTF object to check.
    i
        The current iteration (0-indexed).
    ctf_iters_to_check
        Minimum number of iterations coarse-to-fine must run at each scale.
        If self.coarse_to_fine is False, then this is ignored.

    Returns
    -------
    ctf_enough
        Whether we've been doing coarse to fine synthesis for long enough.
    """
    all_scales = synth.scales[0] == "all"
    # synth.scales_timing['all'] will only be a non-empty list if all_scales is
    # True, so we only check it then. This is equivalent to checking if both
    # conditions are true
    if all_scales:
        return (i - synth.scales_timing["all"][0]) > ctf_iters_to_check
    else:
        return False


def pixel_change_convergence(
    synth: OptimizedSynthesis,
    stop_criterion: float,
    stop_iters_to_check: int,
) -> bool:
    """
    Check whether the pixel change norm has stabilized and, if so, return True.

    We check whether:

    - We have been synthesizing for ``stop_iters_to_check`` iterations, i.e.
      ``len(synth.pixel_change_norm) > stop_iters_to_check``.

    - The ``pixel_change_norm`` has changed by less than ``stop_criterion`` over the
      past ``stop_iters_to_check`` iterations.

    If both conditions are met, we return ``True``. Else, we return ``False``.

    Parameters
    ----------
    synth
        The OptimizedSynthesis object to check.
    stop_criterion
        If the pixel change norm has been less than ``stop_criterion`` for all
        of the past ``stop_iters_to_check``, we terminate synthesis.
    stop_iters_to_check
        How many iterations back to check in order to see if the
        pixel change norm has stopped decreasing (for ``stop_criterion``).

    Returns
    -------
    loss_stabilized :
        Whether the pixel change norm has stabilized or not.
    """
    return (
        len(synth.pixel_change_norm) > stop_iters_to_check
        and (synth.pixel_change_norm[-stop_iters_to_check:] < stop_criterion).all()
    )
