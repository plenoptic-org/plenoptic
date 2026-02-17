"""
Functions that check for optimization convergence/stabilization.

The functions herein generally differ in what they are checking for
convergence: loss, pixel change, etc.

They should probably be able to accept the following arguments, in this order
(they can accept more):

- ``synth``: the synthesis object to check.

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
    from ..synthesize.mad_competition import MADCompetition
    from ..synthesize.metamer import Metamer, MetamerCTF


__all__ = []


def __dir__() -> list[str]:
    return __all__


def _loss_convergence(
    synth: "Metamer | MetamerCTF | MADCompetition",
    stop_criterion: float,
    stop_iters_to_check: int,
) -> bool:
    r"""
    Check whether the loss has stabilized and, if so, return ``True``.

    We check whether:

    - We have been synthesizing for ``stop_iters_to_check`` iterations,
      i.e. ``len(synth.losses) > stop_iters_to_check``.

    - Loss has decreased by less than ``stop_criterion`` over the past
      ``stop_iters_to_check`` iterations.

    If both conditions are met, we return ``True``. Else, we return ``False``.

    Parameters
    ----------
    synth
        The synthesis object to check.
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
    try:
        # get the private losses if available.
        losses = synth._losses
    except AttributeError:
        losses = synth.losses
    return (
        len(losses) > stop_iters_to_check
        and abs(losses[-stop_iters_to_check] - losses[-1]) < stop_criterion
    )


def _coarse_to_fine_enough(
    synth: "MetamerCTF",
    i: int,
    ctf_iters_to_check: int,
) -> bool:
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
        If self.coarse_to_fine is ``False``, then this is ignored.

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


def _pixel_change_convergence(
    synth: "Metamer | MetamerCTF | MADCompetition",
    stop_criterion: float,
    stop_iters_to_check: int,
) -> bool:
    """
    Check whether the pixel change norm has stabilized and, if so, return ``True``.

    We check whether:

    - We have been synthesizing for ``stop_iters_to_check`` iterations, i.e.
      ``len(synth.pixel_change_norm) > stop_iters_to_check``.

    - The ``pixel_change_norm`` has changed by less than ``stop_criterion`` over the
      past ``stop_iters_to_check`` iterations.

    If both conditions are met, we return ``True``. Else, we return ``False``.

    Parameters
    ----------
    synth
        The synthesis object to check.
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
