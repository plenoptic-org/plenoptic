"""Plots for understanding synthesis objects."""  # numpydoc ignore=EX01

import re
import warnings
from collections.abc import Callable
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from .. import tensors
from .._synthesize import Eigendistortion, MADCompetition, Metamer
from . import display
from .metamer import _representation_error, metamer_representation_error

__all__ = [
    "synthesis_loss",
    "synthesis_imshow",
    "synthesis_animshow",
    "synthesis_histogram",
    "synthesis_status",
]


def __dir__() -> list[str]:
    return __all__


def synthesis_loss(
    synthesis_object: Metamer | MADCompetition,
    iteration: int | None = None,
    plot_penalties: bool = False,
    axes: list[mpl.axes.Axes] | mpl.axes.Axes | None = None,
    **kwargs: Any,
) -> dict[str, mpl.axes.Axes]:
    """
    Plot synthesis loss.

    .. versionadded:: 2.0
       Combines previously separate loss plotting functions for
       :class:`~plenoptic.Metamer` and :class:`~plenoptic.MADCompetition`, and adds
       support for plotting penalties. Note that behavior for
       :class:`~plenoptic.Metamer` is different: we now plot the metamer loss, not the
       objective function value (see below for details).

    The behavior of this function is slightly different depending on the type of
    ``synthesis_object``:

    - :class:`~plenoptic.Metamer`: creates a single axis object whose y-axis is
      log-scaled and shows the metamer loss and, if ``plot_penalties=True``,
      :attr:`~plenoptic.Metamer.penalties`. Returned dictionary has key ``"loss"``.

    - :class:`~plenoptic.MADCompetition`: creates multiple axes objects, one each for
      :attr:`~plenoptic.MADCompetition.reference_metric_loss`,
      :attr:`~plenoptic.MADCompetition.optimized_metric_loss`, and (if
      ``plot_penalties=True``) :attr:`~plenoptic.MADCompetition.penalties`. The y-axis
      is linearly-scaled for all plots. Returned dictionary has keys
      ``"reference_metric_loss"``, ``"optimized_metric_loss"``, and ``"penalties"``.

    In all cases, plots a red dot at ``iteration``, to highlight the loss there. If
    ``iteration=None``, then the dot will be at the final iteration.

    .. attention::
       In all cases, we plot the components of the objective function, not the
       objective function itself (whose values are stored in the
       :attr:`plenoptic.Metamer.losses` or :attr:`plenoptic.MADCompetition.losses`
       attribute). See Examples section and :func:`plenoptic.Metamer.objective_function`
       or :func:`plenoptic.MADCompetition.objective_function` for more details.

    Parameters
    ----------
    synthesis_object
        Synthesis object whose loss we want to plot.
    iteration
        Which iteration to display. If ``None``,  we show the most recent one.
        Negative values are also allowed.
    plot_penalties
        Whether to plot the output of the penalty function as well. See above
        for behavior.
    axes
        Pre-existing axes for plot. If ``None``, we call
        :func:`matplotlib.pyplot.gca()`. If ``synthesis_object`` is
        :class:`~plenoptic.MADCompetition`, then if ``ax`` is a single axis, we split it
        horizontally; if ``ax`` is a list, it must contain two (or three, if
        ``plot_penalties=True``) axes to plot on. If ``synthesis_object`` is
        :class:`~plenoptic.Metamer`, then passing a list will result in a
        ``ValueError``.
    **kwargs
        Passed to :func:`matplotlib.pyplot.plot`.

    Returns
    -------
    axes
        A dictionary whose keys are strings describing the created plots and whose
        values are the corresponding matplotlib axes. See above for details.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.
    ValueError
        If ``ax`` is a list and ``synthesis_object`` is a :class:`~plenoptic.Metamer`.
    ValueError
        If ``synthesis_object`` is a :class:`~plenoptic.MADCompetition` and ``ax`` is a
        list of the wrong length.
    TypeError
        If ``synthesis_object`` is not :class:`~plenoptic.MADCompetition` or
        :class:`~plenoptic.Metamer`

    See Also
    --------
    synthesis_status
        Create a figure combining this with other axis-level plots to summarize
        synthesis status at a given iteration.
    synthesis_animshow
        Create a video animating this and other axis-level plots changing over
        the course of synthesis.

    Examples
    --------
    Plot loss for :class:`~plenoptic.Metamer` object:

    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import matplotlib.pyplot as plt
      >>> import torch
      >>> img = po.data.einstein()
      >>> model = po.models.Gaussian(30).eval()
      >>> po.remove_grad(model)
      >>> met = po.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> po.plot.synthesis_loss(met)
      {'loss': <Axes: ... ylabel='Loss'>}

    Include the penalties:

    .. plot::
      :context: close-figs

      >>> po.plot.synthesis_loss(met, plot_penalties=True)
      {'loss': <Axes: ... ylabel='Loss'>}

    Specify an iteration:

    .. plot::
      :context: close-figs

      >>> po.plot.synthesis_loss(met, iteration=10, plot_penalties=True)
      {'loss': <Axes: ... ylabel='Loss'>}

    Plot on an axis in an existing figure:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 2)
      >>> po.plot.synthesis_loss(met, axes=axes[1], plot_penalties=True)
      {'loss': <Axes: ... ylabel='Loss'>}

    Note that we are not plotting the output of
    :func:`plenoptic.Metamer.objective_function`, which is stored in
    :attr:`plenoptic.Metamer.losses`. Instead, we are plotting the output of
    :attr:`plenoptic.Metamer.loss_function`, which is the "metamer loss" (which does not
    include the penalty). The following example illustrates the difference:

    .. plot::
      :context: close-figs

      >>> axes = po.plot.synthesis_loss(met)
      >>> axes["loss"].plot(met.losses, label="objective function")
      [<matplotlib.lines.Line2D ...>]
      >>> # Some tweaks to the marker and size to aid visibility.
      >>> axes["loss"].plot(
      ...     met.losses - met.penalty_lambda * met.penalties,
      ...     "k.",
      ...     ms=2,
      ...     label="reconstructed metamer loss",
      ... )
      [<matplotlib.lines.Line2D ...>]
      >>> axes["loss"].legend()
      <matplotlib.legend.Legend ...>

    Notice how the objective function line is above the one created by the this
    function, and how we compute the metamer loss alone.

    Plot loss for :class:`~plenoptic.MADCompetition` object:

    .. plot::
      :context: close-figs

      >>> img = po.data.curie().to(torch.float64)
      >>> def ds_ssim(x, y):
      ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
      >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
      >>> mad.load(po.data.fetch_data("example_mad.pt"))
      >>> po.plot.synthesis_loss(mad)
      {'reference_metric_loss': <Axes: ...>, 'optimized_metric_loss': <Axes: ...>}

    When plotting :class:`~plenoptic.MADCompetition` loss on an existing figure, you can
    either pass a single axis, in which case we sub-divide it into the necessary number
    of axes, or a list with the appropriate number of axes:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 2)
      >>> po.plot.synthesis_loss(mad, axes=axes[1])
      {'reference_metric_loss': <Axes: ...>, 'optimized_metric_loss': <Axes: ...>}
      >>> fig, axes = plt.subplots(1, 2)
      >>> po.plot.synthesis_loss(mad, axes=axes)
      {'reference_metric_loss': <Axes: ...>, 'optimized_metric_loss': <Axes: ...>}

    Note that, as with :class:`~plenoptic.Metamer`, we are not plotting the output of
    :func:`plenoptic.MADCompetition.objective_function`, which is stored in
    :attr:`plenoptic.MADCompetition.losses`. Instead, we are plotting the output of the
    two metrics we are comparing. If you wish to plot the objective function output, you
    can do so directly:

    .. plot::
      :context: close-figs

      >>> plt.plot(mad.losses)
      [<matplotlib.lines.Line2D ...>]
    """
    if not isinstance(synthesis_object, (Metamer, MADCompetition)):
        raise TypeError(
            "synthesis_object must be a MADCompetition or Metamer object but got"
            f" {type(synthesis_object)}"
        )
    # this warning is not relevant for this plotting function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="loss iteration and iteration for")
        progress = synthesis_object.get_progress(iteration)

    if axes is None:
        axes = plt.gca()

    if isinstance(synthesis_object, Metamer):
        if hasattr(axes, "__iter__"):
            raise ValueError("if synthesis_object is a Metamer, axes cannot be a list!")
        met_loss = (
            synthesis_object.losses
            - synthesis_object.penalty_lambda * synthesis_object.penalties
        )
        axes.plot(met_loss, label="metamer loss", **kwargs)
        axes.scatter(
            progress["iteration"],
            progress["losses"]
            - synthesis_object.penalty_lambda * progress["penalties"],
            c="r",
        )
        axes.set(xlabel="Synthesis iteration", ylabel="Loss", yscale="log")
        if plot_penalties:
            axes.plot(synthesis_object.penalties, label="penalty", **kwargs)
            axes.legend()
        axes_dict = {"loss": axes}
    elif isinstance(synthesis_object, MADCompetition):
        right_length = 3 if plot_penalties else 2
        if not hasattr(axes, "__iter__"):
            axes = display._clean_up_axes(
                axes, False, ["top", "right", "bottom", "left"], ["x", "y"]
            )
            gs = axes.get_subplotspec().subgridspec(1, right_length)
            fig = axes.figure
            axes = [fig.add_subplot(gs[0, i]) for i in range(right_length)]
        else:
            if len(axes) != right_length:
                raise ValueError(
                    f"axes is a list of the wrong length! Must contain {right_length}"
                    " axes."
                )
        losses = [
            synthesis_object.reference_metric_loss,
            synthesis_object.optimized_metric_loss,
        ]
        names = ["reference_metric_loss", "optimized_metric_loss"]
        axes_dict = {}
        if plot_penalties:
            losses.append(synthesis_object.penalties)
            names.append("penalties")
        for ax, loss, name in zip(axes, losses, names):
            ax.plot(loss, **kwargs)
            ax.scatter(progress["iteration"], progress[name], c="r")
            ax.set(
                xlabel="Synthesis iteration", ylabel=name.capitalize().replace("_", " ")
            )
            axes_dict[name] = ax
    return axes_dict


def _get_synthesis_image(
    synthesis_object: Metamer | MADCompetition | Eigendistortion,
    batch_idx: int | None = None,
    iteration: int | None = None,
    return_ref_image: bool = False,
) -> tuple[list[torch.Tensor], list[int]]:
    """
    Grab images from synthesis objects to plot.

    This function:

    - Grabs the synthesized image tensor.

    - Grabs the correct iteration, if possible, raising an error if ``iteration`` is set
      for an Eigendistortion or when ``store_progress=False``.

    - If ``batch_idx is None``, unpack all batches into list of 4d tensors. If not, and
      ``synthesis_object`` is an Eigendistortion, convert from eigenindex values to
      actual indices (see :func:`plenoptic.Eigendistortion._indexer`).

    - If ``return_ref_image``, return the reference image as well.

    Parameters
    ----------
    synthesis_object
        Synthesis object with the images we want to plot.
    batch_idx
        Which index to take from the batch dimension. Note that for
        :class:`~plenoptic.Eigendistortion`, this is the
        :attr:`~plenoptic.Eigendistortion.eigenindex`. If ``None``, we grab all
        batches.
    iteration
        Which iteration to display, for :class:`~plenoptic.Metamer` and
        :class:`~plenoptic.MADCompetition` objects. If ``None``, we show the most recent
        one. Negative values are also allowed. If ``iteration!=None`` and
        ``synthesis_object.store_progress>1`` (that is, the synthesized image was not
        cached on every iteration), then we use the cached image from the nearest
        iteration. For :class:`~plenoptic.Eigendistortion`, this must be ``None``.
    return_ref_image
        Whether to include the reference image (``synthesis_object.image``).

    Returns
    -------
    images
        The corresponding images. Either a single 4d image tensor or a list of such
        tensors.
    batch_idx
        Corresponding ``batch_idx``. If input ``batch_idx`` was ``None``, these are the
        explicit indices. If ``synthesis_object`` was a
        :class:`~plenoptic.Eigendistortion`, these have been remapped so they're now
        indices.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.
    ValueError
        If ``iteration`` is not ``None`` and ``synthesis_object`` is an
        :class:`~plenoptic.Eigendistortion` object.

    Warns
    -----
    UserWarning
        If the iteration used for cached image is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``synthesis_object.store_progress=2``).
    """
    if isinstance(synthesis_object, Eigendistortion):
        if iteration is not None:
            raise ValueError(
                "When synthesis_object is an Eigendistortion, iteration must be None!"
            )
        image = synthesis_object.eigendistortions
        if batch_idx is not None:
            try:
                batch_idx = [synthesis_object._indexer(i) for i in batch_idx]
            except TypeError:
                # we're here because we can't iterate over batch_idx (can't just check
                # attributes because 0d tensors have both __iter__ and __len__)
                batch_idx = synthesis_object._indexer(batch_idx)
    else:
        progress = synthesis_object.get_progress(iteration)
        if isinstance(synthesis_object, Metamer):
            name = "metamer"
        elif isinstance(synthesis_object, MADCompetition):
            name = "mad_image"
        try:
            image = progress[f"saved_{name}"]
        except KeyError:
            if iteration is not None:
                raise IndexError(
                    "When synthesis_object.store_progress=False, iteration must be"
                    " None!"
                )
            image = eval(f"synthesis_object.{name}")
    if batch_idx is None:
        image = [im.unsqueeze(0) for im in image]
    if return_ref_image:
        if isinstance(image, list):
            image.append(synthesis_object.image)
        else:
            image = [image, synthesis_object.image]
    return image, batch_idx


def _get_synthesis_title(
    synthesis_object: Metamer | MADCompetition | Eigendistortion,
    batch_idx: int | None = None,
    iteration: int | None = None,
    return_ref_image: bool = False,
) -> list[str]:
    """
    Grab titles for synthesis images to plot.

    This should be run before :func:`_get_synthesis_image`, as its input ``batch_idx``
    should be the unremapped one -- we want it to match the user input.

    Parameters
    ----------
    synthesis_object
        Synthesis object with the images we want to plot.
    batch_idx
        Which index to take from the batch dimension. Note that for
        :class:`~plenoptic.Eigendistortion`, this is the
        :attr:`~plenoptic.Eigendistortion.eigenindex`. If ``None``, we grab all
        batches.
    iteration
        Which iteration to display, for :class:`~plenoptic.Metamer` and
        :class:`~plenoptic.MADCompetition` objects. If ``None``, we show the most recent
        one. Negative values are also allowed. If ``iteration!=None`` and
        ``synthesis_object.store_progress>1`` (that is, the synthesized image was not
        cached on every iteration), then we use the cached image from the nearest
        iteration. For :class:`~plenoptic.Eigendistortion`, this must be ``None``.
    return_ref_image
        Whether to include the reference image (``synthesis_object.image``).

    Returns
    -------
    titles
        Corresponding titles. These include the ``batch_idx`` and, if relevant
        ``iteration`` in them.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.
    ValueError
        If ``iteration`` is not ``None`` and ``synthesis_object`` is an
        :class:`~plenoptic.Eigendistortion` object.

    Warns
    -----
    UserWarning
        If the iteration used for cached image is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``synthesis_object.store_progress=2``).
    """
    if isinstance(synthesis_object, Eigendistortion):
        if iteration is not None:
            raise ValueError(
                "When synthesis_object is an Eigendistortion, iteration must be None!"
            )
        title_names = ["Eigendistortion[{batch_idx}]", "Reference"]
        if batch_idx is None:
            batch_idx = synthesis_object.eigenindex
    else:
        progress = synthesis_object.get_progress(iteration)
        if isinstance(synthesis_object, Metamer):
            title_names = ["Metamer[{batch_idx}] [iteration={iter}]", "Target"]
            max_batch = synthesis_object.metamer.shape[0]
        elif isinstance(synthesis_object, MADCompetition):
            title_names = ["MAD[{batch_idx}] [iteration={iter}]", "Reference"]
            max_batch = synthesis_object.mad_image.shape[0]
        try:
            iteration = progress["store_progress_iteration"]
        except KeyError:
            if iteration is not None:
                raise IndexError(
                    "When synthesis_object.store_progress=False, iteration must be"
                    " None!"
                )
            # losses will always have one extra value, the current loss.
            iteration = len(synthesis_object.losses) - 1
        if batch_idx is None:
            batch_idx = range(max_batch)
    try:
        titles = [title_names[0].format(batch_idx=i, iter=iteration) for i in batch_idx]
    except TypeError:
        # we're here because we can't iterate over batch_idx (can't just check
        # attributes because 0d tensors have both __iter__ and __len__)
        titles = [title_names[0].format(batch_idx=batch_idx, iter=iteration)]
    if return_ref_image:
        titles += [f"{title_names[1]} image"]
    return titles


def synthesis_histogram(
    synthesis_object: Metamer | MADCompetition | Eigendistortion,
    batch_idx: int | None = None,
    channel_idx: int | None = None,
    iteration: int | None = None,
    ylim: tuple[float, float] | Literal[False] = False,
    xlim: tuple[float, float] | Literal[False, "range"] = "range",
    ax: mpl.axes.Axes | None = None,
    alpha: float = 0.4,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Plot histogram of values of synthesis objects.

    .. versionadded:: 2.0
       Combines previously separate loss plotting functions for
       :class:`~plenoptic.Metamer` and :class:`~plenoptic.MADCompetition`, and adds
       support for :class:`~plenoptic.Eigendistortion`.

    As a way to check whether there's any values outside the preferred range.

    The behavior of this function is slightly different depending on the type of
    ``synthesis_object``:

    - :class:`~plenoptic.Metamer` and :class:`~plenoptic.MADCompetition`: compare the
      synthesized tensor against the target / reference image. ``iteration`` can be
      specified.

    - :class:`~plenoptic.Eigendistortion`: create histograms for eigendistortions.
      ``iteration`` must be ``None``.

    Parameters
    ----------
    synthesis_object
        Synthesis object with the images whose values we want to plot.
    batch_idx
        Which index to take from the batch dimension. Note that for
        :class:`~plenoptic.Eigendistortion`, this is the
        :attr:`~plenoptic.Eigendistortion.eigenindex`. If ``None``, we plot all
        batches as separate histograms (intended use-case is for multiple
        eigendistortions).
    channel_idx
        Which index to take from the channel dimension. If ``None``, we use all
        channels (assumed use-case is RGB(A) images).
    iteration
        Which iteration to display, for :class:`~plenoptic.Metamer` and
        :class:`~plenoptic.MADCompetition` objects. If ``None``, we show the most recent
        one. Negative values are also allowed. If ``iteration!=None`` and
        ``synthesis_object.store_progress>1`` (that is, the synthesized image was not
        cached on every iteration), then we use the cached image from the nearest
        iteration. For :class:`~plenoptic.Eigendistortion`, this must be ``None``.
    ylim
        If tuple, the ylimit to set for this axis. If ``False``, we leave
        it untouched.
    xlim
        If ``"range"``, set the xlimits to the range across plotted data.
        If tuple, the xlimit to set for this axis. If ``False``, we leave
        it untouched.
    ax
        Pre-existing axes for plot. If ``None``, we call
        :func:`matplotlib.pyplot.gca()`.
    alpha
        Alpha value for the histogram bars.
    **kwargs
        Passed to :func:`matplotlib.pyplot.hist`.

    Returns
    -------
    ax :
        Creates axes.

    Raises
    ------
    IndexError
        If ``iteration`` takes an illegal value.
    ValueError
        If ``iteration`` is not ``None`` and ``synthesis_object`` is an
        :class:`~plenoptic.Eigendistortion` object.

    Warns
    -----
    UserWarning
        If the iteration used for cached image is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``synthesis_object.store_progress=2``).

    See Also
    --------
    :func:`~plenoptic.plot.histogram`
        The plotting function used to created this plot.
    synthesis_status
        Create a figure combining this with other axis-level plots to summarize
        synthesis status at a given iteration.
    synthesis_animshow
        Create a video animating this and other axis-level plots changing over
        the course of synthesis.

    Examples
    --------
    Plot histogram for :class:`~plenoptic.Metamer` object:

    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import torch
      >>> img = po.data.einstein()
      >>> model = po.models.Gaussian(30).eval()
      >>> po.remove_grad(model)
      >>> met = po.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> po.plot.synthesis_histogram(met)
      <Axes: ... 'Histogram of tensor values'...>

    Plot pixel values from a specified iteration (requires setting ``store_progress``
    when :meth:`~plenoptic.Metamer.synthesize` was called):

    .. plot::
      :context: close-figs

      >>> po.plot.synthesis_histogram(met, iteration=10)
      <Axes: ... 'Histogram of tensor values'...>

    Plot on an existing axis:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 2, figsize=(8, 4))
      >>> po.plot.synthesis_histogram(met, ax=axes[1])
      <Axes: ... 'Histogram of tensor values'...>

    Plot histogram for :class:`~plenoptic.MADCompetition` object:

    .. plot::
      :context: close-figs

      >>> img = po.data.curie().to(torch.float64)
      >>> def ds_ssim(x, y):
      ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
      >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
      >>> mad.load(po.data.fetch_data("example_mad.pt"))
      >>> po.plot.synthesis_histogram(mad)
      <Axes: ... 'Histogram of tensor values'...>

    Plot histogram for :class:`~plenoptic.Eigendistortion` object. Notice how
    here we plot just the values from the synthesized eigendistortions, not the base
    image.

    .. plot::
      :context: close-figs

      >>> img = po.data.einstein().to(torch.float64)
      >>> lg = po.models.LuminanceGainControl(
      ...     (31, 31), pad_mode="circular", pretrained=True, cache_filt=True
      ... ).eval()
      >>> lg = lg.to(torch.float64)
      >>> po.remove_grad(lg)
      >>> eig = po.Eigendistortion(img, lg)
      >>> eig.load(
      ...     po.data.fetch_data("example_eigendistortion.pt"),
      ...     map_location="cpu",
      ... )
      >>> po.plot.synthesis_histogram(eig)
      <Axes: ... 'Histogram of tensor values'...>
    """
    # For eigendistortion, we don't plot histogram against the reference image
    return_ref_image = not isinstance(synthesis_object, Eigendistortion)
    titles = _get_synthesis_title(
        synthesis_object, batch_idx, iteration, return_ref_image
    )
    images, batch_idx = _get_synthesis_image(
        synthesis_object, batch_idx, iteration, return_ref_image
    )
    return display.histogram(
        images,
        titles,
        batch_idx,
        channel_idx,
        ylim,
        xlim,
        ax=ax,
        alpha=alpha,
        **kwargs,
    )


def synthesis_imshow(
    synthesis_object: Metamer | MADCompetition | Eigendistortion,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    distortion_scale: float = 5.0,
    process_image: Callable[[torch.Tensor], torch.Tensor] | None = None,
    zoom: float | None = None,
    iteration: int | None = None,
    ax: mpl.axes.Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Display image of synthesis object.

    .. versionadded:: 2.0
       Combines previously separate loss plotting functions for
       :class:`~plenoptic.Metamer`, :class:`~plenoptic.MADCompetition`, and
       :class:`~plenoptic.Eigendistortion`.

    We use :func:`~plenoptic.plot.imshow` to display the synthesized image and
    attempt to automatically find the most reasonable zoom value. You can override this
    value using the zoom arg, but remember that :func:`~plenoptic.plot.imshow`
    is opinionated about the size of the resulting image and will throw an Exception if
    the axis created is not big enough for the selected zoom.

    The behavior of this function is slightly different depending on the type of
    ``synthesis_object``:

    - :class:`~plenoptic.Metamer` and :class:`~plenoptic.MADCompetition`: process and
      display the synthesized image. ``iteration`` can be specified,
      ``distortion_scale`` must be unchanged.

    - :class:`~plenoptic.Eigendistortion`: process and display
      ``image + (distortion_scale * eigendistortion)``. ``iteration`` must be ``None``,
      ``distortion_scale`` can be set.

    Parameters
    ----------
    synthesis_object
        Synthesis object with the images we wish to display.
    batch_idx
        Which index to take from the batch dimension. Note that for
        :class:`~plenoptic.Eigendistortion`, this is the
        :attr:`~plenoptic.Eigendistortion.eigenindex`.
    channel_idx
        Which index to take from the channel dimension. If ``None``, plot all channels;
        if image has more than 1 channel, will attempt to plot as RGB(A) image.
    distortion_scale
        Amount by which to scale eigendistortion for
        ``image + (distortion_scale * eigendistortion)`` for display. If
        ``synthesis_object`` is not :class:`~plenoptic.Eigendistortion`, must not be
        set.
    process_image
        A function to process the plotted image. E.g., multiplying by the stdev ImageNet
        then adding the mean of ImageNet to undo image preprocessing or clamping between
        0 and 1. If ``None``, then no processing is performed.
    zoom
        How much to zoom in / enlarge the synthesized image, the ratio of display pixels
        to image pixels. If ``None``, we attempt to find the best value ourselves.
    iteration
        Which iteration to display, for :class:`~plenoptic.Metamer` and
        :class:`~plenoptic.MADCompetition` objects. If ``None``, we show the most recent
        one. Negative values are also allowed. If ``iteration!=None`` and
        ``synthesis_object.store_progress>1`` (that is, the synthesized image was not
        cached on every iteration), then we use the cached image from the nearest
        iteration.
    ax
        Pre-existing axes for plot. If ``None``, we call :func:`matplotlib.pyplot.gca`.
    title
        Title to add to axis. If ``None``, we pick appropriate title based on the type
        of ``synthesis_object``.
    **kwargs
        Passed to :func:`~plenoptic.plot.imshow`.

    Returns
    -------
    ax :
        The matplotlib axes containing the plot.

    Raises
    ------
    ValueError
        If ``batch_idx`` is not an int.
    IndexError
        If ``iteration`` takes an illegal value.

    Warns
    -----
    UserWarning
        If the iteration used for cached image is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``synthesis_object.store_progress=2``).

    See Also
    --------
    :func:`~plenoptic.plot.imshow`
        Function used by this one to visualize the metamer image.
    synthesis_status
        Create a figure combining this with other axis-level plots to summarize
        synthesis status at a given iteration.
    synthesis_animshow
        Create a video animating this and other axis-level plots changing over
        the course of synthesis.

    Examples
    --------
    Plot for :class:`~plenoptic.Metamer` object:

    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import matplotlib.pyplot as plt
      >>> import torch
      >>> img = po.data.einstein()
      >>> model = po.models.Gaussian(30).eval()
      >>> po.remove_grad(model)
      >>> met = po.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> po.plot.synthesis_imshow(met)
      <Axes: title=...Metamer[0] [iteration=107]...>

    If a matplotlib figure exists, this function will use it (using
    :func:`matplotlib.pyplot.gca`):

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 2)
      >>> po.plot.synthesis_imshow(met)
      <Axes: title=...Metamer[0] [iteration=107]...>

    Display metamer from a specified iteration (requires setting ``store_progress``
    when :meth:`~plenoptic.Metamer.synthesize` was called):

    .. plot::
      :context: close-figs

      >>> po.plot.synthesis_imshow(met, iteration=10)
      <Axes: title=...Metamer[0] [iteration=10]...>

    Explicitly define the axis to use:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 2, figsize=(8, 4))
      >>> po.plot.synthesis_imshow(met, ax=axes[1])
      <Axes: title=...Metamer[0] [iteration=107]...>

    When plotting on an existing axis, if ``zoom=None``, this function will determine
    the best zoom level for the axis size.

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 1, figsize=(8, 8))
      >>> po.plot.synthesis_imshow(met, ax=axes)
      <Axes: title=...Metamer[0] [iteration=107]...dims: [256, 256] * 2.0'}>

    Plot for :class:`~plenoptic.MADCompetition` object:

    .. plot::
      :context: close-figs

      >>> img = po.data.curie().to(torch.float64)
      >>> def ds_ssim(x, y):
      ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
      >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
      >>> mad.load(po.data.fetch_data("example_mad.pt"))
      >>> po.plot.synthesis_imshow(mad)
      <Axes: title=...MAD[0] [iteration=400]...>

    Plot for :class:`~plenoptic.Eigendistortion` object. Note here that we plot
    the distortion multiplied by ``distortion_scale`` and added to the target image.

    .. plot::
      :context: close-figs

      >>> img = po.data.einstein().to(torch.float64)
      >>> lg = po.models.LuminanceGainControl(
      ...     (31, 31), pad_mode="circular", pretrained=True, cache_filt=True
      ... ).eval()
      >>> lg = lg.to(torch.float64)
      >>> po.remove_grad(lg)
      >>> eig = po.Eigendistortion(img, lg)
      >>> eig.load(
      ...     po.data.fetch_data("example_eigendistortion.pt"),
      ...     map_location="cpu",
      ... )
      >>> po.plot.synthesis_imshow(eig)
      <Axes: title=...5.0 * Eigendistortion[0]...range: [-1.4e-01, 1.0e+00]...>

    Use the ``process_image`` argument to apply a preprocessing function to the
    image before plotting it:

    .. plot::
      :context: close-figs

      >>> po.plot.synthesis_imshow(eig, process_image=lambda x: x.clip(0, 1))
      <Axes: title=...5.0 * Eigendistortion[0]...range: [0.0e+00, 1.0e+00]...>
    """
    try:
        batch_idx = int(batch_idx)
    except (TypeError, ValueError):
        raise ValueError("batch_idx must be a single integer!")

    if title is None:
        title = _get_synthesis_title(synthesis_object, batch_idx, iteration)
        if isinstance(synthesis_object, Eigendistortion):
            title = [f"{distortion_scale} * {t}" for t in title]
    image, batch_idx = _get_synthesis_image(synthesis_object, batch_idx, iteration)
    if isinstance(synthesis_object, Eigendistortion):
        image = synthesis_object.image + distortion_scale * image
    else:
        # if distortion_scale is not default value
        if distortion_scale != 5:
            raise ValueError(
                f"If synthesis_object is type {type(synthesis_object)}, "
                "distortion_scale cannot be set"
            )

    if process_image is not None:
        image = process_image(image)

    # we're only plotting one image here, so if the user wants multiple
    # channels, they must be RGB
    as_rgb = bool(channel_idx is None and image.shape[1] > 1)
    if ax is None:
        ax = plt.gca()
    display.imshow(
        image,
        ax=ax,
        title=title,
        zoom=zoom,
        batch_idx=batch_idx,
        channel_idx=channel_idx,
        as_rgb=as_rgb,
        **kwargs,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax


def _check_plot_consistency(
    included_plots: list[str],
    width_ratios: dict[str, float],
    axes_idx: dict[str, int],
):
    """
    Raise ValueError if width_ratios or axes_idx reference plots not in included_plots.

    Because I'm unsure how to behave then.
    """  # noqa: DOC501
    # numpydoc ignore=PR01
    if extra_plots := set(width_ratios) - set(included_plots):
        raise ValueError(
            "width_ratios contains keys referencing plots not included in "
            f"included_plots! {extra_plots}"
        )
    extra_plots = set(axes_idx) - set(included_plots)
    if extra_plots and extra_plots != {"misc"}:
        raise ValueError(
            "axes_idx contains keys referencing plots not included in included_plots!"
            f" {extra_plots}"
        )


def _check_included_plots(
    to_check: list[str] | dict[str, float],
    to_check_name: str,
    synthesis_object: Metamer | MADCompetition | Eigendistortion,
):
    """
    Check whether the user wanted us to create plots that we can't.

    Helper function for :func:`synthesis_status` and :func:`synthesis_animshow`.

    Raises a ``ValueError`` if ``to_check`` contains any values that are not allowed.

    Parameters
    ----------
    to_check
        The variable to check. We ensure that it doesn't contain any extra (not
        allowed) values. If a list, we check its contents. If a dict, we check
        its keys.
    to_check_name
        Name of the ``to_check`` variable, used in the error message.
    synthesis_object
        Synthesis object we're producing the figure for, so we know what the allowed
        plots are.

    Raises
    ------
    ValueError
        If ``to_check`` takes an illegal value.
    """  # numpydoc ignore=EX01
    allowed_vals = [
        "synthesis_imshow",
        "synthesis_histogram",
        "misc",
    ]
    if isinstance(synthesis_object, Metamer):
        allowed_vals.extend(["synthesis_loss", "metamer_representation_error"])
    elif isinstance(synthesis_object, MADCompetition):
        allowed_vals.extend(["synthesis_loss"])
    try:
        vals = to_check.keys()
    except AttributeError:
        vals = to_check
    not_allowed = [v for v in vals if v not in allowed_vals]
    if not_allowed:
        raise ValueError(
            f"{to_check_name} contained value(s) {not_allowed}! "
            f"For {type(synthesis_object)} only {allowed_vals} are permissible!"
        )


def _setup_synthesis_fig(
    included_plots: list[str],
    synthesis_object: Metamer | MADCompetition | Eigendistortion,
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float, float] | None = None,
    width_ratios: dict[str, int] = {},
) -> tuple[mpl.figure.Figure, dict[str, mpl.axes.Axes | list[mpl.axes.Axes]]]:
    """
    Set up figure for :func:`synthesis_status`.

    Creates figure with enough axes for the all the plots you want. Will
    also create index in ``axes_idx`` for them if you haven't done so already.

    If ``fig=None``, all axes will be on the same row and have the same width.
    If you want them to be on different rows, will need to initialize ``fig``
    yourself and pass that in. For changing width, change the corresponding
    value in ``width_ratios``, which gives width relative to other axes. So
    if you want the axis for the ``synthesis_loss`` plot to be twice as wide
    as the others, pass ``width_ratios={"synthesis_loss": 2}``.

    .. attention::
       This function does not raise errors if ``included_plots``,
       ``width_ratios``, or ``axes_idx`` contains improper values, it assumes
       that validation has already been handled.

    Parameters
    ----------
    included_plots
        Which plots to include.
    synthesis_object
        Synthesis object we're producing the figure for, so we know what widths
        to use if unset.
    fig
        The figure to plot on or ``None``. If ``None``, we create a new figure.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows for more
        fine-grained control of the resulting figure. Possible keys are the possible
        values of ``included_plots``, plus ``"misc"``. Values should all be ints. If you
        tell this function to create a plot that doesn't have a corresponding key, we
        find the lowest int that is not already in the dict, so if you have axes that
        you want unchanged, place their idx in ``"misc"``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have relative width=1 correspond to 5.
    width_ratios
        If ``width_ratios`` is an empty dictionary, plot widths will depend on
        ``synthesis_object`` class: for :class:`~plenoptic.MADCompetition`,
        :func:`synthesis_loss` will have double the width of the rest; for other
        classes, all will be the same width.  To change that, specify their relative
        widths; keys should be strings (possible values same as ``included_plots``)
        and values should be floats specifying their relative width.

    Returns
    -------
    fig
        The figure to plot on.
    axes_dict
        Dictionary mapping between plot types and axis objects.
    """  # numpydoc ignore=EX01
    n_subplots = 0
    axes_idx = axes_idx.copy()
    # start with the defaults
    actual_width_ratios = {
        "synthesis_imshow": 1,
        "synthesis_histogram": 1,
        "metamer_representation_error": 1,
        "synthesis_loss": 2 if isinstance(synthesis_object, MADCompetition) else 1,
    }
    # overwrite with any user-specified values
    actual_width_ratios.update(width_ratios)
    all_possible_plots = [
        "synthesis_imshow",
        "synthesis_loss",
        "metamer_representation_error",
        "synthesis_histogram",
    ]
    # make sure that we skip any axes user told us to.
    misc_axes = axes_idx.get("misc", [])
    if not hasattr(misc_axes, "__iter__"):
        misc_axes = [misc_axes]
    n_subplots += len(misc_axes)
    figure_width_ratios = [1] * len(misc_axes)
    for plot in all_possible_plots:
        if plot in included_plots:
            n_subplots += 1
            figure_width_ratios.append(actual_width_ratios[plot])
            if plot not in axes_idx:
                axes_idx[plot] = tensors._find_min_int(axes_idx.values())
    if fig is None:
        figure_width_ratios = np.array(figure_width_ratios)
        if figsize is None:
            # we want (5, 5) for each subplot, with a bit of room between
            # each subplot
            figsize = (
                (figure_width_ratios * 5).sum() + figure_width_ratios.sum() - 1,
                5,
            )
        figure_width_ratios = figure_width_ratios / figure_width_ratios.sum()
        fig, axes = plt.subplots(
            1,
            n_subplots,
            figsize=figsize,
            gridspec_kw={"width_ratios": figure_width_ratios},
        )
        if n_subplots == 1:
            axes = [axes]
    else:
        axes = fig.axes
    all_axes = []
    # make sure misc contains all the empty axes. this will catch additional axes if
    # e.g., the user created a figure with 10 axes and then passed it to this function
    for i in axes_idx.values():
        # so if it's a list of ints
        if hasattr(i, "__iter__"):
            all_axes.extend(i)
        else:
            all_axes.append(i)
    misc_axes += [i for i, _ in enumerate(fig.axes) if i not in all_axes]
    axes_idx["misc"] = misc_axes
    # now remap from idx to axes objects
    axes_dict = {}
    for k, v in axes_idx.items():
        if hasattr(v, "__iter__"):
            axes_dict[k] = [axes[v_] for v_ in v]
        else:
            axes_dict[k] = axes[v]
    return fig, axes_dict


def _get_default_included_plots(
    synthesis_object: Metamer | MADCompetition | Eigendistortion,
) -> list[str]:
    """
    Return value for ``included_plots``, based on ``synthesis_object`` class.

    - :class:`~plenoptic.Metamer`: :func:`synthesis_imshow`,
      :func:`synthesis_loss`, :func:`~plenoptic.plot.metamer_representation_error`

    - :class:`~plenoptic.MADCompetition`: :func:`synthesis_imshow`,
      :func:`synthesis_loss`

    - :class:`~plenoptic.Eigendistortion`: :func:`synthesis_imshow`,
      :func:`synthesis_histogram`

    Parameters
    ----------
    synthesis_object
        Synthesis object we're producing the figure for.

    Returns
    -------
    included_plots
        Included plots.
    """
    if isinstance(synthesis_object, Metamer):
        return ["synthesis_imshow", "synthesis_loss", "metamer_representation_error"]
    if isinstance(synthesis_object, MADCompetition):
        return ["synthesis_imshow", "synthesis_loss"]
    if isinstance(synthesis_object, Eigendistortion):
        return ["synthesis_imshow", "synthesis_histogram"]


def _synthesis_status(
    synthesis_object: Metamer | MADCompetition | Eigendistortion,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    iteration: int | None = None,
    included_plots: list[str] | None = None,
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int | list[int]] = {},
    figsize: tuple[float, float] | None = None,
    width_ratios: dict[str, float] = {},
    **kwargs: Any,
) -> tuple[mpl.figure.Figure, dict[str, mpl.axes.Axes | list[mpl.axes.Axes]]]:
    r"""
    Help create synthesis status figure, returning extra info.

    This helper figure is used by :func:`synthesis_status` and :func:`synthesis_animate`
    and returns additional shared information they need.

    See :func:`synthesis_status` for more complete docstring.

    Parameters
    ----------
    synthesis_object
        Synthesis object with status to plot.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, plot all channels;
        if image has more than 1 channel, will attempt to plot as RGB(A) image.
    iteration
        Which iteration to display, for :class:`~plenoptic.Metamer` and
        :class:`~plenoptic.MADCompetition` objects. If ``None``, we show the most recent
        one. Negative values are also allowed. If ``iteration!=None`` and
        ``synthesis_object.store_progress>1`` (that is, the synthesized image was not
        cached on every iteration), then we use the cached image from the nearest
        iteration.
    included_plots
        Which plots to include. See above for behavior if ``None``, otherwise must be a
        list of strings whose values are names of plotting functions that can accept
        ``synthesis_object``, see above for list.
    fig
        If ``None``, we create a new figure. Otherwise we assume this is
        a figure that has the appropriate size and number of subplots.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure.
        Keys must be strings matching the names of the included plots, see above
        for possible values, or ``"misc"``. All axes in ``"misc"`` will be ignored
        by this function. If you tell this function to create a plot that doesn't
        have a corresponding key, we find the lowest int that is not already in
        the dict, so if you have axes that you want unchanged, place their idx
        in ``'misc'``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have each axis be of size ``(5, 5)``.
    width_ratios
        If ``width_ratios`` is an empty dictionary, plot widths will depend on
        ``synthesis_object`` class: for :class:`~plenoptic.MADCompetition`,
        :func:`synthesis_loss` will have double the width of the rest; for other
        classes, all will be the same width.  To change that, specify their relative
        widths; keys should be strings (possible values same as ``included_plots``)
        and values should be floats specifying their relative width.
    **kwargs
        Additional keyword arguments to pass to plotting functions. Keys must be the
        of the form ``{plot_func}_kwargs``, where ``{plot_func}`` name of the
        plotting function. See Examples for examples.

    Returns
    -------
    fig
        The figure containing this plot.
    axes_dict
        Dictionary mapping between plot types and axis objects.
    """
    if included_plots is None:
        included_plots = _get_default_included_plots(synthesis_object)
    _check_included_plots(included_plots, "included_plots", synthesis_object)
    _check_included_plots(width_ratios, "width_ratios", synthesis_object)
    _check_included_plots(axes_idx, "axes_idx", synthesis_object)
    _check_plot_consistency(included_plots, width_ratios, axes_idx)
    fig, axes_dict = _setup_synthesis_fig(
        included_plots, synthesis_object, fig, axes_idx, figsize, width_ratios
    )

    if "synthesis_imshow" in included_plots:
        synthesis_imshow(
            synthesis_object,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=iteration,
            ax=axes_dict["synthesis_imshow"],
            **kwargs.get("synthesis_imshow_kwargs", {}),
        )
    if "synthesis_loss" in included_plots:
        loss_axes = synthesis_loss(
            synthesis_object,
            iteration=iteration,
            axes=axes_dict["synthesis_loss"],
            **kwargs.get("synthesis_loss_kwargs", {}),
        )
        # synthesis_loss may create new axes, so make sure it's up-to-date here
        axes_dict["synthesis_loss"] = list(loss_axes.values())
    if "metamer_representation_error" in included_plots:
        rep_axes = metamer_representation_error(
            synthesis_object,
            batch_idx=batch_idx,
            iteration=iteration,
            ax=axes_dict["metamer_representation_error"],
            **kwargs.get("metamer_representation_error_kwargs", {}),
        )
        # metamer_representation_error may create new axes, so make sure it's
        # up-to-date here
        axes_dict["metamer_representation_error"] = rep_axes
    if "synthesis_histogram" in included_plots:
        synthesis_histogram(
            synthesis_object,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            iteration=iteration,
            ax=axes_dict["synthesis_histogram"],
            **kwargs.get("synthesis_histogram_kwargs", {}),
        )
    return fig, axes_dict


def synthesis_status(
    synthesis_object: Metamer | MADCompetition | Eigendistortion,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    iteration: int | None = None,
    included_plots: list[str] | None = None,
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int | list[int]] = {},
    figsize: tuple[float, float] | None = None,
    width_ratios: dict[str, float] = {},
    **kwargs: Any,
) -> mpl.figure.Figure:
    r"""
    Make a plot showing synthesis status.

    .. versionadded:: 2.0
       Combines previously separate loss plotting functions for
       :class:`~plenoptic.Metamer`, :class:`~plenoptic.MADCompetition`, and adds support
       for :class:`~plenoptic.Eigendistortion`.

    We create several subplots to analyze this. The plots to include are
    specified by including their name in the ``included_plots`` list. All plots
    can be created separately using the method with the same name.

    This function's behavior when ``included_plots is None``, and allowed values for
    that variable, depends upon the type of ``synthesis_object``:

    - :class:`~plenoptic.Metamer`: :func:`synthesis_imshow`,
      :func:`synthesis_loss`, :func:`~plenoptic.plot.metamer_representation_error`.
      Additional allowed values: :func:`synthesis_histogram`.

    - :class:`~plenoptic.MADCompetition`: :func:`synthesis_imshow`,
      :func:`synthesis_loss`. Additional allowed values: :func:`synthesis_histogram`.

    - :class:`~plenoptic.Eigendistortion`: :func:`synthesis_imshow`,
      :func:`synthesis_histogram`.

    Parameters
    ----------
    synthesis_object
        Synthesis object with status to plot.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, plot all channels.
    iteration
        Which iteration to display, for :class:`~plenoptic.Metamer` and
        :class:`~plenoptic.MADCompetition` objects. If ``None``, we show the most recent
        one. Negative values are also allowed. If ``iteration!=None`` and
        ``synthesis_object.store_progress>1`` (that is, the synthesized image was not
        cached on every iteration), then we use the cached image from the nearest
        iteration.
    included_plots
        Which plots to include. See above for behavior if ``None``, otherwise must be a
        list of strings whose values are names of plotting functions that can accept
        ``synthesis_object``, see above for list.
    fig
        If ``None``, we create a new figure. Otherwise we assume this is
        a figure that has the appropriate size and number of subplots.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure.
        Keys must be strings matching the names of the included plots, see above
        for possible values, or ``"misc"``. All axes in ``"misc"`` will be ignored
        by this function. If you tell this function to create a plot that doesn't
        have a corresponding key, we find the lowest int that is not already in
        the dict, so if you have axes that you want unchanged, place their idx
        in ``'misc'``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have each axis be of size ``(5, 5)``.
    width_ratios
        If ``width_ratios`` is an empty dictionary, plot widths will depend on
        ``synthesis_object`` class: for :class:`~plenoptic.MADCompetition`,
        :func:`synthesis_loss` will have double the width of the rest; for other
        classes, all will be the same width.  To change that, specify their relative
        widths; keys should be strings (possible values same as ``included_plots``)
        and values should be floats specifying their relative width.
    **kwargs
        Additional keyword arguments to pass to plotting functions. Keys must be the
        of the form ``{plot_func}_kwargs``, where ``{plot_func}`` name of the
        plotting function. See Examples for examples.

    Returns
    -------
    fig
        The figure containing this plot.

    Raises
    ------
    ValueError
        If the ``iteration is not None`` and the given ``synthesis_object`` object is
        :class:`~plenoptic.Eigendistortion` or synthesis was run with
        ``store_progress=False``.
    ValueError
        If any of ``width_ratios``, ``included_plots``, or ``axes_idx`` reference an
        plot that is incompatible with ``synthesis_object``. See list at top of
        docstring for compatible plots for each class.

    Warns
    -----
    UserWarning
        If the iteration used for cached image is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``synthesis_object.store_progress=2``).

    See Also
    --------
    synthesis_imshow
        One of this function's axis-level component functions: display synthesized
        image at a given synthesis iteration.
    synthesis_loss
        One of this function's axis-level component functions: plot synthesis loss
        over iterations.
    :func:`~plenoptic.plot.metamer_representation_error`
        One of this function's axis-level component functions: plot error in model
        representation at a given metamer synthesis iteration.
    synthesis_histogram
        One of this function's axis-level component functions: plot histogram of
        values from synthesized object.
    synthesis_animshow
        Create a video that animates this figure over synthesis iteration.

    Examples
    --------
    Plot for a :class:`~plenoptic.Metamer` object:

    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import torch
      >>> img = po.data.einstein()
      >>> model = po.models.Gaussian(30).eval()
      >>> po.remove_grad(model)
      >>> met = po.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> po.plot.synthesis_status(met)
      <Figure size ...>

    If model has its own ``plot_representation`` method, this function will use it
    for plotting the representation error (see
    :func:`~plenoptic.models.PortillaSimoncelli.plot_representation`):

    .. plot::
      :context: close-figs

      >>> img = po.data.reptile_skin()
      >>> model = po.models.PortillaSimoncelli(img.shape[-2:])
      >>> met = po.MetamerCTF(img, model, po.loss.l2_norm)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamerCTF_ps.pt"))
      >>> po.plot.synthesis_status(met)
      <Figure size ...>

    Plot a different iteration of synthesis:

    .. plot::
      :context: close-figs

      >>> po.plot.synthesis_status(met, iteration=10)
      <Figure size ...>

    Change the included plots:

    .. plot::
      :context: close-figs

      >>> included_plots = ["synthesis_loss", "synthesis_histogram"]
      >>> po.plot.synthesis_status(met, included_plots=included_plots)
      <Figure size ...>

    Adjust width of included plots:

    .. plot::
      :context: close-figs

      >>> width_ratios = {"metamer_representation_error": 3}
      >>> po.plot.synthesis_status(met, width_ratios=width_ratios)
      <Figure size ...>

    Change the arrangement of the plots, creating some empty axes:

    .. plot::
      :context: close-figs

      >>> axes_idx = {"misc": [0, 3], "synthesis_loss": 4}
      >>> po.plot.synthesis_status(met, axes_idx=axes_idx)
      <Figure size ...>

    Plot on an existing figure, with already existing plots:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 4, figsize=(16, 4))
      >>> axes[0].plot(torch.rand(100))
      [<matplotlib.lines.Line2D ...>]
      >>> # specify misc: 0 so we don't plot on top of this axis.
      >>> axes_idx = {"misc": 0}
      >>> po.plot.synthesis_status(met, fig=fig, axes_idx=axes_idx)
      <Figure size ...>

    Note that if you pass a figure, it must already have axes created:

    >>> fig = plt.figure()
    >>> po.plot.synthesis_status(met, fig=fig)
    Traceback (most recent call last):
    IndexError: list index out of range

    Specify additional keyword arguments to one of the underlying plots:

    .. plot::
      :context: close-figs

      >>> po.plot.synthesis_status(
      ...     met,
      ...     synthesis_loss_kwargs={"plot_penalties": True},
      ...     synthesis_imshow_kwargs={"zoom": 0.5},
      ... )
      <Figure size ...>

    Plot for :class:`~plenoptic.MADCompetition` object. Note the plots
    are different:

    .. plot::
      :context: close-figs

      >>> img = po.data.curie().to(torch.float64)
      >>> def ds_ssim(x, y):
      ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
      >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
      >>> mad.load(po.data.fetch_data("example_mad.pt"))
      >>> po.plot.synthesis_status(mad)
      <Figure size ...>

    Plot for :class:`~plenoptic.Eigendistortion` object. Note the plots
    are different:

    .. plot::
      :context: close-figs

      >>> img = po.data.einstein().to(torch.float64)
      >>> lg = po.models.LuminanceGainControl(
      ...     (31, 31), pad_mode="circular", pretrained=True, cache_filt=True
      ... ).eval()
      >>> lg = lg.to(torch.float64)
      >>> po.remove_grad(lg)
      >>> eig = po.Eigendistortion(img, lg)
      >>> eig.load(
      ...     po.data.fetch_data("example_eigendistortion.pt"),
      ...     map_location="cpu",
      ... )
      >>> po.plot.synthesis_status(eig)
      <Figure size ...>
    """
    fig, _ = _synthesis_status(
        synthesis_object,
        batch_idx,
        channel_idx,
        iteration,
        included_plots,
        fig,
        axes_idx,
        figsize,
        width_ratios,
        **kwargs,
    )
    return fig


def _get_ylim_rescale(
    metamer: Metamer,
    ylim: str | None | tuple[float, float] | Literal[False] = "rescale",
) -> tuple[None | tuple[float, float] | Literal[False], int]:
    """
    Prepare ylim, ylim_rescale_interval for :func:`synthesis_animshow`.

    This only works with a :class:`~plenoptic.Metamer` object, and is intended to be
    used with the :func:`~plenoptic.plot.metamer_representation_error` plot. It thus
    checks ``metamer.target_representation``.

    Parameters
    ----------
    metamer
        Metamer object we'll be animating.
    ylim
        User-provided ylim value for the ``metamer_representation_error`` plot. Ignored
        if model output is not 3d:

        * If a tuple, then this is the ylim of all plots

        * If ``None``, then all plots have the same limits, all
          symmetric about 0 with a limit of
          ``np.abs(representation_error).max()`` (for the initial
          representation_error).

        * If ``False``, don't modify limits.

        * If a string, must be ``"rescale"`` or of the form ``"rescaleN"``,
          where N can be any integer. If ``"rescaleN"``, we rescale the
          limits every N frames (we rescale as if ``ylim=None``). If
          ``"rescale"``, then we do this 10 times over the course of the
          animation.

    Returns
    -------
    ylim
        Initial ylim value.
    ylim_rescale_interval
        How often to update the ylim, in frames.

    Raises
    ------
    ValueError
        If ``synthesis_object`` is a :class:`~plenoptic.Metamer` object whose
        :attr:`~plenoptic.Metamer.target_representation` is 4d and ``ylim`` has been
        set -- we do not know how to best rescale color ranges.
    """
    if metamer.target_representation.ndimension() == 4:
        # we have to do this here so that we set the
        # ylim_rescale_interval such that we never rescale ylim
        # (rescaling ylim messes up an image axis)
        if isinstance(ylim, str) and ylim != "rescale" and ylim.startswith("rescale"):
            raise ValueError(
                "Looks like representation is image-like, haven't fully"
                " thought out how to best handle rescaling color ranges yet!"
            )
        ylim = False
    try:
        if ylim.startswith("rescale"):
            try:
                ylim_rescale_interval = int(ylim.replace("rescale", ""))
            except ValueError:
                # then there's nothing we can convert to an int there
                ylim_rescale_interval = int((metamer.saved_metamer.shape[0] - 1) // 10)
                if ylim_rescale_interval == 0:
                    ylim_rescale_interval = int(metamer.saved_metamer.shape[0] - 1)
            ylim = None
        else:
            raise ValueError(f"Don't know how to handle ylim {ylim}!")
    except AttributeError:
        # this way we'll never rescale
        ylim_rescale_interval = len(metamer.saved_metamer) + 1
    return ylim, ylim_rescale_interval


def synthesis_animshow(
    synthesis_object: Metamer | MADCompetition,
    framerate: int = 10,
    batch_idx: int = 0,
    channel_idx: int | None = None,
    included_plots: list[str] | None = None,
    ylim: str | None | tuple[float, float] | Literal[False] = "rescale",
    fig: mpl.figure.Figure | None = None,
    axes_idx: dict[str, int] = {},
    figsize: tuple[float, float] | None = None,
    width_ratios: dict[str, float] = {},
    **kwargs: Any,
) -> mpl.animation.FuncAnimation:
    r"""
    Animate synthesis progress.

    .. versionadded:: 2.0
       Combines previously separate loss plotting functions for
       :class:`~plenoptic.Metamer` and :class:`~plenoptic.MADCompetition`.

    This animates the figure produced by :func:`synthesis_status` over time, for each
    stored iteration. It begins by calling that function to initialize the first frame
    of the movie.

    This function's behavior when ``included_plots is None``, and allowed values for
    that variable, depends upon the type of ``synthesis_object``:

    - :class:`~plenoptic.Metamer`: :func:`synthesis_imshow`,
      :func:`synthesis_loss`, :func:`~plenoptic.plot.metamer_representation_error`.
      Additional allowed values: :func:`synthesis_histogram`.

    - :class:`~plenoptic.MADCompetition`: :func:`synthesis_imshow`,
      :func:`synthesis_loss`. Additional allowed values: :func:`synthesis_histogram`.

    Parameters
    ----------
    synthesis_object
        Synthesis object with the images we wish to display.
    framerate
        How many frames a second to display.
    batch_idx
        Which index to take from the batch dimension.
    channel_idx
        Which index to take from the channel dimension. If ``None``, plot all channels.
    included_plots
        Which plots to include. See above for behavior if ``None``, otherwise must be a
        list of strings whose values are names of plotting functions that can accept
        ``synthesis_object``, see above for list.
    ylim
        The y-limits of the ``metamer_representation_error`` plot. Ignored if model
        output is not 3d:

        * If a tuple, then this is the ylim of all plots

        * If ``None``, then all plots have the same limits, all
          symmetric about 0 with a limit of
          ``np.abs(representation_error).max()`` (for the initial
          representation_error).

        * If ``False``, don't modify limits.

        * If a string, must be ``"rescale"`` or of the form ``"rescaleN"``,
          where N can be any integer. If ``"rescaleN"``, we rescale the
          limits every N frames (we rescale as if ``ylim=None``). If
          ``"rescale"``, then we do this 10 times over the course of the
          animation.

    fig
        If ``None``, we create a new figure. Otherwise we assume this is
        a figure that has the appropriate size and number of subplots.
    axes_idx
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure.
        Keys must be strings matching the names of the included plots, see above
        for possible values, or ``"misc"``. All axes in ``"misc"`` will be ignored
        by this function. If you tell this function to create a plot that doesn't
        have a corresponding key, we find the lowest int that is not already in
        the dict, so if you have axes that you want unchanged, place their idx
        in ``'misc'``.
    figsize
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If ``None``, we attempt to
        make our best guess, aiming to have each axis be of size ``(5, 5)``.
    width_ratios
        If ``width_ratios`` is an empty dictionary, plot widths will depend on
        ``synthesis_object`` class: for :class:`~plenoptic.MADCompetition`,
        :func:`synthesis_loss` will have double the width of the rest; for other
        classes, all will be the same width.  To change that, specify their relative
        widths; keys should be strings (possible values same as ``included_plots``)
        and values should be floats specifying their relative width.
    **kwargs
        Additional keyword arguments to pass to plotting functions. Keys must be the
        of the form ``{plot_func}_kwargs``, where ``{plot_func}`` name of the
        plotting function. See Examples for examples.

    Returns
    -------
    anim
        The animation object. In order to view, must convert to HTML or save.

    Raises
    ------
    ValueError
        If synthesis for this ``synthesis_object`` was run with
        ``store_progress=False``.
    ValueError
        If we do not know how to interpret the value of ``ylim``.
    ValueError
        If ``synthesis_object`` is a :class:`~plenoptic.Metamer` object whose
        :attr:`~plenoptic.Metamer.target_representation` is 4d and ``ylim`` has been
        set -- we do not know how to best rescale color ranges.
    ValueError
        If any of ``width_ratios``, ``included_plots``, or ``axes_idx`` reference an
        plot that is incompatible with ``synthesis_object``. See list at top of
        docstring for compatible plots for each class.
    TypeError
        If ``synthesis_object`` is not :class:`~plenoptic.MADCompetition` or
        :class:`~plenoptic.Metamer`

    See Also
    --------
    :func:`~plenoptic.plot.update_plot`
        Function used by this one to update ``synthesis_imshow`` and
        ``metamer_representation_error`` plots.
    synthesis_imshow
        One of this function's axis-level component functions: display synthesized
        image at a given synthesis iteration.
    synthesis_loss
        One of this function's axis-level component functions: plot synthesis loss
        over iterations.
    metamer_representation_error
        One of this function's axis-level component functions: plot error in model
        representation at a given synthesis iteration.
    synthesis_histogram
        One of this function's axis-level component functions: plot histogram of
        values from synthesized object.
    synthesis_status
        Create a figure that shows a frame from this movie: the synthesis status at
        a given iteration.

    Notes
    -----
    - This functions returns a matplotlib FuncAnimation object. See below for how
      to view to view it in a Jupyter notebook. See Examples section for how to save to
      disk. In either case, this can take a while and you'll need the appropriate writer
      installed and on your path, e.g., ffmpeg, imagemagick, etc). See :doc:`matplotlib
      documentation <matplotlib:api/animation_api>` for more details.

    - Unless specified, we use the ffmpeg backend, which requires that you have
      ffmpeg installed and on your path (https://ffmpeg.org/download.html). To use
      a different, use the matplotlib rcParams:
      ``matplotlib.rcParams['animation.writer'] = writer``, see `matplotlib
      documentation
      <https://matplotlib.org/stable/api/animation_api.html#writer-classes>`_ for
      more details.

    - To view in a Jupyter notebook, we recommend adding the following to the first cell
      of your notebook (requires ffmpeg):

      .. code:: python

          import matplotlib.pyplot as plt
          plt.rcParams["animation.html"] = "html5"
          # use single-threaded ffmpeg for animation writer
          plt.rcParams["animation.writer"] = "ffmpeg"
          plt.rcParams["animation.ffmpeg_args"] = ["-threads", "1"]

    Examples
    --------
    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import torch
      >>> img = po.data.einstein()
      >>> model = po.models.Gaussian(30).eval()
      >>> po.remove_grad(model)
      >>> met = po.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> ani = po.plot.synthesis_animshow(met)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-1.gif")

    .. image:: animate-example-1.gif

    This function can only be used if
    :meth:`~plenoptic.Metamer.synthesize` was called with
    ``store_progress``.

    >>> import plenoptic as po
    >>> img = po.data.einstein()
    >>> model = po.models.Gaussian(30).eval()
    >>> po.remove_grad(model)
    >>> met = po.Metamer(img, model)
    >>> met.to(torch.float64)
    >>> met.synthesize(5)
    >>> ani = po.plot.synthesis_animshow(met)
    Traceback (most recent call last):
    ValueError: synthesize() was run with store_progress=False...

    If model has its own ``plot_representation`` method, this function will use it
    for plotting the representation error (see
    :func:`~plenoptic.models.PortillaSimoncelli.plot_representation`
    ):

    .. plot::
      :context: close-figs

      >>> img = po.data.reptile_skin()
      >>> model = po.models.PortillaSimoncelli(img.shape[-2:])
      >>> met = po.MetamerCTF(img, model, po.loss.l2_norm)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamerCTF_ps.pt"))
      >>> ani = po.plot.synthesis_animshow(met)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-2.gif")

    .. image:: animate-example-2.gif

    Change the included plots:

    .. plot::
      :context: close-figs

      >>> included_plots = ["synthesis_loss", "synthesis_histogram"]
      >>> ani = po.plot.synthesis_animshow(met, included_plots=included_plots)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-3.gif")

    .. image:: animate-example-3.gif

    Adjust width of included plots:

    .. plot::
      :context: close-figs

      >>> width_ratios = {"metamer_representation_error": 3}
      >>> ani = po.plot.synthesis_animshow(met, width_ratios=width_ratios)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-4.gif")

    .. image:: animate-example-4.gif

    Change the arrangement of the plots, creating some empty axes:

    .. plot::
      :context: close-figs

      >>> axes_idx = {"misc": [0, 3], "synthesis_loss": 4}
      >>> ani = po.plot.synthesis_animshow(met, axes_idx=axes_idx)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-5.gif")

    .. image:: animate-example-5.gif

    Plot on an existing figure, with already existing plots:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 4, figsize=(16, 4))
      >>> axes[0].plot(torch.rand(100))
      [<matplotlib.lines.Line2D ...>]
      >>> # specify misc: 0 so we don't plot on top of this axis.
      >>> axes_idx = {"misc": 0}
      >>> ani = po.plot.synthesis_animshow(met, fig=fig, axes_idx=axes_idx)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-6.gif")

    .. image:: animate-example-6.gif

    Specify additional keyword arguments to one of the underlying plots:

    .. plot::
      :context: close-figs

      >>> ani = po.plot.synthesis_animshow(
      ...     met,
      ...     synthesis_loss_kwargs={"plot_penalties": True},
      ...     synthesis_imshow_kwargs={"zoom": 0.5},
      ... )
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-7.gif")

    .. image:: animate-example-7.gif

    Plot for :class:`~plenoptic.MADCompetition` object. Note the plots
    are different:

    .. plot::
      :context: close-figs

      >>> img = po.data.curie().to(torch.float64)
      >>> def ds_ssim(x, y):
      ...     return 1 - po.metric.ssim(x, y, weighted=True, pad="reflect")
      >>> mad = po.MADCompetition(img, ds_ssim, po.metric.mse, "max", 1e6)
      >>> mad.load(po.data.fetch_data("example_mad.pt"))
      >>> ani = po.plot.synthesis_animshow(mad)
      >>> # Save the video (here we're saving it as a .gif)
      >>> ani.save("animate-example-8.gif")

    .. image:: animate-example-8.gif
    """
    if not isinstance(synthesis_object, (Metamer, MADCompetition)):
        raise TypeError(
            "synthesis_object must be a MADCompetition or Metamer object but got"
            f" {type(synthesis_object)}"
        )
    if not synthesis_object.store_progress:
        raise ValueError(
            "synthesize() was run with store_progress=False, cannot animate!"
        )
    # ylim only relevant for metamer_representation_error plot
    if isinstance(synthesis_object, Metamer):
        ylim, ylim_rescale_interval = _get_ylim_rescale(synthesis_object, ylim)
        # we run synthesis_status to initialize the figure
        rep_error_kwargs = kwargs.pop("metamer_representation_error_kwargs", {})
        rep_error_kwargs["ylim"] = ylim
        kwargs["metamer_representation_error_kwargs"] = rep_error_kwargs
    fig, axes_dict = _synthesis_status(
        synthesis_object,
        batch_idx,
        channel_idx,
        0,
        included_plots,
        fig,
        axes_idx,
        figsize,
        width_ratios,
        **kwargs,
    )

    # grab the artist for the loss plot (we don't need to do this for the
    # metamer or representation plot, because we use the update_plot
    # function for that)
    if "synthesis_loss" in axes_dict:
        scat = [ax.collections[0] for ax in axes_dict["synthesis_loss"]]
    if "synthesis_imshow" in axes_dict:
        ax = axes_dict["synthesis_imshow"]
        # replace the bit of the title that specifies the range,
        # since we don't make any promises about that. we have to do
        # this here because we need the figure to have been created
        ax.set_title(re.sub(r"\n range: .* \n", "\n\n", ax.get_title()))
    if (
        "metamer_representation_error" in axes_dict
        and synthesis_object.target_representation.ndimension() == 4
    ):
        # replace the bit of the title that specifies the range,
        # since we don't make any promises about that. we have to do
        # this here because we need the figure to have been created
        for ax in axes_dict["metamer_representation_error"]:
            ax.set_title(re.sub(r"\n range: .* \n", "\n\n", ax.get_title()))

    if isinstance(synthesis_object, Metamer):
        saved_synth = synthesis_object.saved_metamer
        losses = [synthesis_object.losses]
    elif isinstance(synthesis_object, MADCompetition):
        saved_synth = synthesis_object.saved_mad_image
        losses = [
            synthesis_object.reference_metric_loss,
            synthesis_object.optimized_metric_loss,
        ]

    def movie_plot(i: int) -> list[mpl.artist.Artist]:
        """
        Matplotlib function for animation.

        Update plots for frame ``i``.

        Parameters
        ----------
        i
            The frame to plot.

        Returns
        -------
        artists
            The updated matplotlib artists.
        """  # numpydoc ignore=EX01
        artists = []
        if "synthesis_imshow" in axes_dict:
            artists.extend(
                display.update_plot(
                    axes_dict["synthesis_imshow"],
                    data=saved_synth[i],
                    batch_idx=batch_idx,
                )
            )
        if "metamer_representation_error" in axes_dict:
            # this warning is not relevant for animate
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="loss iteration and iteration for"
                )
                rep_error = _representation_error(
                    synthesis_object,
                    iteration=min(
                        i * synthesis_object.store_progress,
                        len(synthesis_object.losses) - 1,
                    ),
                )

            # we pass rep_error_axes to update, and we've grabbed
            # the right things above
            artists.extend(
                display.update_plot(
                    axes_dict["metamer_representation_error"],
                    batch_idx=batch_idx,
                    model=synthesis_object.model,
                    data=rep_error,
                )
            )
            if (
                (i + 1) % ylim_rescale_interval == 0
                and synthesis_object.target_representation.ndimension() == 3
            ):
                display._rescale_ylim(
                    axes_dict["metamer_representation_error"], rep_error
                )

        if "synthesis_histogram" in axes_dict:
            # this is the dumbest way to do this, but it's simple --
            # clearing the axes can cause problems if the user has, for
            # example, changed the tick locator or formatter. not sure how
            # to handle this best right now
            axes_dict["synthesis_histogram"].clear()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="loss iteration and iteration for"
                )
                synthesis_histogram(
                    synthesis_object,
                    batch_idx=batch_idx,
                    channel_idx=channel_idx,
                    iteration=min(
                        i * synthesis_object.store_progress,
                        len(synthesis_object.losses) - 1,
                    ),
                    ax=axes_dict["synthesis_histogram"],
                )
        if "synthesis_loss" in axes_dict:
            # loss always contains values from every iteration, but everything
            # else will be subsampled.
            x_val = synthesis_object._convert_iteration(i, False)
            for sc_artist, loss in zip(scat, losses):
                sc_artist.set_offsets((x_val, loss[x_val]))
                artists.append(sc_artist)
        # as long as blitting is True, need to return a sequence of artists
        return artists

    # don't need an init_func, since we handle initialization ourselves
    anim = mpl.animation.FuncAnimation(
        fig,
        movie_plot,
        frames=len(saved_synth),
        blit=True,
        interval=1000.0 / framerate,
        repeat=False,
    )
    plt.close(fig)
    return anim
