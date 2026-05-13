"""Plots for understanding synthesis objects."""  # numpydoc ignore=EX01

import warnings
from collections.abc import Callable
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from .._synthesize import Eigendistortion, MADCompetition, Metamer
from . import display

__all__ = [
    "synthesis_loss",
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

    .. versionadded:: 2.0.0
       Added in version 2.0.0, combines previously separate loss plotting functions for
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
            # losses will always have one extra value, the current loss.
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

    .. versionadded:: 2.0.0
       Added in version 2.0.0, combines previously separate loss plotting functions for
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
      >>> po.plot.metamer_pixel_values(met, ax=axes[1])
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
    alpha: float = 5.0,
    process_image: Callable[[torch.Tensor], torch.Tensor] | None = None,
    zoom: float | None = None,
    iteration: int | None = None,
    ax: mpl.axes.Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Display image of synthesis object.

    We use :func:`~plenoptic.plot.imshow` to display the synthesized image and
    attempt to automatically find the most reasonable zoom value. You can override this
    value using the zoom arg, but remember that :func:`~plenoptic.plot.imshow`
    is opinionated about the size of the resulting image and will throw an Exception if
    the axis created is not big enough for the selected zoom.

    The behavior of this function is slightly different depending on the type of
    ``synthesis_object``:

    - :class:`~plenoptic.Metamer` and :class:`~plenoptic.MADCompetition`: process and
      display the synthesized image. ``iteration`` can be specified, ``alpha`` must be
      unchanged.

    - :class:`~plenoptic.Eigendistortion`: process and display ``image + (alpha *
      eigendistortion)``. ``iteration`` must be ``None``, ``alpha`` can be set.

    Parameters
    ----------
    synthesis_object
        Synthesis object with the images we wish to display.
    batch_idx
        Which index to take from the batch dimension. Note that for
        :class:`~plenoptic.Eigendistortion`, this is the
        :attr:`~plenoptic.Eigendistortion.eigenindex`.
    channel_idx
        Which index to take from the channel dimension. If ``None``, we assume
        image is RGB(A) and show all channels.
    alpha
        Amount by which to scale eigendistortion for
        ``image + (alpha * eigendistortion)`` for display. If ``synthesis_object``
        is not :class:`~plenoptic.Eigendistortion`, must not be set.
    process_image
        A function to process the plotted image. E.g., multiplying by the stdev ImageNet
        then adding the mean of ImageNet to undo image preprocessing or clamping between
        0 and 1.
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
        If the iteration used for ``saved_mad_image`` is not the same as the argument
        ``iteration`` (because e.g., you set ``iteration=3`` but
        ``mad.store_progress=2``).

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
    Plot for :class:`~plenoptic.Metamer` object. If a matplotlib figure exists, this
    function will use it (using :func:`matplotlib.pyplot.gca`):

    .. plot::
      :context: reset

      >>> import plenoptic as po
      >>> import matplotlib.pyplot as plt
      >>> import torch
      >>> plt.figure()
      <Figure size ...>
      >>> img = po.data.einstein()
      >>> model = po.models.Gaussian(30).eval()
      >>> po.remove_grad(model)
      >>> met = po.Metamer(img, model)
      >>> met.to(torch.float64)
      >>> met.load(po.data.fetch_data("example_metamer_gaussian.pt"))
      >>> po.plot.synthesis_imshow(met)
      <Axes: title=...Metamer[0] [iteration=107]...>

    If no matplotlib figure exists, this function will create a new one:

    .. plot::
      :context: close-figs

      >>> # close all open figures to ensure none exist
      >>> plt.close("all")
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

    Plot for :class:`~plenoptic.Eigendistortion` object. note here that we plot
    the distortion multiplied by some alpha and added to the target image.

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
            title = [f"{alpha} * {t}" for t in title]
    image, batch_idx = _get_synthesis_image(synthesis_object, batch_idx, iteration)
    if isinstance(synthesis_object, Eigendistortion):
        image = synthesis_object.image + alpha * image
    else:
        # if alpha is not default value
        if alpha != 5:
            raise ValueError(
                f"If synthesis_object is type {type(synthesis_object)}, alpha cannot be"
                " set"
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
