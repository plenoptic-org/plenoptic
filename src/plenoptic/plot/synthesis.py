"""Plots for understanding synthesis objects."""  # numpydoc ignore=EX01

import warnings
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt

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
    Plot histogram of values of synthesized and original tensors.

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

    - :class:`~plenoptic.Eigendistortion`: create separate histograms for all
      eigendistortions. ``iteration`` must be ``None``.

    Parameters
    ----------
    synthesis_object
        Synthesis object with the images whose values we want to compare.
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
        iteration.
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
    animshow
        Create a video animating this and other axis-level plots changing over
        the course of synthesis.

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
      >>> po.plot.metamer_pixel_values(met)
      <Axes: ... 'Histogram of pixel values'...>

    Plot pixel values from a specified iteration (requires setting ``store_progress``
    when :meth:`~plenoptic.Metamer.synthesize` was called):

    .. plot::
      :context: close-figs

      >>> po.plot.metamer_pixel_values(met, iteration=10)
      <Axes: ... 'Histogram of pixel values'...>

    Plot on an existing axis:

    .. plot::
      :context: close-figs

      >>> fig, axes = plt.subplots(1, 2, figsize=(8, 4))
      >>> po.plot.metamer_pixel_values(met, ax=axes[1])
      <Axes: ... 'Histogram of pixel values'...>
    """
    if isinstance(synthesis_object, Eigendistortion):
        name = "eigendistortion"
        if iteration is not None:
            raise ValueError(
                "When synthesis_object is an Eigendistortion, iteration must be None!"
            )
        image = synthesis_object.eigendistortions
        if batch_idx is None:
            image = [im for im in image]
            titles = [f"Eigendistortion[{i}]" for i in synthesis_object.eigenindex]
        else:
            titles = [f"Eigendistortion[{batch_idx}]"]
    else:
        progress = synthesis_object.get_progress(iteration)
        if isinstance(synthesis_object, Metamer):
            name = "metamer"
            title_names = ["Metamer", "Target"]
        elif isinstance(synthesis_object, MADCompetition):
            name = "mad_image"
            title_names = ["MAD", "Reference"]
        try:
            image = progress[f"saved_{name}"]
            iter = progress["store_progress_iteration"]
        except KeyError:
            if iteration is not None:
                raise IndexError(
                    "When synthesis_object.store_progress=False, iteration must be"
                    " None!"
                )
            image = eval(f"synthesis_object.{name}")
            # losses will always have one extra value, the current loss.
            iter = len(synthesis_object.losses) - 1
        if batch_idx is None:
            titles = [
                f"{title_names[0]}[{i}] [iteration={iter}]"
                for i in range(image.shape[0])
            ]
            image = [im for im in image] + [synthesis_object.image]
        else:
            titles = [f"{title_names[0]}[{batch_idx}] [iteration={iter}]"]
            image = [image, synthesis_object.image]
        titles += [f"{title_names[1]} image"]
    return display.histogram(
        image,
        titles,
        batch_idx,
        channel_idx,
        ylim,
        xlim,
        ax=ax,
        alpha=alpha,
        **kwargs,
    )
