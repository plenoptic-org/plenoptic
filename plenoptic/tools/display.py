"""various helpful utilities for plotting or displaying information
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
try:
    from IPython.display import HTML
except ImportError:
    warnings.warn("Unable to import IPython.display.HTML")


def clean_up_axes(ax, ylim=None, spines_to_remove=['top', 'right', 'bottom']):
    r"""Clean up an axis, as desired when making a stem plot of the representation

    This helper function takes in an axis

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis to clean up
    ylim : tuple or None
        If not None, the y-limits to use for this plot. If None, we
        use the default, slightly adjusted so that the minimum is 0
    spines_to_remove : list
        Some combination of 'top', 'right', 'bottom', and 'left'. The
        spines we remove from the axis

    Returns
    -------
    ax : matplotlib.pyplot.axis
        The cleaned-up axis

    """
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim((0, ax.get_ylim()[1]))
    ax.xaxis.set_visible(False)
    for s in spines_to_remove:
        ax.spines[s].set_visible(False)
    return ax


def update_stem(stem_container, ydata):
    r"""Update the information in a stem plot

    We update the information in a single stem plot to match that given
    by ``ydata``. We update the position of the markers and and the
    lines connecting them to the baseline, but we don't change the
    baseline at all and assume that the xdata shouldn't change at all.

    This requires that the initial ``plt.stem`` be called with the
    argument ``use_line_collection=True``, as will be the default in
    matplotlib 3.3 (this improves efficiency, so you should do it
    anyway)

    Parameters
    ----------
    stem_container : matplotlib.container.StemContainer
        Single container for the artists created in a ``plt.stem``
        plot. It can be treated like a namedtuple ``(markerline,
        stemlines, baseline)``. In order to get this from an axis
        ``ax``, try ``ax.containers[0]`` (obviously if you have more
        than one container in that axis, it may not be the first one).
    ydata : array_like
        The new y-data to show on the plot. Importantly, must be the
        same length as the existing y-data.

    Returns
    -------
    stem_container : matplotlib.container.StemContainer
        The StemContainer containing the updated artists.

    """
    stem_container.markerline.set_ydata(ydata)
    try:
        segments = stem_container.stemlines.get_segments().copy()
    except AttributeError:
        raise Exception("We require that the initial stem plot be called with use_line_collection="
                        "True in order to correctly update it. This will significantly improve "
                        "performance as well.")
    for s, y in zip(segments, ydata):
        try:
            s[1, 1] = y
        except IndexError:
            # this happens when our segment array is 1x2 instead of 2x2,
            # which is the case when the data there is nan
            continue
    stem_container.stemlines.set_segments(segments)
    return stem_container


def rescale_ylim(axes, data):
    r"""rescale y-limits nicely

    We take the axes and set their limits to be ``(-y_max, y_max)``,
    where ``y_max=np.abs(data).max()``

    Parameters
    ----------
    axes : list
        A list of matplotlib axes to rescale
    data : array_like
        The data to use when rescaling
    """
    y_max = np.abs(data).max()
    for ax in axes:
        ax.set_ylim((-y_max, y_max))


def convert_anim_to_html(anim):
    r"""convert a matplotlib animation object to HTML (for display)

    This is a simple little wrapper function that allows the animation
    to be displayed in a Jupyter notebook

    Parameters
    ----------
    anim : matplotlib.animation.FuncAnimation
        The animation object to convert to HTML
    """
    # to_html5_video will call savefig with a dpi kwarg, so our
    # custom figure class will raise a warning. we don't want to
    # worry people, so we go ahead and suppress it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return HTML(anim.to_html5_video())


def clean_stem_plot(data, ax=None, title='', ylim=None, xvals=None):
    r"""convenience wrapper for plotting stem plots

    This plots the data, baseline, cleans up the axis, and sets the
    title

    Should not be called by users directly, but is a helper function for
    the various plot_representation() functions

    By default, stem plot would have a baseline that covers the entire
    range of the data. We want to be able to break that up visually (so
    there's a line from 0 to 9, from 10 to 19, etc), and passing xvals
    separately allows us to do that. If you want the default stem plot
    behavior, leave xvals as None.

    Parameters
    ----------
    data : np.array
        The data to plot (as a stem plot)
    ax : matplotlib.pyplot.axis or None, optional
        The axis to plot the data on. If None, we plot on the current
        axis
    title : str, optional
        The title to put on the axis.
    ylim : tuple or None, optional
        If not None, the y-limits to use for this plot. If None, we
        use the default, slightly adjusted so that the minimum is 0
    xvals : tuple or None, optional
        A 2-tuple of lists, containing the start (``xvals[0]``) and stop
        (``xvals[1]``) x values for plotting. If None, we use the
        default stem plot behavior.

    Returns
    -------
    ax : matplotlib.pyplot.axis
        The axis with the plot

    Example
    -------
    We allow for breaks in the baseline value if we want to visually
    break up the plot, as we see below.

    ..plot::
      :include-source:

      import plenoptic as po
      import numpy as np
      import matplotlib.pyplot as plt
      # if ylim=None, as in this example, the minimum y-valuewill get
      # set to 0, so we want to make sure our values are all positive
      y = np.abs(np.random.randn(55))
      y[15:20] = np.nan
      y[35:40] = np.nan
      # we want to draw the baseline from 0 to 14, 20 to 34, and 40 to
      # 54, everywhere that we have non-NaN values for y
      xvals = ([0, 20, 40], [14, 34, 54])
      po.tools.display.clean_stem_plot(y,  xvals=xvals)
      plt.show()

    If we don't care about breaking up the x-axis, you can simply use
    the default xvals (``None``). In this case, this function will just
    clean up the plot a little bit

    ..plot::
      :include-source:

      import plenoptic as po
      import numpy as np
      import matplotlib.pyplot as plt
      # if ylim=None, as in this example, the minimum y-valuewill get
      # set to 0, so we want to make sure our values are all positive
      y = np.abs(np.random.randn(55))
      po.tools.display.clean_stem_plot(y)
      plt.show()

    """
    if ax is None:
        ax = plt.gca()
    if xvals is not None:
        basefmt = ' '
        ax.hlines(len(xvals[0])*[0], xvals[0], xvals[1], colors='C3', zorder=10)
    else:
        # this is the default basefmt value
        basefmt = None
    ax.stem(data, basefmt=basefmt, use_line_collection=True)
    ax = clean_up_axes(ax, ylim, ['top', 'right', 'bottom'])
    ax.set_title(title)
    return ax


def update_plot(axes, data):
    r"""Update the information in a stem plot

    This is used for creating an animation over time. In order to create
    the animation, we need to know how to update the matplotlib Artists,
    and this provides a simple way of doing that. It relies on the fact
    that we've created a stem plot with a known structure to create the
    plots we want to update.

    We take the axes containing the information to update (note that
    this is probably a subset of the total number of axes in the figure,
    if we're showing other information, as done by ``Metamer.animate``),
    as well as the data to show on these plots and, since these are both
    lists, iterate through them, updating as we go.

    Note that we do not do anything to the data to get it into the
    correct shape / type; you'll need to do that before calling this
    function. The expected use case is that you'll be using this on the
    plots showing representation information and will thus have some
    function that wrangles the data into the proper shape for you (e.g.,
    ``_representation_for_plotting``)

    In order for this to be used by ``FuncAnimation``, we need to
    return Artists, so we return a list of the relevant artists, the
    ``markerline`` and ``stemlines`` from the ``StemContainer``.

    NOTE: This currently only works for stem plots, will need to add
    support for other types of plots as necessary

    Parameters
    ----------
    axes : list
        A list of axes to update. We assume that these are the axes
        created by ``plot_representation`` and so contain stem plots
        in the correct order.

    Returns
    -------
    stem_artists : list
        A list of the artists used to update the information on the
        stem plots

    """
    stem_artists = []
    axes = [ax for ax in axes if len(ax.containers) == 1]
    for ax, d in zip(axes, data):
        sc = update_stem(ax.containers[0], d)
        stem_artists.extend([sc.markerline, sc.stemlines])
    return stem_artists
