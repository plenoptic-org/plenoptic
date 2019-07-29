"""various helpful utilities for plotting or displaying information
"""
import warnings
import numpy as np
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
