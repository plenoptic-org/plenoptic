"""various helpful utilities for plotting or displaying information
"""
import warnings
import torch
import numpy as np
import pyrtools as pt
import matplotlib.pyplot as plt
from .data import to_numpy
try:
    from IPython.display import HTML
except ImportError:
    warnings.warn("Unable to import IPython.display.HTML")

def clean_up_axes(ax, ylim=None, spines_to_remove=['top', 'right', 'bottom'],
                  axes_to_remove=['x']):
    r"""Clean up an axis, as desired when making a stem plot of the representation

    This helper function takes in an axis

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis to clean up
    ylim : tuple, False, or None
        If a tuple, the y-limits to use for this plot. If None, we use
        the default, slightly adjusted so that the minimum is 0. If
        False, we do nothing.
    spines_to_remove : list
        Some combination of 'top', 'right', 'bottom', and 'left'. The
        spines we remove from the axis
    axes_to_remove : list
        Some combination of 'x', 'y'. The axes to set as invisible

    Returns
    -------
    ax : matplotlib.pyplot.axis
        The cleaned-up axis

    """
    if ylim is not None:
        if ylim:
            ax.set_ylim(ylim)
    else:
        ax.set_ylim((0, ax.get_ylim()[1]))
    if 'x' in axes_to_remove:
        ax.xaxis.set_visible(False)
    if 'y' in axes_to_remove:
        ax.yaxis.set_visible(False)
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
    data : array_like or dict
        The data to use when rescaling (or a dictiontary of those
        values)
    """
    def find_ymax(data):
        try:
            return np.abs(data).max()
        except RuntimeError:
            # then we need to call to_numpy on it because it needs to be
            # detached and converted to an array
            return np.abs(to_numpy(data)).max()
    try:
        y_max = find_ymax(data)
    except TypeError:
        # then this is a dictionary
        y_max = np.max([find_ymax(d) for d in data.values()])
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


def _get_artists_from_axes(axes, data):
    """Grab artists from axes.

    For now, we only grab containers (stem plots), images, or lines

    See the docstring of :meth:`update_plot()` for details on how `axes` and
    `data` should be structured

    Parameters
    ----------
    axes : list or matplotlib.axes.Axes
        The axis/axes to update.
    data : torch.Tensor or dict
        The new data to plot.

    Returns
    -------
    artists : dict
        dictionary of artists for updating plots. values are the artists to
        use, keys are the corresponding keys for data

    """
    if not hasattr(axes, '__iter__'):
        # then we only have one axis, so we may be able to update more than one
        # data element.
        if len(axes.containers) > 0:
            artists = axes.containers
        elif len(axes.images) > 0:
            artists = axes.images
        elif len(axes.lines) > 0:
            artists = axes.lines
        if isinstance(data, dict):
            artists = {ax.get_label(): ax for ax in artists}
        else:
            if data.shape[1] != len(artists):
                raise Exception(f"data has {data.shape[1]} things to plot, but "
                                f"your axis contains {len(artists)} plotting artists, "
                                "so unsure how to continue! Pass data as a dictionary"
                                " with keys corresponding to the labels of the artists"
                                " to update to resolve this.")
    else:
        # then we have multiple axes, so we are only updating one data element
        # per plot
        artists = []
        for ax in axes:
            if len(ax.containers) == 1:
                artists.extend(ax.containers)
            elif len(ax.images) == 1:
                artists.extend(ax.images)
            elif len(ax.lines) == 1:
                artists.extend(ax.lines)
        if isinstance(data, dict):
            if len(data.keys()) != len(artists):
                raise Exception(f"data has {len(data.keys())} things to plot, but "
                                f"you passed {len(axes)} axes , so unsure how "
                                "to continue!")
            artists = {k: a for k, a in zip(data.keys(), artists)}
            print(artists.keys())
        else:
            if data.shape[1] != len(artists):
                raise Exception(f"data has {data.shape[1]} things to plot, but "
                                f"you passed {len(axes)} axes , so unsure how "
                                "to continue!")
    if not isinstance(artists, dict):
        artists = {f"{i:02d}": a for i, a in enumerate(artists)}
    return artists


def update_plot(axes, data, model=None, batch_idx=0):
    r"""Update the information in some axes.

    This is used for creating an animation over time. In order to create
    the animation, we need to know how to update the matplotlib Artists,
    and this provides a simple way of doing that. It assumes the plot
    has been created by something like ``plot_representation``, which
    initializes all the artists.

    We can update stem plots, lines (as returned by ``plt.plot``), or images.
    All artists-to-update do not need to be of the same type.

    There are two modes for this:

    - single axis: axes is a single axis, which may contain multiple artists to
      update. data should be a Tensor with multiple channels (one per artist in
      the same order) or be a dictionary whose keys give the label(s) of the
      corresponding artist(s).

    - multiple axes: axes is a list of axes, each of which contains a single
      artist to update. data should be a Tensor with multiple channels (one per
      axis in the same order) or a dictionary with the same number of keys as
      axes, which we can iterate through in order.

    If you have multiple axes, each with multiple artists you want to update,
    that's too complicated for us, and so you should write a
    ``model.update_plot()`` function which handles that.

    If ``model`` is set, we try to call ``model.update_plot()`` (which
    must also return artists). If model doesn't have an ``update_plot``
    method, then we try to figure out how to update the axes ourselves,
    based on the shape of the data.

    Parameters
    ----------
    axes : list or matplotlib.axes.Axes
        The axis/axes to update.
    data : torch.Tensor or dict
        The new data to plot.
    model : torch.nn.Module or None, optional
        A differentiable model that tells us how to plot ``data``. See
        above for behavior if ``None``.
    batch_idx : int, optional
        Which index to take from the batch dimension

    Returns
    -------
    artists : list
        A list of the artists used to update the information on the
        plots

    """
    try:
        artists = model.update_plot(axes=axes, batch_idx=batch_idx, data=data)
    except AttributeError:
        ax_artists = _get_artists_from_axes(axes, data)
        artists = []
        if not isinstance(data, dict):
            data_dict = {}
            for i, d in enumerate(data.unbind(1)):
                # need to keep the shape the same because of how we
                # check for shape below (unbinding removes a dimension,
                # so we add it back)
                data_dict[f'{i:02d}'] = d.unsqueeze(1)
            data = data_dict
        for k, d in data.items():
            try:
                art = ax_artists[k]
            except KeyError:
                # If the we're grabbing these labels from the line labels and
                # they were originally ints, they will get converted to
                # strings. this catches that
                art = ax_artists[str(k)]
            d = to_numpy(d[batch_idx]).squeeze()
            if d.ndim == 1:
                try:
                    # then it's a line
                    x, _ = art.get_data()
                    art.set_data(x, d)
                    artists.append(art)
                except AttributeError:
                    # then it's a scatterplot
                    sc = update_stem(art, d)
                    artists.extend([sc.markerline, sc.stemlines])
            elif d.ndim == 2:
                art.set_data(d)
                artists.append(art)
    # make sure to always return a list
    if not isinstance(artists, list):
        artists = [artists]
    return artists


def plot_representation(model=None, data=None, ax=None, figsize=(5, 5), ylim=False, batch_idx=0,
                        title=''):
    r"""Helper function for plotting model representation

    We are trying to plot ``data`` on ``ax``, using
    ``model.plot_representation`` method, if it has it, and otherwise
    default to a function that makes sense based on the shape of ``data``.

    All of these arguments are optional, but at least some of them need
    to be set:

    - If ``model`` is ``None``, we fall-back to a type of plot based on
      the shape of ``data``. If it looks image-like, we'll use
      ``pyrtools.imshow`` and if it looks vector-like, we'll use
      ``plenoptic.clean_stem_plot``. If it's a dictionary, we'll assume
      each key, value pair gives the title and data to plot on a
      separate sub-plot.

    - If ``data`` is ``None``, we can only do something if
      ``model.plot_representation`` has some default behavior when
      ``data=None``; this is probably to plot its own ``representation``
      attribute. Thus, this will raise an Exception if both ``model``
      and ``data`` are ``None``, because we have no idea what to plot
      then.

    - If ``ax`` is ``None``, we create a one-subplot figure using
      ``figsize``. If ``ax`` is not ``None``, we therefore ignore
      ``figsize``.

    - If ``ylim`` is ``None``, we call ``rescale_ylim``, which sets the
      axes' y-limits to be ``(-y_max, y_max)``, where
      ``y_max=np.abs(data).max()``. If it's ``False``, we do nothing.

    Parameters
    ----------
    model : torch.nn.Module or None, optional
        A differentiable model that tells us how to plot ``data``. See
        above for behavior if ``None``.
    data : array_like, dict, or None, optional
        The data to plot. See above for behavior if ``None``.
    ax : matplotlib.pyplot.axis or None, optional
        The axis to plot on. See above for behavior if ``None``.
    figsize : tuple, optional
        The size of the figure to create. Ignored if ``ax`` is not
        ``None``.
    ylim : tuple,None, or False, optional
        If not None, the y-limits to use for this plot. See above for
        behavior if ``None``. If False, we do nothing.
    batch_idx : int, optional
        Which index to take from the batch dimension (the first one)
    title : str, optional
        The title to put above this axis. If you want no title, pass
        the empty string (``''``)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        warnings.warn("ax is not None, so we're ignoring figsize...")
        fig = ax.figure
    try:
        # no point in passing figsize, because we've already created
        # and are passing an axis or are passing the user-specified one
        fig, axes = model.plot_representation(ylim=ylim, ax=ax, title=title,
                                              batch_idx=batch_idx,
                                              data=data)
    except AttributeError:
        if data is None:
            data = model.representation
        if not isinstance(data, dict):
            if title is None:
                title = 'Representation'
            data_dict = {}
            for i, d in enumerate(data.unbind(1)):
                # need to keep the shape the same because of how we
                # check for shape below (unbinding removes a dimension,
                # so we add it back)
                data_dict[title+'_%02d' % i] = d.unsqueeze(1)
            data = data_dict
        else:
            warnings.warn("data has keys, so we're ignoring title!")
        # want to make sure the axis we're taking over is basically invisible.
        ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
        axes = []
        if len(list(data.values())[0].shape) == 3:
            # then this is 'vector-like'
            gs = ax.get_subplotspec().subgridspec(min(4, len(data)),
                                                  int(np.ceil(len(data) / 4)))
            for i, (k, v) in enumerate(data.items()):
                ax = fig.add_subplot(gs[i % 4, i // 4])
                # only plot the specified batch, but plot each channel
                # in a separate call. there should probably only be one,
                # and if there's not you probably want to do things
                # differently
                for d in v[batch_idx]:
                    ax = clean_stem_plot(to_numpy(d), ax, k, ylim)
                axes.append(ax)
        elif len(list(data.values())[0].shape) == 4:
            # then this is 'image-like'
            gs = ax.get_subplotspec().subgridspec(int(np.ceil(len(data) / 4)),
                                                  min(4, len(data)))
            for i, (k, v) in enumerate(data.items()):
                ax = fig.add_subplot(gs[i // 4, i % 4])
                ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
                # only plot the specified batch
                pt.imshow(to_numpy(v[batch_idx]), title=k, ax=ax, vrange='indep0')
                axes.append(ax)
            # because we're plotting image data, don't want to change
            # ylim at all
            ylim = False
        else:
            raise Exception("Don't know what to do with data of shape %s" % data.shape)
    if ylim is None:
        if isinstance(data, dict):
            data = torch.cat(list(data.values()), dim=2)
        rescale_ylim(axes, data)
    return fig
