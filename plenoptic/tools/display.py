"""various helpful utilities for plotting or displaying information
"""
import warnings
import torch
import numpy as np
import pyrtools as pt
import matplotlib.pyplot as plt
from .data import torch_complex_to_numpy, to_numpy
try:
    from IPython.display import HTML
except ImportError:
    warnings.warn("Unable to import IPython.display.HTML")


def convert_pyrshow(pyr_coeffs, image_index=0, channel=0):
    r"""Wrapper that makes outputs of the steerable pyramids compatible
    with the display functions of pyrtools.
    Selects pyramid coefficients corresponding to 'image_index' out of
    the images in the batch, and to 'channel' out of the channel indexes
    (eg. RGB channels that undergo steerable pyramid independently)
    
    Parameters
    ----------
    pyr_coeffs : dict
                pyramid coefficients in the standard dictionary format as
                specified in Steerable_Pyramid_Freq
    image_index : int in [0, batch_size] (default=0)
                  index of the image you would like to select from the batch
                  of coefficients
    channel: int (default = 0)
             index of channel to select for image display
             for grayscale images this will be 0. 
    Example
    -------
        >>> size = 32
        >>> signal = torch.randn(2, 3, size,size) # three images, each with three channels
        >>> SPF = po.simul.Steerable_Pyramid_Freq((size, size), order=3, height=3, is_complex=True, downsample=False)
        >>> pyr = SPF(signal)
        >>> pt.pyrshow(po.convert_pyrshow(pyr, 1, 2), is_complex=True, plot_complex='polar', zoom=3);
    """

    pyr_coeffvis = pyr_coeffs.copy()
    for k in pyr_coeffvis.keys():
        im = pyr_coeffvis[k][image_index, channel, ...]
        # imag and real component exist
        if im.shape[-1] == 2:
            pyr_coeffvis[k] = torch_complex_to_numpy(im)
        else:
            pyr_coeffvis[k] = to_numpy(im)

    return pyr_coeffvis


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
    data : array_like
        The data to use when rescaling
    """
    try:
        y_max = np.abs(data).max()
    except RuntimeError:
        # then we need to call to_numpy on it because it needs to be
        # detached and converted to an array
        y_max = np.abs(to_numpy(data)).max()
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


def update_plot(axes, data, model=None, batch_idx=0):
    r"""Update the information in a stem plot or image

    This is used for creating an animation over time. In order to create
    the animation, we need to know how to update the matplotlib Artists,
    and this provides a simple way of doing that. It assumes the plot
    has been created by something like ``plot_representation``, which
    initializes all the artists.

    We take a list of axes containing the information to update (note
    that this is probably a subset of the total number of axes in the
    figure, if we're showing other information, as done by
    ``Metamer.animate``), as well as the data to show on these plots
    and, since these are both lists, iterate through them, updating as
    we go.

    In order for this to be used by ``FuncAnimation``, we need to return
    Artists, so we return a list of the relevant artists, either the
    ``markerline`` and ``stemlines`` from the ``StemContainer`` or the
    image artist, ``ax.images[0]``.

    If ``model`` is set, we try to call ``model.update_plot()`` (which
    must also return artists). If model doesn't have an ``update_plot``
    method, then we try to figure out how to update the axes ourselves,
    based on the shape of the data.

    If ``data`` contains multiple channels or is a dictionary with
    multiple keys, we assume that the different channels/keys each
    belong on a separate axis (and thus, the number of channels/keys and
    the number of entries in the ``axes`` list *must* be the same --
    this will throw a very strange warning otherwise).

    Parameters
    ----------
    axes : list
        A list of axes to update. We assume that these are the axes
        created by ``plot_representation`` and so contain stem plots
        in the correct order.
    data : torch.Tensor or dict
        The new data to plot.
    model : torch.nn.Module or None, optional
        A differentiable model that tells us how to plot ``data``. See
        above for behavior if ``None``.

    Returns
    -------
    artists : list
        A list of the artists used to update the information on the
        plots

    """
    artists = []
    axes = [ax for ax in axes if len(ax.containers) == 1 or len(ax.images) == 1]
    try:
        artists = model.update_plot(axes=axes, batch_idx=batch_idx, data=data)
    except AttributeError:
        if not isinstance(data, dict):
            data_dict = {}
            for i, d in enumerate(data.unbind(1)):
                # need to keep the shape the same because of how we
                # check for shape below (unbinding removes a dimension,
                # so we add it back)
                data_dict['%02d' % i] = d.unsqueeze(1)
            data = data_dict
        for ax, d in zip(axes, data.values()):
            d = to_numpy(d[batch_idx]).squeeze()
            if d.ndim == 1:
                sc = update_stem(ax.containers[0], d)
                artists.extend([sc.markerline, sc.stemlines])
            elif d.ndim == 2:
                image_artist = ax.images[0]
                image_artist.set_data(d)
                artists.append(image_artist)
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
                pt.imshow(to_numpy(v[batch_idx]), title=k, ax=ax)
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
