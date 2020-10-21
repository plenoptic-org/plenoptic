"""various helpful utilities for plotting or displaying information
"""
import warnings
import torch
import numpy as np
import pyrtools as pt
import matplotlib.pyplot as plt
from .data import to_numpy, torch_complex_to_numpy
try:
    from IPython.display import HTML
except ImportError:
    warnings.warn("Unable to import IPython.display.HTML")


def imshow(image, vrange='indep1', zoom=1, title='', col_wrap=None, ax=None,
           cmap=None, plot_complex='rectangular', batch_idx=None,
           channel_idx=None, as_rgb=False, **kwargs):
    """Show image(s) correctly.

    This function shows images correctly, making sure that each element in the
    tensor corresponds to a pixel or an integer number of pixels, to avoid
    aliasing (NOTE: this guarantee only holds for the saved image; it should
    generally hold in notebooks as well, but will fail if, e.g., you plot an
    image that's 2000 pixels wide on an monitor 1000 pixels wide; the notebook
    handles the rescaling in a way we can't control).

    Arguments
    ---------
    image : torch.Tensor or list
        The images to display. Tensors should be 4d (batch, channel, height,
        width) or 5d (if complex). List of tensors should be used for tensors
        of different height and width: all images will automatically be
        rescaled so they're displayed at the same height and width, thus, their
        heights and widths must be scalar multiples of each other.
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to
        the minimum and maximum value of the colormap, respectively. If a
        string:

        * `'auto0'`: all images have same vmin/vmax, which have the same absolute
                     value, and come from the minimum or maximum across all
                     images, whichever has the larger absolute value
        * `'auto/auto1'`: all images have same vmin/vmax, which are the
                          minimum/maximum values across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across
                     all images) minus/ plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the
                     10th/90th percentile values to the 10th/90th percentile of
                     the display intensity range. For example: vmin is the 10th
                     percentile image value minus 1/8 times the difference
                     between the 90th and 10th percentile
        * `'indep0'`: each image has an independent vmin/vmax, which have the
                      same absolute value, which comes from either their
                      minimum or maximum value, whichever has the larger
                      absolute value.
        * `'indep1'`: each image has an independent vmin/vmax, which are their
                      minimum/maximum values
        * `'indep2'`: each image has an independent vmin/vmax, which is their
                      mean minus/plus 2 std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that
                      the 10th/90th percentile values map to the 10th/90th
                      percentile intensities.
    zoom : `float`
        ratio of display pixels to image pixels. if >1, must be an integer. If
        <1, must be 1/d where d is a a divisor of the size of the largest
        image.
    title : `str`, `list`, or None, optional
        Title for the plot. In addition to the specified title, we add a
        subtitle giving the plotted range and dimensionality (with zoom)
        * if `str`, will put the same title on every plot.
        * if `list`, all values must be `str`, must be the same length as img,
          assigning each title to corresponding image.
        * if None, no title will be printed (and subtitle will be removed;
          unsupported for complex tensors).
    col_wrap : `int` or None, optional
        number of axes to have in each row. If None, will fit all axes in a
        single row.
    ax : `matplotlib.pyplot.axis` or None, optional
        if None, we make the appropriate figure. otherwise, we resize the axes
        so that it's the appropriate number of pixels (done by shrinking the
        bbox - if the bbox is already too small, this will throw an Exception!,
        so first define a large enough figure using either make_figure or
        plt.figure)
    cmap : matplotlib colormap, optional
        colormap to use when showing these images
    plot_complex : {'rectangular', 'polar', 'logpolar'}
        specifies handling of complex values.

        * `'rectangular'`: plot real and imaginary components as separate images
        * `'polar'`: plot amplitude and phase as separate images
        * `'logpolar'`: plot log_2 amplitude and phase as separate images
        for any other value, we raise a warning and default to rectangular.
    batch_idx : int or None, optional
        Which element from the batch dimension to plot. If None, we plot all.
    channel_idx : int or None, optional
        Which element from the channel dimension to plot. If None, we plot all.
        Note if this is an int, then `as_rgb=True` will fail, because we
        restrict the channels.
    as_rgb : bool, optional
        Whether to consider the channels as encoding RGB(A) values. If True, we
        attempt to plot the image in color, so your tensor must have 3 (or 4 if
        you want the alpha channel) elements in the channel dimension, or this
        will raise an Exception. If False, we plot each channel as a separate
        grayscale image.
    kwargs :
        Passed to `ax.imshow`

    Returns
    -------
    fig : `PyrFigure`
        figure containing the plotted images

    """
    if not isinstance(image, list):
        image = [image]
    images_to_plot = []
    for im in image:
        if im.ndimension() == 5:
            # this will also call to_numpy on it
            im = torch_complex_to_numpy(im)
        else:
            im = to_numpy(im)
        if im.shape[0] > 1 and batch_idx is not None:
            # this preserves the number of dimensions
            im = im[batch_idx:batch_idx+1]
        if channel_idx is not None:
            # this preserves the number of dimensions
            im = im[:, channel_idx:channel_idx+1]
        # allow RGB and RGBA
        if as_rgb:
            if im.shape[1] not in [3, 4]:
                raise Exception("If as_rgb is True, then channel must have 3 "
                                "or 4 elements!")
            im = im.transpose(0, 2, 3, 1)
            # want to insert a fake "channel" dimension here, so our putting it
            # into a list below works as expected
            im = im.reshape((im.shape[0], 1, *im.shape[1:]))
        elif im.shape[1] > 1 and im.shape[0] > 1:
            raise Exception("Don't know how to plot images with more than one channel and batch!"
                            " Use batch_idx / channel_idx to choose a subset for plotting")
        # by iterating through it twice, we make sure to peel apart the batch
        # and channel dimensions so that they each show up as a separate image.
        # because of how we've handled everything above, we know that im will
        # be (b,c,h,w) or (b,c,h,w,r) where r is the RGB(A) values
        for i in im:
            images_to_plot.extend([i_.squeeze() for i_ in i])
    return pt.imshow(images_to_plot, vrange=vrange, zoom=zoom, title=title,
                     col_wrap=col_wrap, ax=ax, cmap=cmap, plot_complex=plot_complex,
                     **kwargs)


def animshow(video, framerate=2., repeat=False, vrange='indep1', zoom=1,
             title='', col_wrap=None, ax=None, cmap=None,
             plot_complex='rectangular', batch_idx=None, channel_idx=None,
             as_rgb=False, **kwargs):
    """Animate video(s) correctly.

    This function animates videos correctly, making sure that each element in
    the tensor corresponds to a pixel or an integer number of pixels, to avoid
    aliasing (NOTE: this guarantee only holds for the saved animation (assuming
    video compression doesn't interfere); it should generally hold in notebooks
    as well, but will fail if, e.g., your video is 2000 pixels wide on an
    monitor 1000 pixels wide; the notebook handles the rescaling in a way we
    can't control).

    This functions returns the matplotlib FuncAnimation object. In order to
    view it in a Jupyter notebook, use the
    ``plenoptic.convert_anim_to_html(anim)`` function. In order to save, use
    ``anim.save(filename)`` (note for this that you'll need the appropriate
    writer installed and on your path, e.g., ffmpeg, imagemagick, etc).

    Arguments
    ---------
    video : torch.Tensor or list
        The videos to display. Tensors should be 5d (batch, channel, time,
        height, width) or 6d (if complex). List of tensors should be used for
        tensors of different height and width: all videos will automatically be
        rescaled so they're displayed at the same height and width, thus, their
        heights and widths must be scalar multiples of each other. Videos must
        all have the same number of frames as well.
    framerate : `float`
        Temporal resolution of the video, in Hz (frames per second).
    repeat : `bool`
        whether to loop the animation or just play it once
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to
        the minimum and maximum value of the colormap, respectively. If a
        string:

        * `'auto0'`: all images have same vmin/vmax, which have the same absolute
                     value, and come from the minimum or maximum across all
                     images, whichever has the larger absolute value
        * `'auto/auto1'`: all images have same vmin/vmax, which are the
                          minimum/maximum values across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across
                     all images) minus/ plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the
                     10th/90th percentile values to the 10th/90th percentile of
                     the display intensity range. For example: vmin is the 10th
                     percentile image value minus 1/8 times the difference
                     between the 90th and 10th percentile
        * `'indep0'`: each image has an independent vmin/vmax, which have the
                      same absolute value, which comes from either their
                      minimum or maximum value, whichever has the larger
                      absolute value.
        * `'indep1'`: each image has an independent vmin/vmax, which are their
                      minimum/maximum values
        * `'indep2'`: each image has an independent vmin/vmax, which is their
                      mean minus/plus 2 std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that
                      the 10th/90th percentile values map to the 10th/90th
                      percentile intensities.
    zoom : `float`
        ratio of display pixels to image pixels. if >1, must be an integer. If
        <1, must be 1/d where d is a a divisor of the size of the largest
        image.
    title : `str`, `list`, or None, optional
        Title for the plot. In addition to the specified title, we add a
        subtitle giving the plotted range and dimensionality (with zoom)
        * if `str`, will put the same title on every plot.
        * if `list`, all values must be `str`, must be the same length as img,
          assigning each title to corresponding image.
        * if None, no title will be printed (and subtitle will be removed;
          unsupported for complex tensors).
    col_wrap : `int` or None, optional
        number of axes to have in each row. If None, will fit all axes in a
        single row.
    ax : `matplotlib.pyplot.axis` or None, optional
        if None, we make the appropriate figure. otherwise, we resize the axes
        so that it's the appropriate number of pixels (done by shrinking the
        bbox - if the bbox is already too small, this will throw an Exception!,
        so first define a large enough figure using either
        pyrtools.make_figure or plt.figure)
    cmap : matplotlib colormap, optional
        colormap to use when showing these images
    plot_complex : {'rectangular', 'polar', 'logpolar'}
        specifies handling of complex values.

        * `'rectangular'`: plot real and imaginary components as separate images
        * `'polar'`: plot amplitude and phase as separate images
        * `'logpolar'`: plot log_2 amplitude and phase as separate images
        for any other value, we raise a warning and default to rectangular.
    batch_idx : int or None, optional
        Which element from the batch dimension to plot. If None, we plot all.
    channel_idx : int or None, optional
        Which element from the channel dimension to plot. If None, we plot all.
        Note if this is an int, then `as_rgb=True` will fail, because we
        restrict the channels.
    as_rgb : bool, optional
        Whether to consider the channels as encoding RGB(A) values. If True, we
        attempt to plot the image in color, so your tensor must have 3 (or 4 if
        you want the alpha channel) elements in the channel dimension, or this
        will raise an Exception. If False, we plot each channel as a separate
        grayscale image.
    kwargs :
        Passed to `ax.imshow`

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object. In order to view, must convert to HTML
        or save.

    """
    if not isinstance(video, list):
        video = [video]
    videos_to_show = []
    for vid in video:
        if vid.ndimension() == 6:
            vid = torch_complex_to_numpy(vid)
        else:
            vid = to_numpy(vid)
        if vid.shape[0] > 1 and batch_idx is not None:
            # this preserves the number of dimensions
            vid = vid[batch_idx:batch_idx+1]
        if channel_idx is not None:
            # this preserves the number of dimensions
            vid = vid[:, channel_idx:channel_idx+1]
        # allow RGB and RGBA
        if as_rgb:
            if vid.shape[1] not in [3, 4]:
                raise Exception("If as_rgb is True, then channel must have 3 "
                                "or 4 elements!")
            vid = vid.transpose(0, 2, 3, 4, 1)
            # want to insert a fake "channel" dimension here, so our putting it
            # into a list below works as expected
            vid = vid.reshape((vid.shape[0], 1, *vid.shape[1:]))
        elif vid.shape[1] > 1 and vid.shape[0] > 1:
            raise Exception("Don't know how to plot images with more than one channel and batch!"
                            " Use batch_idx / channel_idx to choose a subset for plotting")
        # by iterating through it twice, we make sure to peel apart the batch
        # and channel dimensions so that they each show up as a separate video.
        # because of how we've handled everything above, we know that vid will
        # be (b,c,t,h,w) or (b,c,t,h,w,r) where r is the RGB(A) values
        for v in vid:
            videos_to_show.extend([v_.squeeze() for v_ in v])
    return pt.animshow(videos_to_show, framerate=framerate, as_html5=False,
                       repeat=repeat, vrange=vrange, zoom=zoom, title=title,
                       col_wrap=col_wrap, ax=ax, cmap=cmap,
                       plot_complex=plot_complex, **kwargs)


def convert_pyrshow(pyr_coeffs, image_index=0, channel=0):
    r"""Wrapper that makes outputs of the steerable pyramids compatible
    with the display functions of pyrtools.
    Selects pyramid coefficients corresponding to 'image_index' out of
    the images in the batch, and to 'channel' out of the channel indexes
    (eg. RGB channels that undergo steerable pyramid independently)
    
    Parameters
    ----------
    pyr_coeffs : `dict`
                pyramid coefficients in the standard dictionary format as
                specified in Steerable_Pyramid_Freq
    image_index : `int` in [0, batch_size]
                  index of the image you would like to select from the batch
                  of coefficients
    channel: `int`
             index of channel to select for image display
             for grayscale images this will be 0.

    Examples
    --------
        >>> size = 32
        >>> signal = torch.randn(2, 3, size,size) # three images, each with three channels
        >>> SPF = po.simul.Steerable_Pyramid_Freq((size, size), order=3, height=3, is_complex=True, downsample=False)
        >>> pyr = SPF(signal)
        >>> pt.pyrshow(po.convert_pyrshow(pyr, 1, 2), is_complex=True, plot_complex='polar', zoom=3);
    """

    pyr_coeffvis = pyr_coeffs.copy()
    for k in pyr_coeffvis.keys():
        im = pyr_coeffvis[k]
        # imag and real component exist
        if im.shape[-1] == 2:
            im = torch_complex_to_numpy(im)
        else:
            im = to_numpy(im)
        pyr_coeffvis[k] = im[image_index, channel, ...]

    return pyr_coeffvis


def clean_up_axes(ax, ylim=None, spines_to_remove=['top', 'right', 'bottom'],
                  axes_to_remove=['x']):
    r"""Clean up an axis, as desired when making a stem plot of the representation

    Parameters
    ----------
    ax : `matplotlib.pyplot.axis`
        The axis to clean up.
    ylim : `tuple`, False, or None
        If a tuple, the y-limits to use for this plot. If None, we use the default, slightly adjusted so that the
        minimum is 0. If False, we do nothing.
    spines_to_remove : `list`
        Some combination of 'top', 'right', 'bottom', and 'left'. The spines we remove from the axis.
    axes_to_remove : `list`
        Some combination of 'x', 'y'. The axes to set as invisible.

    Returns
    -------
    ax : matplotlib.pyplot.axis
        The cleaned-up axis

    """
    if spines_to_remove is None:
        spines_to_remove = ['top', 'right', 'bottom']
    if axes_to_remove is None:
        axes_to_remove = ['x']

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
    stem_container : `matplotlib.container.StemContainer`
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
    stem_container : `matplotlib.container.StemContainer`
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
    axes : `list`
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
    anim : `matplotlib.animation.FuncAnimation`
        The animation object to convert to HTML
    """

    # to_html5_video will call savefig with a dpi kwarg, so our
    # custom figure class will raise a warning. we don't want to
    # worry people, so we go ahead and suppress it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return HTML(anim.to_html5_video())


def clean_stem_plot(data, ax=None, title='', ylim=None, xvals=None, **kwargs):
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
    data : `np.ndarray`
        The data to plot (as a stem plot)
    ax : `matplotlib.pyplot.axis` or `None`, optional
        The axis to plot the data on. If None, we plot on the current
        axis
    title : str or None, optional
        The title to put on the axis if not None. If None, we don't call
        ``ax.set_title`` (useful if you want to avoid changing the title
        on an existing plot)
    ylim : tuple or None, optional
        If not None, the y-limits to use for this plot. If None, we
        use the default, slightly adjusted so that the minimum is 0
    xvals : `tuple` or `None`, optional
        A 2-tuple of lists, containing the start (``xvals[0]``) and stop
        (``xvals[1]``) x values for plotting. If None, we use the
        default stem plot behavior.
    kwargs :
        passed to ax.stem

    Returns
    -------
    ax : `matplotlib.pyplot.axis`
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
    ax.stem(data, basefmt=basefmt, use_line_collection=True, **kwargs)
    ax = clean_up_axes(ax, ylim, ['top', 'right', 'bottom'])
    if title is not None:
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
            data_check = 1
            artists = axes.containers
        elif len(axes.images) > 0:
            # images are weird, so don't check them like this
            data_check = None
            artists = axes.images
        elif len(axes.lines) > 0:
            data_check = 1
            artists = axes.lines
        elif len(axes.collections) > 0:
            data_check = 2
            artists = axes.collections
        if isinstance(data, dict):
            artists = {ax.get_label(): ax for ax in artists}
        else:
            if data_check == 1 and data.shape[1] != len(artists):
                raise Exception(f"data has {data.shape[1]} things to plot, but "
                                f"your axis contains {len(artists)} plotting artists, "
                                "so unsure how to continue! Pass data as a dictionary"
                                " with keys corresponding to the labels of the artists"
                                " to update to resolve this.")
            elif data_check == 2 and data.ndim > 2 and data.shape[-3] != len(artists):
                raise Exception(f"data has {data.shape[-3]} things to plot, but "
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
                data_check = 1
                artists.extend(ax.containers)
            elif len(ax.images) == 1:
                data_check = None
                artists.extend(ax.images)
            elif len(ax.lines) == 1:
                artists.extend(ax.lines)
                data_check = 1
            elif len(ax.collections) == 1:
                artists.extend(ax.collections)
                data_check = 2
        if isinstance(data, dict):
            if len(data.keys()) != len(artists):
                raise Exception(f"data has {len(data.keys())} things to plot, but "
                                f"you passed {len(axes)} axes , so unsure how "
                                "to continue!")
            artists = {k: a for k, a in zip(data.keys(), artists)}
        else:
            if data_check == 1 and data.shape[1] != len(artists):
                raise Exception(f"data has {data.shape[1]} things to plot, but "
                                f"you passed {len(axes)} axes , so unsure how "
                                "to continue!")
            if data_check == 2 and data.ndim > 2 and data.shape[-3] != len(artists):
                raise Exception(f"data has {data.shape[-3]} things to plot, but "
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

    We can update stem plots, lines (as returned by ``plt.plot``), scatter
    plots, or images. All artists-to-update do not need to be of the same type.

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
    axes : `list`
        A list of axes to update. We assume that these are the axes
        created by ``plot_representation`` and so contain stem plots
        in the correct order.
    data : `torch.Tensor` or `dict`
        The new data to plot.
    model : `torch.nn.Module` or `None`, optional
        A differentiable model that tells us how to plot ``data``. See
        above for behavior if ``None``.
    batch_idx : int, optional
        Which index to take from the batch dimension

    Returns
    -------
    artists : `list`
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
                    # then it's a stemplot
                    sc = update_stem(art, d)
                    artists.extend([sc.markerline, sc.stemlines])
            elif d.ndim == 2:
                try:
                    # then it's an image
                    art.set_data(d)
                    artists.append(art)
                except AttributeError:
                    # then it's a scatterplot
                    art.set_offsets(d)
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

    - If ``model`` is ``None``, we fall-back to a type of plot based on the
      shape of ``data``. If it looks image-like, we'll use ``plenoptic.imshow``
      and if it looks vector-like, we'll use ``plenoptic.clean_stem_plot``. If
      it's a dictionary, we'll assume each key, value pair gives the title and
      data to plot on a separate sub-plot.

    - If ``data`` is ``None``, we can only do something if
      ``model.plot_representation`` has some default behavior when
      ``data=None``; this is probably to plot its own ``representation``
      attribute. Thus, this will raise an Exception if both ``model`` and
      ``data`` are ``None``, because we have no idea what to plot then.

    - If ``ax`` is ``None``, we create a one-subplot figure using ``figsize``.
      If ``ax`` is not ``None``, we therefore ignore ``figsize``.

    - If ``ylim`` is ``None``, we call ``rescale_ylim``, which sets the axes'
      y-limits to be ``(-y_max, y_max)``, where ``y_max=np.abs(data).max()``.
      If it's ``False``, we do nothing.

    Parameters
    ----------
    model : `torch.nn.Module` or None, optional
        A differentiable model that tells us how to plot ``data``. See
        above for behavior if ``None``.
    data : `array_like`, `dict`, or `None`, optional
        The data to plot. See above for behavior if ``None``.
    ax : matplotlib.pyplot.axis or None, optional
        The axis to plot on. See above for behavior if ``None``.
    figsize : `tuple`, optional
        The size of the figure to create. Ignored if ``ax`` is not
        ``None``.
    ylim : `tuple`, `None`, or `False`, optional
        If not None, the y-limits to use for this plot. See above for
        behavior if ``None``. If False, we do nothing.
    batch_idx : `int`, optional
        Which index to take from the batch dimension (the first one)
    title : `str`, optional
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
                imshow(v, batch_idx=batch_idx, title=title, ax=ax, vrange='indep0')
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
