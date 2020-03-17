"""functions related to sampling

handful of functions here, related to sampling and checking whether
you're sampling correctly, in order to avoid aliasing

when doing something like strided convolution or using the pooling
windows in this package, you want to make sure you're sampling the image
appropriately, in order to avoid aliasing. this file contains some
functions to help you with that, see the Sampling_and_Aliasing notebook
for some examples

"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from .pooling import gaussian
from ...tools.data import to_numpy


def check_sampling(val_sampling=.5, pix_sampling=None, func=gaussian, x=torch.linspace(-5, 5, 101),
                   **func_kwargs):
    r"""check how sampling relates to interpolation quality

    Given a function, a domain, and how to sample that domain, this
    function will use linear algebra (``np.linalg.lstsq``) to determine
    how to interpolate the function so that it's centered on each
    pixel. You can then use functions like ``plot_coeffs`` and
    ``create_movie`` to see the quality of this interpolation

    The idea here is to take a function (for example,
    ``po.simul.pooling.gaussian``) and say that we have this function
    defined at, e.g., every 10 pixels on the array ``linspace(-5, 5,
    101)``. We want to answer then, the question of how well we can
    interpolate to all the intermediate functions, that is, the
    functions centered on each pixel in the array.

    You can either specify the spacing in pixels (``pix_sampling``) xor
    in x values (``val_sampling``), but exactly one of them must be set.

    Your function can either be a torch or numpy function, but ``x``
    must be the appropriate type, we will not cast it for you.

    Parameters
    ----------
    val_sampling : float or None, optional.
        If float, how far apart (in x-values) each sampled function
        should be. This doesn't have to align perfectly with the pixels,
        but should be close. If None, we use ``pix_sampling`` instead.
    pix_sampling : int or None, optional
        If int, how far apart (in pixels) each sampled function should
        be. If None, we use ``val_sampling`` instead.
    func : callable, optional
        the function to check interpolation for. must take ``x`` as its
        first input, all additional kwargs can be specified in
        ``func_kwargs``
    x : torch.tensor or np.array, optional
        the 1d tensor/array to evaluate ``func`` on.
    func_kwargs :
        additional kwargs to pass to ``func``

    Returns
    -------
    sampled : np.array
        the array of sampled functions. will have shape ``(len(x),
        ceil(len(x)/pix_sampling))``
    full : np.array
        the array of functions centered at each pixel. will have shape
        ``(len(x), len(x))``
    interpolated : np.array
        the array of functions interpolated to each pixel. will have
        shape ``(len(x), len(x))``
    coeffs : np.array
        the array of coefficients to transform ``sampled`` to
        ``full``. This has been transposed from the array returned by
        ``np.linalg.lstsq`` and thus will have the same shape as
        ``sampled`` (this is to make it easier to restrict which coeffs
        to look at, since they'll be more easily indexed along first
        dimension)
    residuals : np.array
        the errors for each interpolation, will have shape ``len(x)``

    """
    if val_sampling is not None:
        if pix_sampling is not None:
            raise Exception("One of val_sampling or pix_sampling must be None!")
        # this will get us the closest value, if there's no exactly
        # correct one.
        pix_sampling = np.argmin(abs((x+val_sampling)[0] - x))
        if pix_sampling == 0 or pix_sampling == (len(x)-1):
            # the above works if x is increasing. if it's decreasing,
            # then pix_sampling will be one of the extremal values, and
            # we need to try the following
            pix_sampling = np.argmin(abs((x-val_sampling)[0] - x))
    try:
        X = x.unsqueeze(1) + x[::pix_sampling]
        sampled = to_numpy(func(X, **func_kwargs))
        full_X = x.unsqueeze(1) + x
        full = to_numpy(func(full_X, **func_kwargs))
    except AttributeError:
        # numpy arrays don't have unsqueeze, so we use this `[:, None]`
        # syntax to get the same outcome
        X = x[:, None] + x[::pix_sampling]
        sampled = func(X, **func_kwargs)
        full_X = x.unsqueeze(1) + x
        full = func(full_X, func_kwargs)
    coeffs, residuals, rank, s = np.linalg.lstsq(sampled, full, rcond=None)
    interpolated = np.matmul(sampled, coeffs)
    return sampled, full, interpolated, coeffs.T, residuals


def plot_coeffs(coeffs, ncols=5, ax_size=(5, 5)):
    r"""plot interpolatoin coefficients

    Simple functoin to plot a bunch of interpolation coefficients on the
    same figure as stem plots

    Parameters
    ----------
    coeffs : np.array
        the array of coefficients to transform ``sampled`` to
        ``full``. In order to show fewer coefficients (because they're
        so many), index along the first dimension (e.g., ``coeffs[:10]``
        to view first 10)
    ncols : int, optional
        the number of columns to create in the plot
    ax_size : tuple, optional
        the size of each subplots axis

    Returns
    -------
    fig : plt.Figure
        the figure containing the plot
    """
    nrows = int(np.ceil(coeffs.shape[0] / ncols))
    ylim = max(abs(coeffs.max()), abs(coeffs.min()))
    ylim += ylim/10
    fig, axes = plt.subplots(nrows, ncols, figsize=[i*j for i, j in zip(ax_size, [ncols, nrows])])
    for i, ax in enumerate(axes.flatten()):
        ax.stem(coeffs[i], use_line_collection=True)
        ax.set_ylim((-ylim, ylim))
    return fig


def interpolation_plot(interpolated, residuals, pix=0, val=None, x=np.linspace(-5, 5, 101),
                       full=None):
    r"""create plot showing interpolation results at specified pixel or value

    We have two subplots: the interpolation (with optional actual
    values) and the residuals

    Either ``pix`` or ``val`` must be set, and the other must be
    ``None``. They specify which interpolated function to display

    Parameters
    ----------
    interpolated : np.array
        the array of functions interpolated to each pixel
    residuals : np.array
        the errors for each interpolation
    pix : int or None, optional
        we plot the interpolated function centered at this pixel
    val : float or None, optional
        we plot the interpolated function centered at this x-value
    x : torch.tensor or np.array, optional
        the 1d tensor/array passed to ``check_sampling()``. the default
        here is the default there. plotted on x-axis
    full : np.array, optional
        the array of functions centered at each pixel. If None, won't
        plot. If not None, will plot as dashed line behind the
        interpolation for comparison
    framerate : int, optional
        How many frames a second to display.

    Returns
    -------
    fig  : plt.Figure
        figure containing the plot

    """
    if val is not None:
        if pix is not None:
            raise Exception("One of val_sampling or pix_sampling must be None!")
        # this will get us the closest value, if there's no exactly
        # correct one.
        pix = np.argmin(abs(x-val))
    x = to_numpy(x)
    ylim = [interpolated.min(), interpolated.max()]
    ylim = [ylim[0] - np.diff(ylim)/10, ylim[1] + np.diff(ylim)/10]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_ylim(ylim)
    axes[0].plot(x, interpolated[:, pix], label='interpolation')
    if full is not None:
        axes[0].plot(x, full[:, pix], '--', zorder=0, label='actual')
        axes[0].legend()
    axes[1].stem(x, residuals, use_line_collection=True)
    axes[1].scatter(x[pix], residuals[pix], c='r', zorder=10)
    axes[0].set_title("Interpolated function centered at highlighted pixel")
    axes[1].set_title("Error for interpolation centered at highlighted pixel")
    return fig


def create_movie(interpolated, residuals, x=np.linspace(-5, 5, 101), full=None, framerate=10):
    r"""create movie showing the interpolation results

    We create a simple movie to show this in action. we have two
    subplots: the interpolation (with optional actual values) and the
    residuals.

    the more finely sampled your ``x`` was when calling
    ``check_sampling()`` (and thus the larger your ``interpolated`` and
    ``full`` arrays), the longer this will take. Calling this function
    will not take too long, but displaying or saving the returned
    animation will.

    Parameters
    ----------
    interpolated : np.array
        the array of functions interpolated to each pixel
    residuals : np.array
        the errors for each interpolation
    x : torch.tensor or np.array, optional
        the 1d tensor/array passed to ``check_sampling()``. the default
        here is the default there. plotted on x-axis
    full : np.array, optional
        the array of functions centered at each pixel. If None, won't
        plot. If not None, will plot as dashed line behind the
        interpolation for comparison
    framerate : int, optional
        How many frames a second to display.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object. In order to view, must convert to HTML
        (call ``po.convert_anim_to_html(anim)``) or save (call
        ``anim.save(movie.mp4)``, must have ``ffmpeg`` installed).

    """
    x = to_numpy(x)
    fig = interpolation_plot(interpolated, residuals, x=x, full=full)
    if full is not None:
        full_line = fig.axes[0].lines[1]
    interp_line = fig.axes[0].lines[0]
    scat = fig.axes[1].collections[1]

    def movie_plot(i):
        interp_line.set_data(x, interpolated[:, i])
        scat.set_offsets((x[i], residuals[i]))
        artists = [interp_line, scat]
        if full is not None:
            full_line.set_data(x, full[:, i])
            artists.append(full_line)
        return artists

    plt.close(fig)
    return animation.FuncAnimation(fig, movie_plot, frames=len(interpolated), blit=True,
                                   interval=1000./framerate, repeat=False)
