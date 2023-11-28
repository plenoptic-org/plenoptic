#!/usr/bin/env python3

import einops
import matplotlib.pyplot as plt
import plenoptic as po
import matplotlib as mpl
from matplotlib import animation

def animate_schematic(metamer, fig, axes_idx, framerate=10,
                      pixel_value_subsample=.01):
    """Animate synthesis_schematic figure.

    Parameters
    ----------
    metamer : po.Metamer
        The Metamer object to grab data from
    iteration : int or None, optional
        Which iteration to display. If None, we show the most recent one.
        Negative values are also allowed.
    pixel_value_subsample : float, optional
        Float between 0 and 1 giving the percentage of image pixels to plot.
        Done to make visualization clearer.
    kwargs :
        passed to metamer.plot_synthesis_status

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object. In order to view, must convert to HTML
        or save.
    """
    def movie_plot(i):
        artists = []
        artists.extend(po.tools.update_plot(fig.axes[axes_idx['display_metamer']],
                                            data=metamer.saved_metamer[i]))
        img_vals = []
        for img in [metamer.image, metamer.saved_metamer[i]]:
            vals = img.flatten()[::int(1/pixel_value_subsample)]
            img_vals.append(vals.detach())
        artists.extend(po.tools.update_plot(fig.axes[axes_idx['plot_pixel_values']],
                                            data=einops.rearrange(img_vals, 'a b -> 1 1 b a')))
        rep_vals = []
        for vals in [metamer.target_representation, metamer.model(metamer.saved_metamer[i])]:
            rep_vals.append(vals.flatten().detach())
        artists.extend(po.tools.update_plot(fig.axes[axes_idx['plot_representation_comparison']],
                                            data=einops.rearrange(rep_vals, 'a b -> 1 1 b a')))
        return artists

    anim = animation.FuncAnimation(fig, movie_plot, frames=len(metamer.saved_metamer),
                                   blit=True, interval=1000./framerate, repeat=False)
    plt.close(fig)
    return anim


def synthesis_schematic(metamer, iteration=0, pixel_value_subsample=.01):
    """Create schematic of synthesis, for animating.

    WARNING: Currently, only works with images of size (256, 256), will need a
    small amount of tweaking to work with differently sized images. (And may
    never look quite as good with them)

    Parameters
    ----------
    metamer : po.Metamer
        The Metamer object to grab data from
    iteration : int or None, optional
        Which iteration to display. If None, we show the most recent one.
        Negative values are also allowed.
    pixel_value_subsample : float, optional
        Float between 0 and 1 giving the percentage of image pixels to plot.
        Done to make visualization clearer.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot
    axes_idx : dict
        dictionary specifying which plot is where, for use with animate()

    Notes
    -----
    To successfully `animate_schematic`, pass the same metamer object,
    as well as the fig and axes_idx as returned by this function, and the same
    value of `pixel_value_subsample`

    """
    if list(metamer.image.shape[:2]) != [1, 1]:
        raise ValueError("Target image must have single channel and batch dim!")
    # arrangement was all made with 72 dpi
    mpl.rc('figure', dpi=72)
    mpl.rc('axes', titlesize=25)
    mpl.rc('axes.spines', right=False, top=False)
    image_shape = metamer.image.shape
    figsize = ((1.5+(image_shape[-1] / image_shape[-2])) * 4.5 + 2.5, 3*4.5+1)
    fig = plt.figure(figsize=figsize)
    gs = mpl.gridspec.GridSpec(3, 10, figure=fig, hspace=.25, bottom=.05,
                               top=.95, left=.05, right=.95)
    fig.add_subplot(gs[0, 0:3], aspect=1)
    fig.add_subplot(gs[0, 4:7], aspect=1)
    fig.add_subplot(gs[1, 1:4], aspect=1)
    fig.add_subplot(gs[1, 6:9], aspect=1)
    fig.add_subplot(gs[2, 0:3], aspect=1)
    fig.add_subplot(gs[2, 4:7], aspect=1)
    axes_idx = {'display_metamer': 0, 'plot_pixel_values': 2, 'plot_representation_comparison': 3,
                'misc': [1] + list(range(4, len(fig.axes)))}
    po.imshow(metamer.image, ax=fig.axes[4], title=None)
    for i in [0] + axes_idx['misc']:
        fig.axes[i].xaxis.set_visible(False)
        fig.axes[i].yaxis.set_visible(False)
        fig.axes[i].set_frame_on(False)
    model_axes = [5]
    model_axes += [1]
    arrowkwargs = {'xycoords': 'axes fraction', 'textcoords': 'axes fraction',
                   'ha': 'center', 'va': 'center'}
    arrowprops = {'color': '0', 'connectionstyle': 'arc3', 'arrowstyle': '->',
                  'lw': 3}
    for i in model_axes:
        p = mpl.patches.Rectangle((0, .25), .5, .5, fill=False)
        p.set_transform(fig.axes[i].transAxes)
        fig.axes[i].add_patch(p)
        fig.axes[i].text(.25, .5, 'M', {'size': 50}, ha='center', va='center',
                         transform=fig.axes[i].transAxes)
        fig.axes[i].annotate('', (0, .5), (-.4, .5), arrowprops=arrowprops,
                             **arrowkwargs)
    arrowprops['connectionstyle'] += ',rad=.3'
    fig.axes[5].annotate('', (1.2, 1.25), (.53, .5), arrowprops=arrowprops,
                         **arrowkwargs)
    arrowprops['connectionstyle'] = 'arc3,rad=.2'
    fig.axes[1].annotate('', (.6, -.8), (.25, .22), arrowprops=arrowprops,
                         **arrowkwargs)
    arrowprops['connectionstyle'] = 'arc3'
    fig.axes[4].annotate('', (.8, 1.25), (.8, 1.03), arrowprops=arrowprops,
                         **arrowkwargs)
    arrowprops['connectionstyle'] += ',rad=.1'
    fig.axes[0].annotate('', (.25, -.8), (.15, -.03), arrowprops=arrowprops,
                         **arrowkwargs)
    img_vals = []
    for img in [metamer.image, metamer.saved_metamer[iteration]]:
        vals = img.flatten()[::int(1/pixel_value_subsample)]
        img_vals.append(po.to_numpy(vals))
    fig.axes[axes_idx['plot_pixel_values']].scatter(*img_vals)
    rep_vals = []
    for vals in [metamer.target_representation, metamer.model(metamer.saved_metamer[iteration])]:
        rep_vals.append(po.to_numpy(vals.flatten()))
    fig.axes[axes_idx['plot_representation_comparison']].scatter(*rep_vals)
    po.imshow(metamer.saved_metamer[iteration], ax=fig.axes[axes_idx['display_metamer']], title=None)
    fig.axes[axes_idx['plot_pixel_values']].set(xlabel='', ylabel='', title='Pixel values', xticks=[], yticks=[])
    fig.axes[axes_idx['plot_representation_comparison']].set(xlabel='', ylabel='', yticks=[], xticks=[], title="Model representation",
                                                             ylim=fig.axes[axes_idx['plot_representation_comparison']].get_xlim())
    return fig, axes_idx
