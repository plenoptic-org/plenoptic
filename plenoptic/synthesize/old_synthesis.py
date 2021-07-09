#!/usr/bin/env python3

"""Old way of handling abstract synthesis super-class
"""
import abc
import re
import torch
from torch import optim
import numpy as np
import warnings
from ..tools.data import to_numpy, _find_min_int
from ..tools.optim import l2_norm, relative_MSE
import matplotlib.pyplot as plt
from ..tools.display import rescale_ylim, plot_representation, update_plot, imshow
from matplotlib import animation
from ..simulate.models.naive import Identity
from tqdm.auto import tqdm
import dill
from .synthesis import Synthesis


class OldSynthesis(Synthesis):
    r"""Abstract super-class for synthesis methods.

    We're in the process of refactoring synthesis classes, so this file is
    temporary while we do so (in particular, we want to overhaul the plotting
    functionality, but that seems like too much to do at once, so the plotting
    functions will remain here for the time being).

    All synthesis methods share a variety of similarities and thus need
    to have similar methods. Some of these can be implemented here and
    simply inherited, some of them will need to be different for each
    sub-class and thus are marked as abstract methods here

    Parameters
    ----------
    base_signal : torch.Tensor or array_like
        A 4d tensor, this is the image whose representation we wish to
        match. If this is not a tensor, we try to cast it as one.
    model : torch.nn.Module or function
        The visual model or metric to synthesize with. See `MAD_Competition`
        for details.
    loss_function : 'l2_norm' or 'relative_MSE' or callable or None, optional
        the loss function to use to compare the representations of the
        models in order to determine their loss. Only used for the
        Module models, ignored otherwise. If None, we use the default:
        the l2 norm. See `MAD_Competition` notebook for more
        details
    model_kwargs : dict
        if model is a function (that is, you're using a metric instead
        of a model), then there might be additional arguments you want
        to pass it at run-time. Note that this means they will be passed
        on every call.

    """

    def representation_error(self, iteration=None, **kwargs):
        r"""Get the representation error

        This is (synthesized_representation - base_representation). If
        ``iteration`` is not None, we use
        ``self.saved_representation[iteration]`` for
        synthesized_representation

        Any kwargs are passed through to self.analyze when computing the
        synthesized/base representation.

        Parameters
        ----------
        iteration: int or None, optional
            Which iteration to create the representation ratio for. If
            None, we use the current ``synthesized_representation``

        Returns
        -------
        torch.Tensor

        """
        if self._rep_warning:
            warnings.warn("Since at least one of your models is a metric, its representation_error"
                          " will be meaningless -- it will just show the pixel-by-pixel difference"
                          ". (Your loss is still meaningful, however, since it's the actual "
                          "metric)")
        if iteration is not None:
            synthesized_rep = self.saved_representation[iteration].to(self.base_representation.device)
        else:
            synthesized_rep = self.analyze(self.synthesized_signal, **kwargs)
        try:
            rep_error = synthesized_rep - self.base_representation
        except RuntimeError:
            # try to use the last scale (if the above failed, it's
            # because they were different shapes), but only if the user
            # didn't give us another scale to use
            if 'scales' not in kwargs.keys():
                kwargs['scales'] = [self.scales[-1]]
            rep_error = synthesized_rep - self.analyze(self.base_signal, **kwargs)
        return rep_error

    def plot_representation_error(self, batch_idx=0, iteration=None, figsize=(5, 5), ylim=None,
                                  ax=None, title=None, as_rgb=False):
        r"""Plot distance ratio showing how close we are to convergence.

        We plot ``self.representation_error(iteration)``

        The goal is to use the model's ``plot_representation``
        method. However, in order for this to work, it needs to not only
        have that method, but a way to make a 'mock copy', a separate
        model that has the same initialization parameters, but whose
        representation we can set. For the VentralStream models, we can
        do this using their ``state_dict_reduced`` attribute. If we can't
        do this, then we'll fall back onto using ``plt.plot``

        In order for this to work, we also count on
        ``plot_representation`` to return the figure and the axes it
        modified (axes should be a list)

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension
        iteration: int or None, optional
            Which iteration to create the representation ratio for. If
            None, we use the current ``synthesized_representation``
        figsize : tuple, optional
            The size of the figure to create
        ylim : tuple or None, optional
            If not None, the y-limits to use for this plot. If None, we
            scale the y-limits so that it's symmetric about 0 with a
            limit of ``np.abs(representation_error).max()``
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        title : str, optional
            The title to put above this axis. If you want no title, pass
            the empty string (``''``)
        as_rgb : bool, optional
            The representation can be image-like with multiple channels, and we
            have no way to determine whether it should be represented as an RGB
            image or not, so the user must set this flag to tell us. It will be
            ignored if the representation doesn't look image-like or if the
            model has its own plot_representation_error() method. Else, it will
            be passed to `po.imshow()`, see that methods docstring for details.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot

        """
        representation_error = self.representation_error(iteration=iteration)
        return plot_representation(self.model, representation_error, ax, figsize, ylim,
                                   batch_idx, title, as_rgb)

    def plot_loss(self, iteration=None, figsize=(5, 5), ax=None, title='Loss', **kwargs):
        """Plot the synthesis loss.

        We plot ``self.loss`` over all iterations. We also plot a red
        dot at ``iteration``, to highlight the loss there. If
        ``iteration=None``, then the dot will be at the final iteration.

        Parameters
        ----------
        iteration : int or None, optional
            Which iteration to display. If None, the default, we show
            the most recent one. Negative values are also allowed.
        figsize : tuple, optional
            The size of the figure to create. Ignored if ax is not None
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        kwargs :
            passed to plt.semilogy

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        if iteration is None:
            loss_idx = len(self.loss) - 1
        else:
            if iteration < 0:
                # in order to get the x-value of the dot to line up,
                # need to use this work-around
                loss_idx = len(self.loss) + iteration
            else:
                loss_idx = iteration
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        ax.semilogy(self.loss, **kwargs)
        try:
            ax.scatter(loss_idx, self.loss[loss_idx], c='r')
        except IndexError:
            # then there's no loss here
            pass
        ax.set_title(title)
        return fig

    def plot_synthesized_image(self, batch_idx=0, channel_idx=None, iteration=None, title=None,
                               figsize=(5, 5), ax=None, imshow_zoom=None, vrange=(0, 1)):
        """Show the synthesized image.

        You can specify what iteration to view by using the ``iteration`` arg.
        The default, ``None``, shows the final one.

        We use ``plenoptic.imshow`` to display the synthesized image and
        attempt to automatically find the most reasonable zoom value. You can
        override this value using the imshow_zoom arg, but remember that
        ``plenoptic.imshow`` is opinionated about the size of the resulting
        image and will throw an Exception if the axis created is not big enough
        for the selected zoom.

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we assume
            image is RGB(A) and show all channels.
        iteration : int or None, optional
            Which iteration to display. If None, the default, we show
            the most recent one. Negative values are also allowed.
        title : str or None, optional
            The title for this subplot. If None, will use the class's
            name (e.g., Metamer, MADCompetition). If you want no title,
            set this equal to the empty str (``''``)
        figsize : tuple, optional
            The size of the figure to create. Ignored if ax is not None
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        imshow_zoom : None or float, optional
            How much to zoom in / enlarge the synthesized image, the ratio
            of display pixels to image pixels. If None (the default), we
            attempt to find the best value ourselves, but we cannot find a
            value <1. Else, if >1, must be an integer. If <1, must be 1/d where
            d is a a divisor of the size of the largest image.
        vrange : tuple or str, optional
            The vrange option to pass to ``pyrtools.imshow``. See that
            function for details

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        if iteration is None:
            image = self.synthesized_signal
        else:
            image = self.saved_signal[iteration]
        if batch_idx is None:
            raise Exception("batch_idx must be an integer!")
        # we're only plotting one image here, so if the user wants multiple
        # channels, they must be RGB
        if channel_idx is None and image.shape[1] > 1:
            as_rgb = True
        else:
            as_rgb = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        if imshow_zoom is None:
            # image.shape[-2] is the height of the image
            imshow_zoom = ax.bbox.height // image.shape[-2]
            if imshow_zoom == 0:
                raise Exception("imshow_zoom would be 0, cannot display synthesized image! Enlarge"
                                " your figure")
        if title is None:
            title = self.__class__.__name__
        fig = imshow(image, ax=ax, title=title, zoom=imshow_zoom,
                     batch_idx=batch_idx, channel_idx=channel_idx,
                     vrange=vrange, as_rgb=as_rgb)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return fig

    def plot_image_hist(self, batch_idx=0, channel_idx=None, iteration=None, figsize=(5, 5),
                        ylim=None, ax=None, **kwargs):
        r"""Plot histogram of target and matched image.

        As a way to check the distributions of pixel intensities and see
        if there's any values outside the allowed range

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we use all
            channels (assumed use-case is RGB(A) images).
        iteration : int or None, optional
            Which iteration to display. If None, the default, we show
            the most recent one. Negative values are also allowed.
        figsize : tuple, optional
            The size of the figure to create. Ignored if ax is not None
        ylim : tuple or None, optional
            if tuple, the ylimit to set for this axis. If None, we leave
            it untouched
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        kwargs :
            passed to plt.hist

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        def _freedman_diaconis_bins(a):
            """Calculate number of hist bins using Freedman-Diaconis rule. copied from seaborn"""
            # From https://stats.stackexchange.com/questions/798/
            a = np.asarray(a)
            iqr = np.diff(np.percentile(a, [.25, .75]))[0]
            if len(a) < 2:
                return 1
            h = 2 * iqr / (len(a) ** (1 / 3))
            # fall back to sqrt(a) bins if iqr is 0
            if h == 0:
                return int(np.sqrt(a.size))
            else:
                return int(np.ceil((a.max() - a.min()) / h))

        kwargs.setdefault('alpha', .4)
        if iteration is None:
            image = self.synthesized_signal[batch_idx]
        else:
            image = self.saved_signal[iteration, batch_idx]
        base_signal = self.base_signal[batch_idx]
        if channel_idx is not None:
            image = image[channel_idx]
            base_signal = base_signal[channel_idx]
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        image = to_numpy(image).flatten()
        base_signal = to_numpy(base_signal).flatten()
        ax.hist(image, bins=min(_freedman_diaconis_bins(image), 50),
                label='synthesized image', **kwargs)
        ax.hist(base_signal, bins=min(_freedman_diaconis_bins(image), 50),
                label='base image', **kwargs)
        ax.legend()
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title("Histogram of pixel values")
        return fig

    def _grab_value_for_comparison(self, value, batch_idx=0, channel_idx=None,
                                   iteration=None, scatter_subsample=1,
                                   **kwargs):
        """Grab and shape values for comparison plot.

        This grabs the appropriate batch_idx, channel_idx, and iteration from
        the saved representation or signal, respectively, and subsamples it if
        necessary.

        We then concatenate thema long the last dimension.

        Parameters
        ----------
        value : {'representation', 'signal'}
            Whether to compare the representations or signals
        batch_idx : int, optional
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we use all
            channels (assumed use-case is RGB(A) image).
        iteration : int or None, optional
            Which iteration to display. If None, the default, we show
            the most recent one. Negative values are also allowed.
        scatter_subsample : float, optional
            What percentage of points to plot. If less than 1, will select that
            proportion of the points to plot. Done to make visualization
            clearer. Note we don't do this randomly (so that animate looks
            reasonable).
        kwargs :
            passed to self.analyze

        Returns
        -------
        plot_vals : torch.Tensor
            4d tensor containing the base and synthesized value (indexed along
            last dimension). First two dims are dummy dimensions and will
            always have value 1 (update_plot needs them)

        """
        if value == 'representation':
            if iteration is not None:
                synthesized_val = self.saved_representation[iteration]
            else:
                synthesized_val = self.analyze(self.synthesized_signal, **kwargs)
            base_val = self.base_representation
        elif value == 'signal':
            if iteration is not None:
                synthesized_val = self.saved_signal[iteration]
            else:
                synthesized_val = self.synthesized_signal
            base_val = self.base_signal
        else:
            raise Exception(f"Don't know how to handle value {value}!")
        # if this is 4d, this will convert it to 3d (if it's 3d, nothing
        # changes)
        base_val = base_val.flatten(2, -1).cpu()
        synthesized_val = synthesized_val.flatten(2, -1).cpu()
        plot_vals = torch.stack((base_val, synthesized_val), -1)
        if scatter_subsample < 1:
            plot_vals = plot_vals[:, :, ::int(1/scatter_subsample)]
        plot_vals = plot_vals[batch_idx]
        if channel_idx is not None:
            plot_vals = plot_vals[channel_idx]
        else:
            plot_vals = plot_vals.flatten(0, 1)
        return plot_vals.unsqueeze(0).unsqueeze(0)


    def plot_value_comparison(self, value='representation', batch_idx=0,
                              channel_idx=None, iteration=None, figsize=(5, 5),
                              ax=None, func='scatter', hist2d_nbins=21,
                              hist2d_cmap='Blues', scatter_subsample=1,
                              **kwargs):
        """Plot comparison of base vs. synthesized representation or signal.

        Plotting representation is another way of visualizing the
        representation error, while plotting signal is similar to
        plot_image_hist, but allows you to see whether there's any pattern of
        individual correspondence.

        Parameters
        ----------
        value : {'representation', 'signal'}
            Whether to compare the representations or signals
        batch_idx : int, optional
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we use all
            channels (assumed use-case is RGB(A) image).
        iteration : int or None, optional
            Which iteration to display. If None, the default, we show
            the most recent one. Negative values are also allowed.
        figsize : tuple, optional
            The size of the figure to create. Ignored if ax is not None
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        func : {'scatter', 'hist2d'}, optional
            Whether to use a scatter plot or 2d histogram to plot this
            comparison. When there are many values (as often happens when
            plotting signal), then hist2d will be clearer
        hist2d_nbins: int, optional
            Number of bins between 0 and 1 to use for hist2d
        hist2d_cmap : str or matplotlib colormap, optional
            Colormap to use for hist2d
        scatter_subsample : float, optional
            What percentage of points to plot. If less than 1, will select that
            proportion of the points to plot. Done to make visualization
            clearer. Note we don't do this randomly (so that animate looks
            reasonable).
        kwargs :
            passed to self.analyze

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        if self._rep_warning and value=='representation':
            warnings.warn("Since at least one of your models is a metric, its representation"
                          " will be meaningless -- it will just show the pixel values"
                          ". (Your loss is still meaningful, however, since it's the actual "
                          "metric)")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'aspect': 1})
        else:
            fig = ax.figure
        plot_vals = to_numpy(self._grab_value_for_comparison(value, batch_idx,
                                                             channel_idx, iteration,
                                                             scatter_subsample,
                                                             **kwargs)).squeeze()
        if func == 'scatter':
            ax.scatter(plot_vals[..., 0], plot_vals[..., 1])
            ax.set(xlim=ax.get_ylim())
        elif func == 'hist2d':
            ax.hist2d(plot_vals[..., 0].flatten(), plot_vals[..., 1].flatten(),
                      bins=np.linspace(0, 1, hist2d_nbins),
                      cmap=hist2d_cmap, cmin=0)
        ax.set(ylabel=f'Synthesized {value}', xlabel=f'Base {value}')
        return fig

    def _setup_synthesis_fig(self, fig, axes_idx, figsize,
                             plot_synthesized_image=True, plot_loss=True,
                             plot_representation_error=True,
                             plot_image_hist=False, plot_rep_comparison=False,
                             plot_signal_comparison=False,
                             synthesized_image_width=1, loss_width=1,
                             representation_error_width=1, image_hist_width=1,
                             rep_comparison_width=1, signal_comparison_width=1):
        """Set up figure for plot_synthesis_status.

        Creates figure with enough axes for the all the plots you want. Will
        also create index in axes_idx for them if you haven't done so already.

        By default, all axes will be on the same row and have the same width.
        If you want them to be on different rows, will need to initialize fig
        yourself and pass that in. For changing width, change the corresponding
        *_width arg, which gives width relative to other axes. So if you want
        the axis for the representation_error plot to be twice as wide as the
        others, set representation_error_width=2.

        Parameters
        ----------
        fig : matplotlib.pyplot.Figure or None
            The figure to plot on or None. If None, we create a new figure
        axes_idx : dict
            Dictionary specifying which axes contains which type of plot,
            allows for more fine-grained control of the resulting figure.
            Probably only helpful if fig is also defined. Possible keys: image,
            loss, rep_error, hist, rep_comp, signal_comp, misc. Values should
            all be ints. If you tell this function to create a plot that doesn't
            have a corresponding key, we find the lowest int that is not
            already in the dict, so if you have axes that you want unchanged,
            place their idx in misc.
        figsize : tuple or None, optional
            The size of the figure to create. It may take a little bit of
            playing around to find a reasonable value. If None, we attempt to
            make our best guess, aiming to have relative width=1 correspond to 5
        plot_synthesized_image : bool, optional
            Whether to include axis for plot of the synthesized image or not.
        plot_loss : bool, optional
            Whether to include axis for plot of the loss or not.
        plot_representation_error : bool, optional
            Whether to include axis for plot of the representation ratio or not.
        plot_image_hist : bool, optional
            Whether to include axis for plot of the histograms of image pixel
            intensities or not.
        plot_rep_comparison : bool, optional
            Whether to include axis for plot of a scatter plot comparing the
            synthesized and base representation.
        plot_signal_comparison : bool, optional
            Whether to include axis for plot of the comparison of the
            synthesized and base signal.
        synthesized_image_width : float, optional
            Relative width of the axis for the synthesized image.
        loss_width : float, optional
            Relative width of the axis for loss plot.
        representation_error_width : float, optional
            Relative width of the axis for representation error plot.
        image_hist_width : float, optional
            Relative width of the axis for image pixel intensities histograms.
        rep_comparison_width : float, optional
            Relative width of the axis for representation comparison plot.
        signal_comparison_width : float, optional
            Relative width of the axis for signal comparison plot.

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure to plot on
        axes : array_like
            List or array of axes contained in fig
        axes_idx : dict
            Dictionary identifying the idx for each plot type

        """
        n_subplots = 0
        axes_idx = axes_idx.copy()
        width_ratios = []
        if plot_synthesized_image:
            n_subplots += 1
            width_ratios.append(synthesized_image_width)
            if 'image' not in axes_idx.keys():
                axes_idx['image'] = _find_min_int(axes_idx.values())
        if plot_loss:
            n_subplots += 1
            width_ratios.append(loss_width)
            if 'loss' not in axes_idx.keys():
                axes_idx['loss'] = _find_min_int(axes_idx.values())
        if plot_representation_error:
            n_subplots += 1
            width_ratios.append(representation_error_width)
            if 'rep_error' not in axes_idx.keys():
                axes_idx['rep_error'] = _find_min_int(axes_idx.values())
        if plot_image_hist:
            n_subplots += 1
            width_ratios.append(image_hist_width)
            if 'hist' not in axes_idx.keys():
                axes_idx['hist'] = _find_min_int(axes_idx.values())
        if plot_rep_comparison:
            n_subplots += 1
            width_ratios.append(rep_comparison_width)
            if 'rep_comp' not in axes_idx.keys():
                axes_idx['rep_comp'] = _find_min_int(axes_idx.values())
        if plot_signal_comparison:
            n_subplots += 1
            width_ratios.append(signal_comparison_width)
            if 'signal_comp' not in axes_idx.keys():
                axes_idx['signal_comp'] = _find_min_int(axes_idx.values())
        if fig is None:
            width_ratios = np.array(width_ratios)
            if figsize is None:
                # we want (5, 5) for each subplot, with a bit of room between
                # each subplot
                figsize = ((width_ratios*5).sum() + width_ratios.sum()-1, 5)
            width_ratios = width_ratios / width_ratios.sum()
            fig, axes = plt.subplots(1, n_subplots, figsize=figsize,
                                     gridspec_kw={'width_ratios': width_ratios})
            if n_subplots == 1:
                axes = [axes]
        else:
            axes = fig.axes
        return fig, axes, axes_idx

    def plot_synthesis_status(self, batch_idx=0, channel_idx=None, iteration=None,
                              figsize=None, ylim=None,
                              plot_synthesized_image=True, plot_loss=True,
                              plot_representation_error=True, imshow_zoom=None,
                              vrange=(0, 1), fig=None, plot_image_hist=False,
                              plot_rep_comparison=False,
                              plot_signal_comparison=False,
                              signal_comp_func='scatter',
                              signal_comp_subsample=.01, axes_idx={},
                              plot_representation_error_as_rgb=False,
                              width_ratios={}):
        r"""Make a plot showing synthesis status.

        We create several subplots to analyze this. By default, we create three
        subplots on a new figure: the first one contains the synthesized image,
        the second contains the loss, and the third contains the representation
        error.

        There are several optional additional plots: image_hist, rep_comparison, and
        signal_comparison:

        - image_hist contains a histogram of pixel values of the synthesized
          and base images.

        - rep_comparison is a scatter plot comparing the representation of the
          synthesized and base images.

        - signal_comparison is a scatter plot (by default) or 2d histogram (if
          signal_comp_func='hist2d') of the pixel values in the synthesized and
          base images.

        All of these (including the default plots) can be toggled using their
        corresponding boolean flags, and can be created separately using the
        method with the same name as the flag.

        You can specify what iteration to view by using the
        ``iteration`` arg. The default, ``None``, shows the final one.

        The loss plot shows the loss as a function of iteration for all
        iterations (even if we didn't save the representation or
        synthesized image at each iteration), with a red dot showing the
        location of the iteration.

        We use ``pyrtools.imshow`` to display the synthesized image and
        attempt to automatically find the most reasonable zoom
        value. You can override this value using the imshow_zoom arg,
        but remember that ``pyrtools.imshow`` is opinionated about the
        size of the resulting image and will throw an Exception if the
        axis created is not big enough for the selected zoom. We
        currently cannot shrink the image, so figsize must be big enough
        to display the image

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we use all
            channels (assumed use-case is RGB(A) image).
        iteration : int or None, optional
            Which iteration to display. If None, the default, we show
            the most recent one. Negative values are also allowed.
        figsize : tuple or None, optional
            The size of the figure to create. It may take a little bit of
            playing around to find a reasonable value. If None, we attempt to
            make our best guess, aiming to have each axis be of size (5, 5)
        ylim : tuple or None, optional
            The ylimit to use for the representation_error plot. We pass
            this value directly to ``self.plot_representation_error``
        plot_synthesized_image : bool, optional
            Whether to plot the synthesized image or not.
        plot_loss : bool, optional
            Whether to plot the loss or not.
        plot_representation_error : bool, optional
            Whether to plot the representation ratio or not.
        imshow_zoom : None or float, optional
            How much to zoom in / enlarge the synthesized image, the ratio
            of display pixels to image pixels. If None (the default), we
            attempt to find the best value ourselves. Else, if >1, must
            be an integer.  If <1, must be 1/d where d is a a divisor of
            the size of the largest image.
        vrange : tuple or str, optional
            The vrange option to pass to ``pyrtools.imshow``. See that
            function for details
        fig : None or matplotlib.pyplot.Figure
            if None, we create a new figure. otherwise we assume this is
            an empty figure that has the appropriate size and number of
            subplots
        plot_image_hist : bool, optional
            Whether to plot the histograms of image pixel intensities or
            not.
        plot_rep_comparison : bool, optional
            Whether to plot a scatter plot comparing the synthesized and base
            representation.
        plot_signal_comparison : bool, optional
            Whether to plot the comparison of the synthesized and base
            signal.
        signal_comp_func : {'scatter', 'hist2d'}, optional
            Whether to use a scatter plot or 2d histogram to plot this signal
            comparison. When there are many values (as often happens), then
            hist2d will be clearer
        signal_comp_subsample : float, optional
            What percentage of signal points to plot. If less than 1, will
            randomly select that proportion of the points to plot. Done to make
            visualization clearer.
        axes_idx : dict, optional
            Dictionary specifying which axes contains which type of plot,
            allows for more fine-grained control of the resulting figure.
            Probably only helpful if fig is also defined. Possible keys: image,
            loss, rep_error, hist, rep_comp, signal_comp, misc. Values should
            all be ints. If you tell this function to create a plot that doesn't
            have a corresponding key, we find the lowest int that is not
            already in the dict, so if you have axes that you want unchanged,
            place their idx in misc.
        plot_representation_error_as_rgb : bool, optional
            The representation can be image-like with multiple channels, and we
            have no way to determine whether it should be represented as an RGB
            image or not, so the user must set this flag to tell us. It will be
            ignored if the representation doesn't look image-like or if the
            model has its own plot_representation_error() method. Else, it will
            be passed to `po.imshow()`, see that methods docstring for details.
        width_ratios : dict, optional
            By defualt, all plots axes will have the same width. To change
            that, specify their relative widths using keys of the format
            "{x}_width", where `x` in ['synthesized_image', 'loss',
            'representation_error', 'image_hist', 'rep_comparison',
            'signal_comparison']

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        if iteration is not None and not self.store_progress:
            raise Exception("synthesis() was run with store_progress=False, "
                            "cannot specify which iteration to plot (only"
                            " last one, with iteration=None)")
        if self.synthesized_signal.ndim not in [3, 4]:
            raise Exception("plot_synthesis_status() expects 3 or 4d data;"
                            "unexpected behavior will result otherwise!")
        fig, axes, axes_idx = self._setup_synthesis_fig(fig, axes_idx, figsize,
                                                        plot_synthesized_image,
                                                        plot_loss,
                                                        plot_representation_error,
                                                        plot_image_hist,
                                                        plot_rep_comparison,
                                                        plot_signal_comparison,
                                                        **width_ratios)

        def check_iterables(i, vals):
            for j in vals:
                try:
                    # then it's an iterable
                    if i in j:
                        return True
                except TypeError:
                    # then it's not an iterable
                    if i == j:
                        return True

        if plot_synthesized_image:
            self.plot_synthesized_image(batch_idx=batch_idx,
                                        channel_idx=channel_idx,
                                        iteration=iteration, title=None,
                                        ax=axes[axes_idx['image']],
                                        imshow_zoom=imshow_zoom, vrange=vrange)
        if plot_loss:
            self.plot_loss(iteration=iteration, ax=axes[axes_idx['loss']])
        if plot_representation_error:
            fig = self.plot_representation_error(batch_idx=batch_idx,
                                                 iteration=iteration,
                                                 ax=axes[axes_idx['rep_error']],
                                                 ylim=ylim,
                                                 as_rgb=plot_representation_error_as_rgb)
            # this can add a bunch of axes, so this will try and figure
            # them out
            new_axes = [i for i, _ in enumerate(fig.axes) if not
                        check_iterables(i, axes_idx.values())] + [axes_idx['rep_error']]
            axes_idx['rep_error'] = new_axes
        if plot_image_hist:
            fig = self.plot_image_hist(batch_idx=batch_idx,
                                       channel_idx=channel_idx,
                                       iteration=iteration,
                                       ax=axes[axes_idx['hist']])
        if plot_rep_comparison:
            fig = self.plot_value_comparison(value='representation',
                                             batch_idx=batch_idx,
                                             channel_idx=channel_idx,
                                             iteration=iteration,
                                             ax=axes[axes_idx['rep_comp']])
            # this can add some axes, so this will try and figure them out
            new_axes = [i for i, _ in enumerate(fig.axes) if not
                        check_iterables(i, axes_idx.values())] + [axes_idx['rep_comp']]
            axes_idx['rep_comp'] = new_axes
        if plot_signal_comparison:
            fig = self.plot_value_comparison(value='signal',
                                             batch_idx=batch_idx,
                                             channel_idx=channel_idx,
                                             iteration=iteration,
                                             ax=axes[axes_idx['signal_comp']],
                                             func=signal_comp_func,
                                             scatter_subsample=signal_comp_subsample)
        self._axes_idx = axes_idx
        return fig

    def animate(self, batch_idx=0, channel_idx=None, figsize=None,
                framerate=10, ylim='rescale', plot_synthesized_image=True,
                plot_loss=True, plot_representation_error=True,
                imshow_zoom=None, plot_data_attr=['loss'], rep_func_kwargs={},
                plot_image_hist=False, plot_rep_comparison=False,
                plot_signal_comparison=False, fig=None,
                signal_comp_func='scatter', signal_comp_subsample=.01,
                axes_idx={}, init_figure=True,
                plot_representation_error_as_rgb=False,
                width_ratios={}):
        r"""Animate synthesis progress.

        This is essentially the figure produced by
        ``self.plot_synthesis_status`` animated over time, for each stored
        iteration.

        We return the matplotlib FuncAnimation object. In order to view
        it in a Jupyter notebook, use the
        ``plenoptic.convert_anim_to_html(anim)`` function. In order to
        save, use ``anim.save(filename)`` (note for this that you'll
        need the appropriate writer installed and on your path, e.g.,
        ffmpeg, imagemagick, etc). Either of these will probably take a
        reasonably long amount of time.

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we use all
            channels (assumed use-case is RGB(A) image).
        figsize : tuple or None, optional
            The size of the figure to create. It may take a little bit of
            playing around to find a reasonable value. If None, we attempt to
            make our best guess, aiming to have each axis be of size (5, 5)
        framerate : int, optional
            How many frames a second to display.
        ylim : str, None, or tuple, optional
            The y-limits of the representation_error plot (ignored if
            ``plot_representation_error`` arg is False).

            * If a tuple, then this is the ylim of all plots

            * If None, then all plots have the same limits, all
              symmetric about 0 with a limit of
              ``np.abs(representation_error).max()`` (for the initial
              representation_error)

            * If a string, must be 'rescale' or of the form 'rescaleN',
              where N can be any integer. If 'rescaleN', we rescale the
              limits every N frames (we rescale as if ylim = None). If
              'rescale', then we do this 10 times over the course of the
              animation

        plot_representation_error : bool, optional
            Whether to plot the representation ratio or not.
        imshow_zoom : int, float, or None, optional
            Either an int or an inverse power of 2, how much to zoom the
            images by in the plots we'll create. If None (the default), we
            attempt to find the best value ourselves.
        plot_data_attr : list, optional
            list of strs giving the names of the attributes with data
            plotted on the second subplot. this allows us to update
            whatever is in there if your plot_synthesis_status() plots
            something other than loss or if you plotted more than one
            attribute (e.g., MADCompetition plots two losses)
        rep_func_kwargs : dict, optional
            a dictionary of additional kwargs to pass through to the repeated
            calls to representation_error() or _grab_value_for_comparison()
            (for plotting representation error and representation comparison,
            respectively)
        plot_image_hist : bool, optional
            Whether to plot the histograms of image pixel intensities or
            not. Note that we update this in the most naive way possible
            (by clearing and replotting the values), so it might not
            look as good as the others and may take some time.
        plot_rep_comparison : bool, optional
            Whether to plot a scatter plot comparing the synthesized and base
            representation.
        plot_signal_comparison : bool, optional
            Whether to plot a 2d histogram comparing the synthesized and base
            representation. Note that we update this in the most naive way
            possible (by clearing and replotting the values), so it might not
            look as good as the others and may take some time.
        fig : plt.Figure or None, optional
            If None, create the figure from scratch. Else, should be an empty
            figure with enough axes (the expected use here is have same-size
            movies with different plots).
        signal_comp_func : {'scatter', 'hist2d'}, optional
            Whether to use a scatter plot or 2d histogram to plot this signal
            comparison. When there are many values (as often happens), then
            hist2d will be clearer
        signal_comp_subsample : float, optional
            What percentage of signal points to plot. If less than 1, will
            randomly select that proportion of the points to plot. Done to make
            visualization clearer.
        axes_idx : dict, optional
            Dictionary specifying which axes contains which type of plot,
            allows for more fine-grained control of the resulting figure.
            Probably only helpful if fig is also defined. Possible keys: image,
            loss, rep_error, hist, rep_comp, signal_comp, misc. Values should
            all be ints. If you tell this function to create a plot that doesn't
            have a corresponding key, we find the lowest int that is not
            already in the dict, so if you have axes that you want unchanged,
            place their idx in misc.
        init_figure : bool, optional
            If True, we call plot_synthesis_status to initialize the figure. If
            False, we assume fig has already been intialized with the proper
            plots (e.g., you already called plot_synthesis_status and are
            passing that figure as the fig argument). In this case, axes_idx
            must also be set and include keys for each of the included plots,
        plot_representation_error_as_rgb : bool, optional
            The representation can be image-like with multiple channels, and we
            have no way to determine whether it should be represented as an RGB
            image or not, so the user must set this flag to tell us. It will be
            ignored if the representation doesn't look image-like or if the
            model has its own plot_representation_error() method. Else, it will
            be passed to `po.imshow()`, see that methods docstring for details.
            since plot_synthesis_status normally sets it up for us
        width_ratios : dict, optional
            By defualt, all plots axes will have the same width. To change
            that, specify their relative widths using keys of the format
            "{x}_width", where `x` in ['synthesized_image', 'loss',
            'representation_error', 'image_hist', 'rep_comparison',
            'signal_comparison']

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object. In order to view, must convert to HTML
            or save.

        Notes
        -----

        By default, we use the ffmpeg backend, which requires that you have
        ffmpeg installed and on your path (https://ffmpeg.org/download.html).
        To use a different, use the matplotlib rcParams:
        `matplotlib.rcParams['animation.writer'] = writer`, see
        https://matplotlib.org/stable/api/animation_api.html#writer-classes for
        more details.

        For displaying in a jupyter notebook, ffmpeg appears to be required.

        """
        if not self.store_progress:
            raise Exception("synthesize() was run with store_progress=False,"
                            " cannot animate!")
        if self.saved_representation is not None and len(self.saved_signal) != len(self.saved_representation):
            raise Exception("saved_signal and saved_representation need to be the same length in "
                            "order for this to work!")
        if self.synthesized_signal.ndim not in [3, 4]:
            raise Exception("animate() expects 3 or 4d data; unexpected"
                            " behavior will result otherwise!")
        # every time we call synthesize(), store_progress gets one extra
        # element compared to loss. this uses that fact to figure out
        # how many times we've called sythesize())
        times_called = ((self.saved_signal.shape[0] * self.store_progress - len(self.loss)) //
                        self.store_progress)
        # which we use in order to pad out the end of plot_data so that
        # the lengths work out correctly (technically, should be
        # inserting this at the moments synthesize() was called, but I
        # don't know how to figure that out and the difference shouldn't
        # be noticeable except in extreme circumstances, e.g., you
        # called synthesize(max_iter=5) 100 times).

        plot_data = [getattr(self, d) + self.store_progress*times_called*[getattr(self, d)[-1]]
                     for d in plot_data_attr]
        if self.base_representation.ndimension() == 4:
            # we have to do this here so that we set the
            # ylim_rescale_interval such that we never rescale ylim
            # (rescaling ylim messes up an image axis)
            ylim = False
        try:
            if ylim.startswith('rescale'):
                try:
                    ylim_rescale_interval = int(ylim.replace('rescale', ''))
                except ValueError:
                    # then there's nothing we can convert to an int there
                    ylim_rescale_interval = int((self.saved_representation.shape[0] - 1) // 10)
                    if ylim_rescale_interval == 0:
                        ylim_rescale_interval = int(self.saved_representation.shape[0] - 1)
                ylim = None
            else:
                raise Exception("Don't know how to handle ylim %s!" % ylim)
        except AttributeError:
            # this way we'll never rescale
            ylim_rescale_interval = len(self.saved_signal)+1
        if init_figure:
            # initialize the figure
            fig = self.plot_synthesis_status(batch_idx=batch_idx, channel_idx=channel_idx,
                                             iteration=0, figsize=figsize, ylim=ylim,
                                             plot_loss=plot_loss,
                                             plot_representation_error=plot_representation_error,
                                             imshow_zoom=imshow_zoom, fig=fig,
                                             plot_synthesized_image=plot_synthesized_image,
                                             plot_image_hist=plot_image_hist,
                                             plot_signal_comparison=plot_signal_comparison,
                                             plot_rep_comparison=plot_rep_comparison,
                                             signal_comp_func=signal_comp_func,
                                             signal_comp_subsample=signal_comp_subsample,
                                             axes_idx=axes_idx,
                                             plot_representation_error_as_rgb=plot_representation_error_as_rgb,
                                             width_ratios=width_ratios)
            # plot_synthesis_status creates a hidden attribute, _axes_idx, a dict
            # which tells us which axes contains which plot
            axes_idx = self._axes_idx
        # grab the artists for the second plot (we don't need to do this
        # for the synthesized image or representation plot, because we
        # use the update_plot function for that)
        if plot_loss:
            scat = fig.axes[axes_idx['loss']].collections
        # can have multiple plots
        if plot_representation_error:
            try:
                rep_error_axes = [fig.axes[i] for i in axes_idx['rep_error']]
            except TypeError:
                # in this case, axes_idx['rep_error'] is not iterable and so is
                # a single value
                rep_error_axes = [fig.axes[axes_idx['rep_error']]]
        else:
            rep_error_axes = []
        # can also have multiple plots
        if plot_rep_comparison:
            try:
                rep_comp_axes = [fig.axes[i] for i in axes_idx['rep_comp']]
            except TypeError:
                # in this case, axes_idx['rep_comp'] is not iterable and so is
                # a single value
                rep_comp_axes = [fig.axes[axes_idx['rep_comp']]]
        else:
            rep_comp_axes = []

        if self.base_representation.ndimension() == 4:
            warnings.warn("Looks like representation is image-like, haven't fully thought out how"
                          " to best handle rescaling color ranges yet!")
            # replace the bit of the title that specifies the range,
            # since we don't make any promises about that. we have to do
            # this here because we need the figure to have been created
            for ax in rep_error_axes:
                ax.set_title(re.sub(r'\n range: .* \n', '\n\n', ax.get_title()))

        def movie_plot(i):
            artists = []
            if plot_synthesized_image:
                artists.extend(update_plot(fig.axes[axes_idx['image']],
                                           data=self.saved_signal[i],
                                           batch_idx=batch_idx))
            if plot_representation_error:
                representation_error = self.representation_error(iteration=i,
                                                                 **rep_func_kwargs)
                # we pass rep_error_axes to update, and we've grabbed
                # the right things above
                artists.extend(update_plot(rep_error_axes,
                                           batch_idx=batch_idx,
                                           model=self.model,
                                           data=representation_error))
                # again, we know that rep_error_axes contains all the axes
                # with the representation ratio info
                if ((i+1) % ylim_rescale_interval) == 0:
                    if self.base_representation.ndimension() == 3:
                        rescale_ylim(rep_error_axes,
                                     representation_error)
            if plot_image_hist:
                # this is the dumbest way to do this, but it's simple --
                # clearing the axes can cause problems if the user has, for
                # example, changed the tick locator or formatter. not sure how
                # to handle this best right now
                fig.axes[axes_idx['hist']].clear()
                self.plot_image_hist(batch_idx=batch_idx,
                                     channel_idx=channel_idx, iteration=i,
                                     ax=fig.axes[axes_idx['hist']])
            if plot_signal_comparison:
                if signal_comp_func == 'hist2d':
                    # this is the dumbest way to do this, but it's simple --
                    # clearing the axes can cause problems if the user has, for
                    # example, changed the tick locator or formatter. not sure how
                    # to handle this best right now
                    fig.axes[axes_idx['signal_comp']].clear()
                    self.plot_value_comparison(value='signal', batch_idx=batch_idx,
                                               channel_idx=channel_idx, iteration=i,
                                               ax=fig.axes[axes_idx['signal_comp']],
                                               func=signal_comp_func)
                else:
                    plot_vals = self._grab_value_for_comparison('signal',
                                                                batch_idx,
                                                                channel_idx, i,
                                                                signal_comp_subsample)
                    artists.extend(update_plot(fig.axes[axes_idx['signal_comp']],
                                               plot_vals))
            if plot_loss:
                # loss always contains values from every iteration, but
                # everything else will be subsampled
                for s, d in zip(scat, plot_data):
                    s.set_offsets((i*self.store_progress, d[i*self.store_progress]))
                artists.extend(scat)
            if plot_rep_comparison:
                plot_vals = self._grab_value_for_comparison('representation',
                                                            batch_idx, channel_idx,
                                                            i, **rep_func_kwargs)
                artists.extend(update_plot(rep_comp_axes, plot_vals))
            # as long as blitting is True, need to return a sequence of artists
            return artists


        # don't need an init_func, since we handle initialization ourselves
        anim = animation.FuncAnimation(fig, movie_plot, frames=len(self.saved_signal),
                                       blit=True, interval=1000./framerate, repeat=False)
        plt.close(fig)
        return anim
