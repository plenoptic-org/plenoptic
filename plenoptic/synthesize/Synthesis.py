"""abstract synthesis super-class
"""
import abc
import re
import torch
from torch import optim
import numpy as np
import warnings
from ..tools.data import to_numpy
import matplotlib.pyplot as plt
import pyrtools as pt
from ..tools.display import rescale_ylim, plot_representation, update_plot
from matplotlib import animation


class Synthesis(torch.nn.Module, metaclass=abc.ABCMeta):
    r"""Abstract super-class for synthesis methods

    All synthesis methods share a variety of similarities and thus need
    to have similar methods. Some of these can be implemented here and
    simply inherited, some of them will need to be different for each
    sub-class and thus are marked as abstract methods here

    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def synthesize():
        r"""synthesize an image
        """
        pass

    def analyze(self, x, **kwargs):
        r"""Analyze the image, that is, obtain the model's representation of it

        Any kwargs are passed to the model's forward method

        Parameters
        ----------
        x : torch.tensor
            The image to analyze

        Returns
        -------
        y : torch.tensor
            The model's representation of x
        """
        y = self.model(x, **kwargs)
        if isinstance(y, list):
            return torch.cat([s.squeeze().view(-1) for s in y]).unsqueeze(1)
        else:
            return y

    def objective_function(self, x, y):
        r"""Calculate the loss between x and y

        This is what we minimize. Currently it's the L2-norm of their
        difference: ``torch.norm(x-y, p=2)``.

        Parameters
        ----------
        x : torch.tensor
            the first element
        y : torch.tensor
            the second element

        Returns
        -------
        loss : torch.tensor
            single-element tensor containing the L2-norm of the
            difference between x and y

        """
        return torch.norm(x - y, p=2)

    def representation_error(self, iteration=None, **kwargs):
        r"""Get the representation error

        This is (matched_representation - target_representation). If
        ``iteration`` is not None, we use
        ``self.saved_representation[iteration]`` for
        matched_representation

        Any kwargs are passed through to self.analyze when computing the
        matched/target representation.

        Parameters
        ----------
        iteration: int or None, optional
            Which iteration to create the representation ratio for. If
            None, we use the current ``matched_representation``

        Returns
        -------
        torch.Tensor

        """
        if iteration is not None:
            matched_rep = self.saved_representation[iteration].to(self.target_representation.device)
        else:
            matched_rep = self.analyze(self.matched_image, **kwargs)
        try:
            rep_error = matched_rep - self.target_representation
        except RuntimeError:
            # try to use the last scale (if the above failed, it's
            # because they were different shapes), but only if the user
            # didn't give us another scale to use
            if 'scales' not in kwargs.keys():
                kwargs['scales'] = [self.scales[-1]]
            rep_error = matched_rep - self.analyze(self.target_image, **kwargs)
        return rep_error

    def normalized_mse(self, iteration=None, **kwargs):
        r"""Get the normalized mean-squared representation error

        Following the method used in [1]_ to check for convergence, here
        we take the mean-squared error between the target_representation
        and matched_representation, then divide by the variance of
        target_representation.

        If ``iteration`` is not None, we use
        ``self.saved_representation[iteration]`` for
        matched_representation.

        Any kwargs are passed through to self.analyze when computing the
        matched/target representation

        Parameters
        ----------
        iteration: int or None, optional
            Which iteration to create the representation ratio for. If
            None, we use the current ``matched_representation``

        Returns
        -------
        torch.Tensor

        References
        ----------
        .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
           ventral stream. Nature Neuroscience, 14(9),
           1195–1201. http://dx.doi.org/10.1038/nn.2889

        """
        if iteration is not None:
            matched_rep = self.saved_representation[iteration].to(self.target_representation.device)
        else:
            matched_rep = self.analyze(self.matched_image, **kwargs)
        try:
            rep_error = matched_rep - self.target_representation
            target_rep = self.target_representation
        except RuntimeError:
            # try to use the last scale (if the above failed, it's
            # because they were different shapes), but only if the user
            # didn't give us another scale to use
            if 'scales' not in kwargs.keys():
                kwargs['scales'] = [self.scales[-1]]
            target_rep = self.analyze(self.target_image, **kwargs)
            rep_error = matched_rep - target_rep
        return torch.pow(rep_error, 2).mean() / torch.var(target_rep)

    def _init_optimizer(self, optimizer, lr, scheduler=True, **optimizer_kwargs):
        """Initialize the optimzer and learning rate scheduler

        This gets called at the beginning of synthesize() and can also
        be called at other moments to make sure we're using the original
        learning rate (e.g., when moving to a different scale for
        coarse-to-fine optimization).

        we also (optionally) initialize a learning rate scheduler which
        will reduce the LR on plateau by a factor of .5. To turn this
        behavior off, pass ``scheduler=False``

        optimizer options. each has some default arguments which are
        explained below. with the exception of ``'GD'``, each of these
        can be overwritten by values passed as ``optimizer_kwargs``:
        - 'GD': gradient descent, ``optim.SGD(nesterov=False,
          momentum=0, weight_decay=0)`` (these cannot be modified)
        - 'SGD': stochastic gradient descent, ``optim.SGD(nesterov=True,
          momentum=.8)``
        - 'LBFGS': limited-memory BFGS , ``optim.LBFGS(history_size=10,
          max_iter=4)``
        - 'Adam': Adam, ``optim.Adam(amsgrad=True)``
        - 'AdamW': AdamW, ``optim.AdamW(amsgrad=True)``

        Parameters
        ----------
        optimizer : {'GD', 'SGD', 'LBFGS', 'Adam', 'AdamW'}
            the optimizer to initialize.
        lr : float
            the learning rate of the optimizer
        scheduler : bool, optional
            whether to initialize the scheduler or not. If False, the
            learning rate will never decrease. Setting this to True
            seems to improve performance, but it might be useful to turn
            it off in order to better work through what's happening
        optimizer_kwargs :
            passed to the optimizer's initializer

        """
        if optimizer == 'GD':
            # std gradient descent
            self.optimizer = optim.SGD([self.matched_image], lr=lr, nesterov=False, momentum=0,
                                       weight_decay=0, **optimizer_kwargs)
        elif optimizer == 'SGD':
            for k, v in zip(['nesterov', 'momentum'], [True, .8]):
                if k not in optimizer_kwargs:
                    optimizer_kwargs[k] = v
            self.optimizer = optim.SGD([self.matched_image], lr=lr, **optimizer_kwargs)
        elif optimizer == 'LBFGS':
            for k, v in zip(['history_size', 'max_iter'], [10, 4]):
                if k not in optimizer_kwargs:
                    optimizer_kwargs[k] = v
            self.optimizer = optim.LBFGS([self.matched_image], lr=lr, **optimizer_kwargs)
            warnings.warn('This second order optimization method is more intensive')
            if self.fraction_removed > 0:
                warnings.warn('For now the code is not designed to handle LBFGS and random'
                              ' subsampling of coeffs')
        elif optimizer == 'Adam':
            if 'amsgrad' not in optimizer_kwargs:
                optimizer_kwargs['amsgrad'] = True
            self.optimizer = optim.Adam([self.matched_image], lr=lr, **optimizer_kwargs)
        elif optimizer == 'AdamW':
            if 'amsgrad' not in optimizer_kwargs:
                optimizer_kwargs['amsgrad'] = True
            self.optimizer = optim.AdamW([self.matched_image], lr=lr, **optimizer_kwargs)
        else:
            raise Exception("Don't know how to handle optimizer %s!" % optimizer)
        if scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=.5)
        else:
            self.scheduler = None

    def _closure(self):
        r"""An abstraction of the gradient calculation, before the optimization step.

        This enables optimization algorithms that perform several
        evaluations of the gradient before taking a step (ie. second
        order methods like LBFGS).

        Note that the fraction removed also happens here, and for now a
        fresh sample of noise is drawn at each iteration.
            1) that means for now we do not support LBFGS with a random
               fraction removed.
            2) beyond removing random fraction of the coefficients, one
               could schedule the optimization (eg. coarse to fine)

        Additionally, this is where:
        - ``matched_representation`` is updated
        - ``loss.backward()`` is called

        """
        self.optimizer.zero_grad()
        analyze_kwargs = {}
        if self.coarse_to_fine:
            # if we've reached 'all', we act the same as if
            # coarse_to_fine was False
            if self.scales[-1] != 'all':
                analyze_kwargs['scales'] = [self.scales[-1]]
        self.matched_representation = self.analyze(self.matched_image, **analyze_kwargs)
        target_rep = self.analyze(self.target_image, **analyze_kwargs)
        if self.store_progress:
            self.matched_representation.retain_grad()

        # here we get a boolean mask (bunch of ones and zeroes) for all
        # the statistics we want to include. We only do this if the loss
        # appears to be roughly unchanging for some number of iterations
        if (len(self.loss) > self.loss_change_iter and
            self.loss[-self.loss_change_iter] - self.loss[-1] < self.loss_change_thresh):
            error_idx = self.representation_error(**analyze_kwargs).flatten().abs().argsort(descending=True)
            error_idx = error_idx[:int(self.loss_change_fraction * error_idx.numel())]
        # else, we use all of the statistics
        else:
            error_idx = torch.nonzero(torch.ones_like(self.matched_representation.flatten()))
        # for some reason, pytorch doesn't have the equivalent of
        # np.random.permutation, something that returns a shuffled copy
        # of a tensor, so we use numpy's version
        idx_shuffled = torch.LongTensor(np.random.permutation(to_numpy(error_idx)))
        # then we optionally randomly select some subset of those.
        idx_sub = idx_shuffled[:int((1 - self.fraction_removed) * idx_shuffled.numel())]
        loss = self.objective_function(self.matched_representation.flatten()[idx_sub],
                                       target_rep.flatten()[idx_sub])

        loss.backward(retain_graph=True)

        return loss

    def _optimizer_step(self, pbar=None, **kwargs):
        r"""Compute and propagate gradients, then step the optimizer to update matched_image

        Parameters
        ----------
        pbar : tqdm.tqdm or None, optional
            A tqdm progress-bar, which we update with a postfix
            describing the current loss, gradient norm, and learning
            rate (it already tells us which iteration and the time
            elapsed). If None, then we don't display any progress
        kwargs :
            will also display in the progress bar's postfix

        Returns
        -------
        loss : torch.tensor
            1-element tensor containing the loss on this step
        gradient : torch.tensor
            1-element tensor containing the gradient on this step
        learning_rate : torch.tensor
            1-element tensor containing the learning rate on this step

        """
        postfix_dict = {}
        if self.coarse_to_fine:
            # the last scale will be 'all', and we never remove
            # it. Otherwise, check to see if it looks like loss has
            # stopped declining and, if so, switch to the next scale
            if (len(self.scales) > 1 and len(self.scales_loss) > self.loss_change_iter and
                abs(self.scales_loss[-1] - self.scales_loss[-self.loss_change_iter]) < self.loss_change_thresh and
                len(self.loss) - self.scales_timing[self.scales[-1]][0] > self.loss_change_iter):
                self.scales_timing[self.scales[-1]].append(len(self.loss)-1)
                self.scales = self.scales[:-1]
                self.scales_timing[self.scales[-1]].append(len(self.loss))
                # reset scheduler and optimizer
                self._init_optimizer(**self.optimizer_kwargs)
            # we have some extra info to include in the progress bar if
            # we're doing coarse-to-fine
            postfix_dict['current_scale'] = self.scales[-1]
        loss = self.optimizer.step(self._closure)
        # we have this here because we want to do the above checking at
        # the beginning of each step, before computing the loss
        # (otherwise there's an error thrown because self.scales[-1] is
        # not the same scale we computed matched_representation using)
        if self.coarse_to_fine:
            postfix_dict['current_scale_loss'] = loss.item()
            # and we also want to keep track of this
            self.scales_loss.append(loss.item())
        g = self.matched_image.grad.data
        # optionally step the scheduler
        if self.scheduler is not None:
            self.scheduler.step(loss.item())

        if self.coarse_to_fine and self.scales[-1] != 'all':
            with torch.no_grad():
                full_matched_rep = self.analyze(self.matched_image)
                loss = self.objective_function(full_matched_rep, self.target_representation)
        else:
            loss = self.objective_function(self.matched_representation, self.target_representation)

        # for display purposes, always want loss to be positive
        postfix_dict.update(dict(loss="%.4e" % abs(loss.item()),
                                 gradient_norm="%.4e" % g.norm().item(),
                                 learning_rate=self.optimizer.param_groups[0]['lr'], **kwargs))
        # add extra info here if you want it to show up in progress bar
        if pbar is not None:
            pbar.set_postfix(**postfix_dict)
        return loss, g.norm(), self.optimizer.param_groups[0]['lr']

    @abc.abstractmethod
    def save(self, file_path, save_model_reduced=False, attrs=['model']):
        r"""save all relevant variables in .pt file

        This is an abstractmethod only because you need to specify which
        attributes to save. See ``metamer.save`` as an example, but the
        save method in your synthesis object should probably should have
        a line defining the attributes to save and then a call to
        ``super().save(file_path, save_model_reduced, attrs)``

        Note that if store_progress is True, this will probably be very
        large

        Parameters
        ----------
        file_path : str
            The path to save the synthesis object to
        save_model_reduced : bool
            Whether we save the full model or just its attribute
            ``state_dict_reduced`` (this is a custom attribute of ours,
            the basic idea being that it only contains the attributes
            necessary to initialize the model, none of the (probably
            much larger) ones it gets during run-time).
        attrs : list
            List of strs containing the names of the attributes of this
            object to save.

        """
        save_dict = {}
        if 'model' in attrs:
            model = self.model
            try:
                if save_model_reduced:
                    model = self.model.state_dict_reduced
            except AttributeError:
                warnings.warn("self.model doesn't have a state_dict_reduced attribute, will pickle "
                              "the whole model object")
            save_dict['model'] = model
            attrs.remove('model')
        for k in attrs:
            attr = getattr(self, k)
            # detaching the tensors avoids some headaches like the
            # tensors having extra hooks or the like
            if isinstance(attr, torch.Tensor):
                attr = attr.detach()
            save_dict[k] = attr
        torch.save(save_dict, file_path)

    @classmethod
    def load(cls, file_path, model_constructor=None, map_location='cpu', **state_dict_kwargs):
        r"""load all relevant stuff from a .pt file

        We will iterate through any additional key word arguments
        provided and, if the model in the saved representation is a
        dictionary, add them to the state_dict of the model. In this
        way, you can replace, e.g., paths that have changed between
        where you ran the model and where you are now.

        Parameters
        ----------
        file_path : str
            The path to load the synthesis object from
        model_constructor : callable or None, optional
            When saving the synthesis object, we have the option to only
            save the ``state_dict_reduced`` (in order to save space). If
            we do that, then we need some way to construct that model
            again and, not knowing its class or anything, this object
            doesn't know how. Therefore, a user must pass a constructor
            for the model that takes in the ``state_dict_reduced``
            dictionary and returns the initialized model. See the
            VentralModel class for an example of this.
        map_location : str, optional
            map_location argument to pass to ``torch.load``. If you save
            stuff that was being run on a GPU and are loading onto a
            CPU, you'll need this to make sure everything lines up
            properly. This should be structured like the str you would
            pass to ``torch.device``

        Returns
        -------
        synthesis : plenoptic.synth.Synthesis
            The loaded synthesis object


        Examples
        --------
        >>> metamer = po.synth.Metamer(img, model)
        >>> metamer.synthesize(max_iter=10, store_progress=True)
        >>> metamer.save('metamers.pt')
        >>> metamer_copy = po.synth.Metamer.load('metamers.pt')

        Things are slightly more complicated if you saved a reduced
        representation of the model by setting the
        ``save_model_reduced`` flag to ``True``. In that case, you also
        need to pass a model constructor argument, like so:

        >>> model = po.simul.RetinalGanglionCells(1)
        >>> metamer = po.synth.Metamer(img, model)
        >>> metamer.synthesize(max_iter=10, store_progress=True)
        >>> metamer.save('metamers.pt', save_model_reduced=True)
        >>> metamer_copy = po.synth.Metamer.load('metamers.pt',
                                                 po.simul.RetinalGanglionCells.from_state_dict_reduced)

        You may want to update one or more of the arguments used to
        initialize the model. The example I have in mind is where you
        run the metamer synthesis on a cluster but then load it on your
        local machine. The VentralModel classes have a ``cache_dir``
        attribute which you will want to change so it finds the
        appropriate location:

        >>> model = po.simul.RetinalGanglionCells(1)
        >>> metamer = po.synth.Metamer(img, model)
        >>> metamer.synthesize(max_iter=10, store_progress=True)
        >>> metamer.save('metamers.pt', save_model_reduced=True)
        >>> metamer_copy = po.synth.Metamer.load('metamers.pt',
                                                 po.simul.RetinalGanglionCells.from_state_dict_reduced,
                                                 cache_dir="/home/user/Desktop/metamers/windows_cache")

        """
        tmp_dict = torch.load(file_path, map_location=map_location)
        device = torch.device(map_location)
        model = tmp_dict.pop('model')
        target_image = tmp_dict.pop('target_image').to(device)
        if isinstance(model, dict):
            for k, v in state_dict_kwargs.items():
                warnings.warn("Replacing state_dict key %s, value %s with kwarg value %s" %
                              (k, model.pop(k, None), v))
                model[k] = v
            # then we've got a state_dict_reduced and we need the model_constructor
            model = model_constructor(model)
            # want to make sure the dtypes match up as well
            model = model.to(device, target_image.dtype)
        synth = cls(target_image, model)
        for k, v in tmp_dict.items():
            setattr(synth, k, v)
        return synth

    @abc.abstractmethod
    def to(self, *args, attrs=[], **kwargs):
        r"""Moves and/or casts the parameters and buffers.

        Similar to ``save``, this is an abstract method only because you
        need to define the attributes to call to on.

        NOTE: We always call ``model.to(*args, **kwargs)`` (and thus
        ``'model'`` does not need to be in the ``attrs`` argument), but
        we only raise a Warning (not an Exception) if ``model`` does not
        have a ``to()`` method

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)

        .. function:: to(dtype, non_blocking=False)

        .. function:: to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            attrs (:class:`list`): list of strs containing the attributes of
                this object to move to the specified device/dtype

        Returns:
            Module: self

        """
        try:
            self.model = self.model.to(*args, **kwargs)
        except AttributeError:
            warnings.warn("model has no `to` method, so we leave it as is...")
        for k in attrs:
            if hasattr(self, k):
                attr = getattr(self, k)
                if isinstance(attr, torch.Tensor):
                    attr = attr.to(*args, **kwargs)
                    if isinstance(getattr(self, k), torch.nn.Parameter):
                        attr = torch.nn.Parameter(attr)
                    setattr(self, k, attr)
                elif isinstance(attr, list):
                    setattr(self, k, [a.to(*args, **kwargs) for a in attr])
        return self

    def plot_representation_error(self, batch_idx=0, iteration=None, figsize=(5, 5), ylim=None,
                                  ax=None, title=None):
        r"""Plot distance ratio showing how close we are to convergence

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
            Which index to take from the batch dimension (the first one)
        iteration: int or None, optional
            Which iteration to create the representation ratio for. If
            None, we use the current ``matched_representation``
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

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot

        """
        representation_error = self.representation_error(iteration=iteration)
        return plot_representation(self.model, representation_error, ax, figsize, ylim,
                                   batch_idx, title)

    def plot_loss(self, iteration=None, figsize=(5, 5), ax=None, title='Loss', **kwargs):
        """Plot the synthesis loss

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

    def plot_synthesized_image(self, batch_idx=0, channel_idx=0, iteration=None, title=None,
                               figsize=(5, 5), ax=None, imshow_zoom=None, vrange=(0, 1)):
        """show the synthesized image

        You can specify what iteration to view by using the
        ``iteration`` arg. The default, ``None``, shows the final one.

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
            Which index to take from the batch dimension (the first one)
        channel_idx : int, optional
            Which index to take from the channel dimension (the second one)
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
            attempt to find the best value ourselves. Else, if >1, must
            be an integer.  If <1, must be 1/d where d is a a divisor of
            the size of the largest image.
        vrange : tuple or str, optional
            The vrange option to pass to ``pyrtools.imshow``. See that
            function for details

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        if iteration is None:
            image = self.matched_image[batch_idx, channel_idx]
        else:
            image = self.saved_image[iteration, batch_idx, channel_idx]
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        if imshow_zoom is None:
            # image.shape[0] is the height of the image
            imshow_zoom = ax.bbox.height // image.shape[0]
            if imshow_zoom == 0:
                raise Exception("imshow_zoom would be 0, cannot display synthesized image! Enlarge"
                                " your figure")
        if title is None:
            title = self.__class__.__name__
        fig = pt.imshow(to_numpy(image), ax=ax, title=title, zoom=imshow_zoom, vrange=vrange)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return fig

    def plot_synthesis_status(self, batch_idx=0, channel_idx=0, iteration=None, figsize=(17, 5),
                              ylim=None, plot_representation_error=True, imshow_zoom=None,
                              vrange=(0, 1), fig=None):
        r"""Make a plot showing synthesized image, loss, and (optionally) representation ratio

        We create two or three subplots on a new figure. The first one
        contains the synthesized image, the second contains the loss,
        and the (optional) third contains the representation ratio, as
        plotted by ``self.plot_representation_error``.

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
            Which index to take from the batch dimension (the first one)
        channel_idx : int, optional
            Which index to take from the channel dimension (the second one)
        iteration : int or None, optional
            Which iteration to display. If None, the default, we show
            the most recent one. Negative values are also allowed.
        figsize : tuple, optional
            The size of the figure to create. It may take a little bit
            of playing around to find a reasonable value. If you're not
            showing the representation, (12, 5) probably makes sense. If
            you are showing the representation, it depends on the level
            of detail in that plot. If it only creates one set of axes,
            like ``RetinalGanglionCells`, then (17,5) is probably fine,
            but you may need much larger if it's more complicated; e.g.,
            for PrimaryVisualCortex, try (39, 11).
        ylim : tuple or None, optional
            The ylimit to use for the representation_error plot. We pass
            this value directly to ``self.plot_representation_error``
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

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        if fig is None:
            if plot_representation_error:
                n_subplots = 3
            else:
                n_subplots = 2
            fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
        else:
            axes = fig.axes
        self.plot_synthesized_image(batch_idx, channel_idx, iteration, None, ax=axes[0],
                                    imshow_zoom=imshow_zoom, vrange=vrange)
        self.plot_loss(iteration, ax=axes[1])
        if plot_representation_error:
            fig = self.plot_representation_error(batch_idx, iteration, ax=axes[2], ylim=ylim)
        return fig

    def animate(self, batch_idx=0, channel_idx=0, figsize=(17, 5), framerate=10, ylim='rescale',
                plot_representation_error=True, imshow_zoom=None, plot_data_attr=['loss'],
                rep_error_kwargs={}):
        r"""Animate synthesis progress!

        This is essentially the figure produced by
        ``self.plot_synthesis_status`` animated over time, for each stored
        iteration.

        It's difficult to determine a reasonable figsize, because we
        don't know how much information is in the plot showing the
        representation ratio. Therefore, it's recommended you play
        around with ``plot_synthesis_status`` until you find a
        good-looking value for figsize.

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
            Which index to take from the batch dimension (the first one)
        channel_idx : int, optional
            Which index to take from the channel dimension (the second one)
        figsize : tuple, optional
            The size of the figure to create. It may take a little bit
            of playing around to find a reasonable value. If you're not
            showing the representation, (12, 5) probably makes sense. If
            you are showing the representation, it depends on the level
            of detail in that plot. If it only creates one set of axes,
            like ``RetinalGanglionCells`, then (17,5) is probably fine,
            but you may need much larger if it's more complicated; e.g.,
            for PrimaryVisualCortex, try (39, 11).
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
        rep_error_kwargs : dict, optional
            a dictionary of kwargs to pass through to the repeated calls
            to representation_error() (in addition to the iteration)

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object. In order to view, must convert to HTML
            or save.

        """
        if len(self.saved_image) != len(self.saved_representation):
            raise Exception("saved_image and saved_representation need to be the same length in "
                            "order for this to work!")
        # this recovers the store_progress arg used with the call to
        # synthesize(), which we need for updating the progress of the
        # loss
        saved_subsample = len(self.loss) // (self.saved_representation.shape[0] - 1)
        # we have one extra frame of saved_image compared to loss, so we
        # just duplicate the loss value at the end
        plot_data = [getattr(self, d) + [getattr(self, d)[-1]] for d in plot_data_attr]
        if self.target_representation.ndimension() == 4:
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
            ylim_rescale_interval = len(self.saved_image)+1
        # initialize the figure
        fig = self.plot_synthesis_status(batch_idx, channel_idx, 0, figsize, ylim,
                                         plot_representation_error, imshow_zoom=imshow_zoom)
        # grab the artists for the second plot (we don't need to do this
        # for the synthesized image or representation plot, because we
        # use the update_plot function for that)
        scat = fig.axes[1].collections

        if self.target_representation.ndimension() == 4:
            warnings.warn("Looks like representation is image-like, haven't fully thought out how"
                          " to best handle rescaling color ranges yet!")
            # replace the bit of the title that specifies the range,
            # since we don't make any promises about that. we have to do
            # this here because we need the figure to have been created
            for ax in fig.axes[2:]:
                ax.set_title(re.sub(r'\n range: .* \n', '\n\n', ax.get_title()))

        def movie_plot(i):
            artists = []
            artists.extend(update_plot([fig.axes[0]], data=self.saved_image[i],
                                       batch_idx=batch_idx))
            if plot_representation_error:
                representation_error = self.representation_error(iteration=i, **rep_error_kwargs)
                # we know that the first two axes are the image and
                # loss, so we pass everything after that to update
                artists.extend(update_plot(fig.axes[2:], batch_idx=batch_idx, model=self.model,
                                           data=representation_error))
                # again, we know that fig.axes[2:] contains all the axes
                # with the representation ratio info
                if ((i+1) % ylim_rescale_interval) == 0:
                    if self.target_representation.ndimension() == 3:
                        rescale_ylim(fig.axes[2:], representation_error)
            # loss always contains values from every iteration, but
            # everything else will be subsampled
            for s, d in zip(scat, plot_data):
                s.set_offsets((i*saved_subsample, d[i*saved_subsample]))
            artists.extend(scat)
            # as long as blitting is True, need to return a sequence of artists
            return artists

        # don't need an init_func, since we handle initialization ourselves
        anim = animation.FuncAnimation(fig, movie_plot, frames=len(self.saved_image),
                                       blit=True, interval=1000./framerate, repeat=False)
        plt.close(fig)
        return anim
