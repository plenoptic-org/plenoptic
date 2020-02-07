"""abstract synthesis super-class
"""
import abc
import torch
from torch import optim
import numpy as np
import warnings
from ..tools.data import to_numpy


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
