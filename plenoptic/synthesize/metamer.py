"""Synthesize model metamers."""
import torch
import re
import numpy as np
from torch import Tensor
from tqdm.auto import tqdm
from ..tools import optim, display, signal, data
from typing import Union, Tuple, Callable, List, Dict
from typing_extensions import Literal
from .synthesis import Synthesis
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict


class Metamer(Synthesis):
    r"""Synthesize metamers for image-computable differentiable models.

    Following the basic idea in [1]_, this class creates a metamer for a given
    model on a given image. We start with ``initial_image`` and iteratively
    adjust the pixel values so as to match the representation of the
    ``metamer`` and ``image``.

    All ``saved_`` attributes are initialized as empty lists and will be
    non-empty if the ``store_progress`` arg to ``synthesize()`` is not
    ``False``. They will be appended to on every iteration if
    ``store_progress=True`` or every ``store_progress`` iterations if it's an
    ``int``.

    Parameters
    ----------
    image :
        A 4d tensor, this is the image whose representation we wish to
        match. If this is not a tensor, we try to cast it as one.
    model :
        A visual model, see `MAD_Competition` notebook for more details
    loss_function :
        the loss function to use to compare the representations of the
        models in order to determine their loss.
    range_penalty_lambda :
        Lambda to multiply by range penalty and add to loss.
    allowable_range :
        Range (inclusive) of allowed pixel values. Any values outside this
        range will be penalized.
    initial_image :
        4d Tensor to initialize our metamer with. If None, will draw a sample
        of uniform noise within ``allowed_range``.

    Attributes
    ----------
    target_representation : torch.Tensor
        Whatever is returned by ``model(image)``, this is what we match
        in order to create a metamer
    metamer : torch.Tensor
        The metamer. This may be unfinished depending on how many
        iterations we've run for.
    losses : list
        A list of our loss over iterations.
    gradient_norm : list
        A list of the gradient_norm over iterations.
    learning_rate : list
        A list of the learning_rate over iterations. We use a scheduler
        that gradually reduces this over time, so it won't be constant.
    pixel_change : list
        A list containing the max pixel change over iterations
        (``pixel_change[i]`` is the max pixel change in
        ``metamer`` between iterations ``i`` and ``i-1``). note
        this is calculated before any clamping, so may have some very
        large numbers in the beginning
    saved_metamer : torch.Tensor or list
        Saved ``self.metamer`` for later examination.
    scales : list or None
        The list of scales in optimization order (i.e., from coarse to fine).
        Will be modified during the course of optimization.
    scales_loss : list or None
        The scale-specific loss at each iteration
    scales_timing : dict or None
        Keys are the values found in ``scales``, values are lists, specifying
        the iteration where we started and stopped optimizing this scale.
    scales_finished : list or None
        List of scales that we've finished optimizing.

    References
    ----------
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model
       based on Joint Statistics of Complex Wavelet Coefficients. Int'l
       Journal of Computer Vision. 40(1):49-71, October, 2000.
       http://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
       http://www.cns.nyu.edu/~lcv/texture/

    """

    def __init__(self, image: Tensor, model: torch.nn.Module,
                 loss_function: Callable[[Tensor, Tensor], Tensor] = optim.mse,
                 range_penalty_lambda: float = .1,
                 allowed_range: Tuple[float] = (0, 1),
                 initial_image: Union[None, Tensor] = None):
        self.model = model
        self.image = image
        if image.ndimension() < 4:
            raise Exception("image must be torch.Size([n_batch, "
                            "n_channels, im_height, im_width]) but got "
                            f"{image.size()}")
        self._signal_shape = image.shape
        self.target_representation = self.model(self.image).detach()
        self.optimizer = None
        self.scheduler = None
        self.losses = []
        self.learning_rate = []
        self.gradient_norm = []
        self.pixel_change = []
        self.loss_function = loss_function
        self.range_penalty_lambda = range_penalty_lambda
        self.allowed_range = allowed_range
        self._init_metamer(initial_image)
        self.coarse_to_fine = False
        self.scales = None
        self.scales_loss = None
        self.scales_timing = None
        self.scales_finished = None
        self.store_progress = None
        self.saved_metamer = []

    def _init_metamer(self, initial_image: Union[None, Tensor] = None):
        """Initialize the metamer.

        Set the ``self.metamer`` attribute to be a parameter with the
        user-supplied data, making sure it's the right shape.

        Parameters
        ----------
        initial_image :
            The tensor we use to initialize the metamer. If None (the default),
            we initialize with uniformly-distributed random noise lying between
            0 and 1.

        """
        if initial_image is None:
            metamer = torch.rand_like(self.image)
            # rescale metamer to lie within the interval
            # self.allowed_range
            metamer = signal.rescale(metamer, *self.allowed_range)
            metamer.requires_grad_()
        else:
            metamer = initial_image.clone().detach()
            metamer = metamer.to(dtype=self.image.dtype, device=self.image.device)
            metamer.requires_grad_()
            if metamer.ndimension() < 4:
                raise Exception("metamer must be torch.Size([n_batch"
                                ", n_channels, im_height, im_width]) but got "
                                f"{metamer.size()}")
            if metamer.size() != self.image.size():
                raise Exception("metamer and image must be"
                                " same size!")
        self.metamer = metamer
        self.losses.append(self.objective_function(self.model(metamer)).item())

    def _init_ctf(self, coarse_to_fine: Literal['together', 'separate', False],
                  change_scale_criterion: Union[float, None],
                  stop_criterion: float):
        """Initialize stuff related to coarse-to-fine."""
        if coarse_to_fine not in [False, 'separate', 'together']:
            raise Exception(f"Don't know how to handle value {coarse_to_fine}!"
                            " Must be one of: False, 'separate', 'together'")
        if coarse_to_fine and self.scales is None:
            # if self.scales is not None, we're continuing a previous version
            # and want to continue. this list comprehension creates a new
            # object, so we don't modify model.scales
            try:
                self.scales = [i for i in self.model.scales[:-1]]
            except AttributeError:
                raise AttributeError(f"Model '{self.model._get_name()}' has no"
                                     " attribute " " 'scales', and therefore "
                                     "we cannot do coarse-to-fine synthesis.")
            if coarse_to_fine == 'separate':
                self.scales += [self.model.scales[-1]]
            self.scales += ['all']
            self.scales_timing = dict((k, []) for k in self.scales)
            self.scales_timing[self.scales[0]].append(0)
            self.scales_finished = []
            self.scales_loss = []
            if (change_scale_criterion is not None) and (stop_criterion >= change_scale_criterion):
                raise Exception("stop_criterion must be strictly less than "
                                "coarse-to-fine's change_scale_criterion, or"
                                " things get weird!")
        self.coarse_to_fine = coarse_to_fine

    def _init_optimizer(self, optimizer, scheduler):
        """Initialize optimizer and scheduler."""
        if optimizer is None:
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam([self.metamer],
                                                  lr=.01, amsgrad=True)
        else:
            if self.optimizer is not None:
                raise Exception("When resuming synthesis, optimizer arg must be None!")
            params = optimizer.param_groups[0]['params']
            if len(params) != 1 or not torch.equal(params[0], self.metamer):
                raise Exception("For metamer synthesis, optimizer must have one "
                                "parameter, the metamer we're synthesizing.")
            self.optimizer = optimizer
        self.scheduler = scheduler
        for pg in self.optimizer.param_groups:
            # initialize initial_lr if it's not here. Scheduler should add it
            # if it's not None.
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg['lr']


    def _init_store_progress(self, store_progress: Union[bool, int]):
        """Initialize store_progress-related attributes.

        Sets the ``self.store_progress`` attribute, as well as changing the
        ``saved_metamer`` attibute to a list so we can append to them. finally,
        adds first value to ``saved_metamer`` if it's empty.

        Parameters
        ----------
        store_progress : bool or int, optional
            Whether we should store the model response and the metamer image in
            progress on every iteration. If False, we don't save anything. If
            True, we save every iteration. If an int, we save every
            ``store_progress`` iterations (note then that 0 is the same as
            False and 1 the same as True). If True or int>0,
            ``self.saved_metamer`` contains the stored images.

        """
        if store_progress:
            if store_progress is True:
                store_progress = 1
            # if this is not the first time synthesize is being run for this
            # metamer object, saved_metamer will be a tensor instead of a list.
            # This converts it back to a list so we can use append. If it's the
            # first time, it will be an empty list and this does nothing
            self.saved_metamer = list(self.saved_metamer)
            # first time synthesize() is called, add the initial metamer (on
            # subsequent calls, this is already part of saved_metamer).
            if len(self.saved_metamer) == 0:
                self.saved_metamer.append(self.metamer.clone().to('cpu'))
        if self.store_progress is not None and store_progress != self.store_progress:
            # we require store_progress to be the same because otherwise the
            # subsampling relationship between attrs that are stored every
            # iteration (loss, gradient, etc) and those that are stored every
            # store_progress iteration (e.g., saved_metamer) changes partway
            # through and that's annoying
            raise Exception("If you've already run synthesize() before, must "
                            "re-run it with same store_progress arg. You "
                            f"passed {store_progress} instead of "
                            f"{self.store_progress} (True is equivalent to 1)")
        self.store_progress = store_progress

    def _check_nan_loss(self, loss: Tensor) -> bool:
        """Check if loss is nan and, if so, return True.

        This checks if loss is NaN and, if so, updates metamer to be several
        iterations ago (so they're meaningful) and then returns True.

        Parameters
        ----------
        loss :
            the loss from the most recent iteration

        Returns
        -------
        is_nan :
            True if loss was nan, False otherwise

        """
        if np.isnan(loss.item()):
            warnings.warn("Loss is NaN, quitting out! We revert "
                          "metamer to our last saved values (which "
                          "means this will throw an IndexError if you're not "
                          "saving anything)!")
            # need to use the -2 index because the last one will be the one
            # full of NaNs. this happens because the loss is computed before
            # calculating the gradient and updating metamer; therefore the
            # iteration where loss is NaN is the one *after* the iteration
            # where metamer started to have NaN values. this will fail if it
            # hits a nan before store_progress iterations (because then
            # saved_metamer only has a length of 1) but in that case, you have
            # more severe problems
            self.metamer = torch.nn.Parameter(self.saved_metamer[-2])
            return True
        return False

    def _store(self, i: int) -> bool:
        """Store metamer, if appropriate.

        if it's the right iteration, we update: ``saved_metamer``.

        Parameters
        ----------
        i :
            the current iteration (0-indexed)

        Returns
        -------
        stored :
            True if we stored this iteration, False if not.

        """
        stored = False
        with torch.no_grad():
            # i is 0-indexed but in order for the math to work out we want to
            # be checking a 1-indexed thing against the modulo (e.g., if
            # max_iter=10 and store_progress=3, then if it's 0-indexed, we'll
            # try to save this four times, at 0, 3, 6, 9; but we just want to
            # save it three times, at 3, 6, 9)
            if self.store_progress and ((i+1) % self.store_progress == 0):
                # want these to always be on cpu, to reduce memory use for GPUs
                self.saved_metamer.append(self.metamer.clone().to('cpu'))
                stored = True
        return stored

    def _check_for_stabilization(self, i: int, stop_criterion: float,
                                 stop_iters_to_check: int,
                                 ctf_iters_to_check: Union[int, None] = None) -> bool:
        r"""Check whether the loss has stabilized and, if so, return True.

         Have we been synthesizing for ``stop_iters_to_check`` iterations?
         | |
        no yes
         | '---->Is ``abs(self.loss[-1] - self.losses[-stop_iters_to_check] < stop_criterion``?
         |      no |
         |       | yes
         <-------' '---->Is ``coarse_to_fine`` not False?
         |              no  |
         |               |  yes
         |               |  '----> Have we synthesized all scales and done so for ``ctf_iters_to_check`` iterations?
         |               |          |                   |
         |               |          yes                 no
         |               |          |                   |
         |               |          v                   |
         |               '------> return ``True``       |
         |                                              |
         '---------> return ``False``<------------------'

        Parameters
        ----------
        i :
            The current iteration (0-indexed).
        stop_criterion :
            If the loss over the past ``stop_iters_to_check`` has changed
            less than ``stop_criterion``, we terminate synthesis.
        stop_iters_to_check :
            How many iterations back to check in order to see if the
            loss has stopped decreasing (for ``stop_criterion``).
        ctf_iters_to_check :
            Minimum number of iterations coarse-to-fine must run at each scale.
            If self.coarse_to_fine is False, then this is ignored.

        Returns
        -------
        loss_stabilized :
            Whether the loss has stabilized or not.

        """
        if len(self.losses) > stop_iters_to_check:
            if abs(self.losses[-stop_iters_to_check] - self.losses[-1]) < stop_criterion:
                if self.coarse_to_fine:
                    # only break out if we've been doing for long enough
                    if self.scales[0] == 'all' and i - self.scales_timing['all'][0] > ctf_iters_to_check:
                        return True
                else:
                    return True
        return False

    def objective_function(self, metamer_representation: Tensor,
                           target_representation: Union[Tensor, None] = None) -> Tensor:
        """Compute the metamer synthesis loss.

        This calls self.loss_function on ``metamer_representation`` and
        ``target_representation`` and then adds the weighted range penalty.

        Parameters
        ----------
        metamer_representation :
            Model response to ``metamer``.
        target_representation :
            Model response to ``image``. If None, we use
            ``self.target_representation``.

        Returns
        -------
        loss

        """
        if target_representation is None:
            target_representation = self.target_representation
        loss = self.loss_function(metamer_representation,
                                  target_representation)
        range_penalty = optim.penalize_range(self.metamer,
                                             self.allowed_range)
        return loss + self.range_penalty_lambda * range_penalty

    def _closure(self) -> Tensor:
        r"""An abstraction of the gradient calculation, before the optimization step.

        This enables optimization algorithms that perform several evaluations
        of the gradient before taking a step (ie. second order methods like
        LBFGS).

        Additionally, this is where:

        - ``metamer_representation`` is calculated, and thus any modifications
          to the model's forward call (e.g., specifying `scale` kwarg for
          coarse-to-fine) should happen.

        - ``loss`` is calculated and ``loss.backward()`` is called.

        Returns
        -------
        loss

        """
        self.optimizer.zero_grad()
        analyze_kwargs = {}
        if self.coarse_to_fine:
            # if we've reached 'all', we act the same as if
            # coarse_to_fine was False
            if self.scales[0] != 'all':
                analyze_kwargs['scales'] = [self.scales[0]]
                # if 'together', then we also want all the coarser
                # scales
                if self.coarse_to_fine == 'together':
                    analyze_kwargs['scales'] += self.scales_finished
        metamer_representation = self.model(self.metamer, **analyze_kwargs)
        # if analyze_kwargs is empty, we can just compare
        # metamer_representation against our cached target_representation
        if analyze_kwargs:
            target_rep = self.model(self.image, **analyze_kwargs)
        else:
            target_rep = None

        loss = self.objective_function(metamer_representation, target_rep)
        loss.backward(retain_graph=False)

        return loss

    def _optimizer_step(self, pbar: tqdm,
                        change_scale_criterion: float,
                        ctf_iters_to_check: int
                        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Compute and propagate gradients, then step the optimizer to update metamer.

        Parameters
        ----------
        pbar :
            A tqdm progress-bar, which we update with a postfix
            describing the current loss, gradient norm, and learning
            rate (it already tells us which iteration and the time
            elapsed).
        change_scale_criterion :
            How many iterations back to check to see if the loss has stopped
            decreasing and we should thus move to the next scale in
            coarse-to-fine optimization.
        ctf_iters_to_check :
            Minimum number of iterations coarse-to-fine must run at each scale.
            If self.coarse_to_fine is False, then this is ignored.

        Returns
        -------
        loss : torch.Tensor
            1-element tensor containing the loss on this step
        gradient : torch.Tensor
            1-element tensor containing the gradient on this step
        learning_rate : torch.Tensor
            1-element tensor containing the learning rate on this step
        pixel_change : torch.Tensor
            1-element tensor containing the max pixel change in
            metamer between this step and the last

        """
        last_iter_metamer = self.metamer.clone()
        postfix_dict = {}
        if self.coarse_to_fine:
            # The first check here is because the last scale will be 'all', and
            # we never remove it. Otherwise, check to see if it looks like loss
            # has stopped declining and, if so, switch to the next scale. Then
            # we're checking if self.scales_loss is long enough to check
            # ctf_iters_to_check back.
            if len(self.scales) > 1 and len(self.scales_loss) >= ctf_iters_to_check:
                # Now we check whether loss has decreased less than
                # change_scale_criterion
                if ((change_scale_criterion is None) or abs(self.scales_loss[-1] - self.scales_loss[-ctf_iters_to_check]) < change_scale_criterion):
                    # and finally we check whether we've been optimizing this
                    # scale for ctf_iters_to_check
                    if len(self.losses) - self.scales_timing[self.scales[0]][0] >= ctf_iters_to_check:
                        self.scales_timing[self.scales[0]].append(len(self.losses)-1)
                        self.scales_finished.append(self.scales.pop(0))
                        self.scales_timing[self.scales[0]].append(len(self.losses))
                        # reset optimizer's lr.
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = pg['initial_lr']
            # we have some extra info to include in the progress bar if
            # we're doing coarse-to-fine
            postfix_dict['current_scale'] = self.scales[0]
        loss = self.optimizer.step(self._closure)
        # we have this here because we want to do the above checking at
        # the beginning of each step, before computing the loss
        # (otherwise there's an error thrown because self.scales[-1] is
        # not the same scale we computed metamer_representation using)
        if self.coarse_to_fine:
            postfix_dict['current_scale_loss'] = loss.item()
            # and we also want to keep track of this
            self.scales_loss.append(loss.item())
        grad_norm = self.metamer.grad.detach().norm()
        if grad_norm.item() != grad_norm.item():
            raise Exception('found a NaN in the gradients during optimization')

        # optionally step the scheduler
        if self.scheduler is not None:
            self.scheduler.step(loss.item())

        if self.coarse_to_fine and self.scales[0] != 'all':
            with torch.no_grad():
                tmp_im = self.metamer.detach().clone()
                full_metamer_rep = self.model(tmp_im)
                loss = self.objective_function(full_metamer_rep)
        else:
            loss = self.objective_function(self.model(self.metamer))

        pixel_change = torch.max(torch.abs(self.metamer - last_iter_metamer))
        # for display purposes, always want loss to be positive. add extra info
        # here if you want it to show up in progress bar
        pbar.set_postfix(
            OrderedDict(loss=f"{abs(loss.item()):.04e}",
                        learning_rate=self.optimizer.param_groups[0]['lr'],
                        gradient_norm=f"{grad_norm.item():.04e}",
                        pixel_change=f"{pixel_change:.04e}", **postfix_dict))
        return loss, grad_norm, self.optimizer.param_groups[0]['lr'], pixel_change

    def synthesize(self, max_iter: int = 100,
                   optimizer: Union[None, torch.optim.Optimizer] = None,
                   scheduler: Union[None, torch.optim.lr_scheduler._LRScheduler] = None,
                   store_progress: Union[bool, int] = False,
                   stop_criterion: float = 1e-4, stop_iters_to_check: int = 50,
                   coarse_to_fine: Literal['together', 'separate', False] = False,
                   coarse_to_fine_kwargs: Dict[str, float] = {'change_scale_criterion': 1e-2,
                                                              'ctf_iters_to_check': 50}
                   ) -> Tensor:
        r"""Synthesize a metamer.

        This is the main method, which updates the ``initial_image`` until its
        representation matches that of ``image``.

        We run this until either we reach ``max_iter`` or the change over the
        past ``stop_iters_to_check`` iterations is less than
        ``stop_criterion``, whichever comes first

        Parameters
        ----------
        max_iter :
            The maximum number of iterations to run before we end synthesis
            (unless we hit the stop criterion).
        optimizer :
            The optimizer to use. If None and this is the first time calling
            synthesize, we use Adam(lr=.01, amsgrad=True); if synthesize has
            been called before, this must be None and we reuse the previous
            optimizer.
        scheduler :
            The learning rate scheduler to use. If None, we don't use one.
        store_progress :
            Whether we should store the representation of the metamer
            and the metamer image in progress on every iteration. If
            False, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress``
            iterations (note then that 0 is the same as False and 1 the
            same as True).
        stop_criterion :
            If the loss over the past ``stop_iters_to_check`` has changed
            less than ``stop_criterion``, we terminate synthesis.
        stop_iters_to_check :
            How many iterations back to check in order to see if the
            loss has stopped decreasing (for ``stop_criterion``).
        coarse_to_fine :
            If False, don't do coarse-to-fine optimization. Else, there
            are two options for how to do it:
            - 'together': start with the coarsest scale, then gradually
              add each finer scale.
            - 'separate': compute the gradient with respect to each
              scale separately (ignoring the others), then with respect
              to all of them at the end.
            (see ``Metamer`` tutorial for more details).
        coarse_to_fine_kwargs :
            Dictionary with two keys: ``'change_scale_criterion'`` and
            ``'ctf_iters_to_check'``. We use these analogously to
            ``stop_criterion`` and ``stop_iters_to_check``: if the loss has
            changed less than ``'change_scale_criterion'`` in the past
            ``'ctf_iters_to_check'`` iterations, we move on to the next scale in
            coarse-to-fine optimization. `'change_scale_criterion'` can also be
            `None`, in which case we will change scales as soon as we've spent
            `ctf_iters_to_check` on a given scale.

        Returns
        -------
        metamer : torch.Tensor
            The metamer we've created

        """
        # initialize stuff related to coarse-to-fine
        self._init_ctf(coarse_to_fine,
                       coarse_to_fine_kwargs.get('change_scale_criterion', None),
                       stop_criterion)

        # initialize the optimizer and scheduler
        self._init_optimizer(optimizer, scheduler)

        # get ready to store progress
        self._init_store_progress(store_progress)

        pbar = tqdm(range(max_iter))

        for i in pbar:
            loss, g, lr, pixel_change = self._optimizer_step(pbar,
                                                             coarse_to_fine_kwargs.get('change_scale_criterion', None),
                                                             coarse_to_fine_kwargs.get('ctf_iters_to_check', None))
            self.losses.append(loss.item())
            self.pixel_change.append(pixel_change.item())
            self.gradient_norm.append(g.item())
            self.learning_rate.append(lr)

            if self._check_nan_loss(loss):
                break

            # update saved_* attrs
            self._store(i)

            if self._check_for_stabilization(i, stop_criterion, stop_iters_to_check,
                                             coarse_to_fine_kwargs.get('ctf_iters_to_check', None)):
                break

        pbar.close()

        # finally, stack the saved_* attributes
        if self.store_progress:
            self.saved_metamer = torch.stack(self.saved_metamer)

        return self.metamer

    def save(self, file_path: str):
        r"""Save all relevant variables in .pt file.

        Note that if store_progress is True, this will probably be very
        large.

        See ``load`` docstring for an example of use.

        Parameters
        ----------
        file_path : str
            The path to save the metamer object to

        """
        super().save(file_path, attrs=None)

    def to(self, *args, **kwargs):
        r"""Moves and/or casts the parameters and buffers.

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

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module

        Returns:
            Module: self
        """
        attrs = ['image', 'target_representation',
                 'metamer', 'model', 'saved_metamer']
        return super().to(*args, attrs=attrs, **kwargs)

    def load(self, file_path: str,
             map_location: Union[str, None] = None,
             **pickle_load_args):
        r"""Load all relevant stuff from a .pt file.

        This should be called by an initialized ``Metamer`` object -- we will
        ensure that ``image``, ``target_representation`` (and thus
        ``model``), and ``loss_function`` are all identical.

        Note this operates in place and so doesn't return anything.

        Parameters
        ----------
        file_path : str
            The path to load the synthesis object from
        map_location : str, optional
            map_location argument to pass to ``torch.load``. If you save
            stuff that was being run on a GPU and are loading onto a
            CPU, you'll need this to make sure everything lines up
            properly. This should be structured like the str you would
            pass to ``torch.device``
        pickle_load_args :
            any additional kwargs will be added to ``pickle_module.load`` via
            ``torch.load``, see that function's docstring for details.

        Examples
        --------
        >>> metamer = po.synth.Metamer(img, model)
        >>> metamer.synthesize(max_iter=10, store_progress=True)
        >>> metamer.save('metamers.pt')
        >>> metamer_copy = po.synth.Metamer(img, model)
        >>> metamer_copy.load('metamers.pt')

        Note that you must create a new instance of the Synthesis object and
        *then* load.

        """
        check_attributes = ['image', 'target_representation',
                            'range_penalty_lambda', 'allowed_range']
        check_loss_functions = ['loss_function']
        super().load(file_path, map_location=map_location,
                     check_attributes=check_attributes,
                     check_loss_functions=check_loss_functions,
                     **pickle_load_args)
        # make this require a grad again
        self.metamer.requires_grad_()
        # these are always supposed to be on cpu, but may get copied over to
        # gpu on load (which can cause problems when resuming synthesis), so
        # fix that.
        if len(self.saved_metamer) and self.saved_metamer.device.type != 'cpu':
            self.saved_metamer = self.saved_metamer.to('cpu')


def plot_loss(metamer: Metamer,
              iteration: Union[int, None] = None,
              ax: Union[mpl.axes.Axes, None] = None,
              **kwargs) -> mpl.axes.Axes:
    """Plot synthesis loss with log-scaled y axis.

    Plots ``metamer.losses`` over all iterations. Also plots a red dot at
    ``iteration``, to highlight the loss there. If ``iteration=None``, then the
    dot will be at the final iteration.

    Parameters
    ----------
    metamer :
        Metamer object whose loss we want to plot.
    iteration :
        Which iteration to display. If None, the default, we show
        the most recent one. Negative values are also allowed.
    ax :
        Pre-existing axes for plot. If None, we call ``plt.gca()``.
    kwargs :
        passed to plt.semilogy

    Returns
    -------
    ax :
        The matplotlib axes containing the plot.

    """
    if iteration is None:
        loss_idx = len(metamer.losses) - 1
    else:
        if iteration < 0:
            # in order to get the x-value of the dot to line up,
            # need to use this work-around
            loss_idx = len(metamer.losses) + iteration
        else:
            loss_idx = iteration
    if ax is None:
        ax = plt.gca()
    ax.semilogy(metamer.losses, **kwargs)
    try:
        ax.scatter(loss_idx, metamer.losses[loss_idx], c='r')
    except IndexError:
        # then there's no loss here
        pass
    ax.set(xlabel='Synthesis iteration', ylabel='Loss')
    return ax


def display_metamer(metamer: Metamer,
                    batch_idx: int = 0,
                    channel_idx: Union[int, None] = None,
                    zoom: Union[float, None] = None,
                    iteration: Union[int, None] = None,
                    ax: Union[mpl.axes.Axes, None] = None,
                    **kwargs) -> mpl.axes.Axes:
    """Display metamer.

    You can specify what iteration to view by using the ``iteration`` arg.
    The default, ``None``, shows the final one.

    We use ``plenoptic.imshow`` to display the metamer and attempt to
    automatically find the most reasonable zoom value. You can override this
    value using the zoom arg, but remember that ``plenoptic.imshow`` is
    opinionated about the size of the resulting image and will throw an
    Exception if the axis created is not big enough for the selected zoom.

    Parameters
    ----------
    metamer :
        Metamer object whose synthesized metamer we want to display.
    batch_idx :
        Which index to take from the batch dimension
    channel_idx :
        Which index to take from the channel dimension. If None, we assume
        image is RGB(A) and show all channels.
    iteration :
        Which iteration to display. If None, the default, we show
        the most recent one. Negative values are also allowed.
    ax :
        Pre-existing axes for plot. If None, we call ``plt.gca()``.
    zoom :
        How much to zoom in / enlarge the metamer, the ratio of display pixels
        to image pixels. If None (the default), we attempt to find the best
        value ourselves.
    kwargs :
        Passed to ``plenoptic.imshow``

    Returns
    -------
    ax :
        The matplotlib axes containing the plot.

    """
    if iteration is None:
        image = metamer.metamer
    else:
        image = metamer.saved_metamer[iteration]
    if batch_idx is None:
        raise Exception("batch_idx must be an integer!")
    # we're only plotting one image here, so if the user wants multiple
    # channels, they must be RGB
    if channel_idx is None and image.shape[1] > 1:
        as_rgb = True
    else:
        as_rgb = False
    if ax is None:
        ax = plt.gca()
    display.imshow(image, ax=ax, title='Metamer', zoom=zoom,
                   batch_idx=batch_idx, channel_idx=channel_idx,
                   as_rgb=as_rgb, **kwargs)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax


def _representation_error(metamer: Metamer,
                          iteration: Union[int, None] = None,
                          **kwargs) -> Tensor:
    r"""Get the representation error.

    This is ``metamer.model(metamer) - target_representation)``. If
    ``iteration`` is not None, we use
    ``metamer.model(saved_metamer[iteration])`` instead.

    Parameters
    ----------
    metamer :
        Metamer object whose model response error we want to compute.
    iteration :
        Which iteration to compute the model response error for. If None, we
        show the most recent one. Negative values are also allowed.
    kwargs :
        Passed to ``metamer.model.forward``

    Returns
    -------
    representation_error

    """
    if iteration is not None:
        metamer_rep = metamer.model(metamer.saved_metamer[iteration]).to(metamer.target_representation.device)
    else:
        metamer_rep = metamer.model(metamer.metamer, **kwargs)
    return metamer_rep - metamer.target_representation


def plot_representation_error(metamer: Metamer,
                              batch_idx: int = 0,
                              iteration: Union[int, None] = None,
                              ylim: Union[Tuple[float], None, Literal[False]] = None,
                              ax: Union[mpl.axes.Axes, None] = None,
                              as_rgb: bool = False,
                              **kwargs) -> List[mpl.axes.Axes]:
    r"""Plot distance ratio showing how close we are to convergence.

    We plot ``_representation_error(metamer, iteration)``. For more details, see
    ``plenoptic.tools.display.plot_representation``.

    Parameters
    ----------
    metamer :
        Metamer object whose synthesized metamer we want to display.
    batch_idx :
        Which index to take from the batch dimension
    iteration :
        Which iteration to display. If None, the default, we show
        the most recent one. Negative values are also allowed.
    ylim :
        If ``ylim`` is ``None``, we sets the axes' y-limits to be ``(-y_max,
        y_max)``, where ``y_max=np.abs(data).max()``. If it's ``False``, we do
        nothing. If a tuple, we use that range.
    ax :
        Pre-existing axes for plot. If None, we call ``plt.gca()``.
    as_rgb : bool, optional
        The model response can be image-like with multiple channels, and we
        have no way to determine whether it should be represented as an RGB
        image or not, so the user must set this flag to tell us. It will be
        ignored if the response doesn't look image-like or if the model has its
        own plot_representation_error() method. Else, it will be passed to
        `po.imshow()`, see that methods docstring for details.
    kwargs :
        Passed to ``metamer.model.forward``

    Returns
    -------
    axes :
        List of created axes

    """
    representation_error = _representation_error(metamer=metamer,
                                                 iteration=iteration, **kwargs)
    if ax is None:
        ax = plt.gca()
    return display.plot_representation(metamer.model, representation_error, ax,
                                       title="Model response error", ylim=ylim,
                                       batch_idx=batch_idx, as_rgb=as_rgb)


def plot_pixel_values(metamer: Metamer,
                      batch_idx: int = 0,
                      channel_idx: Union[int, None] = None,
                      iteration: Union[int, None] = None,
                      ylim: Union[Tuple[float], Literal[False]] = None,
                      ax: Union[mpl.axes.Axes, None] = None,
                      **kwargs) -> mpl.axes.Axes:
    r"""Plot histogram of pixel values of target signal and its metamer.

    As a way to check the distributions of pixel intensities and see
    if there's any values outside the allowed range

    Parameters
    ----------
    metamer :
        Metamer object with the images whose pixel values we want to compare.
    batch_idx :
        Which index to take from the batch dimension
    channel_idx :
        Which index to take from the channel dimension. If None, we use all
        channels (assumed use-case is RGB(A) images).
    iteration :
        Which iteration to display. If None, the default, we show
        the most recent one. Negative values are also allowed.
    ylim :
        if tuple, the ylimit to set for this axis. If False, we leave
        it untouched
    ax :
        Pre-existing axes for plot. If None, we call ``plt.gca()``.
    kwargs :
        passed to plt.hist

    Returns
    -------
    ax :
        Creates axes.

    """
    def _freedman_diaconis_bins(a):
        """Calculate number of hist bins using Freedman-Diaconis rule. copied from seaborn."""
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
        image = metamer.metamer[batch_idx]
    else:
        image = metamer.saved_metamer[iteration, batch_idx]
    image = metamer.image[batch_idx]
    if channel_idx is not None:
        image = image[channel_idx]
        image = image[channel_idx]
    if ax is None:
        ax = plt.gca()
    image = data.to_numpy(image).flatten()
    image = data.to_numpy(image).flatten()
    ax.hist(image, bins=min(_freedman_diaconis_bins(image), 50),
            label='metamer', **kwargs)
    ax.hist(image, bins=min(_freedman_diaconis_bins(image), 50),
            label='target image', **kwargs)
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title("Histogram of pixel values")
    return ax


def _setup_synthesis_fig(fig: Union[mpl.figure.Figure, None] = None,
                         axes_idx: Dict[str, int] = {},
                         figsize: Union[Tuple[float], None] = None,
                         included_plots: List[str] = ['display_metamer',
                                                      'plot_loss',
                                                      'plot_representation_error'],
                         display_metamer_width: float = 1,
                         plot_loss_width: float = 1,
                         plot_representation_error_width: float = 1,
                         plot_pixel_values_width: float = 1) -> Tuple[mpl.figure.Figure, List[mpl.axes.Axes], Dict[str, int]]:
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
    fig :
        The figure to plot on or None. If None, we create a new figure
    axes_idx :
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: loss, representation_error,
        pixel_values, misc. Values should all be ints. If you tell this
        function to create a plot that doesn't have a corresponding key, we
        find the lowest int that is not already in the dict, so if you have
        axes that you want unchanged, place their idx in misc.
    figsize :
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If None, we attempt to
        make our best guess, aiming to have relative width=1 correspond to 5
    included_plots :
        Which plots to include. Must be some subset of ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values'``.
    display_metamer_width :
        Relative width of the axis for the synthesized metamer.
    plot_loss_width :
        Relative width of the axis for loss plot.
    plot_representation_error_width :
        Relative width of the axis for representation error plot.
    plot_pixel_values_width :
        Relative width of the axis for image pixel intensities histograms.

    Returns
    -------
    fig :
        The figure to plot on
    axes :
        List or array of axes contained in fig
    axes_idx :
        Dictionary identifying the idx for each plot type

    """
    n_subplots = 0
    axes_idx = axes_idx.copy()
    width_ratios = []
    if "display_metamer" in included_plots:
        n_subplots += 1
        width_ratios.append(display_metamer_width)
        if 'display_metamer' not in axes_idx.keys():
            axes_idx['display_metamer'] = data._find_min_int(axes_idx.values())
    if "plot_loss" in included_plots:
        n_subplots += 1
        width_ratios.append(plot_loss_width)
        if 'plot_loss' not in axes_idx.keys():
            axes_idx['plot_loss'] = data._find_min_int(axes_idx.values())
    if "plot_representation_error" in included_plots:
        n_subplots += 1
        width_ratios.append(plot_representation_error_width)
        if 'plot_representation_error' not in axes_idx.keys():
            axes_idx['plot_representation_error'] = data._find_min_int(axes_idx.values())
    if "plot_pixel_values" in included_plots:
        n_subplots += 1
        width_ratios.append(plot_pixel_values_width)
        if 'plot_pixel_values' not in axes_idx.keys():
            axes_idx['plot_pixel_values'] = data._find_min_int(axes_idx.values())
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
    # make sure misc contains all the empty axes
    misc_axes = axes_idx.get('misc', [])
    all_axes = []
    for i in axes_idx.values():
        # so if it's a list of ints
        if hasattr(i, '__iter__'):
            all_axes.extend(i)
        else:
            all_axes.append(i)
    misc_axes += [i for i, _ in enumerate(fig.axes) if i not in all_axes]
    axes_idx['misc'] = misc_axes
    return fig, axes, axes_idx


def plot_synthesis_status(metamer: Metamer,
                          batch_idx: int = 0,
                          channel_idx: Union[int, None] = None,
                          iteration: Union[int, None] = None,
                          ylim: Union[Tuple[float], Literal[False]] = None,
                          vrange: Union[Tuple[float], str] = 'indep1',
                          zoom: Union[float, None] = None,
                          plot_representation_error_as_rgb: bool = False,
                          fig: Union[mpl.figure.Figure, None] = None,
                          axes_idx: Dict[str, int] = {},
                          figsize: Union[Tuple[float], None] = None,
                          included_plots: List[str] = ['display_metamer',
                                                       'plot_loss',
                                                       'plot_representation_error'],
                          width_ratios: Dict[str, float] = {},
                          ) -> Tuple[mpl.figure.Figure, Dict[str, int]]:
    r"""Make a plot showing synthesis status.

    We create several subplots to analyze this. By default, we create three
    subplots on a new figure: the first one contains the synthesized metamer,
    the second contains the loss, and the third contains the model response
    error.

    There is an optional additional plot: ``plot_pixel_values``, a histogram of
    pixel values of the metamer and target image.

    The plots to include are specified by including their name in the
    ``included_plots`` list. All plots can be created separately using the
    method with the same name.

    Parameters
    ----------
    metamer :
        Metamer object whose status we want to plot.
    batch_idx :
        Which index to take from the batch dimension
    channel_idx :
        Which index to take from the channel dimension. If None, we use all
        channels (assumed use-case is RGB(A) image).
    iteration :
        Which iteration to display. If None, the default, we show
        the most recent one. Negative values are also allowed.
    ylim :
        The ylimit to use for the representation_error plot. We pass
        this value directly to ``plot_representation_error``
    vrange :
        The vrange option to pass to ``display_metamer()``. See
        docstring of ``imshow`` for possible values.
    zoom :
        How much to zoom in / enlarge the metamer, the ratio
        of display pixels to image pixels. If None (the default), we
        attempt to find the best value ourselves.
    plot_representation_error_as_rgb : bool, optional
        The model response can be image-like with multiple channels, and we
        have no way to determine whether it should be represented as an RGB
        image or not, so the user must set this flag to tell us. It will be
        ignored if the response doesn't look image-like or if the
        model has its own plot_representation_error() method. Else, it will
        be passed to `po.imshow()`, see that methods docstring for details.
    fig :
        if None, we create a new figure. otherwise we assume this is
        an empty figure that has the appropriate size and number of
        subplots
    axes_idx :
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values',
        'misc'``. Values should all be ints. If you tell this function to
        create a plot that doesn't have a corresponding key, we find the lowest
        int that is not already in the dict, so if you have axes that you want
        unchanged, place their idx in ``'misc'``.
    figsize :
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If None, we attempt to
        make our best guess, aiming to have each axis be of size (5, 5)
    included_plots :
        Which plots to include. Must be some subset of ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values'``.
    width_ratios :
        By default, all plots axes will have the same width. To change
        that, specify their relative widths using the keys: ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values'`` and floats
        specifying their relative width. Any not included will be assumed to be
        1.

    Returns
    -------
    fig :
        The figure containing this plot
    axes_idx :
        Dictionary giving index of each plot.

    """
    if iteration is not None and not metamer.store_progress:
        raise Exception("synthesis() was run with store_progress=False, "
                        "cannot specify which iteration to plot (only"
                        " last one, with iteration=None)")
    if metamer.metamer.ndim not in [3, 4]:
        raise ValueError("plot_synthesis_status() expects 3 or 4d data;"
                         "unexpected behavior will result otherwise!")
    width_ratios = {f'{k}_width': v for k, v in width_ratios.items()}
    fig, axes, axes_idx = _setup_synthesis_fig(fig, axes_idx, figsize,
                                               included_plots,
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

    if "display_metamer" in included_plots:
        display_metamer(metamer, batch_idx=batch_idx,
                        channel_idx=channel_idx,
                        iteration=iteration,
                        ax=axes[axes_idx['display_metamer']],
                        zoom=zoom, vrange=vrange)
    if "plot_loss" in included_plots:
        plot_loss(metamer, iteration=iteration, ax=axes[axes_idx['plot_loss']])
    if "plot_representation_error" in included_plots:
        plot_representation_error(metamer, batch_idx=batch_idx,
                                  iteration=iteration,
                                  ax=axes[axes_idx['plot_representation_error']],
                                  ylim=ylim,
                                  as_rgb=plot_representation_error_as_rgb)
        # this can add a bunch of axes, so this will try and figure
        # them out
        new_axes = [i for i, _ in enumerate(fig.axes) if not
                    check_iterables(i, axes_idx.values())] + [axes_idx['plot_representation_error']]
        axes_idx['plot_representation_error'] = new_axes
    if "plot_pixel_values" in included_plots:
        plot_pixel_values(metamer, batch_idx=batch_idx,
                          channel_idx=channel_idx,
                          iteration=iteration,
                          ax=axes[axes_idx['plot_pixel_values']])
    return fig, axes_idx


def animate(metamer: Metamer,
            framerate: int = 10,
            batch_idx: int = 0,
            channel_idx: Union[int, None] = None,
            ylim: Union[str, Tuple[float], Literal[False]] = None,
            vrange: Union[Tuple[float], str] = (0, 1),
            zoom: Union[float, None] = None,
            plot_representation_error_as_rgb: bool = False,
            fig: Union[mpl.figure.Figure, None] = None,
            axes_idx: Dict[str, int] = {},
            figsize: Union[Tuple[float], None] = None,
            included_plots: List[str] = ['display_metamer',
                                         'plot_loss',
                                         'plot_representation_error'],
            width_ratios: Dict[str, float] = {},
            ) -> mpl.animation.FuncAnimation:
    r"""Animate synthesis progress.

    This is essentially the figure produced by
    ``metamer.plot_synthesis_status`` animated over time, for each stored
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
    metamer :
        Metamer object whose synthesis we want to animate.
    framerate :
        How many frames a second to display.
    batch_idx :
        Which index to take from the batch dimension
    channel_idx :
        Which index to take from the channel dimension. If None, we use all
        channels (assumed use-case is RGB(A) image).
    ylim :
        The y-limits of the representation_error plot:

        * If a tuple, then this is the ylim of all plots

        * If None, then all plots have the same limits, all
          symmetric about 0 with a limit of
          ``np.abs(representation_error).max()`` (for the initial
          representation_error)

        * If False, don't modify limits.

        * If a string, must be 'rescale' or of the form 'rescaleN',
          where N can be any integer. If 'rescaleN', we rescale the
          limits every N frames (we rescale as if ylim = None). If
          'rescale', then we do this 10 times over the course of the
          animation

    vrange :
        The vrange option to pass to ``display_metamer()``. See
        docstring of ``imshow`` for possible values.
    zoom :
        How much to zoom in / enlarge the metamer, the ratio
        of display pixels to image pixels. If None (the default), we
        attempt to find the best value ourselves.
    plot_representation_error_as_rgb :
        The representation can be image-like with multiple channels, and we
        have no way to determine whether it should be represented as an RGB
        image or not, so the user must set this flag to tell us. It will be
        ignored if the representation doesn't look image-like or if the
        model has its own plot_representation_error() method. Else, it will
        be passed to `po.imshow()`, see that methods docstring for details.
        since plot_synthesis_status normally sets it up for us
    fig :
        If None, create the figure from scratch. Else, should be an empty
        figure with enough axes (the expected use here is have same-size
        movies with different plots).
    axes_idx :
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values',
        'misc'``. Values should all be ints. If you tell this function to
        create a plot that doesn't have a corresponding key, we find the lowest
        int that is not already in the dict, so if you have axes that you want
        unchanged, place their idx in ``'misc'``.
    figsize :
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If None, we attempt to
        make our best guess, aiming to have each axis be of size (5, 5)
    included_plots :
        Which plots to include. Must be some subset of ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values'``.
    width_ratios :
        By default, all plots axes will have the same width. To change
        that, specify their relative widths using the keys: ``'display_metamer',
        'plot_loss', 'plot_representation_error', 'plot_pixel_values'`` and floats
        specifying their relative width. Any not included will be assumed to be
        1.

    Returns
    -------
    anim :
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
    if not metamer.store_progress:
        raise Exception("synthesize() was run with store_progress=False,"
                        " cannot animate!")
    if metamer.metamer.ndim not in [3, 4]:
        raise ValueError("animate() expects 3 or 4d data; unexpected"
                         " behavior will result otherwise!")
    if metamer.target_representation.ndimension() == 4:
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
                ylim_rescale_interval = int((metamer.saved_metamer.shape[0] - 1) // 10)
                if ylim_rescale_interval == 0:
                    ylim_rescale_interval = int(metamer.saved_metamer.shape[0] - 1)
            ylim = None
        else:
            raise Exception("Don't know how to handle ylim %s!" % ylim)
    except AttributeError:
        # this way we'll never rescale
        ylim_rescale_interval = len(metamer.saved_metamer)+1
    # we run plot_synthesis_status to initialize the figure if either fig is
    # None or if there are no titles on any axes, which we assume means that
    # it's an empty figure
    if fig is None or not any([ax.get_title() for ax in fig.axes]):
        fig, axes_idx = plot_synthesis_status(metamer_obj=metamer_obj,
                                              batch_idx=batch_idx,
                                              channel_idx=channel_idx,
                                              iteration=0, figsize=figsize,
                                              ylim=ylim, vrange=vrange,
                                              loss=loss,
                                              zoom=zoom, fig=fig,
                                              axes_idx=axes_idx,
                                              included_plots=included_plots,
                                              plot_representation_error_as_rgb=plot_representation_error_as_rgb,
                                              width_ratios=width_ratios)
    # grab the artist for the second plot (we don't need to do this for the
    # metamer or representation plot, because we use the update_plot
    # function for that)
    if 'plot_loss' in included_plots:
        scat = fig.axes[axes_idx['loss']].collections[0]
    # can have multiple plots
    if 'plot_representation_error' in included_plots:
        try:
            model_resp_error_axes = [fig.axes[i] for i in axes_idx['plot_representation_error']]
        except TypeError:
            # in this case, axes_idx['plot_representation_error'] is not iterable and so is
            # a single value
            model_resp_error_axes = [fig.axes[axes_idx['plot_representation_error']]]
    else:
        model_resp_error_axes = []
    # can also have multiple plots

    if metamer.target_representation.ndimension() == 4:
        warnings.warn("Looks like representation is image-like, haven't fully thought out how"
                      " to best handle rescaling color ranges yet!")
        # replace the bit of the title that specifies the range,
        # since we don't make any promises about that. we have to do
        # this here because we need the figure to have been created
        for ax in model_resp_error_axes:
            ax.set_title(re.sub(r'\n range: .* \n', '\n\n', ax.get_title()))

    def movie_plot(i):
        artists = []
        if metamer:
            artists.extend(display.update_plot(fig.axes[axes_idx['display_metamer']],
                                               data=metamer.saved_metamer[i],
                                               batch_idx=batch_idx))
        if 'plot_representation_error' in included_plots:
            model_resp_error = _representation_error(metamer,
                                                     iteration=i)

            # we pass model_resp_error_axes to update, and we've grabbed
            # the right things above
            artists.extend(display.update_plot(model_resp_error_axes,
                                               batch_idx=batch_idx,
                                               model=metamer.model,
                                               data=model_resp_error))
            # again, we know that model_resp_error_axes contains all the axes
            # with the representation ratio info
            if ((i+1) % ylim_rescale_interval) == 0:
                if metamer.target_representation.ndimension() == 3:
                    display.rescale_ylim(model_resp_error_axes,
                                         model_resp_error)
        if 'plot_pixel_values' in included_plots:
            # this is the dumbest way to do this, but it's simple --
            # clearing the axes can cause problems if the user has, for
            # example, changed the tick locator or formatter. not sure how
            # to handle this best right now
            fig.axes[axes_idx['plot_pixel_values']].clear()
            plot_pixel_values(metamer, batch_idx=batch_idx,
                              channel_idx=channel_idx, iteration=i,
                              ax=fig.axes[axes_idx['plot_pixel_values']])
        if loss:
            # loss always contains values from every iteration, but everything
            # else will be subsampled.
            x_val = i*metamer.store_progress
            scat.set_offsets((x_val, metamer.losses[x_val]))
            artists.append(scat)
        # as long as blitting is True, need to return a sequence of artists
        return artists

    # don't need an init_func, since we handle initialization ourselves
    anim = mpl.animation.FuncAnimation(fig, movie_plot,
                                       frames=len(metamer.saved_metamer),
                                       blit=True, interval=1000./framerate,
                                       repeat=False)
    plt.close(fig)
    return anim
