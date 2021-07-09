"""Synthesize model metamers."""
import torch
import numpy as np
from torch import Tensor
from tqdm.auto import tqdm
from ..tools import optim
from typing import Union, Tuple, Callable
from typing_extensions import Literal
from .synthesis import Synthesis
import warnings


class Metamer(Synthesis):
    r"""Synthesize metamers for image-computable differentiable models.

    Following the basic idea in [1]_, this module creates a metamer for a given
    model on a given image. We start with some random noise and iteratively
    adjust the pixel values so as to match the representation of the
    ``synthesized_signal`` and ``base_signal``. This is optimization though, so
    you'll probably need to experiment with the optimization hyper-parameters
    before you find a good solution.

    There are two types of objects you can pass as your models: torch.nn.Module
    or functions, which correspond to using a visual model or metric,
    respectively. See the `MAD_Competition` notebook for more details on this.

    All ``saved_`` attributes are initialized as empty lists and will be
    non-empty if the ``store_progress`` arg to ``synthesize()`` is not
    ``False``. They will be appended to on every iteration if
    ``store_progress=True`` or every ``store_progress`` iterations if it's an
    ``int``.

    All ``scales`` attributes will only be non-None if ``coarse_to_fine`` is
    not ``False``. See ``Metamer`` tutorial for more details.

    Parameters
    ----------
    base_signal : torch.Tensor or array_like
        A 4d tensor, this is the image whose representation we wish to
        match. If this is not a tensor, we try to cast it as one.
    model : torch.nn.Module or function
        A visual model or metric, see `MAD_Competition` notebook for more
        details
    loss_function : callable or None, optional
        the loss function to use to compare the representations of the
        models in order to determine their loss. Only used for the
        Module models, ignored otherwise. If None, we use the default:
        the element-wise 2-norm. See `MAD_Competition` notebook for more
        details
    model_kwargs :
        if model is a function (that is, you're using a metric instead
        of a model), then there might be additional arguments you want
        to pass it at run-time. Note that this means they will be passed
        on every call.

    Attributes
    ----------
    base_representation : torch.Tensor
        Whatever is returned by ``model(base_signal)``, this is
        what we match in order to create a metamer
    synthesized_signal : torch.Tensor
        The metamer. This may be unfinished depending on how many
        iterations we've run for.
    synthesized_represetation: torch.Tensor
        Whatever is returned by ``model(synthesized_signal)``; we're
        trying to make this identical to ``self.base_representation``
    seed : int
        Number with which we seeded pytorch and numpy's random number
        generators
    loss : list
        A list of our loss over iterations.
    gradient : list
        A list of the gradient over iterations.
    learning_rate : list
        A list of the learning_rate over iterations. We use a scheduler
        that gradually reduces this over time, so it won't be constant.
    pixel_change : list
        A list containing the max pixel change over iterations
        (``pixel_change[i]`` is the max pixel change in
        ``synthesized_signal`` between iterations ``i`` and ``i-1``). note
        this is calculated before any clamping, so may have some very
        large numbers in the beginning
    saved_signal : torch.Tensor or list
        Saved ``self.synthesized_signal`` for later examination.
    saved_representation : torch.Tensor or list
        Saved ``self.synthesized_representation`` for later examination.
    saved_signal_gradient : torch.Tensor or list
        Saved ``self.synthesized_signal.grad`` for later examination.
    saved_representation_gradient : torch.Tensor or list
        Saved ``self.synthesized_representation.grad`` for later examination.
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

    def __init__(self, model: torch.nn.Module, target_signal: Tensor,
                 loss_function: Callable[[Tensor, Tensor], Tensor] = optim.mse,
                 range_penalty_lambda: float = .1,
                 allowed_range: Tuple[float] = (0, 1),
                 initial_image: Union[None, Tensor] = None):
        self.model = model
        self.target_signal = target_signal
        if target_signal.ndimension() < 4:
            raise Exception("target_signal must be torch.Size([n_batch, "
                            "n_channels, im_height, im_width]) but got "
                            f"{target_signal.size()}")
        self.target_model_response = self.model(self.target_signal)
        self.optimizer = None
        self.scheduler = None
        self.losses = []
        self.learning_rate = []
        self.gradient_norm = []
        self.pixel_change = []
        self.loss_function = loss_function
        self.range_penalty_lambda = range_penalty_lambda
        self.allowed_range = allowed_range
        self._init_synthesized_signal(initial_image)
        self.coarse_to_fine = False
        self.scales = None
        self.store_progress = None
        self.saved_signal = []
        self.saved_model_response = []

    def _init_synthesized_signal(self,
                                 initial_image: Union[None, Tensor] = None):
        """Initialize the synthesized image.

        Set the ``self.synthesized_signal`` attribute to be a parameter with
        the user-supplied data, making sure it's the right shape.

        Also initialize the ``self.synthesized_representation`` attribute

        Parameters
        ----------
        initial_image :
            The tensor we use to initialize the metamer. If None (the
            default), we initialize with uniformly-distributed random
            noise lying between 0 and 1.

        """
        if initial_image is None:
            synthesized_signal = torch.rand_like(self.target_signal,
                                                 requires_grad=True)
        else:
            synthesized_signal = initial_image.clone().detach()
            synthesized_signal = synthesized_signal.to(dtype=self.target_signal.dtype,
                                                       device=self.target_signal.device)
            synthesized_signal.requires_grad_()
            if synthesized_signal.ndimension() < 4:
                raise Exception("synthesized_signal must be torch.Size([n_batch"
                                ", n_channels, im_height, im_width]) but got "
                                f"{synthesized_signal.size()}")
            if synthesized_signal.size() != self.target_signal.size():
                raise Exception("synthesized_signal and target_signal must be"
                                " same size!")
        self.synthesized_signal = synthesized_signal

    def _init_ctf(self, coarse_to_fine, change_scale_criterion,
                  stop_criterion):
        """Initialize stuff related to coarse-to-fine."""
        if coarse_to_fine not in [False, 'separate', 'together']:
            raise Exception(f"Don't know how to handle value {coarse_to_fine}!"
                            " Must be one of: False, 'separate', 'together'")
        if coarse_to_fine and self.scales is None:
            # if self.scales is not None, we're continuing a previous version
            # and want to continue. this list comprehension creates a new
            # object, so we don't modify model.scales
            self.scales = [i for i in self.model.scales[:-1]]
            if coarse_to_fine == 'separate':
                self.scales += [self.model.scales[-1]]
            self.scales += ['all']
            self.scales_timing = dict((k, []) for k in self.scales)
            self.scales_timing[self.scales[0]].append(0)
            self.scales_finished = []
            self.scales_loss = []
            if stop_criterion >= change_scale_criterion:
                raise Exception("stop_criterion must be strictly less than "
                                "coarse-to-fine's change_scale_criterion, or"
                                " things get weird!")
        self.coarse_to_fine = coarse_to_fine

    def _init_optimizer(self, optimizer, scheduler):
        """Initialize optimizer and scheduler."""
        if optimizer is None:
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam([self.synthesized_signal],
                                                  lr=.01, amsgrad=True)
        else:
            if self.optimizer is not None:
                raise Exception("When resuming synthesis, optimizer arg must be None!")
            params = optimizer.param_groups[0]['params']
            if len(params) != 1 or not torch.equal(params[0], self.synthesized_signal):
                raise Exception("For metamer synthesis, optimizer must have one "
                                "parameter, the metamer we're synthesizing.")
            self.optimizer = optimizer
        self.scheduler = scheduler

    def _init_store_progress(self, store_progress: Union[bool, int]):
        """Initialize store_progress-related attributes.

        Sets the ``self.store_progress`` attribute, as well as changing
        ``saved_signal`` and ``saved_model_response`` attibutes to lists so we
        can append to them. finally, adds first value to ``saved_signal`` and
        ``saved_model_response`` if they're empty.

        Parameters
        ----------
        store_progress : bool or int, optional
            Whether we should store the representation of the metamer
            and the metamer image in progress on every iteration. If
            False, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress``
            iterations (note then that 0 is the same as False and 1 the
            same as True). If True or int>0, ``self.saved_signal``
            contains the stored images, and ``self.saved_model_response``
            contains the stored model response.

        """
        # python's implicit boolean-ness means we can do this! it will evaluate
        # to False for False and 0, and True for True and every int >= 1
        if store_progress is None and self.store_progress is not None:
            store_progress = self.store_progress
        if store_progress:
            if store_progress is True:
                store_progress = 1
            # if this is not the first time synthesize is being run for this
            # metamer object, saved_signal/saved_model_response will be tensors
            # instead of lists. This converts them back to lists so we can use
            # append. If it's the first time, they'll be empty lists and this
            # does nothing
            self.saved_signal = list(self.saved_signal)
            self.saved_model_response = list(self.saved_model_response)
            # first time synthesize() is called, add the initial synthesized
            # signal and model response (on subsequent calls, this is already
            # part of saved_signal / saved_model_response).
            if len(self.saved_signal) == 0:
                self.saved_signal.append(self.synthesized_signal.clone().to('cpu'))
                self.saved_model_response.append(self.model(self.synthesized_signal).to('cpu'))
        if self.store_progress is not None and store_progress != self.store_progress:
            # we require store_progress to be the same because otherwise the
            # subsampling relationship between attrs that are stored every
            # iteration (loss, gradient, etc) and those that are stored every
            # store_progress iteration (e.g., saved_signal,
            # saved_model_response) changes partway through and that's annoying
            raise Exception("If you've already run synthesize() before, must "
                            "re-run it with same store_progress arg. You "
                            f"passed {store_progress} instead of "
                            f"{self.store_progress} (True is equivalent to 1)")
        self.store_progress = store_progress

    def _check_nan_loss(self, loss: Tensor) -> bool:
        """Check if loss is nan and, if so, return True.

        This checks if loss is NaN and, if so, updates
        synthesized_signal/model_response to be several iterations ago (so
        they're meaningful) and then returns True.

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
                          "synthesized_signal to our last saved values (which "
                          "means this will throw an IndexError if you're not "
                          "saving anything)!")
            # need to use the -2 index because the last one will be the one
            # full of NaNs. this happens because the loss is computed before
            # calculating the gradient and updating synthesized_signal;
            # therefore the iteration where loss is NaN is the one *after* the
            # iteration where synthesized_signal (and thus
            # synthesized_representation) started to have NaN values. this will
            # fail if it hits a nan before store_progress iterations (because
            # then saved_signal only has a length of 1) but in that case, you
            # have more severe problems
            self.synthesized_signal = torch.nn.Parameter(self.saved_signal[-2])
            return True
        return False

    def _store(self, i: int) -> bool:
        """Store synthesized_signal anbd model response, if appropriate.

        if it's the right iteration, we update: ``saved_signal,
        saved_model_response``

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
                self.saved_signal.append(self.synthesized_signal.clone().to('cpu'))
                self.saved_model_response.append(self.model(self.synthesized_signal).to('cpu'))
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
         <-------' '---->Is ``coarse_to_fine`` True?
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

    def objective_function(self, synthesized_model_response: Tensor,
                           target_model_response: Union[Tensor, None] = None) -> Tensor:
        """Compute the metamer synthesis loss.

        This calls self.loss_function on ``synthesized_model_response`` and
        ``target_model_response`` and then adds the weighted range penalty.

        Parameters
        ----------
        synthesized_model_response :
            Model response to ``synthesized_signal``.
        target_model_response :
            Model response to ``target_signal``. If None, we use
            ``self.target_model_response``.

        Returns
        -------
        loss

        """
        if target_model_response is None:
            target_model_response = self.target_model_response
        loss = self.loss_function(synthesized_model_response,
                                  target_model_response)
        range_penalty = optim.penalize_range(self.synthesized_signal,
                                             self.allowed_range)
        return loss + self.range_penalty_lambda * range_penalty

    def _closure(self) -> Tensor:
        r"""An abstraction of the gradient calculation, before the optimization step.

        This enables optimization algorithms that perform several evaluations
        of the gradient before taking a step (ie. second order methods like
        LBFGS).

        Additionally, this is where:

        - ``synthesized_model_response`` is calculated, and thus any
          modifications to the model's forward call (e.g., specifying `scale`
          kwarg for coarse-to-fine) should happen.

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
        synthesized_model_response = self.model(self.synthesized_signal,
                                                **analyze_kwargs)
        if analyze_kwargs:
            target_resp = self.model(self.target_signal, **analyze_kwargs)
        else:
            target_resp = None

        loss = self.objective_function(synthesized_model_response, target_resp)
        loss.backward(retain_graph=True)

        return loss

    def _optimizer_step(self, pbar: tqdm,
                        **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Compute and propagate gradients, then step the optimizer to update synthesized_signal.

        Parameters
        ----------
        pbar :
            A tqdm progress-bar, which we update with a postfix
            describing the current loss, gradient norm, and learning
            rate (it already tells us which iteration and the time
            elapsed).
        kwargs :
            Will also display in the progress bar's postfix

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
            synthesized_signal between this step and the last

        """
        last_iter_synthesized_signal = self.synthesized_signal.clone()
        postfix_dict = {}
        if self.coarse_to_fine:
            # the last scale will be 'all', and we never remove
            # it. Otherwise, check to see if it looks like loss has
            # stopped declining and, if so, switch to the next scale
            if (len(self.scales) > 1 and len(self.scales_loss) > self.loss_change_iter and
                abs(self.scales_loss[-1] - self.scales_loss[-self.loss_change_iter]) < self.loss_change_thresh and
                len(self.loss) - self.scales_timing[self.scales[0]][0] > self.loss_change_iter):
                self.scales_timing[self.scales[0]].append(len(self.loss)-1)
                self.scales_finished.append(self.scales.pop(0))
                self.scales_timing[self.scales[0]].append(len(self.loss))
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
        # not the same scale we computed synthesized_representation using)
        if self.coarse_to_fine:
            postfix_dict['current_scale_loss'] = loss.item()
            # and we also want to keep track of this
            self.scales_loss.append(loss.item())
        grad_norm = self.synthesized_signal.grad.detach().norm()
        if grad_norm.item() != grad_norm.item():
            raise Exception('found a NaN in the gradients during optimization')

        # optionally step the scheduler
        if self.scheduler is not None:
            self.scheduler.step(loss.item())

        if self.coarse_to_fine and self.scales[0] != 'all':
            with torch.no_grad():
                tmp_im = self.synthesized_signal.detach().clone()
                full_synthesized_rep = self.model(tmp_im)
                loss = self.objective_function(full_synthesized_rep)
        else:
            loss = self.objective_function(self.model(self.synthesized_signal))

        pixel_change = torch.max(torch.abs(self.synthesized_signal - last_iter_synthesized_signal))
        # for display purposes, always want loss to be positive
        postfix_dict.update(dict(loss=f"{abs(loss.item()):.04e}",
                                 gradient_norm=f"{grad_norm.item():.04e}",
                                 learning_rate=self.optimizer.param_groups[0]['lr'],
                                 pixel_change=f"{pixel_change:.04e}",
                                 **kwargs))
        # add extra info here if you want it to show up in progress bar
        pbar.set_postfix(**postfix_dict)
        return loss, grad_norm, self.optimizer.param_groups[0]['lr'], pixel_change

    def synthesize(self, max_iter: int = 100,
                   optimizer: Union[None, torch.optim.Optimizer] = None,
                   scheduler: Union[None, torch.optim.lr_scheduler._LRScheduler] = None,
                   store_progress: Union[bool, int] = False,
                   stop_criterion: float = 1e-4, stop_iters_to_check: int = 50,
                   coarse_to_fine: Literal['together', 'separate', False] = False,
                   coarse_to_fine_kwargs: dict = {'change_scale_criterion': 1e-2,
                                                  'ctf_iters_to_check': 50}):
        r"""Synthesize a metamer.

        This is the main method, which updates the ``initial_image`` until its
        representation matches that of ``base_signal``.

        We run this until either we reach ``max_iter`` or the change
        over the past ``loss_change_iter`` iterations is less than
        ``loss_thresh``, whichever comes first

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
            coarse-to-fine optimization.

        Returns
        -------
        synthesized_signal : torch.Tensor
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
            loss, g, lr, pixel_change = self._optimizer_step(pbar)
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
            self.saved_model_response = torch.stack(self.saved_model_response)
            self.saved_signal = torch.stack(self.saved_signal)

        # return metamer
        return self.synthesized_signal

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
        attrs = ['target_signal', 'target_model_response',
                 'synthesized_signal', 'model', 'saved_signal',
                 'saved_model_response']
        return super().to(*args, attrs=attrs, **kwargs)

    def load(self, file_path: str,
             map_location: Union[str, None] = None,
             **pickle_load_args):
        r"""Load all relevant stuff from a .pt file.

        This should be called by an initialized ``Metamer`` object -- we will
        ensure that ``target_signal``, ``target_model_response`` (and thus
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
        check_attributes = ['target_signal', 'target_model_response',
                            'range_penalty_lambda', 'allowed_range']
        check_loss_functions = ['loss_function']
        super().load(file_path, map_location=map_location,
                     check_attributes=check_attributes,
                     check_loss_functions=check_loss_functions,
                     **pickle_load_args)

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
        fig = super().plot_value_comparison(value=value, batch_idx=batch_idx,
                                            channel_idx=channel_idx,
                                            iteration=iteration,
                                            figsize=figsize, ax=ax, func=func,
                                            hist2d_nbins=hist2d_nbins,
                                            hist2d_cmap=hist2d_cmap,
                                            scatter_subsample=scatter_subsample,
                                            **kwargs)
        if ax is None:
            ax = fig.axes[0]
        ax.set(xlabel=f'Target {value}')
        return fig
