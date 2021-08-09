"""Run MAD Competition."""
import torch
import numpy as np
from torch import Tensor
from tqdm.auto import tqdm
from ..tools import optim, display, data
from typing import Union, Tuple, Callable, List, Dict
from typing_extensions import Literal
from .synthesis import Synthesis
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt


class MADCompetition(Synthesis):
    r"""Synthesize a single maximally-differentiating image for two metrics.

    Following the basic idea in [1]_, this class synthesizes a
    maximally-differentiating image for two given metrics, based on a given
    image. We start by adding noise to this image and then iteratively
    adjusting its pixels so as to either minimize or maximize
    ``synthesis_metric`` while holding the value of ``fixed_metric`` constant.

    Note that a full set of images MAD Competition images consists of two
    pairs: a maximal and a minimal image for each metric. A single
    instantiation of ``MADCompetition`` will generate one of these four images.

    Parameters
    ----------
    reference_signal :
        A 4d tensor, this is the image whose representation we wish to
        match. If this is not a tensor, we try to cast it as one.
    synthesis_metric :
        The metric whose value you wish to minimize or maximize, which takes
        two tensors and returns a scalar.
    fixed_metric :
        The metric whose value you wish to keep fixed, which takes two tensors
        and returns a scalar.
    synthesis_target :
        Whether you wish to minimize or maximize ``synthesis_metric``.
    initial_noise :
        Standard deviation of the Gaussian noise used to initialize
        ``synthesized_signal`` from ``reference_signal``.
    metric_tradeoff_lambda :
        Lambda to multiply by ``fixed_metric`` loss and add to
        ``synthesis_metric`` loss. If ``None``, we pick a value so the two
        initial losses are approximately equal in magnitude.
    range_penalty_lambda :
        Lambda to multiply by range penalty and add to loss.
    allowable_range :
        Range (inclusive) of allowed pixel values. Any values outside this
        range will be penalized.

    Attributes
    ----------
    synthesized_signal : torch.Tensor
        The metamer. This may be unfinished depending on how many
        iterations we've run for.
        generators
    losses : list
        A list of the objective function's loss over iterations.
    synthesis_metric_loss : list
        A list of the ``synthesis_metric`` loss over iterations.
    fixed_metric_loss : list
        A list of the ``fixed_metric`` loss over iterations.
    gradient_norm : list
        A list of the gradient_norm over iterations.
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

    References
    ----------
    .. [1] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
           competition: A methodology for comparing computational models of
           perceptual discriminability. Journal of Vision, 8(12), 1–13.
           http://dx.doi.org/10.1167/8.12.8

    """

    def __init__(self, reference_signal: Tensor,
                 synthesis_metric: Union[torch.nn.Module, Callable[[Tensor, Tensor], Tensor]],
                 fixed_metric: Union[torch.nn.Module, Callable[[Tensor, Tensor], Tensor]],
                 synthesis_target: Literal['min', 'max'],
                 initial_noise: float = .1,
                 metric_tradeoff_lambda: Union[None, float] = None,
                 range_penalty_lambda: float = .1,
                 allowed_range: Tuple[float] = (0, 1),):
        self.synthesis_metric = synthesis_metric
        self.fixed_metric = fixed_metric
        self.reference_signal = reference_signal
        if reference_signal.ndimension() < 4:
            raise Exception("reference_signal must be torch.Size([n_batch, "
                            "n_channels, im_height, im_width]) but got "
                            f"{reference_signal.size()}")
        self.optimizer = None
        self.scheduler = None
        self.losses = []
        self.synthesis_metric_loss = []
        self.fixed_metric_loss = []
        if synthesis_target not in ['min', 'max']:
            raise Exception("synthessi_target must be one of {'min', 'max'}, but got "
                            f"value {synthesis_target} instead!")
        self.synthesis_target = synthesis_target
        self.learning_rate = []
        self.gradient_norm = []
        self.pixel_change = []
        self.range_penalty_lambda = range_penalty_lambda
        self.allowed_range = allowed_range
        self._init_synthesized_signal(initial_noise)
        # If no metric_tradeoff_lambda is specified, pick one that gets them to
        # approximately the same magnitude
        if metric_tradeoff_lambda is None:
            loss_ratio = torch.tensor(self.synthesis_metric_loss[-1] / self.fixed_metric_loss[-1],
                                      dtype=torch.float32)
            metric_tradeoff_lambda = torch.pow(torch.tensor(10),
                                               torch.round(torch.log10(loss_ratio)))
            warnings.warn("Since metric_tradeoff_lamda was None, automatically set"
                          f" to {metric_tradeoff_lambda} to roughly balance metrics.")
        self.metric_tradeoff_lambda = metric_tradeoff_lambda
        self.losses.append(self.objective_function().item())
        self.store_progress = None
        self.saved_signal = []

    def _init_synthesized_signal(self,
                                 initial_noise: float = .1):
        """Initialize the synthesized image.

        Initialize ``self.synthesized_signal`` attribute to be a
        ``reference_signal`` plus Gaussian noise with user-specified standard
        deviation.

        Parameters
        ----------
        initial_noise :
            Standard deviation of the Gaussian noise used to initialize
            ``synthesized_signal`` from ``reference_signal``.

        """
        synthesized_signal = (self.reference_signal + initial_noise *
                              torch.randn_like(self.reference_signal))
        synthesized_signal = synthesized_signal.clamp(*self.allowed_range)
        self.initial_signal = synthesized_signal.clone()
        synthesized_signal.requires_grad_()
        self.synthesized_signal = synthesized_signal
        self._fixed_metric_target = self.fixed_metric(self.reference_signal,
                                                      self.synthesized_signal).item()
        self.fixed_metric_loss.append(self._fixed_metric_target)
        self.synthesis_metric_loss.append(self.synthesis_metric(self.reference_signal,
                                                                self.synthesized_signal).item())

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
            # first time synthesize() is called, add the initial synthesized
            # signal and model response (on subsequent calls, this is already
            # part of saved_signal / saved_model_response).
            if len(self.saved_signal) == 0:
                self.saved_signal.append(self.synthesized_signal.clone().to('cpu'))
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
            # iteration where synthesized_signal started to have NaN values.
            # this will fail if it hits a nan before store_progress iterations
            # (because then saved_signal only has a length of 1) but in that
            # case, you have more severe problems
            self.synthesized_signal = torch.nn.Parameter(self.saved_signal[-2])
            return True
        return False

    def _store(self, i: int) -> bool:
        """Store synthesized_signal anbd model response, if appropriate.

        if it's the right iteration, we update: ``saved_signal``

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
                stored = True
        return stored

    def _check_for_stabilization(self, i: int, stop_criterion: float,
                                 stop_iters_to_check: int) -> bool:
        r"""Check whether the loss has stabilized and, if so, return True.

         Have we been synthesizing for ``stop_iters_to_check`` iterations?
         | |
        no yes
         | '---->Is ``abs(self.loss[-1] - self.losses[-stop_iters_to_check] < stop_criterion``?
         |      no |
         |       | yes
         <-------' |
         |         '------> return ``True``
         |
         '---------> return ``False``

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

        Returns
        -------
        loss_stabilized :
            Whether the loss has stabilized or not.

        """
        if len(self.losses) > stop_iters_to_check:
            if abs(self.losses[-stop_iters_to_check] - self.losses[-1]) < stop_criterion:
                return True
        return False

    def objective_function(self,
                           synthesized_signal: Union[Tensor, None] = None,
                           reference_signal: Union[Tensor, None] = None) -> Tensor:
        r"""Compute the MADCompetition synthesis loss.

        This computes:

        .. math::

            t L_1(x, \hat{x}) &+ \lambda_1 [L_2(x, x+\epsilon) - L_2(x, \hat{x})]^2 \\
                              &+ \lambda_2 \mathcal{B}(\hat{x})


        where :math:`t` is 1 if ``self.synthesis_target`` is ``'min'`` and -1
        if it's ``'max'``, :math:`L_1` is ``self.synthesis_metric``,
        :math:`L_2` is ``self.fixed_metric``, :math:`x` is
        ``self.reference_signal``, :math:`\hat{x}` is
        ``self.synthesized_signal``, :math:`\epsilon` is the initial noise,
        :math:`\mathcal{B}` is the quadratic bound penalty, :math:`\lambda_1`
        is ``self.metric_tradeoff_lambda`` and :math:`\lambda_2` is
        ``self.range_penalty_lambda``.

        Parameters
        ----------
        synthesized_signal :
            Proposed ``synthesized_signal``, :math:`\hat{x}` in the above
            equation. If None, use ``self.synthesized_signal``.
        reference_signal :
            Proposed ``reference_signal``, :math:`x` in the above equation. If
            None, use ``self.reference_signal``.

        Returns
        -------
        loss

        """
        if reference_signal is None:
            reference_signal = self.reference_signal
        if synthesized_signal is None:
            synthesized_signal = self.synthesized_signal
        synth_target = {'min': 1, 'max': -1}[self.synthesis_target]
        synthesis_loss = self.synthesis_metric(reference_signal, synthesized_signal)
        fixed_loss = (self._fixed_metric_target -
                      self.fixed_metric(reference_signal, synthesized_signal)).pow(2)
        range_penalty = optim.penalize_range(synthesized_signal,
                                             self.allowed_range)
        # print(synthesis_loss, fixed_loss, range_penalty)
        return (synth_target * synthesis_loss +
                self.metric_tradeoff_lambda * fixed_loss +
                self.range_penalty_lambda * range_penalty)

    def _closure(self) -> Tensor:
        r"""An abstraction of the gradient calculation, before the optimization step.

        This enables optimization algorithms that perform several evaluations
        of the gradient before taking a step (ie. second order methods like
        LBFGS).

        Additionally, this is where ``loss`` is calculated and
        ``loss.backward()`` is called.

        Returns
        -------
        loss

        """
        self.optimizer.zero_grad()
        loss = self.objective_function()
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
        synthesis_metric : torch.Tensor
            1-element tensor containing the synthesis_metric on this step
        fixed_metric : torch.Tensor
            1-element tensor containing the fixed_metric on this step
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
        loss = self.optimizer.step(self._closure)
        grad_norm = self.synthesized_signal.grad.detach().norm()
        if grad_norm.item() != grad_norm.item():
            raise Exception('found a NaN in the gradients during optimization')

        fm = self.fixed_metric(self.reference_signal, self.synthesized_signal)
        sm = self.synthesis_metric(self.reference_signal, self.synthesized_signal)

        # optionally step the scheduler
        if self.scheduler is not None:
            self.scheduler.step(loss.item())

        pixel_change = torch.max(torch.abs(self.synthesized_signal -
                                           last_iter_synthesized_signal))
        # for display purposes, always want loss to be positive
        postfix_dict.update(dict(loss=f"{abs(loss.item()):.04e}",
                                 synthesis_metric=f'{sm.item():.04e}',
                                 fixed_metric=f'{fm.item():.04e}',
                                 gradient_norm=f"{grad_norm.item():.04e}",
                                 learning_rate=self.optimizer.param_groups[0]['lr'],
                                 pixel_change=f"{pixel_change:.04e}",
                                 **kwargs))
        # add extra info here if you want it to show up in progress bar
        pbar.set_postfix(**postfix_dict)
        return loss, sm, fm, grad_norm, self.optimizer.param_groups[0]['lr'], pixel_change

    def synthesize(self, max_iter: int = 100,
                   optimizer: Union[None, torch.optim.Optimizer] = None,
                   scheduler: Union[None, torch.optim.lr_scheduler._LRScheduler] = None,
                   store_progress: Union[bool, int] = False,
                   stop_criterion: float = 1e-4, stop_iters_to_check: int = 50):
        r"""Synthesize a metamer.

        This is the main method, which updates the ``initial_image`` until its
        representation matches that of ``target_signal``.

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

        Returns
        -------
        synthesized_signal : torch.Tensor
            The metamer we've created

        """
        # initialize the optimizer and scheduler
        self._init_optimizer(optimizer, scheduler)

        # get ready to store progress
        self._init_store_progress(store_progress)

        pbar = tqdm(range(max_iter))

        for i in pbar:
            loss, sm, fm, g, lr, pixel_change = self._optimizer_step(pbar)
            self.losses.append(loss.item())
            self.fixed_metric_loss.append(fm.item())
            self.synthesis_metric_loss.append(sm.item())
            self.pixel_change.append(pixel_change.item())
            self.gradient_norm.append(g.item())
            self.learning_rate.append(lr)
            if self._check_nan_loss(loss):
                break

            # update saved_* attrs
            self._store(i)

            if self._check_for_stabilization(i, stop_criterion,
                                             stop_iters_to_check):
                break

        pbar.close()

        # finally, stack the saved_* attributes
        if self.store_progress:
            self.saved_signal = torch.stack(self.saved_signal)

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
        # this copies the attributes dict so we don't actually remove the
        # model attribute in the next line
        attrs = {k: v for k, v in vars(self).items()}
        # if the metrics are Modules, then we don't want to save them. If
        # they're functions then saving them is fine.
        if isinstance(self.synthesis_metric, torch.nn.Module):
            attrs.pop('synthesis_metric')
        if isinstance(self.fixed_metric, torch.nn.Module):
            attrs.pop('fixed_metric')
        super().save(file_path, attrs=attrs)

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
        attrs = ['reference_signal', 'synthesized_signal', 'saved_signal']
        super().to(*args, attrs=attrs, **kwargs)
        # if the metrics are Modules, then we should pass them as well. If
        # they're functions then nothing needs to be done.
        try:
            self.fixed_metric.to(*args, **kwargs)
        except AttributeError:
            pass
        try:
            self.synthesis_metric.to(*args, **kwargs)
        except AttributeError:
            pass

    def load(self, file_path: str,
             map_location: Union[str, None] = None,
             **pickle_load_args):
        r"""Load all relevant stuff from a .pt file.

        This should be called by an initialized ``MADCompetition`` object -- we will
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
        >>> metamer = po.synth.MADCompetition(img, model)
        >>> metamer.synthesize(max_iter=10, store_progress=True)
        >>> metamer.save('metamers.pt')
        >>> metamer_copy = po.synth.MADCompetition(img, model)
        >>> metamer_copy.load('metamers.pt')

        Note that you must create a new instance of the Synthesis object and
        *then* load.

        """
        check_attributes = ['reference_signal', 'metric_tradeoff_lambda',
                            'range_penalty_lambda', 'allowed_range',
                            'synthesis_target']
        check_loss_functions = ['fixed_metric', 'synthesis_metric']
        super().load(file_path, map_location=map_location,
                     check_attributes=check_attributes,
                     check_loss_functions=check_loss_functions,
                     **pickle_load_args)
        # make this require a grad again
        self.synthesized_signal.requires_grad_()


def plot_loss(mad: MADCompetition,
              iteration: Union[int, None] = None,
              ax: Union[mpl.axes.Axes, None] = None,
              **kwargs) -> mpl.axes.Axes:
    """Plot synthesis loss with log-scaled y axis.

    Plots ``abs(mad.losses)``, ``mad.synthesis_metric_loss`` and
    ``mad.metric_tradeoff_lambda * mad.fixed_metric_loss`` over all iterations.
    Also plots a red dot at ``iteration``, to highlight the loss there. If
    ``iteration=None``, then the dot will be at the final iteration.

    Parameters
    ----------
    mad :
        MADCompetition object whose loss we want to plot.
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

    Notes
    -----
    We plot ``abs(mad.losses)`` because if we're maximizing the synthesis
    metric, we minimized its negative. By plotting the absolute value, we get
    them all on the same scale.

    """
    if iteration is None:
        loss_idx = len(mad.losses) - 1
    else:
        if iteration < 0:
            # in order to get the x-value of the dot to line up,
            # need to use this work-around
            loss_idx = len(mad.losses) + iteration
        else:
            loss_idx = iteration
    if ax is None:
        ax = plt.gca()
    # if we're maximizing synthesis_metric, our loss will be negative. for
    # plotting purposes, make it positive.
    losses = np.abs(mad.losses)
    ax.semilogy(losses, label='abs(objective function)', **kwargs)
    try:
        ax.scatter(loss_idx, losses[loss_idx], c='r')
    except IndexError:
        # then there's no loss here
        pass
    ax.semilogy(mad.synthesis_metric_loss, label='synthesis metric')
    ax.scatter(loss_idx, mad.synthesis_metric_loss[loss_idx], c='r')
    fixed_metric = data.to_numpy(mad.metric_tradeoff_lambda *
                                 np.array(mad.fixed_metric_loss))
    ax.semilogy(fixed_metric, label=r'$\lambda$ * fixed metric')
    ax.scatter(loss_idx, fixed_metric[loss_idx], c='r')
    ax.set(xlabel='Synthesis iteration', ylabel='Loss')
    ax.legend()
    return ax


def display_synthesized_signal(mad: MADCompetition,
                               batch_idx: int = 0,
                               channel_idx: Union[int, None] = None,
                               zoom: Union[float, None] = None,
                               iteration: Union[int, None] = None,
                               ax: Union[mpl.axes.Axes, None] = None,
                               **kwargs) -> mpl.axes.Axes:
    """Display synthesized_signal.

    You can specify what iteration to view by using the ``iteration`` arg.
    The default, ``None``, shows the final one.

    We use ``plenoptic.imshow`` to display the synthesized image and attempt to
    automatically find the most reasonable zoom value. You can override this
    value using the zoom arg, but remember that ``plenoptic.imshow`` is
    opinionated about the size of the resulting image and will throw an
    Exception if the axis created is not big enough for the selected zoom.

    Parameters
    ----------
    mad :
        MADCompetition object whose synthesized signal we want to display.
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
        How much to zoom in / enlarge the synthesized image, the ratio
        of display pixels to image pixels. If None (the default), we
        attempt to find the best value ourselves.
    kwargs :
        Passed to ``plenoptic.imshow``

    Returns
    -------
    ax :
        The matplotlib axes containing the plot.

    """
    if iteration is None:
        image = mad.synthesized_signal
    else:
        image = mad.saved_signal[iteration]
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
    display.imshow(image, ax=ax, title='MADCompetition', zoom=zoom,
                   batch_idx=batch_idx, channel_idx=channel_idx,
                   as_rgb=as_rgb, **kwargs)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax


def plot_pixel_values(mad: MADCompetition,
                      batch_idx: int = 0,
                      channel_idx: Union[int, None] = None,
                      iteration: Union[int, None] = None,
                      ylim: Union[Tuple[float], Literal[False]] = None,
                      ax: Union[mpl.axes.Axes, None] = None,
                      **kwargs) -> mpl.axes.Axes:
    r"""Plot histogram of pixel values of reference and synthesized signals.

    As a way to check the distributions of pixel intensities and see
    if there's any values outside the allowed range

    Parameters
    ----------
    mad :
        MADCompetition object with the images whose pixel values we want to compare.
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
        image = mad.synthesized_signal[batch_idx]
    else:
        image = mad.saved_signal[iteration, batch_idx]
    reference_signal = mad.reference_signal[batch_idx]
    if channel_idx is not None:
        image = image[channel_idx]
        reference_signal = reference_signal[channel_idx]
    if ax is None:
        ax = plt.gca()
    image = data.to_numpy(image).flatten()
    reference_signal = data.to_numpy(reference_signal).flatten()
    ax.hist(image, bins=min(_freedman_diaconis_bins(image), 50),
            label='Synthesized image', **kwargs)
    ax.hist(reference_signal, bins=min(_freedman_diaconis_bins(image), 50),
            label='reference image', **kwargs)
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title("Histogram of pixel values")
    return ax


def _setup_synthesis_fig(fig: Union[mpl.figure.Figure, None] = None,
                         axes_idx: Dict[str, int] = {},
                         figsize: Union[Tuple[float], None] = None,
                         synthesized_signal: bool = True,
                         loss: bool = True,
                         pixel_values: bool = False,
                         synthesized_signal_width: float = 1,
                         loss_width: float = 1,
                         pixel_values_width: float = 1) -> Tuple[mpl.figure.Figure, List[mpl.axes.Axes], Dict[str, int]]:
    """Set up figure for plot_synthesis_status.

    Creates figure with enough axes for the all the plots you want. Will
    also create index in axes_idx for them if you haven't done so already.

    By default, all axes will be on the same row and have the same width. If
    you want them to be on different rows, will need to initialize fig yourself
    and pass that in. For changing width, change the corresponding *_width arg,
    which gives width relative to other axes. So if you want the axis for the
    loss plot to be twice as wide as the others, set loss_width=2.

    Parameters
    ----------
    fig :
        The figure to plot on or None. If None, we create a new figure
    axes_idx :
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: loss, pixel_values,
        misc. Values should all be ints. If you tell this function to create a
        plot that doesn't have a corresponding key, we find the lowest int that
        is not already in the dict, so if you have axes that you want
        unchanged, place their idx in misc.
    figsize :
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If None, we attempt to
        make our best guess, aiming to have relative width=1 correspond to 5
    synthesized_signal :
        Whether to include axis for plot of the synthesized image or not.
    loss :
        Whether to include axis for plot of the loss or not.
    pixel_values :
        Whether to include axis for plot of the histograms of image pixel
        intensities or not.
    synthesized_signal_width :
        Relative width of the axis for the synthesized image.
    loss_width :
        Relative width of the axis for loss plot.
    pixel_values_width :
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
    if synthesized_signal:
        n_subplots += 1
        width_ratios.append(synthesized_signal_width)
        if 'synthesized_signal' not in axes_idx.keys():
            axes_idx['synthesized_signal'] = data._find_min_int(axes_idx.values())
    if loss:
        n_subplots += 1
        width_ratios.append(loss_width)
        if 'loss' not in axes_idx.keys():
            axes_idx['loss'] = data._find_min_int(axes_idx.values())
    if pixel_values:
        n_subplots += 1
        width_ratios.append(pixel_values_width)
        if 'pixel_values' not in axes_idx.keys():
            axes_idx['pixel_values'] = data._find_min_int(axes_idx.values())
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


def plot_synthesis_status(mad: MADCompetition,
                          batch_idx: int = 0,
                          channel_idx: Union[int, None] = None,
                          iteration: Union[int, None] = None,
                          vrange: Union[Tuple[float], str] = 'indep1',
                          zoom: Union[float, None] = None,
                          fig: Union[mpl.figure.Figure, None] = None,
                          axes_idx: Dict[str, int] = {},
                          figsize: Union[Tuple[float], None] = None,
                          synthesized_signal: bool = True,
                          loss: bool = True,
                          pixel_values: bool = False,
                          width_ratios: Dict[str, float] = {},
                          ) -> Tuple[mpl.figure.Figure, Dict[str, int]]:
    r"""Make a plot showing synthesis status.

    We create several subplots to analyze this. By default, we create two
    subplots on a new figure: the first one contains the synthesized signal and
    the second contains the loss.

    There is an optional additional plot: pixel_values, a histogram of pixel
    values of the synthesized and target images.

    All of these (including the default plots) can be toggled using their
    corresponding boolean flags, and can be created separately using the
    method with the name `plot_{flag}`.

    Parameters
    ----------
    mad :
        MADCompetition object whose status we want to plot.
    batch_idx :
        Which index to take from the batch dimension
    channel_idx :
        Which index to take from the channel dimension. If None, we use all
        channels (assumed use-case is RGB(A) image).
    iteration :
        Which iteration to display. If None, the default, we show
        the most recent one. Negative values are also allowed.
    vrange :
        The vrange option to pass to ``display_synthesized_signal()``. See
        docstring of ``imshow`` for possible values.
    zoom :
        How much to zoom in / enlarge the synthesized image, the ratio
        of display pixels to image pixels. If None (the default), we
        attempt to find the best value ourselves.
    fig :
        if None, we create a new figure. otherwise we assume this is
        an empty figure that has the appropriate size and number of
        subplots
    axes_idx :
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'synthesized_signal',
        'loss', 'pixel_values', 'misc'``. Values should all be ints. If you
        tell this function to create a plot that doesn't have a corresponding
        key, we find the lowest int that is not already in the dict, so if you
        have axes that you want unchanged, place their idx in ``'misc'``.
    figsize :
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If None, we attempt to
        make our best guess, aiming to have each axis be of size (5, 5)
    synthesized_signal :
        Whether to display the synthesized image or not.
    loss :
        Whether to plot the loss or not.
    pixel_values :
        Whether to plot the histograms of image pixel intensities or
        not.
    width_ratios :
        By default, all plots axes will have the same width. To change
        that, specify their relative widths using the keys:
        ['synthesized_signal', 'loss', 'pixel_values'] and floats specifying
        their relative width. Any not included will be assumed to be 1.

    Returns
    -------
    fig :
        The figure containing this plot
    axes_idx :
        Dictionary giving index of each plot.

    """
    if iteration is not None and not mad.store_progress:
        raise Exception("synthesis() was run with store_progress=False, "
                        "cannot specify which iteration to plot (only"
                        " last one, with iteration=None)")
    if mad.synthesized_signal.ndim not in [3, 4]:
        raise Exception("plot_synthesis_status() expects 3 or 4d data;"
                        "unexpected behavior will result otherwise!")
    width_ratios = {f'{k}_width': v for k, v in width_ratios.items()}
    fig, axes, axes_idx = _setup_synthesis_fig(fig, axes_idx, figsize,
                                               synthesized_signal,
                                               loss,
                                               pixel_values,
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

    if synthesized_signal:
        display_synthesized_signal(mad, batch_idx=batch_idx,
                                   channel_idx=channel_idx,
                                   iteration=iteration,
                                   ax=axes[axes_idx['synthesized_signal']],
                                   zoom=zoom, vrange=vrange)
    if loss:
        plot_loss(mad, iteration=iteration, ax=axes[axes_idx['loss']])
    if pixel_values:
        plot_pixel_values(mad, batch_idx=batch_idx,
                          channel_idx=channel_idx,
                          iteration=iteration,
                          ax=axes[axes_idx['pixel_values']])
    return fig, axes_idx


def animate(mad: MADCompetition,
            framerate: int = 10,
            batch_idx: int = 0,
            channel_idx: Union[int, None] = None,
            zoom: Union[float, None] = None,
            fig: Union[mpl.figure.Figure, None] = None,
            axes_idx: Dict[str, int] = {},
            figsize: Union[Tuple[float], None] = None,
            synthesized_signal: bool = True,
            loss: bool = True,
            pixel_values: bool = False,
            width_ratios: Dict[str, float] = {},
            ) -> mpl.animation.FuncAnimation:
    r"""Animate synthesis progress.

    This is essentially the figure produced by
    ``mad.plot_synthesis_status`` animated over time, for each stored
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
    mad :
        MADCompetition object whose synthesis we want to animate.
    framerate :
        How many frames a second to display.
    batch_idx :
        Which index to take from the batch dimension
    channel_idx :
        Which index to take from the channel dimension. If None, we use all
        channels (assumed use-case is RGB(A) image).
    zoom :
        How much to zoom in / enlarge the synthesized image, the ratio
        of display pixels to image pixels. If None (the default), we
        attempt to find the best value ourselves.
    fig :
        If None, create the figure from scratch. Else, should be an empty
        figure with enough axes (the expected use here is have same-size
        movies with different plots).
    axes_idx :
        Dictionary specifying which axes contains which type of plot, allows
        for more fine-grained control of the resulting figure. Probably only
        helpful if fig is also defined. Possible keys: ``'synthesized_signal',
        'loss', 'pixel_values', 'misc'``. Values should all be ints. If you
        tell this function to create a plot that doesn't have a corresponding
        key, we find the lowest int that is not already in the dict, so if you
        have axes that you want unchanged, place their idx in ``'misc'``.
    figsize :
        The size of the figure to create. It may take a little bit of
        playing around to find a reasonable value. If None, we attempt to
        make our best guess, aiming to have each axis be of size (5, 5)
    synthesized_signal :
        Whether to display the synthesized image or not.
    loss :
        Whether to plot the loss or not.
    pixel_values :
        Whether to plot the histograms of image pixel intensities or
        not.
        By default, all plots axes will have the same width. To change
        that, specify their relative widths using the keys:
        ['synthesized_signal', 'loss', 'pixel_values'] and floats specifying
        their relative width. Any not included will be assumed to be 1.

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
    if not mad.store_progress:
        raise Exception("synthesize() was run with store_progress=False,"
                        " cannot animate!")
    if mad.synthesized_signal.ndim not in [3, 4]:
        raise Exception("animate() expects 3 or 4d data; unexpected"
                        " behavior will result otherwise!")
    # we run plot_synthesis_status to initialize the figure if either fig is
    # None or if there are no titles on any axes, which we assume means that
    # it's an empty figure
    if fig is None or not any([ax.get_title() for ax in fig.axes]):
        fig, axes_idx = plot_synthesis_status(mad=mad,
                                              batch_idx=batch_idx,
                                              channel_idx=channel_idx,
                                              iteration=0, figsize=figsize,
                                              loss=loss,
                                              zoom=zoom, fig=fig,
                                              synthesized_signal=synthesized_signal,
                                              pixel_values=pixel_values,
                                              axes_idx=axes_idx,
                                              width_ratios=width_ratios)
    # grab the artist for the second plot (we don't need to do this for the
    # synthesized image or model_response plot, because we use the update_plot
    # function for that)
    if loss:
        scat = fig.axes[axes_idx['loss']].collections
    # can also have multiple plots

    def movie_plot(i):
        artists = []
        if synthesized_signal:
            artists.extend(display.update_plot(fig.axes[axes_idx['synthesized_signal']],
                                               data=mad.saved_signal[i],
                                               batch_idx=batch_idx))
        if pixel_values:
            # this is the dumbest way to do this, but it's simple --
            # clearing the axes can cause problems if the user has, for
            # example, changed the tick locator or formatter. not sure how
            # to handle this best right now
            fig.axes[axes_idx['pixel_values']].clear()
            plot_pixel_values(mad, batch_idx=batch_idx,
                              channel_idx=channel_idx, iteration=i,
                              ax=fig.axes[axes_idx['pixel_values']])
        if loss:
            # loss always contains values from every iteration, but everything
            # else will be subsampled.
            x_val = i*mad.store_progress
            scat[0].set_offsets((x_val, np.abs(mad.losses[x_val])))
            scat[1].set_offsets((x_val, mad.synthesis_metric_loss[x_val]))
            fixed_metric = data.to_numpy(mad.metric_tradeoff_lambda *
                                         np.array(mad.fixed_metric_loss))
            scat[2].set_offsets((x_val, fixed_metric[x_val]))
            artists.extend(scat)
        # as long as blitting is True, need to return a sequence of artists
        return artists

    # don't need an init_func, since we handle initialization ourselves
    anim = mpl.animation.FuncAnimation(fig, movie_plot,
                                       frames=len(mad.saved_signal),
                                       blit=True, interval=1000./framerate,
                                       repeat=False)
    plt.close(fig)
    return anim
