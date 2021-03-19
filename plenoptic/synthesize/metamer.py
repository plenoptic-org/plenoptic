import torch
from tqdm.auto import tqdm
from .synthesis import Synthesis
from ..tools.metamer_utils import RangeClamper


class Metamer(Synthesis):
    r"""Synthesize metamers for image-computable differentiable models!

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
    -----
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model
       based on Joint Statistics of Complex Wavelet Coefficients. Int'l
       Journal of Computer Vision. 40(1):49-71, October, 2000.
       http://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
       http://www.cns.nyu.edu/~lcv/texture/

    """

    def __init__(self, base_signal, model, loss_function=None, model_kwargs={},
                 loss_function_kwargs={}):
        super().__init__(base_signal, model, loss_function, model_kwargs, loss_function_kwargs)

    def _init_synthesized_signal(self, initial_image, clamper=RangeClamper((0, 1)),
                                 clamp_each_iter=True):
        """initialize the synthesized image

        set the ``self.synthesized_signal`` attribute to be a parameter with
        the user-supplied data, making sure it's the right shape and
        calling clamper on it, if set

        also initialize the ``self.synthesized_representation`` attribute

        Parameters
        ----------
        initial_image : torch.Tensor, array_like, or None, optional
            The 2d tensor we use to initialize the metamer. If None (the
            default), we initialize with uniformly-distributed random
            noise lying between 0 and 1 or, if ``self.saved_signal`` is
            not empty, use the final value there. If this is not a
            tensor or None, we try to cast it as a tensor.
        clamper : Clamper or None, optional
            will set ``self.clamper`` attribute to this, and if not
            None, will call ``clamper.clamp`` on synthesized_signal
        clamp_each_iter : bool, optional
            If True (and ``clamper`` is not ``None``), we clamp every
            iteration. If False, we only clamp at the very end, after
            the last iteration
        """
        if initial_image is None:
            try:
                # then we have a previous run to resume
                synthesized_signal_data = self.saved_signal[-1]
            except IndexError:
                # else we're starting over
                synthesized_signal_data = torch.rand_like(self.base_signal, dtype=torch.float32,
                                                          device=self.base_signal.device)
        else:
            synthesized_signal_data = torch.tensor(initial_image, dtype=torch.float32,
                                                   device=self.base_signal.device)
        super()._init_synthesized_signal(synthesized_signal_data.clone(), clamper, clamp_each_iter)

    def synthesize(self, initial_image=None, seed=0, max_iter=100, learning_rate=.01,
                   scheduler=True, optimizer='SGD', optimizer_kwargs={},
                   clamper=RangeClamper((0, 1)), clamp_each_iter=True,
                   store_progress=False, save_progress=False,
                   save_path='metamer.pt', loss_thresh=1e-4, loss_change_iter=50,
                   loss_change_thresh=1e-2, coarse_to_fine=False, clip_grad_norm=False):
        r"""Synthesize a metamer

        This is the main method, which updates the ``initial_image`` until its
        representation matches that of ``base_signal``.

        We run this until either we reach ``max_iter`` or the change
        over the past ``loss_change_iter`` iterations is less than
        ``loss_thresh``, whichever comes first

        Parameters
        ----------
        initial_image : torch.Tensor, array_like, or None, optional
            The 4d tensor we use to initialize the metamer. If None (the
            default), we initialize with uniformly-distributed random
            noise lying between 0 and 1 or, if ``self.saved_signal`` is
            not empty, use the final value there. If this is not a
            tensor or None, we try to cast it as a tensor.
        seed : int or None, optional
            Number with which to seed pytorch and numy's random number
            generators. If None, won't set the seed.
        max_iter : int, optinal
            The maximum number of iterations to run before we end
        learning_rate : float or None, optional
            The learning rate for our optimizer. None is only accepted
            if we're resuming synthesis, in which case we use the last
            learning rate from the previous instance.
        scheduler : bool, optional
            whether to initialize the scheduler or not. If False, the
            learning rate will never decrease.
        optimizer: {'GD', 'Adam', 'SGD', 'LBFGS', 'AdamW'}
            The choice of optimization algorithm. 'GD' is regular
            gradient descent.
        optimizer_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the optimizer (in
            addition to learning_rate). What these should be depend on
            the specific optimizer you're using
        clamper : plenoptic.Clamper or None, optional
            Clamper makes a change to the image in order to ensure that
            it stays reasonable. The classic example (and default
            option) is making sure the range lies between 0 and 1, see
            plenoptic.RangeClamper for an example.
        clamp_each_iter : bool, optional
            If True (and ``clamper`` is not ``None``), we clamp every
            iteration. If False, we only clamp at the very end, after
            the last iteration
        store_progress : bool or int, optional
            Whether we should store the representation of the metamer
            and the metamer image in progress on every iteration. If
            False, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress``
            iterations (note then that 0 is the same as False and 1 the
            same as True).
        save_progress : bool or int, optional
            Whether to save the metamer as we go. If True, we save to
            ``save_path`` every ``store_progress`` iterations. If an int, we
            save every ``save_progress`` iterations. Note that this can end up
            actually taking a fair amount of time.
        save_path : str, optional
            The path to save the synthesis-in-progress to (ignored if
            ``save_progress`` is False)
        loss_thresh : float, optional
            If the loss over the past ``loss_change_iter`` has changed
            less than ``loss_thresh``, we stop.
        loss_change_iter : int, optional
            How many iterations back to check in order to see if the
            loss has stopped decreasing (for loss_change_thresh).
        loss_change_thresh : float, optional
            The threshold below which we consider the loss as unchanging and so
            should switch scales if `coarse_to_fine is not False`. Ignored
            otherwise.
        coarse_to_fine : { 'together', 'separate', False}, optional
            If False, don't do coarse-to-fine optimization. Else, there
            are two options for how to do it:
            - 'together': start with the coarsest scale, then gradually
              add each finer scale.
            - 'separate': compute the gradient with respect to each
              scale separately (ignoring the others), then with respect
              to all of them at the end.
            (see ``Metamer`` tutorial for more details).
        clip_grad_norm : bool or float, optional
            Clip the gradient norm to avoid issues with numerical overflow.
            Gradient norm will be clipped to the specified value (True is
            equivalent to 1).

        Returns
        -------
        synthesized_signal : torch.Tensor
            The metamer we've created
        synthesized_representation : torch.Tensor
            The model's representation of the metamer

        """
        # set seed
        self._set_seed(seed)

        # initialize synthesized_signal
        self._init_synthesized_signal(initial_image, clamper, clamp_each_iter)

        # initialize stuff related to coarse-to-fine and randomization
        self._init_ctf_and_randomizer(loss_thresh, coarse_to_fine,
                                      loss_change_thresh, loss_change_iter)

        # initialize the optimizer
        self._init_optimizer(optimizer, learning_rate, scheduler, clip_grad_norm,
                             optimizer_kwargs)

        # get ready to store progress
        self._init_store_progress(store_progress, save_progress, save_path)

        pbar = tqdm(range(max_iter))

        for i in pbar:
            loss, g, lr, pixel_change = self._optimizer_step(pbar)
            self.loss.append(loss.item())
            self.pixel_change.append(pixel_change.item())
            self.gradient.append(g.item())
            self.learning_rate.append(lr)

            if self._check_nan_loss(loss):
                break

            # clamp and update saved_* attrs
            self._clamp_and_store(i)

            if self._check_for_stabilization(i):
                break

        pbar.close()

        # finally, stack the saved_* attributes
        self._finalize_stored_progress()

        # return data
        return self.synthesized_signal.data, self.synthesized_representation.data

    def save(self, file_path):
        r"""Save all relevant variables in .pt file.

        Note that if store_progress is True, this will probably be very
        large.

        See ``load`` docstring for an example of use.

        Parameters
        ----------
        file_path : str
            The path to save the metamer object to

        """
        attrs = ['synthesized_signal', 'base_signal', 'seed', 'loss', 'base_representation',
                 'synthesized_representation', 'saved_representation', 'gradient', 'saved_signal',
                 'learning_rate', 'saved_representation_gradient', 'saved_signal_gradient',
                 'coarse_to_fine', 'scales', 'scales_timing', 'scales_loss', 'loss_function',
                 'scales_finished', 'store_progress', 'save_progress', 'save_path', 'pixel_change']
        super().save(file_path, attrs)

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
        attrs = ['base_signal', 'base_representation', 'synthesized_signal',
                 'synthesized_representation', 'saved_signal', 'saved_representation',
                 'saved_signal_gradient', 'saved_representation_gradient']
        return super().to(*args, attrs=attrs, **kwargs)

    def load(self, file_path, map_location='cpu', **pickle_load_args):
        r"""Load all relevant stuff from a .pt file.

        This should be called by an initialized ``Metamer`` object -- we will
        ensure that ``base_signal``, ``base_representation`` (and thus
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
        We will iterate through any additional key word arguments
        provided and, if the model in the saved representation is a
        dictionary, add them to the state_dict of the model. In this
        way, you can replace, e.g., paths that have changed between
        where you ran the model and where you are now.

        """
        super().load(file_path, map_location,
                     ['base_signal', 'base_representation', 'loss_function'],
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
