import torch
import warnings
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from .Synthesis import Synthesis


class Metamer(Synthesis):
    r"""Synthesize metamers for image-computable differentiable models!

    Following the basic idea in [1]_, this module creates a metamer for
    a given model on a given image. We start with some random noise
    (typically, though users can choose to start with something else)
    and iterative adjust the pixel values so as to match the
    representation of this metamer-to-be and the ``target_image``. This
    is optimization though, so you'll probably need to experiment with
    the optimization hyper-parameters before you find a good solution.

    Currently we do not: support batch creation of images.

    Parameters
    ----------
    target_image : torch.tensor or array_like
        A 2d tensor, this is the image whose representation we wish to
        match. If this is not a tensor, we try to cast it as one.
    model : torch.nn.Module
        A differentiable model that takes an image as an input and
        transforms it into a representation of some sort. We only
        require that it has a forward method, which returns the
        representation to match. However, if you want to use the various
        plot and animate function, it should also have
        ``plot_representation`` and ``_update_plot`` functions.

    Attributes
    ----------
    target_image : torch.tensor
        A 2d tensor, this is the image whose representation we wish to
        match.
    model : torch.nn.Module
        A differentiable model that takes an image as an input and
        transforms it into a representation of some sort. We only
        require that it has a forward method, which returns the
        representation to match.
    target_representation : torch.tensor
        Whatever is returned by ``model.foward(target_image)``, this is
        what we match in order to create a metamer
    matched_image : torch.tensor
        The metamer. This may be unfinished depending on how many
        iterations we've run for.
    matched_represetation: torch.tensor
        Whatever is returned by ``model.forward(matched_image)``; we're
        trying to make this identical to ``self.target_representation``
    optimizer : torch.optim.Optimizer
        A pytorch optimization method.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        A pytorch scheduler, which tells us how to change the learning
        rate over iterations. Currently, user cannot set and we use
        ReduceLROnPlateau (so that the learning rate gets reduced if it
        seems like we're on a plateau i.e., the loss isn't changing
        much)
    loss : list
        A list of our loss over iterations.
    gradient : list
        A list of the gradient over iterations.
    learning_rate : list
        A list of the learning_rate over iterations. We use a scheduler
        that gradually reduces this over time, so it won't be constant.
    saved_representation : torch.tensor
        If the ``store_progress`` arg in ``synthesize`` is set to
        True or an int>0, we will save ``self.matched_representation``
        at each iteration (or each ``store_progress`` iteration, if it's an
        int), for later examination.
    saved_image : torch.tensor
        If the ``store_progress`` arg in ``synthesize`` is set to True
        or an int>0, we will save ``self.matched_image`` at each
        iteration (or each ``store_progress`` iteration, if it's an
        int), for later examination.
    seed : int Number with which
        to seed pytorch and numy's random number generators
    saved_image_gradient : torch.tensor
        If the ``store_progress`` arg in ``synthesize`` is set to True
        or an int>0, we will save ``self.matched_image.grad`` at each
        iteration (or each ``store_progress`` iteration, if it's an
        int), for later examination.
    saved_representation_gradient : torch.tensor
        If the ``store_progress`` arg in ``synthesize`` is set to
        True or an int>0, we will save
        ``self.matched_representation.grad`` at each iteration (or each
        ``store_progress`` iteration, if it's an int), for later
        examination.
    scales_loss : list
        If ``coarse_to_fine`` is True, this contains the scale-specific
        loss at each iteration (that is, the loss computed on just the
        scale we're optimizing on that iteration; which we use to
        determine when to switch scales). If ``coarse_to_fine`` is
        False, this will be empty
    scales : list or None
        If ``coarse_to_fine`` is True, this is a list of the scales in
        reverse optimization order (i.e., from fine to coarse). The
        first entry will be 'all' (since after we've optimized each
        individual scale, we move on to optimizing all at once) This
        will be modified by the synthesize() method and is used to track
        which scale we're currently optimizing (the last one). When
        we've gone through all the scales present, this will just
        contain a single value: 'all'. If ``coarse_to_fine`` is False,
        this will be None.
    scales_timing : dict or None
        If ``coarse_to_fine`` is True, this is a dictionary whose keys
        are the values of scales. The values are lists, with 0 through 2
        entries: the first entry is the iteration where we started
        optimizing this scale, the second is when we stopped (thus if
        it's an empty list, we haven't started optimzing it yet). If
        ``coarse_to_fine`` is False, this will be None.

    References
    -----
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model
       based on Joint Statistics of Complex Wavelet Coefficients. Int'l
       Journal of Computer Vision. 40(1):49-71, October, 2000.
       http://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
       http://www.cns.nyu.edu/~lcv/texture/

    TODO
    ----
    (musts)
    - [ ] synthesize an image of a different size than the target image
    - [ ] flexible objective function: make objective_function an attribute, have user set it
          during optimization, have variety of standard ones as static methods
          (https://realpython.com/instance-class-and-static-methods-demystified/) to choose from?
    - [x] flexibility on the optimizer / scheduler (or at least parameterize the stuff): do similar
          to above? -- not as important right now, but added some flexibility here
    - [x] should we initialize optimizer / scheduler at initialization
          or during the call to synthesize? seems reasonable to me that
          you'd want to change it I guess... -- not important right now,
          same as above. we initialize during synthesize because you may
          want to make multiple calls with different optimizers /
          options and we need to re-initialize optimizer during
          coarse-to-fine
    - [x] is that note in analyze still up-to-date? -- No
    - [x] add save method
    - [x] add example for load method
    - [x] add animate method, which creates a three-subplot animation: the metamer over time, the
          plot of differences in representation over time, and the loss over time (as a red point
          on the loss curve) -- some models' representation might not be practical to plot, add the
          ability to take a function for the plot representation and if it's set to None, don't
          plot anything; make this a separate class or whatever because we'll want to be able to do
          this for eigendistortions, etc (this will require standardizing our API, which we want to
          do anyway)
    - [x] how to handle device? -- get rid of device in here, expect the user to set .to(device)
          (and then check self.target_image.device when initializing any tensors)
    - [x] how do we handle continuation? right now the way to do it is to just pass matched_im
          again, but is there a better way? how then to handle self.time and
          self.saved_image/representation? -- don't worry about this, add note about how this works
          but don't worry about this; add ability to save every n steps, not just or every

    (other)
    - [ ] batch
    - [ ] return multiple samples

    """

    def __init__(self, target_image, model):
        super().__init__()

        if not isinstance(target_image, torch.Tensor):
            target_image = torch.tensor(target_image, torch.float32)
        self.target_image = target_image
        self.model = model
        self.seed = None

        self.target_representation = self.analyze(self.target_image)
        self.matched_image = None
        self.matched_representation = None
        self.optimizer = None
        self.scheduler = None
        self.fraction_removed = 0

        self.loss = []
        self.gradient = []
        self.learning_rate = []
        self.saved_representation = []
        self.saved_image = []
        self.saved_image_gradient = []
        self.saved_representation_gradient = []
        self.scales_loss = []
        self.scales = None
        self.scales_timing = None

    def synthesize(self, seed=0, learning_rate=.01, max_iter=100, initial_image=None,
                   clamper=None, optimizer='SGD', fraction_removed=0., loss_thresh=1e-4,
                   store_progress=False, save_progress=False, save_path='metamer.pt',
                   loss_change_thresh=1e-2, loss_change_iter=50, loss_change_fraction=1.,
                   coarse_to_fine=False, **optimizer_kwargs):
        r"""synthesize a metamer

        This is the main method, trying to update the ``initial_image``
        until its representation matches that of ``target_image``. If
        ``initial_image`` is not set, we initialize with
        uniformly-distributed random noise between 0 and 1. NOTE: This
        means that the value of ``target_image`` should probably lie
        between 0 and 1. If that's not the case, you might want to pass
        something to act as the initial image.

        We run this until either we reach ``max_iter`` or the change
        over the past ``loss_change_iter`` iterations is less than
        ``loss_thresh``, whichever comes first

        Note that you can run this several times in sequence by setting
        ``initial_image`` to ``metamer.matched_image`` (I would detach
        and clone it just to make sure things don't get weird:
        ``initial_image=metamer.matched_image.detach().clone()``). Everything
        that stores the progress of the optimization (``loss``,
        ``saved_representation``, ``saved_image``) will persist between
        calls and so potentially get very large.

        We provide three ways to try and add some more randomness to
        this optimization, in order to either improve the diversity of
        generated metamers or avoid getting stuck in local optima:

        1. Use a different optimizer (and change its hyperparameters)
           with ``optimizer`` and ``optimizer_kwargs``

        2. Only calculate the gradient with respect to some random
           subset of the model's representation. By setting
           ``fraction_removed`` to some number between 0 and 1, the
           gradient and loss are computed using a random subset of the
           representation on each iteration (this random subset is drawn
           independently on each trial). Therefore, if you wish to
           disable this (to use all of the representation), this should
           be set to 0.

        3. Only calculate the gradient with respect to the parts of the
           representation that have the highest error. If we think the
           loss has stopped changing (by seeing that the loss
           ``loss_change_iter`` iterations ago is within
           ``loss_change_thresh`` of the most recent loss), then only
           compute the loss and gradient using the top
           ``loss_change_fraction`` of the representation. This can be
           combined wth ``fraction_removed`` so as to randomly subsample
           from this selection. To disable this (and use all the
           representation), this should be set to 1.

        We also provide the ability of using a coarse-to-fine
        optimization. Unlike the above methods, this will not work
        out-of-the-box with every model, as the model object must have a
        ``scales`` attributes (which gives the scales in fine-to-coarse
        order, i.e., reverse order that we will be optimizing) and that
        it's forward method can accept a ``scales`` keyword argument, a
        list that specifies which scales to use to compute the
        representation. If ``coarse_to_fine`` is True, then we optimize
        each scale until we think it's reached convergence before moving
        on. Once we've done each scale individually, we spend the rest
        of the iterations doing them all together, as if
        ``coarse_to_fine`` was False. This can be combined with the
        above three methods. We determine if a scale has converged in
        the same way as method 3 above: if the scale-specific loss
        ``loss_change_iter`` iterations ago is within
        ``loss_change_thresh`` of the most recent loss.

        Parameters
        ----------
        seed : int, optional
            Number with which to seed pytorch and numy's random number
            generators
        learning_rate : float, optional
            The learning rate for our optimizer
        max_iter : int, optinal
            The maximum number of iterations to run before we end
        initial_image : torch.tensor, array_like, or None, optional
            The 2d tensor we use to initialize the metamer. If None (the
            default), we initialize with uniformly-distributed random
            noise lying between 0 and 1. If this is not a tensor or
            None, we try to cast it as a tensor.
        clamper : plenoptic.Clamper or None, optional
            Clamper makes a change to the image in order to ensure that
            it stays reasonable. The classic example is making sure the
            range lies between 0 and 1, see plenoptic.RangeClamper for
            an example.
        optimizer: {'Adam', 'SGD', 'LBFGS'}
            The choice of optimization algorithm
        fraction_removed: float, optional
            The fraction of the representation that will be ignored
            when computing the loss. At every step the loss is computed
            using the remaining fraction of the representation only.
            A new sample is drawn a every step. This gives a stochastic
            estimate of the gradient and might help optimization.
        loss_thresh : float, optional
            If the loss over the past ``loss_change_iter`` is less than
            ``loss_thresh``, we stop.
        store_progress : bool or int, optional
            Whether we should store the representation of the metamer
            and the metamer image in progress on every iteration. If
            False, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress``
            iterations (note then that 0 is the same as False and 1 the
            same as True). If True or int>0, ``self.saved_image``
            contains the stored images, and ``self.saved_representation
            contains the stored representations.
        save_progress : bool or int, optional
            Whether to save the metamer as we go (so that you can check
            it periodically and so you don't lose everything if you have
            to kill the job / it dies before it finishes running). If
            True, we save to ``save_path`` every time we update the
            saved_representation. We attempt to save with the
            ``save_model_reduced`` flag set to True. If an int, we save
            every ``save_progress`` iterations. Note that this can end
            up actually taking a fair amount of time, especially for
            large numbers of iterations (and thus, presumably, larger
            saved history tensors) -- it's therefore recommended that
            you set this to a relatively large integer (say, one-tenth
            ``max_iter``) for the purposes of speeding up your
            synthesis.
        save_path : str, optional
            The path to save the synthesis-in-progress to (ignored if
            ``save_progress`` is False)
        loss_change_iter : int, optional
            How many iterations back to check in order to see if the
            loss has stopped decreasing in order to determine whether we
            should only calculate the gradient with respect to the
            ``loss_change_fraction`` fraction of statistics with
            the highest error.
        loss_change_thresh : float, optional
            The threshold below which we consider the loss as unchanging
            in order to determine whether we should only calculate the
            gradient with respect to the
            ``loss_change_fraction`` fraction of statistics with
            the highest error.
        loss_change_fraction : float, optional
            If we think the loss has stopped decreasing (based on
            ``loss_change_iter`` and ``loss_change_thresh``), the
            fraction of the representation with the highest loss that we
            use to calculate the gradients
        coarse_to_fine : bool, optional
            If True, we attempt to use the coarse-to-fine optimization
            (see above for more details on what's required of the model
            for this to work).
        optimizer_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the optimizer (in
            addition to learning_rate). What these should be depend on
            the specific optimizer you're using

        Returns
        -------
        matched_image : torch.tensor
            The metamer we've created
        matched_representation : torch.tensor
            The model's representation of the metamer

        """
        self.seed = seed
        # random initialization
        torch.manual_seed(seed)
        np.random.seed(seed)

        if initial_image is None:
            self.matched_image = torch.rand_like(self.target_image, dtype=torch.float32,
                                                 device=self.target_image.device)
            self.matched_image.requires_grad = True
        else:
            if not isinstance(initial_image, torch.Tensor):
                initial_image = torch.tensor(initial_image, dtype=torch.float32,
                                             device=self.target_image.device)
            self.matched_image = torch.nn.Parameter(initial_image, requires_grad=True)

        while self.matched_image.ndimension() < 4:
            self.matched_image = self.matched_image.unsqueeze(0)
        if isinstance(self.matched_representation, torch.nn.Parameter):
            # for some reason, when saving and loading the metamer
            # object after running it, self.matched_representation ends
            # up as a parameter, which we don't want. This resets it.
            delattr(self, 'matched_representation')
            self.matched_representation = None

        self.fraction_removed = fraction_removed
        self.loss_change_thresh = loss_change_thresh
        self.loss_change_iter = loss_change_iter
        self.loss_change_fraction = loss_change_fraction
        self.coarse_to_fine = coarse_to_fine
        if coarse_to_fine:
            # this creates a new object, so we don't modify model.scales
            self.scales = ['all'] + [i for i in self.model.scales]
            self.scales_timing = dict((k, []) for k in self.scales)
            self.scales_timing[self.scales[-1]].append(0)
        if loss_thresh >= loss_change_thresh:
            raise Exception("loss_thresh must be strictly less than loss_change_thresh, or things"
                            " get weird!")

        optimizer_kwargs.update({'optimizer': optimizer, 'lr': learning_rate})
        self.optimizer_kwargs = optimizer_kwargs
        self._init_optimizer(**self.optimizer_kwargs)

        # python's implicit boolean-ness means we can do this! it will evaluate to False for False
        # and 0, and True for True and every int >= 1
        if store_progress:
            if store_progress is True:
                store_progress = 1
            # if this is not the first time synthesize is being run for
            # this metamer object,
            # saved_image/saved_representation(_gradient) will be
            # tensors instead of lists. This converts them back to lists
            # so we can use append. If it's the first time, they'll be
            # empty lists and this does nothing
            self.saved_image = list(self.saved_image)
            self.saved_representation = list(self.saved_representation)
            self.saved_image_gradient = list(self.saved_image_gradient)
            self.saved_representation_gradient = list(self.saved_representation_gradient)
            self.saved_image.append(self.matched_image.clone().to('cpu'))
            self.saved_representation.append(self.analyze(self.matched_image).to('cpu'))
        else:
            if save_progress:
                raise Exception("Can't save progress if we're not storing it! If save_progress is"
                                " True, store_progress must be not False")
        self.store_progress = store_progress

        pbar = tqdm(range(max_iter))

        for i in pbar:
            loss, g, lr = self._optimizer_step(pbar)
            self.loss.append(loss.item())
            self.gradient.append(g.item())
            self.learning_rate.append(lr)
            if np.isnan(loss.item()):
                warnings.warn("Loss is NaN, quitting out! We revert matched_image / matched_"
                              "representation to our last saved values (which means this will "
                              "throw an IndexError if you're not saving anything)!")
                # need to use the -2 index because the last one will be
                # the one full of NaNs. this happens because the loss is
                # computed before calculating the gradient and updating
                # matched_image; therefore the iteration where loss is
                # NaN is the one *after* the iteration where
                # matched_image (and thus matched_representation)
                # started to have NaN values. this will fail if it hits
                # a nan before store_progress iterations (because then
                # saved_image/saved_representation only has a length of
                # 1) but in that case, you have more severe problems
                self.matched_image = nn.Parameter(self.saved_image[-2])
                self.matched_representation = nn.Parameter(self.saved_representation[-2])
                break

            with torch.no_grad():
                if clamper is not None:
                    self.matched_image.data = clamper.clamp(self.matched_image.data)

                # i is 0-indexed but in order for the math to work out we want to be checking a
                # 1-indexed thing against the modulo (e.g., if max_iter=10 and
                # store_progress=3, then if it's 0-indexed, we'll try to save this four times,
                # at 0, 3, 6, 9; but we just want to save it three times, at 3, 6, 9)
                if store_progress and ((i+1) % store_progress == 0):
                    # want these to always be on cpu, to reduce memory use for GPUs
                    self.saved_image.append(self.matched_image.clone().to('cpu'))
                    self.saved_representation.append(self.analyze(self.matched_image).to('cpu'))
                    self.saved_image_gradient.append(self.matched_image.grad.clone().to('cpu'))
                    self.saved_representation_gradient.append(self.matched_representation.grad.clone().to('cpu'))
                    if save_progress is True:
                        self.save(save_path, True)
                if type(save_progress) == int and ((i+1) % save_progress == 0):
                    self.save(save_path, True)

            if len(self.loss) > self.loss_change_iter:
                if abs(self.loss[-self.loss_change_iter] - self.loss[-1]) < loss_thresh:
                    if self.coarse_to_fine:
                        # only break out if we've been doing for long enough
                        if self.scales[-1] == 'all' and i - self.scales_timing['all'][0] > self.loss_change_iter:
                            break
                    else:
                        break

        pbar.close()

        if store_progress:
            self.saved_representation = torch.stack(self.saved_representation)
            self.saved_image = torch.stack(self.saved_image)
            self.saved_image_gradient = torch.stack(self.saved_image_gradient)
            # we can't stack the gradients if we used coarse-to-fine
            # optimization, because then they'll be different shapes, so
            # we have to keep them as a list
            try:
                self.saved_representation_gradient = torch.stack(self.saved_representation_gradient)
            except RuntimeError:
                pass
        return self.matched_image.data.squeeze(), self.matched_representation.data.squeeze()

    def save(self, file_path, save_model_reduced=False):
        r"""save all relevant variables in .pt file

        Note that if store_progress is True, this will probably be very
        large

        Parameters
        ----------
        file_path : str
            The path to save the metamer object to
        save_model_reduced : bool
            Whether we save the full model or just its attribute
            ``state_dict_reduced`` (this is a custom attribute of ours,
            the basic idea being that it only contains the attributes
            necessary to initialize the model, none of the (probably
            much larger) ones it gets during run-time).

        """
        attrs = ['model', 'matched_image', 'target_image', 'seed', 'loss', 'target_representation',
                 'matched_representation', 'saved_representation', 'gradient', 'saved_image',
                 'learning_rate', 'saved_representation_gradient', 'saved_image_gradient']
        super().save(file_path, save_model_reduced,  attrs)

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
        attrs = ['target_image', 'target_representation', 'matched_image',
                 'matched_representation', 'saved_image', 'saved_representation',
                 'saved_image_gradient', 'saved_representation_gradient']
        return super().to(*args, attrs=attrs, **kwargs)
