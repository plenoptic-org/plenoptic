import torch
import re
import warnings
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import pyrtools as pt
from ..tools.display import rescale_ylim, plot_representation, update_plot
from ..tools.data import to_numpy
from ..tools.metamer_utils import RangeClamper
from matplotlib import animation


class Metamer(nn.Module):
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
    - [ ] flexibility on the optimizer / scheduler (or at least parameterize the stuff): do similar
          to above? -- not as important right now
    - [ ] should we initialize optimizer / scheduler at initialization or during the call to
          synthesize? seems reasonable to me that you'd want to change it I guess... --  not
          important right now, same as above
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

    def analyze(self, x, **kwargs):
        r"""Analyze the image, that is, obtain the model's representation of it

        Any kwargs are passed to the model's forward method
        """
        y = self.model(x, **kwargs)
        if isinstance(y, list):
            return torch.cat([s.squeeze().view(-1) for s in y]).unsqueeze(1)
        else:
            return y

    def objective_function(self, x, y):
        r"""Calculate the loss between x and y

        This is what we minimize. Currently it's the L2-norm
        """
        return torch.norm(x - y, p=2)

    def _init_optimizer(self, optimizer, lr, **optimizer_kwargs):
        """Initialize the optimzer and job scheduler

        This gets called at the beginning of synthesize() and can also
        be called at other moments to make sure we're using the original
        learning rate (e.g., when moving to a different scale for
        coarse-to-fine optimization).

        """
        if optimizer == 'SGD':
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
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=.5)

    def _closure(self):
        r"""An abstraction of the gradient calculation, before the optimization step. This enables optimization algorithms
        that perform several evaluations of the gradient before taking a step (ie. second order methods like LBFGS).

        Note that the fraction removed also happens here, and for now a fresh sample of noise is drwan at each iteration.
            i) that means for now we do not support LBFGS with a random fraction removed.
            ii) beyond removing random fraction of the coefficients, one could schedule the optimization (eg. coarse to fine)
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

    def _optimizer_step(self, pbar):
        r"""Compute and propagate gradients, then step the optimizer to update matched_image

        Parameters
        ----------
        pbar : tqdm.tqdm
            A tqdm progress-bar, which we update with a postfix
            describing the current loss, gradient norm, and learning
            rate (it already tells us which iteration and the time
            elapsed)

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
        self.scheduler.step(loss.item())

        if self.coarse_to_fine and self.scales[-1] != 'all':
            with torch.no_grad():
                tmp_im = self.matched_image.detach().clone()
                # if the model has a cone_power attribute, it's going to
                # raise its input to some power and if that power is
                # fractional, it won't handle negative values well. this
                # should be generally handled by clamping, but clamping
                # happens after this, which is just intended to give a
                # sense of the overall loss, so we clamp with a min of 0
                if hasattr(self.model, 'cone_power'):
                    if self.model.cone_power != int(self.model.cone_power):
                        tmp_im = torch.clamp(tmp_im, min=0)
                full_matched_rep = self.analyze(tmp_im)
                loss = self.objective_function(full_matched_rep, self.target_representation)
        else:
            loss = self.objective_function(self.matched_representation, self.target_representation)

        postfix_dict.update(dict(loss="%.4e" % loss.item(), gradient_norm="%.4e" % g.norm().item(),
                                 learning_rate=self.optimizer.param_groups[0]['lr']))
        # add extra info here if you want it to show up in progress bar
        pbar.set_postfix(**postfix_dict)
        return loss, g.norm(), self.optimizer.param_groups[0]['lr']

    def synthesize(self, seed=0, learning_rate=.01, max_iter=100, initial_image=None,
                   clamper=RangeClamper((0, 1)), clamp_each_iter=True, optimizer='SGD',
                   fraction_removed=0., loss_thresh=1e-4, store_progress=False,
                   save_progress=False, save_path='metamer.pt', loss_change_thresh=1e-2,
                   loss_change_iter=50, loss_change_fraction=1., coarse_to_fine=False,
                   **optimizer_kwargs):
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
            it stays reasonable. The classic example (and default
            option) is making sure the range lies between 0 and 1, see
            plenoptic.RangeClamper for an example.
        clamp_each_iter : bool, optional
            If True (and ``clamper`` is not ``None``), we clamp every
            iteration. If False, we only clamp at the very end, after
            the last iteration
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
                if clamper is not None and clamp_each_iter:
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

        # if clamp_each_iter is True, then we've done this above and so
        # this gains us nothing, but would leave us open to weird edge
        # cases
        if clamper is not None and not clamp_each_iter:
            self.matched_image.data = clamper.clamp(self.matched_image.data)
            self.matched_representation = self.analyze(self.matched_image)

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
        model = self.model
        try:
            if save_model_reduced:
                model = self.model.state_dict_reduced
        except AttributeError:
            warnings.warn("self.model doesn't have a state_dict_reduced attribute, will pickle "
                          "the whole model object")
        save_dict = {}
        for k in ['matched_image', 'target_image', 'seed', 'loss', 'target_representation',
                  'matched_representation', 'saved_representation', 'gradient', 'saved_image',
                  'learning_rate', 'saved_representation_gradient', 'saved_image_gradient']:
            attr = getattr(self, k)
            # detaching the tensors avoids some headaches like the
            # tensors having extra hooks or the like
            if isinstance(attr, torch.Tensor):
                attr = attr.detach()
            save_dict[k] = attr
        save_dict['model'] = model
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
            The path to load the metamer object from
        model_constructor : callable or None, optional
            When saving the metamer object, we have the option to only
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
        metamer : plenoptic.synth.Metamer
            The loaded metamer object


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
        metamer = cls(target_image, model)
        for k, v in tmp_dict.items():
            setattr(metamer, k, v)
        return metamer

    def to(self, *args, do_windows=True, **kwargs):
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
        try:
            self.model = self.model.to(*args, **kwargs)
        except AttributeError:
            warnings.warn("model has no `to` method, so we leave it as is...")
        for k in ['target_image', 'target_representation', 'matched_image',
                  'matched_representation', 'saved_image', 'saved_representation',
                  'saved_image_gradient', 'saved_representation_gradient']:
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
        representation_error = self.representation_error(iteration)
        return plot_representation(self.model, representation_error, ax, figsize, ylim,
                                   batch_idx, title)

    def plot_metamer_status(self, batch_idx=0, channel_idx=0, iteration=None, figsize=(17, 5),
                            ylim=None, plot_representation_error=True, imshow_zoom=None,
                            vrange=(0, 1)):
        r"""Make a plot showing metamer, loss, and (optionally) representation ratio

        We create two or three subplots on a new figure. The first one
        contains the metamer, the second contains the loss, and the
        (optional) third contains the representation ratio, as plotted
        by ``self.plot_representation_error``.

        You can specify what iteration to view by using the
        ``iteration`` arg. The default, ``None``, shows the final one.

        The loss plot shows the loss as a function of iteration for all
        iterations (even if we didn't save the representation or metamer
        at each iteration), with a red dot showing the location of the
        iteration.

        We use ``pyrtools.imshow`` to display the metamer and attempt to
        automatically find the most reasonable zoom value. You can
        override this value using the imshow_zoom arg, but remember that
        ``pyrtools.imshow`` is opinionated about the size of the
        resulting image and will throw an Exception if the axis created
        is not big enough for the selected zoom. We currently cannot
        shrink the image, so figsize must be big enough to display the
        image

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
            How much to zoom in / enlarge the metamer image, the ratio
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
        if plot_representation_error:
            n_subplots = 3
        else:
            n_subplots = 2
        if iteration is None:
            image = self.matched_image[batch_idx, channel_idx]
            loss_idx = len(self.loss) - 1
        else:
            image = self.saved_image[iteration, batch_idx, channel_idx]
            if iteration < 0:
                # in order to get the x-value of the dot to line up,
                # need to use this work-around
                loss_idx = len(self.loss) + iteration
            else:
                loss_idx = iteration
        fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
        if imshow_zoom is None:
            # image.shape[0] is the height of the image
            imshow_zoom = axes[0].bbox.height // image.shape[0]
            if imshow_zoom == 0:
                raise Exception("imshow_zoom would be 0, cannot display metamer image! Enlarge "
                                "your figure")
        fig = pt.imshow(to_numpy(image), ax=axes[0], title='Metamer', zoom=imshow_zoom,
                        vrange=vrange)
        axes[0].xaxis.set_visible(False)
        axes[0].yaxis.set_visible(False)
        axes[1].semilogy(self.loss)
        axes[1].scatter(loss_idx, self.loss[loss_idx], c='r')
        axes[1].set_title('Loss')
        if plot_representation_error:
            fig = self.plot_representation_error(batch_idx, iteration, ax=axes[2], ylim=ylim)
        return fig

    def animate(self, batch_idx=0, channel_idx=0, figsize=(17, 5), framerate=10, ylim='rescale',
                plot_representation_error=True, imshow_zoom=None):
        r"""Animate metamer synthesis progress!

        This is essentially the figure produced by
        ``self.plot_metamer_status`` animated over time, for each stored
        iteration.

        It's difficult to determine a reasonable figsize, because we
        don't know how much information is in the plot showing the
        representation ratio. Therefore, it's recommended you play
        around with ``plot_metamer_status`` until you find a
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
        loss = self.loss + [self.loss[-1]]
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
        if self.target_representation.ndimension() == 4:
            ylim = False
        # initialize the figure
        fig = self.plot_metamer_status(batch_idx, channel_idx, 0, figsize, ylim,
                                       plot_representation_error, imshow_zoom=imshow_zoom)
        # grab the artists for the second plot (we don't need to do this
        # for the metamer or representation plot, because we use the
        # update_plot function for that)
        scat = fig.axes[1].collections[0]

        if self.target_representation.ndimension() == 4:
            warnings.warn("Looks like representation is image-like, haven't fully thought out how"
                          " to best handle rescaling color ranges yet!")
            # replace the bit of the title that specifies the range,
            # since we don't make any promises about that
            for ax in fig.axes[2:]:
                ax.set_title(re.sub(r'\n range: .* \n', '\n\n', ax.get_title()))

        def movie_plot(i):
            artists = []
            artists.extend(update_plot([fig.axes[0]], data=self.saved_image[i],
                                       batch_idx=batch_idx))
            if plot_representation_error:
                representation_error = self.representation_error(i)
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
            scat.set_offsets((i*saved_subsample, loss[i*saved_subsample]))
            artists.append(scat)
            # as long as blitting is True, need to return a sequence of artists
            return artists

        # don't need an init_func, since we handle initialization ourselves
        anim = animation.FuncAnimation(fig, movie_plot, frames=len(self.saved_image),
                                       blit=True, interval=1000./framerate, repeat=False)
        plt.close(fig)
        return anim
