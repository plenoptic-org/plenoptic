import torch
import warnings
from tqdm import tqdm
import numpy as np
from .Synthesis import Synthesis


class MADCompetition(Synthesis):
    """Generate maximally-differentiating images for two models

    In MAD Competition, we start with a reference image and generate two
    pairs of images. We end up with two pairs of images, one of which
    contains the images which produce the largest and smallest responses
    in model 1 while keeping model 2's response as close to constant as
    possible, while the other pair of images does the inverse
    (differentiates model 2's responses as much as possible while
    keeping model 1's response as close to constant as possible).

    Note: a complete set of MAD Competition stimuli consists of four
    synthesized images per target image. We modularize this so that a
    call to ``synthesis()`` gets you one of the four images (by setting
    the ``synthesis_target`` arg, you determine which one). You can get
    the full set by calling ``synthesis()`` four times with different
    args (allowing you to parallelize however you see fit), or by
    calling ``synthesize_all()``.

    Because MAD Competition works with two models at once, whereas most
    synthesis methods only work with one, this class has "doubled
    attributes", that is, for many attributes, ``MADCompetition`` has
    two versions. For example, instead of a single ``loss`` attribute,
    we have a ``loss_1`` and ``loss_2`` attribute, containing the loss
    for ``model_1`` and ``model_2``, respectively. For all attributes of
    this type, you can access the one that is currently be modified
    using the base string (e.g., ``loss``), but it is recommended that
    you stick with the explicit attributes unless you're sure you know
    what you're doing.

    Because a full set consists of four synthesized images, many
    attributes below have a corresponding ``_all`` attribute, which is a
    dictionary containing that data from each of the four synthesis
    sets. For example, we have a ``matched_image_all`` attribute, which
    is a dictionary with four keys (the four possible values of
    ``synthesis_target``) that each contain the relevant synthesized
    image.

    Warning
    -------
    There are several limitations to this implementation:

    1. Both two models both have to be functioning `torch.nn.Module`,
       with a `forward()` method and a the ability to propagate
       gradients backwards through them.

    2. Both models must take an arbitrary grayscale image as input
       (currently, we do not support color images, movies, or batches of
       images).

    3. Both models must produce a scalar output (the prediction). We do
       not currently support models that produce a vector of predictions
       (for example, firing rates of a population of neurons or BOLD
       signals of voxels across a brain region)

    Parameters
    ----------
    model_1, model_2 : torch.nn.Module
        The two models to compare.
    target_image : torch.tensor or array_like
        A 4d tensor, this is the image whose representation we wish to
        match. If this is not a tensor, we try to cast it as one.

    Attributes
    ----------
    target_image : torch.tensor
        A 2d tensor, this is the image whose representation we wish to
        match.
    model_1, model_2 : torch.nn.Module
        Two differentiable model that takes an image as an input and
        transforms it into a representation of some sort. We only
        require that they have a forward method, which returns the
        representation to match.
    target_representation_1, target_representation_2 : torch.tensor
        Whatever is returned by ``model_1`` and ``model_2``
        ``forward(target_image)`` methods, respectively. This is the
        representation we're trying to get as close or far away from as
        possible when targeting a given model.
    initial_image : torch.tensor
        target_image with white noise added to it (and clamped, if
        applicable), this is the starting point of our synthesis
    initial_image_all : dict
        Dictionray containing ``initial_image `` for each
        ``synthesis_target`` (if run individually, they can have
        different noise seeds)
    matched_image : torch.tensor
        The synthesized image from the last call to
        ``synthesis()``. This may be unfinished depending on how many
        iterations we've run for.
    matched_image_all : dict
        Dictionary containing ``matched_image`` for each
        ``synthesis_target``
    matched_represetation_1, matched_representation_2: torch.tensor
        Whatever is returned by ``model_1`` and ``model_2``
        ``forward(matched_image)``, respectively.
    matched_representation_1_all, matched_representation_2_all : dict
        Dictionary containing ``matched_representation_1`` and
        ``matched_representation_2``, respectively for each
        ``synthesis_target``
    optimizer : torch.optim.Optimizer
        A pytorch optimization method.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        A pytorch scheduler, which tells us how to change the learning
        rate over iterations. Currently, user cannot set and we use
        ReduceLROnPlateau (so that the learning rate gets reduced if it
        seems like we're on a plateau i.e., the loss isn't changing
        much)
    loss_1, loss_2 : list
        list of the loss with respect to model_1, model_2 over
        iterations.
    loss_1_all, loss_2_all : dict
        Dictionary containing ``loss_1`` and ``loss_2``, respectively
        for each ``synthesis_target``
    gradient : dict
        Dictionary containing the gradient over iterations for each
        ``synthesis_target``
    learning_rate : dict
        Dictionary of the learning_rate over iterations for each
        ``synthesis_target``. We use a scheduler that gradually reduces
        this over time, so it won't be constant.
    nu : dict
        Dictionary of the nu parameter over iterations for each
        ``synthesis_target``. Nu is the parameter used to correct the
        image so that the other model's representation will not change;
        see docstring of ``self._find_nu()`` for more details
    saved_representation_1, saved_representation_2 : dict
        If the ``store_progress`` arg in ``synthesize`` is set to True
        or an int>0, we will save ``self.matched_representation`` at
        each iteration (or each ``store_progress`` iteration, if it's an
        int), for later examination (separately for each model and each
        ``synthesis_target``).
    saved_image : dict
        If the ``store_progress`` arg in ``synthesize`` is set to True
        or an int>0, we will save ``self.matched_image`` at each
        iteration (or each ``store_progress`` iteration, if it's an
        int), for later examination (separately for each
        ``synthesis_target``).
    seed : int
        Number with which to seed pytorch and numy's random number
        generators
    saved_image_gradient : dict
        If the ``store_progress`` arg in ``synthesize`` is set to True
        or an int>0, we will save ``self.matched_image.grad`` at each
        iteration (or each ``store_progress`` iteration, if it's an
        int), for later examination (separately for each
        ``synthesis_target``).
    saved_representation_1_gradient, saved_representation_2_gradient : dict
        If the ``store_progress`` arg in ``synthesize`` is set to
        True or an int>0, we will save
        ``self.matched_representation.grad`` at each iteration (or each
        ``store_progress`` iteration, if it's an int), for later
        examination (separately for each model and
        ``synthesis_target``).
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

    Notes
    -----
    Method described in [1]_.

    References
    -----
    .. [1] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD) competition: A
           methodology for comparing computational models of perceptual discriminability. Journal
           of Vision, 8(12), 1–13. http://dx.doi.org/10.1167/8.12.8

    """

    def __init__(self, model_1, model_2, target_image):
        super().__init__()

        self._names = {'target_image': 'target_image',
                       'matched_image': 'matched_image',
                       'model': 'model_1',
                       'target_representation': 'target_representation_1',
                       'matched_representation': 'matched_representation_1',
                       'initial_representation': 'initial_representation_1',
                       'loss_norm': 'loss_1_norm',
                       'loss': 'loss_1',
                       'saved_representation': 'saved_representation_1',
                       'saved_representation_gradient': 'saved_representation_gradient_1'}

        self.loss_sign = 1
        self.model_1 = model_1
        self.model_2 = model_2
        if not isinstance(target_image, torch.Tensor):
            target_image = torch.tensor(target_image, torch.float32)
        self.target_image = target_image
        self.target_representation_1 = self.analyze(self.target_image)
        self.update_target('model_1_min', 'fix')
        self.target_representation_2 = self.analyze(self.target_image)
        self.matched_image = None
        self.initial_image = None
        self.matched_representation_1 = None
        self.matched_representation_2 = None
        self.update_target('model_1_min', 'main')
        self.step = 'main'
        self.loss_1 = []
        self.loss_2 = []

        def _init_dict():
            return dict((k, []) for k in ['model_1_min', 'model_1_max', 'model_2_min',
                                          'model_2_max'])
        self.saved_representation_1 = _init_dict()
        self.saved_representation_2 = _init_dict()
        self.saved_image = _init_dict()
        self.saved_image_gradient = _init_dict()
        self.saved_representation_1_gradient = _init_dict()
        self.saved_representation_2_gradient = _init_dict()
        self.loss_1_all = _init_dict()
        self.loss_2_all = _init_dict()
        self.gradient = _init_dict()
        self.learning_rate = _init_dict()
        self.nu = _init_dict()
        self.initial_image_all = _init_dict()
        self.matched_image_all = _init_dict()
        self.initial_representation_1_all = _init_dict()
        self.initial_representation_2_all = _init_dict()
        self.matched_representation_1_all = _init_dict()
        self.matched_representation_2_all = _init_dict()

        self.coarse_to_fine = False
        self.scales = []
        self.scales_loss = []
        self.loss_change_thresh = 1e-2
        self.loss_change_iter = 50
        self.loss_change_fraction = 1.
        self.fraction_removed = 0

    def __getattr__(self, name):
        """get an attribute

        this is the standard __getattr__, except we override it for the
        attributes that have two versions, depending on which model
        we're currently targeting: 'target_representation',
        'target_image', 'matched_representation', 'loss',
        'matched_image', 'model', 'loss_norm', 'initial_representation',
        'saved_representation', 'saved_representation_gradient'

        """
        if name in ['target_representation', 'target_image', 'matched_representation', 'loss',
                    'matched_image', 'model', 'loss_norm', 'initial_representation',
                    'saved_representation', 'saved_representation_gradient']:
            name = self._names[name]
        try:
            return self.__dict__[name]
        except KeyError:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        """set attributes

        this is the standard __setattr__, except we override it for the
        attributes that have two versions, depending on which model
        we're currently targeting: 'target_representation',
        'target_image', 'matched_representation', 'loss',
        'matched_image', 'model', 'loss_norm', 'initial_representation',
        'saved_representation', 'saved_representation_gradient'

        """
        if name in ['target_representation', 'target_image', 'matched_representation', 'loss',
                    'matched_image', 'model', 'loss_norm', 'initial_representation',
                    'saved_representation', 'saved_representation_gradient']:
            name = self._names[name]
        super().__setattr__(name, value)

    def update_target(self, synthesis_target, step):
        """Update attributes to target for synthesis

        We use this function to switch back and forth between whether
        we're updating the attributes based on minimizing or maximizing
        model_1's loss or model_2's loss

        Parameters
        ----------
        synthesis_target : {'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which image to synthesize
        step : {'main', 'fix'}
            whether we're minimizing/maximizing the targeted model's
            loss or maintaining the other one's

        """
        # save this history for later, so we can reset to last state
        self._last_update_target_args = [synthesis_target, step]
        if step not in ['main', 'fix']:
            raise Exception(f"Don't know what to do with step {step}!")
        if synthesis_target not in ['model_1_min', 'model_1_max', 'model_2_min', 'model_2_max']:
            raise Exception(f"Don't know what to do with synthesis_target {synthesis_target}!")
        model_num = synthesis_target.split('_')[1]
        other_model_num = {'1': '2', '2': '1'}[model_num]
        synth_direction = synthesis_target.split('_')[2]
        if step == 'main':
            num = model_num
            self.loss_sign = {'min': 1, 'max': -1}[synth_direction]
        else:
            num = other_model_num
            self.loss_sign = 1
        self._names.update({'model': f'model_{num}',
                            'target_representation': f'target_representation_{num}',
                            'matched_representation': f'matched_representation_{num}',
                            'initial_representation': f'initial_representation_{num}',
                            'loss_norm': f'loss_{num}_norm',
                            'loss': f'loss_{num}',
                            'saved_representation': f'saved_representation_{num}',
                            'saved_representation_gradient': f'saved_representation_gradient_{num}'})

    def _find_nu(self, grad, n_iter=10):
        """find the optimum nu to remain on model_2's level set

        While we're minimizing model_1's loss, we do our best to stay on
        model_2's level set. Projecting out model_2's gradient from
        model_1's gradient (which we do in ``_closure()``) helps with
        this, but it's not perfect.

        Call ``grad`` :math:`G`, ``target_image`` :math:`X` and the
        synthesized image on iteration n :math:`Y_{n}` (and thus
        :math:`Y_0` is the initial distorted image), and the proposed
        synthesized image that we've created after the main step of
        synthesis :math:`Y'_{n+1}`. Then the goal of this function is to
        find the :math:`\nu` that minimizes the difference between
        :math:`M_2(X,Y'_{n+1}+\nu G)` and :math:`M_2(X,Y_0)` (see [1]_,
        Appendix C for details).

        We do this using a brief optimization, using Adam with a
        learning rate scheduler. We minimize the above loss, doing
        several iterations to try and find the best nu (number of
        iterations set by ``n_iter``). During each iteration, we have to
        call model_2's forward method, so the amount of time that takes
        will have a direct impact on this function's runtime.

        Parameters
        ----------
        grad : torch.tensor
            the gradient of ``self.matched_image`` for model 2 with
            respect to the loss between ``self.matched_image`` and
            ``self.target_image``
        n_iter : int
            The number of iterations to use when finding the best
            nu. Obviously, the larger this number, the longer it will
            take

        Returns
        -------
        nu : torch.Parameter
            The optimized (scalar) nu value

        """
        lr = self.optimizer.param_groups[0]['lr']
        nu = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        nu_optim = torch.optim.Adam([nu], lr=1, amsgrad=True)
        nu_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(nu_optim, 'min', factor=.5)
        target_loss = self.objective_function(self.initial_representation,
                                              self.target_representation)
        for i in range(n_iter):
            # in Appendix C of the paper, they just add nu*grad to get
            # the proposed image. here we also multiply by a -lr because
            # of how pytorch updates images. see torch/optim/sgd.py, at
            # the very end of the step() function and you'll see that
            # when updating parameters, they add -lr * grad. thus, this
            # is what we need in order to predict what the updated image
            # will be
            proposed_img = self.matched_image - lr * nu * grad
            proposed_loss = self.objective_function(self.analyze(proposed_img),
                                                    self.target_representation)
            actual_loss = torch.abs(target_loss - proposed_loss)
            actual_loss.backward()
            nu_optim.step()
            nu_scheduler.step(actual_loss.item())

        return nu

    def _closure(self):
        """An abstraction of the gradient calculation, before the optimization step

        This is a bit of trickery intended to handle the fact that on
        each iteration of synthesis, we update the image twice: once to
        do our best to minimize/maximize one model's loss, and once to
        correct the image so that the other model's loss hasn't changed
        by much. We do this by checking ``self.step``: if ``'main'``, we
        minimize/maximize the first model's loss; if ``'fix'``, we
        correct for the second model's loss. (note that which model in
        the desription above corresponds to the attributes
        ``self.model_1`` and ``self.model_2`` is handled by the
        ``update_target()`` method and the getter/setter)

        (see [1]_ Appendix C for more details)

        """
        # the main step corresponds to equation C3 in the paper
        if self.step == "main":
            # grab model_stable's gradient
            self.update_target(self.synthesis_target, 'fix')
            loss_stable = super()._closure()
            grad_stable = self.matched_image.grad.clone()
            # grab model_target's gradient
            self.update_target(self.synthesis_target, self.step)
            loss_target = super()._closure()
            grad_target = self.matched_image.grad.clone()
            # we do this reshaping to make these vectors so that this matmul
            # ends up being a dot product, and thus we get a scalar output
            proj_grad = torch.matmul(grad_target.flatten().unsqueeze(0),
                                     grad_stable.flatten().unsqueeze(1))
            grad_stable_norm = torch.matmul(grad_stable.flatten().unsqueeze(0),
                                            grad_stable.flatten().unsqueeze(1))
            # project out model_stable's gradient from model_target's gradient
            self.matched_image.grad = grad_target - (proj_grad / grad_stable_norm) * grad_stable
            # return model_target's loss
            return loss_target
        # the fix step corresponds to equation C5 in the paper
        elif self.step == 'fix':
            # grab model_stable's gradient
            self.update_target(self.synthesis_target, self.step)
            loss = super()._closure()
            grad = self.matched_image.grad.clone()
            # find the best nu
            nu = self._find_nu(grad, self.fix_step_n_iter)
            self.nu[self.synthesis_target].append(nu)
            # update the gradient
            self.matched_image.grad = nu * grad
            self.update_target(self.synthesis_target, 'main')
            return loss

    def objective_function(self, x, y, norm_loss=True):
        r"""Calculate the loss between x and y

        This is what we minimize. Currently it's the L2-norm of their
        difference: ``torch.norm(x-y, p=2)``.

        We can also normalize the loss, if ``norm_loss=True`` and we
        have a ``loss_nom`` attribute. We use this to get the losses of
        our two models at the same magnitude (normalizing by their
        initial loss)

        Finally, we also multiply the loss by ``self.loss_sign``,
        because sometimes we want to minimize the loss and sometimes we
        want to maximize it; maximizing the loss is the same as
        minimizing its negative (the setting of ``self.loss_sign`` is
        handled automatically by the ``update_target()`` method)

        Parameters
        ----------
        x : torch.tensor
            the first element
        y : torch.tensor
            the second element
        norm_loss : bool, optional
            whether to normalize the loss by ``self.loss_norm`` or not.

        Returns
        -------
        loss : torch.tensor
            single-element tensor containing the L2-norm of the
            difference between x and y

        """
        loss = torch.norm(x - y, p=2)
        if norm_loss:
            loss = loss / self.loss_norm
        return self.loss_sign * loss

    def synthesize(self, synthesis_target, seed=0, initial_noise=.1, max_iter=100, learning_rate=1,
                   optimizer='Adam', clamper=None, store_progress=False, save_progress=False,
                   save_path='mad.pt', fix_step_n_iter=10, **optimizer_kwargs):
        r"""Synthesize one maximally-differentiating image

        This synthesizes a single image, minimizing or maximizing either
        model 1 or model 2 while holding the other constant. By setting
        ``synthesis_target``, you can determine which of these you wish
        to synthesize.

        Parameters
        ----------
        synthesis_target : {'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which image to synthesize
        seed : `int`, optional
            seed to initialize the random number generator with
        initial_noise : `float`, optional
            standard deviation of the Gaussian noise used to create the
            initial image from the target image
        max_iter : int, optional
            The maximum number of iterations to run before we end
        learning_rate : float, optional
            The learning rate for our optimizer
        optimizer: {'GD', 'Adam', 'SGD', 'LBFGS'}
            The choice of optimization algorithm. 'GD' is regular
            gradient descent, as decribed in [1]_
        clamper : plenoptic.Clamper or None, optional
            Clamper makes a change to the image in order to ensure that
            it stays reasonable. The classic example is making sure the
            range lies between 0 and 1, see plenoptic.RangeClamper for
            an example.
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
        fix_step_n_iter : int, optional
            Each iteration of synthesis has two steps: update the image
            to increase/decrease one model's loss (main step), then
            update it to ensure that the other model's loss is as
            constant as possible (fix step). In order to do that, we use
            a secondary optimization loop to determine how big a step we
            should take (the value of ``nu``). ``fix_step_n_iter``
            determines how many iterations we should use in that loop to
            find nu. Obviously, the larger this, the longer synthesis
            will take.
        optimizer_kwargs :
            Dictionary of keyword arguments to pass to the optimizer (in
            addition to learning_rate). What these should be depend on
            the specific optimizer you're using

        Returns
        -------
        matched_image : torch.tensor
            The MAD competition image we've created
        matched_representation_1 : torch.tensor
            model_1's representation of this image
        matched_representation_2 : torch.tensor
            The model_2's representation of this image

        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.update_target(synthesis_target, 'main')
        self.fix_step_n_iter = fix_step_n_iter
        self.loss_1 = self.loss_1_all[synthesis_target]
        self.loss_2 = self.loss_2_all[synthesis_target]

        self.synthesis_target = synthesis_target
        self.initial_image = (self.target_image + initial_noise *
                              torch.randn_like(self.target_image))
        self.matched_image = torch.nn.Parameter(self.initial_image.clone())
        # that initial noise can take us outside the clamper
        self.clamper = clamper
        if clamper is not None:
            self.matched_image.data = clamper.clamp(self.matched_image.data)
            self.initial_image.data = clamper.clamp(self.initial_image.data)
        if isinstance(self.matched_representation_1, torch.nn.Parameter):
            # for some reason, when saving and loading the metamer
            # object after running it, self.matched_representation ends
            # up as a parameter, which we don't want. This resets it.
            delattr(self, 'matched_representation_1')
        if isinstance(self.matched_representation_2, torch.nn.Parameter):
            # for some reason, when saving and loading the metamer
            # object after running it, self.matched_representation ends
            # up as a parameter, which we don't want. This resets it.
            delattr(self, 'matched_representation_2')
        self.update_target(self.synthesis_target, 'fix')
        self.target_representation = self.analyze(self.target_image)
        self.matched_representation = self.analyze(self.matched_image)
        self.initial_representation = self.analyze(self.initial_image)
        self.loss_norm = self.objective_function(self.target_representation,
                                                 self.initial_representation, False)

        self.update_target(self.synthesis_target, 'main')
        self.target_representation = self.analyze(self.target_image)
        self.matched_representation = self.analyze(self.matched_image)
        self.initial_representation = self.analyze(self.initial_image)
        # if synthesis target is model_1/2_max, then loss_sign is
        # negative (for main step; because minimizing the negative of
        # the loss is the same as maximizing it). But if we include the
        # negative in both the regular calculation of the loss and the
        # norm, then we end up canceling it out. This will make sure
        # that loss_norm is always positive
        self.loss_norm = abs(self.objective_function(self.target_representation,
                                                     self.initial_representation, False))

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
            self.saved_image[synthesis_target] = list(self.saved_image[synthesis_target])
            self.saved_image_gradient[synthesis_target] = list(self.saved_image_gradient[synthesis_target])
            self.saved_representation_1[synthesis_target] = list(self.saved_representation_1[synthesis_target])
            self.saved_representation_1_gradient[synthesis_target] = list(self.saved_representation_1_gradient[synthesis_target])
            self.saved_representation_2[synthesis_target] = list(self.saved_representation_2[synthesis_target])
            self.saved_representation_2_gradient[synthesis_target] = list(self.saved_representation_2_gradient[synthesis_target])
            self.saved_image[synthesis_target].append(self.matched_image.clone().to('cpu'))
            self.update_target('model_1_min', 'main')
            self.saved_representation_1[synthesis_target].append(self.analyze(self.matched_image).to('cpu'))
            self.update_target('model_1_min', 'fix')
            self.saved_representation_2[synthesis_target].append(self.analyze(self.matched_image).to('cpu'))
        else:
            if save_progress:
                raise Exception("Can't save progress if we're not storing it! If save_progress is"
                                " True, store_progress must be not False")
        self.store_progress = store_progress

        pbar = tqdm(range(max_iter), position=0, leave=True)

        self.update_target(self.synthesis_target, 'main')
        for i in pbar:
            self.update_target(self.synthesis_target, 'fix')
            loss_2 = self.objective_function(self.matched_representation,
                                             self.target_representation).item()
            self.loss.append(loss_2)
            self.update_target(self.synthesis_target, 'main')
            self.step = 'main'
            loss, g, lr = self._optimizer_step(pbar, stable_loss="%.4e" % loss_2)
            self.loss.append(abs(loss.item()))
            self.update_target(self.synthesis_target, 'fix')
            self.step = 'fix'
            self._optimizer_step()
            self.gradient[synthesis_target].append(g.item())
            self.learning_rate[synthesis_target].append(lr)

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
                self.matched_image = torch.nn.Parameter(self.saved_image[-2])
                self.matched_representation_1 = torch.nn.Parameter(self.saved_representation_1[-2])
                self.matched_representation_2 = torch.nn.Parameter(self.saved_representation_2[-2])
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
                    self.saved_image[synthesis_target].append(self.matched_image.clone().to('cpu'))
                    self.saved_image_gradient[synthesis_target].append(self.matched_image.grad.clone().to('cpu'))
                    # we use model_1_min here as the synthesis target
                    # because we know that, whatever the synthesis
                    # target is, this will get us model 1 and model 2,
                    # respectively
                    self.update_target('model_1_min', 'main')
                    self.saved_representation_1[synthesis_target].append(self.analyze(self.matched_image).to('cpu'))
                    self.saved_representation_1_gradient[synthesis_target].append(self.matched_representation_1.grad.clone().to('cpu'))
                    self.update_target('model_1_min', 'fix')
                    self.saved_representation_2[synthesis_target].append(self.analyze(self.matched_image).to('cpu'))
                    self.saved_representation_2_gradient[synthesis_target].append(self.matched_representation_2.grad.clone().to('cpu'))
                    if save_progress is True:
                        self.save(save_path, True)
                if type(save_progress) == int and ((i+1) % save_progress == 0):
                    self.save(save_path, True)

        pbar.close()

        if store_progress:
            self.saved_image[synthesis_target] = torch.stack(self.saved_image[synthesis_target])
            self.saved_image_gradient[synthesis_target] = torch.stack(self.saved_image_gradient[synthesis_target])
            self.saved_representation_1[synthesis_target] = torch.stack(self.saved_representation_1[synthesis_target])
            self.saved_representation_2[synthesis_target] = torch.stack(self.saved_representation_2[synthesis_target])
            # we can't stack the gradients if we used coarse-to-fine
            # optimization, because then they'll be different shapes, so
            # we have to keep them as a list
            try:
                self.saved_representation_1_gradient[synthesis_target] = torch.stack(self.saved_representation_1_gradient[synthesis_target])
                self.saved_representation_2_gradient[synthesis_target] = torch.stack(self.saved_representation_2_gradient[synthesis_target])
            except RuntimeError:
                pass
        self.initial_image_all[synthesis_target] = self.initial_image.clone().to('cpu')
        self.matched_image_all[synthesis_target] = self.matched_image.clone().to('cpu')
        self.initial_representation_1_all[synthesis_target] = self.initial_representation_1.clone().to('cpu')
        self.initial_representation_2_all[synthesis_target] = self.initial_representation_2.clone().to('cpu')
        self.matched_representation_1_all[synthesis_target] = self.matched_representation_1.clone().to('cpu')
        self.matched_representation_2_all[synthesis_target] = self.matched_representation_2.clone().to('cpu')
        return self.matched_image.data, self.matched_representation_1.data, self.matched_representation_2.data

    def synthesize_all(self, seed=0, initial_noise=.1, max_iter=100, learning_rate=1,
                       optimizer='Adam', clamper=None, store_progress=False, save_progress=False,
                       save_path='mad.pt', fix_step_n_iter=10, **optimizer_kwargs):
        """Synthesize two pairs of maximally-differentiating images

        MAD Competitoin consists of two pairs of
        maximally-differentiating images: one pair minimizes and
        maximizes model 1, while holding model 2 constant, and the other
        minimizes and maximizes model 2, while holding model 1
        constant. This creates all four images. We return nothing, but
        all the outputs are stored in attributes.

        All parameters are passed directly through to ``synthesis()`` so
        if you want to synthesize the four images with different
        arguments, you should call ``synthesis()`` directly

        Parameters
        ----------
        synthesis_target : {'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which image to synthesize
        seed : `int`, optional
            seed to initialize the random number generator with
        initial_noise : `float`, optional
            standard deviation of the Gaussian noise used to create the
            initial image from the target image
        max_iter : int, optional
            The maximum number of iterations to run before we end
        learning_rate : float, optional
            The learning rate for our optimizer
        optimizer: {'GD', 'Adam', 'SGD', 'LBFGS'}
            The choice of optimization algorithm. 'GD' is regular
            gradient descent, as decribed in [1]_
        clamper : plenoptic.Clamper or None, optional
            Clamper makes a change to the image in order to ensure that
            it stays reasonable. The classic example is making sure the
            range lies between 0 and 1, see plenoptic.RangeClamper for
            an example.
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
        fix_step_n_iter : int, optional
            Each iteration of synthesis has two steps: update the image
            to increase/decrease one model's loss (main step), then
            update it to ensure that the other model's loss is as
            constant as possible (fix step). In order to do that, we use
            a secondary optimization loop to determine how big a step we
            should take (the value of ``nu``). ``fix_step_n_iter``
            determines how many iterations we should use in that loop to
            find nu. Obviously, the larger this, the longer synthesis
            will take.
        optimizer_kwargs :
            Dictionary of keyword arguments to pass to the optimizer (in
            addition to learning_rate). What these should be depend on
            the specific optimizer you're using

        """
        for target in ['model_1_min', 'model_1_max', 'model_2_min', 'model_2_max']:
            print(f"Synthesizing {target}")
            self.synthesize(target, seed, initial_noise, max_iter, learning_rate, optimizer,
                            clamper, store_progress, save_progress, save_path, fix_step_n_iter,
                            **optimizer_kwargs)

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
        attrs = ['model_1', 'model_2', 'matched_image', 'target_image', 'seed', 'loss',
                 'target_representation_1', 'target_representation_2', 'matched_representation_1',
                 'matched_representation_2', 'saved_representation_1', 'saved_representation_2',
                 'gradient', 'saved_image', 'learning_rate', 'saved_representation_1_gradient',
                 'saved_representation_2_gradient', 'saved_image_gradient']
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
        attrs = ['target_image', 'target_representation_1', 'target_representation_2',
                 'matched_image', 'matched_representation_1', 'matched_representation_2',
                 'saved_image', 'saved_representation_1', 'saved_representation_2',
                 'saved_image_gradient', 'saved_representation_1_gradient',
                 'saved_representation_2_gradient', 'model_1', 'model_2']
        return super().to(*args, attrs=attrs, **kwargs)

    def representation_error(self, synthesis_target=None, model=None, iteration=None, **kwargs):
        r"""Get the representation error

        This is (matched_representation - target_representation). If
        ``iteration`` is not None, we use
        ``self.saved_representation[iteration]`` for
        matched_representation..

        Since a single MADCompetition instance can be used for
        synthesizing multiple targets and has two models with different
        errors, you can specify the target and the model as well. If
        both are None, we use the current target of the synthesis. If
        synthesis_target is not None, but model is, we use the model
        that's the main target (e.g., if
        ``synthesis_target=='model_1_min'``, the we'd use `'model_1'`)

        Regardless, we always reset the target state to what it was
        before this was called

        Any kwargs are passed through to self.analyze when computing the
        matched/target representation.

        Parameters
        ----------
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).
        model : {None, 'model_1', 'model_2'}, optional
            which model's representation to get the error for. If None
            and ``synthesis_targe`` is not None, we use the model that's
            the main target for synthesis_target (so if
            synthesis_target=='model_1_min', then we'd use
            'model_1'). If both are None, we use the current target
        iteration: int or None, optional
            Which iteration to create the representation ratio for. If
            None, we use the current ``matched_representation``

        Returns
        -------
        torch.Tensor

        """
        # if both are None, then we don't update the target at all
        last_state = None
        if synthesis_target is not None or model is not None:
            if synthesis_target is None:
                synthesis_target = self.synthesis_target
            if model not in ['model_1', 'model_2', None]:
                raise Exception(f"Can't handle model {model}, must be one of 'model_1', 'model_2', or "
                                "None")
            if model is None:
                step = 'main'
            elif model.split('_') == synthesis_target.split('_')[:-1]:
                step = 'main'
            else:
                step = 'fix'
            last_state = self._last_update_target_args
            self.update_target(synthesis_target, step)
        if iteration is not None:
            matched_rep = self.saved_representation[synthesis_target][iteration].to(self.target_representation.device)
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
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return rep_error
