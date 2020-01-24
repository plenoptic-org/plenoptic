import torch
import warnings
from tqdm import tqdm
import numpy as np
from .Synthesis import Synthesis


class MADCompetition(Synthesis):
    """Generate maximally-differentiating images for two models

    In MAD Competition, we start with a reference image and generate two
    pairs of images. We proceed as follows:

    - Add Gaussian white noise to the reference image in order to
      perturb it. This gives us the "initial image"
    - do stuff

    And so we end up with two pairs of images, one of which contains the
    images which produce the largest and smallest responses in model 1
    while keeping model 2's response as close to constant as possible,
    while the other pair of images does the inverse (differentiates
    model 2's responses as much as possible while keeping model 1's
    response as close to constant as possible).

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
    model_1, model_2 : `torch.nn.Module`
        The two models to compare.
    reference_image : `array_like`
        The 2d grayscale image to generate the maximally-differentiating
        stimuli from.

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

        self.names = {'target_representation': 'target_representation_1',
                      'target_image': 'target_image',
                      'matched_representation': 'matched_representation_1',
                      'matched_image': 'matched_image',
                      'model': 'model_1',
                      'loss_norm': 'loss_norm'}

        self.model_1 = model_1
        self.model_2 = model_2
        if not isinstance(target_image, torch.Tensor):
            target_image = torch.tensor(target_image, torch.float32)
        self.target_image = target_image
        self.reference_image = target_image
        self.target_representation_1 = self.analyze(self.target_image)
        self.names['model'] = 'model_2'
        self.target_representation_2 = self.analyze(self.target_image)
        self.matched_image = None
        self.matched_representation_1 = None
        self.matched_representation_2 = None
        self.names['model'] = 'model_1'
        self.names['step'] = 'main'

        self.saved_representation_1 = dict((k, []) for k in ['model_1_min', 'model_1_max',
                                                             'model_2_min', 'model_2_max'])
        self.saved_representation_2 = dict((k, []) for k in ['model_1_min', 'model_1_max',
                                                             'model_2_min', 'model_2_max'])
        self.saved_image = dict((k, []) for k in ['model_1_min', 'model_1_max', 'model_2_min',
                                                  'model_2_max'])
        self.saved_image_gradient = dict((k, []) for k in ['model_1_min', 'model_1_max',
                                                           'model_2_min', 'model_2_max'])
        self.saved_representation_1_gradient = dict((k, []) for k in ['model_1_min', 'model_1_max',
                                                                      'model_2_min', 'model_2_max'])
        self.saved_representation_2_gradient = dict((k, []) for k in ['model_1_min', 'model_1_max',
                                                                      'model_2_min', 'model_2_max'])
        self.loss = dict((k, []) for k in ['model_1_min', 'model_1_max', 'model_2_min',
                                           'model_2_max'])
        self.loss_maintain = dict((k, []) for k in ['model_1_min', 'model_1_max', 'model_2_min',
                                                    'model_2_max'])
        self.gradient = dict((k, []) for k in ['model_1_min', 'model_1_max', 'model_2_min',
                                               'model_2_max'])
        self.learning_rate = dict((k, []) for k in ['model_1_min', 'model_1_max', 'model_2_min',
                                                    'model_2_max'])
        self.nu = dict((k, []) for k in ['model_1_min', 'model_1_max', 'model_2_min',
                                         'model_2_max'])

        self.coarse_to_fine = False
        self.scales = []
        self.scales_loss = []
        self.loss_change_thresh = 1e-2
        self.loss_change_iter = 50
        self.loss_change_fraction = 1.
        self.fraction_removed = 0

    def update_target(self, model):
        """Update attributes to target for synthesis

        We use this function to switch back and forth between whether
        we're updating the attributes based on minimizing model_1's loss
        or model_2's loss

        if ``model=='model_1'``
        - model: 'model_1'
        - target_representation: 'target_representation_1'
        - matched_representation: 'matched_representation_1'
        - loss_norm: 'loss_norm'

        elif ``model=='model_2'``
        - model: 'model_2'
        - target_representation: 'target_representation_2'
        - matched_representation: 'matched_representation_2'
        - loss_norm: 'loss_maintain_norm'

        Parameters
        ----------
        model : {'model_1', 'model_2'}
            the model to target, see above for details

        """
        if model == 'model_1':
            self.names.update({'model': 'model_1',
                               'target_representation': 'target_representation_1',
                               'matched_representation': 'matched_representation_1',
                               'loss_norm': 'loss_norm'})
        elif model == 'model_2':
            self.names.update({'model': 'model_2',
                               'target_representation': 'target_representation_2',
                               'matched_representation': 'matched_representation_2',
                               'loss_norm': 'loss_maintain_norm'})
        else:
            raise Exception(f"Don't know what to do with model {model}!")

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
        target_loss = self.objective_function(self.initial_representation_2,
                                              self.target_representation_2)
        for i in range(n_iter):
            proposed_img = self.matched_image - lr * nu * grad
            proposed_loss = self.objective_function(self.analyze(proposed_img),
                                                    self.target_representation_2)
            actual_loss = torch.abs(target_loss - proposed_loss)
            actual_loss.backward()
            nu_optim.step()
            nu_scheduler.step(actual_loss.item())

        # this is a manual line search, which is less efficient than the above
        # with torch.no_grad():
        #     nu = 0
        #     target_loss = self.objective_function(self.initial_representation_2),
        #                                           self.target_representation_2)
        #     for i in [1000, 100, 10, 1, .1]:
        #         nus = np.linspace(nu-i, nu+i, 21)
        #         losses = []
        #         for nu in nus:
        #             tmp_rep = self.analyze(self.matched_image - lr * nu * grad)
        #             proposed_loss = self.objective_function(tmp_rep,
        #                                                     self.target_representation_2)
        #             losses.append(abs(proposed_loss - target_loss))
        #         nu = nus[np.argmin(losses)]
        return nu

    def _closure(self):
        """An abstraction of the gradient calculation, before the optimization step

        This is a bit of trickery intended to handle the fact that on
        each iteration of synthesis, we update the image twice: once to
        do our best to minimize model_1's loss, and once to correct the
        image so that model_2's loss hasn't changed by much. We do this
        by checking ``self.names['step']``: if ``'main'``, we minimize
        model_1's loss; if ``'fix'``, we correct for model_2's loss.
        
        (see [1]_ Appendix C for more details)

        """
        # the main step corresponds to equation C3 in the paper
        if self.names['step'] == "main":
            # grab model_2's gradient
            self.update_target('model_2')
            loss_2 = super()._closure()
            grad_2 = self.matched_image.grad.clone()
            # grab model_1's gradient
            self.update_target('model_1')
            loss_1 = super()._closure()
            grad_1 = self.matched_image.grad.clone()
            # we do this reshaping to make these vectors so that this matmul
            # ends up being a dot product, and thus we get a scalar output
            proj_grad = torch.matmul(grad_1.flatten().unsqueeze(0), grad_2.flatten().unsqueeze(1))
            grad_2_norm = torch.matmul(grad_2.flatten().unsqueeze(0), grad_2.flatten().unsqueeze(1))
            # project out model_2's gradient from model_1's gradient
            self.matched_image.grad = grad_1 - (proj_grad / grad_2_norm) * grad_2
            # return model_1's loss
            return loss_1
        # the fix step corresponds to equation C5 in the paper
        elif self.names['step'] == 'fix':
            # grab model_2's gradient
            self.update_target('model_2')
            loss = super()._closure()
            grad = self.matched_image.grad.clone()
            # find the best nu
            nu = self._find_nu(grad)
            self.nu[self.synthesis_target].append(nu)
            # update the gradient
            self.matched_image.grad = nu * grad
            self.update_target('model_1')
            return loss

    def objective_function(self, x, y):
        r"""Calculate the loss between x and y

        This is what we minimize. Currently it's the L2-norm of their
        difference: ``torch.norm(x-y, p=2)``.

        If ``self.names`` has a ``'loss_norm'`` key and ``self``
        contains that attribute, we will normalize loss by that
        value. We use this to get the losses of our two models at the
        same magnitude (normalizing by their initial loss)

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
        loss = torch.norm(x - y, p=2)
        try:
            loss = loss / getattr(self, self.names['loss_norm'])
        except (AttributeError, KeyError):
            pass
        return loss

    def synthesize(self, synthesis_target, seed=0, initial_noise=.1, max_iter=100, learning_rate=1,
                   optimizer='Adam', clamper=None, store_progress=False, save_progress=False,
                   save_path='mad.pt', **optimizer_kwargs):
        r"""Synthesize two pairs of maximally-differentiation images

        Currently, only ``synthesis_target=='model_1_min'`` is supported.

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
        optimizer_kwargs :
            Dictionary of keyword arguments to pass to the optimizer (in
            addition to learning_rate). What these should be depend on
            the specific optimizer you're using

        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

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
        self.update_target('model_1')
        self.matched_representation_1 = self.analyze(self.matched_image)
        self.initial_representation_1 = self.analyze(self.initial_image)
        self.loss_norm = self.objective_function(self.target_representation_1,
                                                 self.initial_representation_1, False)
        self.update_target('model_2')
        self.target_representation_2 = self.analyze(self.target_image)
        self.matched_representation_2 = self.analyze(self.matched_image)
        self.initial_representation_2 = self.analyze(self.initial_image)
        self.loss_maintain_norm = self.objective_function(self.target_representation_2,
                                                          self.initial_representation_2, False)

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
            self.update_target('model_1')
            self.saved_representation_1[synthesis_target].append(self.analyze(self.matched_image).to('cpu'))
            self.update_target('model_2')
            self.saved_representation_2[synthesis_target].append(self.analyze(self.matched_image).to('cpu'))
        else:
            if save_progress:
                raise Exception("Can't save progress if we're not storing it! If save_progress is"
                                " True, store_progress must be not False")
        self.store_progress = store_progress

        main_pbar = tqdm(range(max_iter), position=0, leave=True)

        for i in main_pbar:
            loss_2 = self.objective_function(self.matched_representation_2,
                                             self.target_representation_2).item()
            self.names['step'] = 'main'
            loss, g, lr = self._optimizer_step(main_pbar, model_2_loss="%.4e" % loss_2)
            self.names['step'] = 'fix'
            self._optimizer_step()
            self.loss[synthesis_target].append(loss.item())
            self.gradient[synthesis_target].append(g.item())
            self.learning_rate[synthesis_target].append(lr)
            self.loss_maintain[synthesis_target].append(loss_2)

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
                    self.update_target('model_1')
                    self.saved_representation_1[synthesis_target].append(self.analyze(self.matched_image).to('cpu'))
                    self.saved_representation_1_gradient[synthesis_target].append(self.matched_representation_1.grad.clone().to('cpu'))
                    self.update_target('model_2')
                    self.saved_representation_2[synthesis_target].append(self.analyze(self.matched_image).to('cpu'))
                    self.saved_representation_2_gradient[synthesis_target].append(self.matched_representation_2.grad.clone().to('cpu'))
                    if save_progress is True:
                        self.save(save_path, True)
                if type(save_progress) == int and ((i+1) % save_progress == 0):
                    self.save(save_path, True)

        main_pbar.close()

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
        return self.matched_image.data, self.matched_representation_1.data, self.matched_representation_2.data

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
