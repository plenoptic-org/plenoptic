import torch
import warnings
from tqdm import tqdm
import pyrtools as pt
from .Synthesis import Synthesis
import matplotlib.pyplot as plt
from ..tools.signal import add_noise
from ..tools.display import plot_representation, clean_up_axes
from ..simulate.models.naive import Identity
from ..tools.metamer_utils import RangeClamper
from ..tools.optim import l2_norm


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
    synthesized images per base image. We modularize this so that a
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
    sets. For example, we have a ``synthesized_signal_all`` attribute, which
    is a dictionary with four keys (the four possible values of
    ``synthesis_target``) that each contain the relevant synthesized
    image.

    There are two types of objects you can pass as your models:
    torch.nn.Module or functions.

    1. Module: in this case, you're passing a visual *model*, which
       takes an image (as a 4d tensor) and returns some representation
       (as a 3d or 4d tensor). The model must have a forward() method
       that we can differentiate through (so, it should use pytorch
       methods, rather than numpy or scipy, unless you manually define
       the gradients in the backward() method). The distance we use is
       the L2-norm of the difference between the model's representation
       of two images (by default, to change, set ``loss_function`` to
       some other callable).

    2. Function: in this case, you're passing a visual *metric*, a
       function which takes two images (as 4d tensors) and returns a
       distance between them (as a single-valued tensor), which is what
       we use as the distance for optimization purposes. This is
       slightly more general than the above, as you can do arbitrary
       calculations on the images, but you'll lose some of the power of
       the helper functions. For example, the plot of the representation
       and representation error will just be the pixel values and
       pixel-wise difference, respectively. This is because we construct
       a "dummy model" that just returns a duplicate of the image and
       use that throughout this class. You may have additional arguments
       you want to pass to your function, in which case you can pass a
       dictionary as ``model_1_kwargs`` (or ``model_2_kwargs``) during
       initialization. These will be passed during every call.

    Parameters
    ----------
    base_signal : torch.tensor or array_like
        A 4d tensor, this is the image whose representation we wish to
        match. If this is not a tensor, we try to cast it as one.
    model_1, model_2 : torch.nn.Module or function
        The two models to compare. See above for the two allowed types
        (Modules and functions)
    loss_function : callable or None, optional
        the loss function to use to compare the representations of the
        models in order to determine their loss. Only used for the
        Module models, ignored otherwise. If None, we use the defualt:
        the element-wise 2-norm. If a callable, must take four keyword
        arguments (synth_rep, base_rep, synth_img, base_img) and
        return some loss between them. Should probably be symmetric but
        that might not be strictly necessary
    model_1_kwargs, model_2_kwargs : dict
        if model_1 or model_2 are functions (that is, you're using a
        metric instead of a model), then there might be additional
        arguments you want to pass it at run-time. Those should be
        included in a dictionary as ``key: value`` pairs. Note that this
        means they will be passed on every call.

    Attributes
    ----------
    base_signal : torch.tensor
        A 2d tensor, this is the image whose representation we wish to
        match.
    model_1, model_2 : torch.nn.Module
        Two differentiable model that takes an image as an input and
        transforms it into a representation of some sort. We only
        require that they have a forward method, which returns the
        representation to match.
    base_representation_1, base_representation_2 : torch.tensor
        Whatever is returned by ``model_1`` and ``model_2``
        ``forward(base_signal)`` methods, respectively. This is the
        representation we're trying to get as close or far away from as
        possible when targeting a given model.
    initial_image : torch.tensor
        base_signal with white noise added to it (and clamped, if
        applicable), this is the starting point of our synthesis
    initial_image_all : dict
        Dictionray containing ``initial_image `` for each
        ``synthesis_target`` (if run individually, they can have
        different noise seeds)
    synthesized_signal : torch.tensor
        The synthesized image from the last call to
        ``synthesis()``. This may be unfinished depending on how many
        iterations we've run for.
    synthesized_signal_all : dict
        Dictionary containing ``synthesized_signal`` for each
        ``synthesis_target``
    synthesized_represetation_1, synthesized_representation_2: torch.tensor
        Whatever is returned by ``model_1`` and ``model_2``
        ``forward(synthesized_signal)``, respectively.
    synthesized_representation_1_all, synthesized_representation_2_all : dict
        Dictionary containing ``synthesized_representation_1`` and
        ``synthesized_representation_2``, respectively for each
        ``synthesis_target``
    loss_1, loss_2 : list
        list of the loss with respect to model_1, model_2 over
        iterations.
    loss_1_all, loss_2_all : dict
        Dictionary containing ``loss_1`` and ``loss_2``, respectively
        for each ``synthesis_target``
    gradient : list
        list containing the gradient over iterations
    gradient_all : dict
        Dictionary containing ``gradient`` for each ``synthesis_target``
    learning_rate : list
        list containing the learning_rate over iterations. We use a
        scheduler that gradually reduces this over time, so it won't be
        constant.
    learning_rate_all : dict
        dictionary containing ``learning_rate`` for each
        ``synthesis_target``.  scheduler that gradually reduces this
        over time, so it won't be constant.
    nu : list
        list containing the nu parameter over iterations. Nu is the
        parameter used to correct the image so that the other model's
        representation will not change; see docstring of
        ``self._find_nu()`` for more details
    nu_all : dict
        Dictionary of ``nu`` for each ``synthesis_target``.
    saved_representation_1, saved_representation_2 : dict
        If the ``store_progress`` arg in ``synthesize`` is set to True
        or an int>0, we will save ``self.synthesized_representation`` at
        each iteration (or each ``store_progress`` iteration, if it's an
        int), for later examination (separately for each model and each
        ``synthesis_target``).
    saved_signal : dict
        If the ``store_progress`` arg in ``synthesize`` is set to True
        or an int>0, we will save ``self.synthesized_signal`` at each
        iteration (or each ``store_progress`` iteration, if it's an
        int), for later examination (separately for each
        ``synthesis_target``).
    seed : int
        Number with which to seed pytorch and numy's random number
        generators
    saved_signal_gradient : dict
        If the ``store_progress`` arg in ``synthesize`` is set to True
        or an int>0, we will save ``self.synthesized_signal.grad`` at each
        iteration (or each ``store_progress`` iteration, if it's an
        int), for later examination (separately for each
        ``synthesis_target``).
    saved_representation_1_gradient, saved_representation_2_gradient : dict
        If the ``store_progress`` arg in ``synthesize`` is set to
        True or an int>0, we will save
        ``self.synthesized_representation.grad`` at each iteration (or each
        ``store_progress`` iteration, if it's an int), for later
        examination (separately for each model and
        ``synthesis_target``).
    scales_loss : list
        If ``coarse_to_fine`` is not False, this contains the
        scale-specific loss at each iteration (that is, the loss
        computed on just the scale(s) we're optimizing on that
        iteration; which we use to determine when to switch scales). If
        ``coarse_to_fine=='together'``, then this will not include the
        coarsest scale, since that scale is equivalent to 'all'.If
        ``coarse_to_fine`` is False, this will be empty
    scales : list or None
        If ``coarse_to_fine`` is not False, this is a list of the scales
        in optimization order (i.e., from coarse to fine). The last
        entry will be 'all' (since after we've optimized each individual
        scale, we move on to optimizing all at once) This will be
        modified by the synthesize() method and is used to track which
        scale we're currently optimizing (the first one). When we've
        gone through all the scales present, this will just contain a
        single value: 'all'. If ``coarse_to_fine=='together'``, then
        this will never include the coarsest scale, since that scale is
        equivalent to 'all'. If ``coarse_to_fine`` is False, this will
        be None.
    scales_timing : dict or None
        If ``coarse_to_fine`` is not False, this is a dictionary whose
        keys are the values of scales. The values are lists, with 0
        through 2 entries: the first entry is the iteration where we
        started optimizing this scale, the second is when we stopped
        (thus if it's an empty list, we haven't started optimzing it
        yet). If ``coarse_to_fine=='together'``, then this will not
        include the coarsest scale, since that scale is equivalent to
        'all'. If ``coarse_to_fine`` is False, this will be None.
    scales_finished : list or None
        If ``coarse_to_fine`` is not False, this is a list of the scales
        that we've finished optimizing (in the order we've finished).
        If ``coarse_to_fine=='together'``, then this will never include
        the coarsest scale, since that scale is equivalent to 'all'. If
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

    def __init__(self, base_signal, model_1, model_2, loss_function=None, model_1_kwargs={},
                 model_2_kwargs={}, loss_function_kwargs={}):
        self._names = {'base_signal': 'base_signal',
                       'synthesized_signal': 'synthesized_signal',
                       'model': 'model_1',
                       'base_representation': 'base_representation_1',
                       'synthesized_representation': 'synthesized_representation_1',
                       'initial_representation': 'initial_representation_1',
                       'loss_norm': 'loss_1_norm',
                       'loss': 'loss_1',
                       'saved_representation': 'saved_representation_1',
                       'saved_representation_gradient': 'saved_representation_1_gradient',
                       'loss_function': 'loss_function_1',
                       'coarse_to_fine': 'coarse_to_fine_1'}

        self.synthesis_target = 'model_1_min'
        super().__init__(base_signal, model_1, loss_function, model_1_kwargs, loss_function_kwargs)

        # initialize the MAD-specific attributes
        self._loss_sign = 1
        self._step = 'main'
        self.nu = []
        self.initial_image = None

        # we initialize all the model 1 versions of these in the
        # super().__init__() call above, so we just need to do the model
        # 2 ones
        if loss_function is None:
            loss_function = l2_norm
        else:
            if not isinstance(model_2, torch.nn.Module):
                warnings.warn("Ignoring custom loss_function for model_2 since it's a metric")

        if isinstance(model_2, torch.nn.Module):
            self.model_2 = model_2

            def wrapped_loss_func(synth_rep, ref_rep, synth_img, ref_img):
                return loss_function(ref_rep=ref_rep, synth_rep=synth_rep, ref_img=ref_img,
                                     synth_img=synth_img, **loss_function_kwargs)
            self.loss_function_2 = wrapped_loss_func
        else:
            self.model_2 = Identity(model_2.__name__).to(base_signal.device)

            def wrapped_model_2(synth_rep, ref_rep, synth_img, ref_img):
                return model_2(synth_rep, ref_rep, **model_2_kwargs)
            self.loss_function_2 = wrapped_model_2
            self._rep_warning = True

        self.update_target('model_1_min', 'fix')
        self.base_representation_2 = self.analyze(self.base_signal)
        self.synthesized_representation_2 = None
        self.loss_2 = []
        self.saved_representation_2 = []
        self.saved_representation_2_gradient = []
        self.coarse_to_fine_2 = False
        self.update_target('model_1_min', 'main')

        # these are the attributes that have 'all' versions of them, and
        # they'll all need to be initialized with a dictionary for each
        # possible target
        self._attrs_all = ['saved_representation_1', 'saved_representation_2', 'saved_signal',
                           'saved_representation_1_gradient', 'saved_representation_2_gradient',
                           'saved_signal_gradient', 'loss_1', 'loss_2', 'gradient', 'learning_rate',
                           'nu', 'initial_image', 'synthesized_signal', 'initial_representation_1',
                           'initial_representation_2', 'synthesized_representation_1',
                           'synthesized_representation_2']

        def _init_dict(none_flag=False):
            if none_flag:
                val = None
            else:
                val = []
            return dict((k, val) for k in ['model_1_min', 'model_1_max', 'model_2_min',
                                           'model_2_max'])
        for attr in self._attrs_all:
            if attr == 'synthesized_signal':
                # synthesized_signal is a parameter and so has to be initialized with None
                setattr(self, attr+'_all', _init_dict(True))
            else:
                setattr(self, attr+'_all', _init_dict())

    def __getattr__(self, name):
        """get an attribute

        this is the standard __getattr__, except we override it for the
        attributes that have two versions, depending on which model
        we're currently targeting: 'base_representation',
        'base_signal', 'synthesized_representation', 'loss',
        'synthesized_signal', 'model', 'loss_norm', 'initial_representation',
        'saved_representation', 'saved_representation_gradient',
        'loss_function'

        """
        # we don't do this for '_names' because if we did we'd run into
        # some infinite recursion nonsense
        if name != '_names':
            # this returns self._names[name] if name is in that dictionary
            # and doesn't change it if not
            name = self._names.get(name, name)
        try:
            return self.__dict__[name]
        except KeyError:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        """set attributes

        this is the standard __setattr__, except we override it for the
        attributes that have two versions, depending on which model
        we're currently targeting: 'base_representation',
        'base_signal', 'synthesized_representation', 'loss',
        'synthesized_signal', 'model', 'loss_norm', 'initial_representation',
        'saved_representation', 'saved_representation_gradient',
        'loss_function'

        """
        # we don't do this for '_names' because if we did we'd run into
        # some infinite recursion nonsense
        if name != '_names':
            # this returns self._names[name] if name is in that dictionary
            # and doesn't change it if not
            name = self._names.get(name, name)
        super().__setattr__(name, value)

    def _get_model_name(self, model):
        """get the name of one of the models

        We first check whether model has a ``name`` attribute and, if
        not, grab the name of the model's class

        Parameters
        ----------
        model : {'model_1', 'model_2'}
            which model's name to get

        Returns
        -------
        model_name : str
            the name of ``model``

        """
        try:
            model_name = getattr(self, model).name
        except AttributeError:
            model_name = getattr(self, model).__class__.__name__
        return model_name

    def update_target(self, synthesis_target, step):
        """Update attributes to target for synthesis

        We use this function to switch back and forth between whether
        we're updating the attributes based on minimizing or maximizing
        model_1's loss or model_2's loss

        Note that if you're switching synthesis_target, you should NOT
        set self.synthesis_target yourself, you should call this
        function with the new synthesis_target and it will do it for you
        (we rely on checking whether self.synthesis_target matches
        synthesis_target arg to correctly update the attrs that have a
        _all version)

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
            self._loss_sign = {'min': 1, 'max': -1}[synth_direction]
        else:
            num = other_model_num
            self._loss_sign = 1
        self._names.update({'model': f'model_{num}',
                            'base_representation': f'base_representation_{num}',
                            'synthesized_representation': f'synthesized_representation_{num}',
                            'initial_representation': f'initial_representation_{num}',
                            'loss_norm': f'loss_{num}_norm',
                            'loss': f'loss_{num}',
                            'saved_representation': f'saved_representation_{num}',
                            'saved_representation_gradient': f'saved_representation_{num}_gradient',
                            'loss_function': f'loss_function_{num}',
                            'coarse_to_fine': f'coarse_to_fine_{num}'})
        if synthesis_target != self.synthesis_target:
            self.synthesis_target = synthesis_target
            for attr in self._attrs_all:
                if attr == 'synthesized_signal':
                    # synthesized_signal needs to be a parameter
                    setattr(self, attr, torch.nn.Parameter(getattr(self, attr+'_all')[synthesis_target]))
                else:
                    try:
                        setattr(self, attr, getattr(self, attr+'_all')[synthesis_target].clone().to('cpu'))
                    except AttributeError:
                        # then this isn't a tensor, it's a list
                        setattr(self, attr, getattr(self, attr+'_all')[synthesis_target].copy())

    def _update_attrs_all(self):
        """copy the data from attributes into their _all version

        in a given call to synthesis, we only update the 'local' version
        of an attribute (e.g., synthesized_signal), which contains the data
        relevant to the synthesis we're currently doing (e.g.,
        minimizing model 1's loss). however, we want to store all these
        attributes across each of the four types of runs, for which we
        use the '_all' versions of the attributes (e.g.,
        mtached_image_all). this copies the information from the local
        into the global for the future (the inverse of this, copying
        from the global into the local, happens in ``update_target``)

        """
        for attr in self._attrs_all:
            attr_all = getattr(self, attr+'_all')
            try:
                attr_all[self.synthesis_target] = getattr(self, attr).clone().to('cpu')
            except AttributeError:
                # then this isn't a tensor, it's a list
                attr_all[self.synthesis_target] = getattr(self, attr).copy()

    def _find_nu(self, grad, n_iter=10):
        """find the optimum nu to remain on model_2's level set

        While we're minimizing model_1's loss, we do our best to stay on
        model_2's level set. Projecting out model_2's gradient from
        model_1's gradient (which we do in ``_closure()``) helps with
        this, but it's not perfect.

        Call ``grad`` :math:`G`, ``base_signal`` :math:`X` and the
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
            the gradient of ``self.synthesized_signal`` for model 2 with
            respect to the loss between ``self.synthesized_signal`` and
            ``self.base_signal``
        n_iter : int
            The number of iterations to use when finding the best
            nu. Obviously, the larger this number, the longer it will
            take

        Returns
        -------
        nu : torch.Parameter
            The optimized (scalar) nu value

        """
        lr = self._optimizer.param_groups[0]['lr']
        nu = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        nu_optim = torch.optim.Adam([nu], lr=1, amsgrad=True)
        nu_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(nu_optim, 'min', factor=.5)
        base_loss = self.objective_function(self.initial_representation,
                                              self.base_representation, self.initial_image,
                                              self.base_signal)
        for i in range(n_iter):
            # in Appendix C of the paper, they just add nu*grad to get
            # the proposed image. here we also multiply by a -lr because
            # of how pytorch updates images. see torch/optim/sgd.py, at
            # the very end of the step() function and you'll see that
            # when updating parameters, they add -lr * grad. thus, this
            # is what we need in order to predict what the updated image
            # will be
            proposed_img = self.synthesized_signal - lr * nu * grad
            proposed_loss = self.objective_function(self.analyze(proposed_img),
                                                    self.base_representation,
                                                    proposed_img, self.base_signal)
            actual_loss = torch.abs(base_loss - proposed_loss)
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
        by much. We do this by checking ``self._step``: if ``'main'``, we
        minimize/maximize the first model's loss; if ``'fix'``, we
        correct for the second model's loss. (note that which model in
        the desription above corresponds to the attributes
        ``self.model_1`` and ``self.model_2`` is handled by the
        ``update_target()`` method and the getter/setter)

        (see [1]_ Appendix C for more details)

        """
        # the main step corresponds to equation C3 in the paper
        if self._step == "main":
            # grab model_stable's gradient
            self.update_target(self.synthesis_target, 'fix')
            loss_stable = super()._closure()
            grad_stable = self.synthesized_signal.grad.clone()
            # grab model_target's gradient
            self.update_target(self.synthesis_target, self._step)
            loss_target = super()._closure()
            grad_target = self.synthesized_signal.grad.clone()
            # we do this reshaping to make these vectors so that this matmul
            # ends up being a dot product, and thus we get a scalar output
            proj_grad = torch.matmul(grad_target.flatten().unsqueeze(0),
                                     grad_stable.flatten().unsqueeze(1))
            grad_stable_norm = torch.matmul(grad_stable.flatten().unsqueeze(0),
                                            grad_stable.flatten().unsqueeze(1))
            # project out model_stable's gradient from model_target's gradient
            self.synthesized_signal.grad = grad_target - (proj_grad / grad_stable_norm) * grad_stable
            # return model_target's loss
            return loss_target
        # the fix step corresponds to equation C5 in the paper
        elif self._step == 'fix':
            # grab model_stable's gradient
            self.update_target(self.synthesis_target, self._step)
            loss = super()._closure()
            grad = self.synthesized_signal.grad.clone()
            # find the best nu
            nu = self._find_nu(grad, self.fix_step_n_iter)
            self.nu.append(nu.clone().to('cpu'))
            # update the gradient
            self.synthesized_signal.grad = nu * grad
            self.update_target(self.synthesis_target, 'main')
            return loss

    def objective_function(self, synth_rep, ref_rep, synth_img, ref_img, norm_loss=True):
        r"""Calculate the loss

        This is what we minimize. By default it's the L2-norm of the
        difference between synth_rep and ref_rep

        We can also normalize the loss, if ``norm_loss=True`` and we
        have a ``loss_nom`` attribute. We use this to get the losses of
        our two models at the same magnitude (normalizing by their
        initial loss)

        Finally, we also multiply the loss by ``self._loss_sign``,
        because sometimes we want to minimize the loss and sometimes we
        want to maximize it; maximizing the loss is the same as
        minimizing its negative (the setting of ``self._loss_sign`` is
        handled automatically by the ``update_target()`` method)

        Parameters
        ----------
        synth_rep : torch.tensor
            model representation of the synthesized image
        ref_rep : torch.tensor
            model representation of the reference image
        synth_img : torch.tensor
            the synthesized image.
        ref_img : torch.tensor
            the reference image

        Returns
        -------
        loss : torch.tensor
            single-element tensor containing the L2-norm of the
            difference between x and y

        """
        loss = super().objective_function(synth_rep, ref_rep, synth_img, ref_img)
        if norm_loss:
            loss = loss / self.loss_norm
        return self._loss_sign * loss

    def _init_synthesized_signal(self, initial_noise=None, clamper=RangeClamper((0, 1)),
                            clamp_each_iter=True, norm_loss=True):
        """initialize the synthesized image

        set the ``self.synthesized_signal`` attribute to be a parameter with
        the user-supplied data, making sure it's the right shape and
        calling clamper on it, if set

        also initialize the ``self.synthesized_representation`` attribute

        Parameters
        ----------
        initial_noise : `float` or None, optional
            standard deviation of the Gaussian noise used to create the
            initial image from the target image. If None (the default),
            we try to grab the final value from ``self.saved_signal``
            (thus, if ``self.saved_signal`` is empty, this will raise an
            Exception)
        clamper : plenoptic.Clamper or None, optional
            Clamper makes a change to the image in order to ensure that
            it stays reasonable. The classic example (and default
            option) is making sure the range lies between 0 and 1, see
            plenoptic.RangeClamper for an example.
        clamp_each_iter : bool, optional
            If True (and ``clamper`` is not ``None``), we clamp every
            iteration. If False, we only clamp at the very end, after
            the last iteration
        norm_loss : bool, optional
            Whether to normalize the loss of each model. You probably
            want them to be normalized so that they are of the same
            magnitude and thus their gradients are also of the same
            magnitude. However, you can turn it off and see how that
            affects performance. It's also useful for debugging
            purposes.

        """
        if initial_noise is not None:
            self.initial_image = add_noise(self.base_signal, initial_noise)
            init_image = self.initial_image
        # we want to keep the initial_image attribute unchanged if
        # initial_noise is None (we still want it to be the initial
        # representation), but we want to make synthesized_signal the last
        # saved_signal
        else:
            init_image = self.saved_signal[-1]
        self.update_target(self.synthesis_target, 'main')
        super()._init_synthesized_signal(init_image.clone(), clamper, clamp_each_iter)
        if clamper is not None:
            # that initial noise can take us outside the clamper
            self.initial_image.data = clamper.clamp(self.initial_image.data)
        self.initial_representation = self.analyze(self.initial_image)
        # if synthesis target is model_1/2_max, then _loss_sign is
        # negative (for main step; because minimizing the negative of
        # the loss is the same as maximizing it). But if we include the
        # negative in both the regular calculation of the loss and the
        # norm, then we end up canceling it out. This will make sure
        # that loss_norm is always positive
        if norm_loss:
            self.loss_norm = abs(self.objective_function(self.initial_representation,
                                                         self.base_representation,
                                                         self.initial_image, self.base_signal,
                                                         norm_loss=False))
        else:
            self.loss_norm = 1
        self.update_target(self.synthesis_target, 'fix')
        self.synthesized_representation = self.analyze(self.synthesized_signal)
        self.initial_representation = self.analyze(self.initial_image)
        if norm_loss:
            self.loss_norm = self.objective_function(self.initial_representation,
                                                     self.base_representation,
                                                     self.initial_image, self.base_signal,
                                                     norm_loss=False)
        else:
            self.loss_norm = 1

    def _init_ctf_and_randomizer(self, loss_thresh=1e-4, fraction_removed=0, coarse_to_fine=False,
                                 loss_change_fraction=1, loss_change_thresh=1e-2,
                                 loss_change_iter=50):
        """initialize stuff related to randomization and coarse-to-fine

        we always make the stable model's coarse to fine False

        Parameters
        ----------
        loss_thresh : float, optional
            If the loss over the past ``loss_change_iter`` is less than
            ``loss_thresh``, we stop.
        fraction_removed: float, optional
            The fraction of the representation that will be ignored
            when computing the loss. At every step the loss is computed
            using the remaining fraction of the representation only.
            A new sample is drawn a every step. This gives a stochastic
            estimate of the gradient and might help optimization.
        coarse_to_fine : { 'together', 'separate', False}, optional
            If False, don't do coarse-to-fine optimization. Else, there
            are two options for how to do it:
            - 'together': start with the coarsest scale, then gradually
              add each finer scale. this is like blurring the objective
              function and then gradually adding details and is probably
              what you want.
            - 'separate': compute the gradient with respect to each
              scale separately (ignoring the others), then with respect
              to all of them at the end.
            (see above for more details on what's required of the model
            for this to work).
        loss_change_fraction : float, optional
            If we think the loss has stopped decreasing (based on
            ``loss_change_iter`` and ``loss_change_thresh``), the
            fraction of the representation with the highest loss that we
            use to calculate the gradients
        loss_change_thresh : float, optional
            The threshold below which we consider the loss as unchanging
            in order to determine whether we should only calculate the
            gradient with respect to the
            ``loss_change_fraction`` fraction of statistics with
            the highest error.
        loss_change_iter : int, optional
            How many iterations back to check in order to see if the
            loss has stopped decreasing in order to determine whether we
            should only calculate the gradient with respect to the
            ``loss_change_fraction`` fraction of statistics with
            the highest error.

        """
        self.update_target(self.synthesis_target, 'main')
        super()._init_ctf_and_randomizer(loss_thresh, fraction_removed, coarse_to_fine,
                                         loss_change_fraction, loss_change_thresh,
                                         loss_change_iter)
        # always want the stable model's coarse to fine to be False
        self.update_target(self.synthesis_target, 'fix')
        self.coarse_to_fine = False

    def _init_store_progress(self, store_progress, save_progress, save_path):
        """initialize store_progress-related attributes

        sets the ``self.save_progress``, ``self.store_progress``, and
        ``self.save_path`` attributes, as well as changing
        ``saved_signal, saved_representation, saved_signal_gradient,
        saved_representation_gradient`` attibutes all to lists so we can
        append to them. finally, adds first value to ``saved_signal`` and
        ``saved_representation``

        Parameters
        ----------
        store_progress : bool or int, optional
            Whether we should store the representation of the metamer
            and the metamer image in progress on every iteration. If
            False, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress``
            iterations (note then that 0 is the same as False and 1 the
            same as True). If True or int>0, ``self.saved_signal``
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

        """
        self.update_target(self.synthesis_target, 'main')
        super()._init_store_progress(store_progress, save_progress, save_path)
        self.update_target(self.synthesis_target, 'fix')
        self.saved_representation = list(self.saved_representation)
        self.saved_representation_gradient = list(self.saved_representation_gradient)
        self.saved_representation.append(self.analyze(self.synthesized_signal).to('cpu'))

    def _clamp_and_store(self, i):
        """clamp synthesized_signal and store/save, if appropriate

        these all happen together because they all happen ``with
        torch.no_grad()``

        if it's the right iteration, we update: ``saved_signal,
        saved_representation, saved_signal_gradient,
        saved_representation_gradient``

        Parameters
        ----------
        i : int
            the current iteration (0-indexed)

        """
        self.update_target(self.synthesis_target, 'main')
        if super()._clamp_and_store(i):
            self.update_target(self.synthesis_target, 'fix')
            # these are the only ones that differ between main and fix
            with torch.no_grad():
                self.saved_representation.append(self.analyze(self.synthesized_signal).to('cpu'))
                self.saved_representation_gradient.append(self.synthesized_representation.grad.clone().to('cpu'))

    def _finalize_stored_progress(self):
        """stack the saved_* attributes

        if we were storing progress, stack the ``saved_representation,
        saved_signal, saved_signal_gradient,
        saved_representation_gradient`` attributes so they're a single
        tensor

        we can't stack the gradients if we used coarse-to-fine
        optimization, because then they'll be different shapes, so we
        have to keep them as a list

        """
        self.update_target(self.synthesis_target, 'main')
        super()._finalize_stored_progress()
        self.update_target(self.synthesis_target, 'fix')
        # these are the only ones that differ between main and fix
        if self.store_progress:
            self.saved_representation = torch.stack(self.saved_representation)
            try:
                self.saved_representation_gradient = torch.stack(self.saved_representation_gradient)
            except RuntimeError:
                pass

    def synthesize(self, synthesis_target, initial_noise=.1, fix_step_n_iter=5, norm_loss=True,
                   seed=0, max_iter=100, learning_rate=1, scheduler=True, optimizer='SGD',
                   clamper=RangeClamper((0, 1)), clamp_each_iter=True, store_progress=False,
                   save_progress=False, save_path='mad.pt', loss_thresh=1e-4, loss_change_iter=50,
                   fraction_removed=0., loss_change_thresh=1e-2, loss_change_fraction=1.,
                   coarse_to_fine=False, clip_grad_norm=False, **optimizer_kwargs):
        r"""Synthesize one maximally-differentiating image

        This synthesizes a single image, minimizing or maximizing either
        model 1 or model 2 while holding the other constant. By setting
        ``synthesis_target``, you can determine which of these you wish
        to synthesize.

        We run this until either we reach ``max_iter`` or the change
        over the past ``loss_change_iter`` iterations is less than
        ``loss_thresh``, whichever comes first

        The synthesis is initialized with the ``base_signal`` plus
        Gaussian noise with mean 0 and standard deviation
        ``initial_noise``.

        If ``store_progress!=False``, you can run this several times in
        sequence by setting ``initial_noise`` to ``None``. In that case,
        the initial image of subsequent calls will be equal to the last
        value of ``self.saved_signal``. Everything that stores the
        progress of the optimization (``loss``,
        ``saved_representation``, ``saved_signal``) will persist between
        calls and so potentially get very large. To most directly resume
        where you left off, it's recommended you set
        ``learning_rate=None``, in which case we use the most recent
        learning rate (since we use a learning rate scheduler, the
        learning rate decreases over time as the gradient shrinks; note
        that we will still reset to the original value in coarse-to-fine
        optimization). Coarse-to-fine optimization will also resume
        where you left off.

        We currently do not exactly preserve the state of the RNG
        between calls (the seed will be reset), because it's difficult
        to figure out which device we should grab the RNG state for. If
        you're interested in doing this yourself, see
        https://pytorch.org/docs/stable/random.html, specifically the
        fork_rng function (I recommend looking at the source code for
        that function to see how to get and set the RNG state). This
        means that there will be a transient increase in loss right
        after resuming synthesis. In every example I've seen, it goes
        away and continues decreasing after a relatively small number of
        iterations, but it means that running synthesis for 500
        iterations is not the same as running it twice for 250
        iterations each.

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
        ``scales`` attributes (which gives the scales in coarse-to-fine
        order, i.e., the order that we will be optimizing) and its
        ``forward`` method can accept a ``scales`` keyword argument, a
        list that specifies which scales to use to compute the
        representation. If ``coarse_to_fine`` is not False, then we
        optimize each scale until we think it's reached convergence
        before moving on (either computing the gradient for each scale
        individually, if ``coarse_to_fine=='separate'`` or for a given
        scale and all coarser scales, if
        ``coarse_to_fine=='together'``). Once we've done each scale, we
        spend the rest of the iterations doing them all together, as if
        ``coarse_to_fine`` was False. This can be combined with the
        above three methods. We determine if a scale has converged in
        the same way as method 3 above: if the scale-specific loss
        ``loss_change_iter`` iterations ago is within
        ``loss_change_thresh`` of the most recent loss.

        Parameters
        ----------
        synthesis_target : {'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which image to synthesize
        initial_noise : `float` or None, optional
            standard deviation of the Gaussian noise used to create the
            initial image from the target image. Can only be None if
            ``self.saved_signal`` is not empty (i.e., this has been
            called at least once before with
            ``store_progress!=False``). In that case, the initial image
            is the last value of ``self.saved_signal``
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
        norm_loss : bool, optional
            Whether to normalize the loss of each model. You probably
            want them to be normalized so that they are of the same
            magnitude and thus their gradients are also of the same
            magnitude. However, you can turn it off and see how that
            affects performance. It's also useful for debugging
            purposes.
        seed : int or None, optional
            Number with which to seed pytorch and numy's random number
            generators. If None, won't set the seed; general use case
            for this is to avoid resetting the seed when resuming
            synthesis
        max_iter : int, optional
            The maximum number of iterations to run before we end
        learning_rate : float or None, optional
            The learning rate for our optimizer. None is only accepted
            if we're resuming synthesis, in which case we use the last
            learning rate from the previous instance.
        scheduler : bool, optional
            whether to initialize the scheduler or not. If False, the
            learning rate will never decrease. Setting this to True
            seems to improve performance, but it might be useful to turn
            it off in order to better work through what's happening
        optimizer: {'GD', 'Adam', 'SGD', 'LBFGS', 'AdamW'}
            The choice of optimization algorithm. 'GD' is regular
            gradient descent, as decribed in [1]_
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
            same as True). If True or int>0, ``self.saved_signal``
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
        loss_thresh : float, optional
            If the loss over the past ``loss_change_iter`` has changed
            less than ``loss_thresh``, we stop.
        loss_change_iter : int, optional
            How many iterations back to check in order to see if the
            loss has stopped decreasing in order to determine whether we
            should only calculate the gradient with respect to the
            ``loss_change_fraction`` fraction of statistics with
            the highest error.
        fraction_removed: float, optional
            The fraction of the representation that will be ignored
            when computing the loss. At every step the loss is computed
            using the remaining fraction of the representation only.
            A new sample is drawn a every step. This gives a stochastic
            estimate of the gradient and might help optimization.
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
        coarse_to_fine : { 'together', 'separate', False}, optional
            If False, don't do coarse-to-fine optimization. Else, there
            are two options for how to do it:
            - 'together': start with the coarsest scale, then gradually
              add each finer scale. this is like blurring the objective
              function and then gradually adding details and is probably
              what you want.
            - 'separate': compute the gradient with respect to each
              scale separately (ignoring the others), then with respect
              to all of them at the end.
            (see above for more details on what's required of the model
            for this to work).
        clip_grad_norm : bool or float, optional
            If the gradient norm gets too large, the optimization can
            run into problems with numerical overflow. In order to avoid
            that, you can clip the gradient norm to a certain maximum by
            setting this to True or a float (if you set this to False,
            we don't clip the gradient norm). If True, then we use 1,
            which seems reasonable. Otherwise, we use the value set
            here.
        optimizer_kwargs :
            Dictionary of keyword arguments to pass to the optimizer (in
            addition to learning_rate). What these should be depend on
            the specific optimizer you're using

        Returns
        -------
        synthesized_signal : torch.tensor
            The MAD competition image we've created
        synthesized_representation_1 : torch.tensor
            model_1's representation of this image
        synthesized_representation_2 : torch.tensor
            The model_2's representation of this image

        """
        self._set_seed(seed)
        self.fix_step_n_iter = fix_step_n_iter
        # self.synthesis_target gets updated in this call, DO NOT do it
        # manually
        self.update_target(synthesis_target, 'main')

        self._init_synthesized_signal(initial_noise, clamper, clamp_each_iter, norm_loss)

        self.update_target(synthesis_target, 'main')
        # initialize stuff related to coarse-to-fine and randomization
        self._init_ctf_and_randomizer(loss_thresh, fraction_removed, coarse_to_fine,
                                      loss_change_fraction, loss_change_thresh, loss_change_iter)

        # initialize the optimizer
        self._init_optimizer(optimizer, learning_rate, scheduler, clip_grad_norm,
                             **optimizer_kwargs)

        self._init_store_progress(store_progress, save_progress, save_path)

        pbar = tqdm(range(max_iter))

        for i in pbar:
            self.update_target(self.synthesis_target, 'fix')
            # first, figure out what the stable model's loss is
            loss_2 = self.objective_function(self.synthesized_representation,
                                             self.base_representation, self.synthesized_signal,
                                             self.base_signal).item()
            self.loss.append(loss_2)
            # then update synthesized_signal to try and min or max (depending
            # on synthesis_target) the targeted model
            self.update_target(self.synthesis_target, 'main')
            self._step = 'main'
            loss, g, lr = self._optimizer_step(pbar, stable_loss="%.4e" % loss_2)
            self.loss.append(abs(loss.item()))
            self.gradient.append(g.item())
            self.learning_rate.append(lr)
            # finally, update synthesized_signal to try and keep the stable
            # model's loss constant
            self.update_target(self.synthesis_target, 'fix')
            self._step = 'fix'
            self._optimizer_step()

            if self._check_nan_loss(loss):
                # synthesized_signal and the other synthesized represntation will
                # be handled in the _check_nan_loss call (because we've
                # got target 'fix')
                self.update_target(self.synthesis_target, 'main')
                self.synthesized_representation = self.saved_representation[-2]
                break

            if self._check_for_stabilization(i):
                break

            # clamp and update saved_* attrs
            self._clamp_and_store(i)

        pbar.close()

        self._finalize_stored_progress()

        self._update_attrs_all()
        return self.synthesized_signal.data, self.synthesized_representation_1.data, self.synthesized_representation_2.data

    def synthesize_all(self, if_existing='skip', initial_noise=.1, fix_step_n_iter=10,
                       norm_loss=True, seed=0, max_iter=100, learning_rate=1, scheduler=True,
                       optimizer='Adam', clamper=RangeClamper((0, 1)), clamp_each_iter=True,
                       store_progress=False, save_progress=False, save_path='mad_{}.pt',
                       loss_thresh=1e-4, loss_change_iter=50, fraction_removed=0.,
                       loss_change_thresh=1e-2, loss_change_fraction=1., coarse_to_fine=False,
                       clip_grad_norm=False, **optimizer_kwargs):
        r"""Synthesize two pairs of maximally-differentiating images

        MAD Competitoin consists of two pairs of
        maximally-differentiating images: one pair minimizes and
        maximizes model 1, while holding model 2 constant, and the other
        minimizes and maximizes model 2, while holding model 1
        constant. This creates all four images. We return nothing, but
        all the outputs are stored in attributes.

        All parameters are passed directly through to ``synthesis()`` so
        if you want to synthesize the four images with different
        arguments, you should call ``synthesis()`` directly. The
        exception is ``save_path`` -- if it contains ``'{}'``, we format
        it to include the target name.

        Parameters
        ----------
        if_existing : {'skip', 're-run', 'continue'}, optional
            what to do if synthesis for one of the targets has been run
            before.
            - ``'skip'``: skip it, doing nothing
            - ``'re-run'``: re-run from scratch, starting over -- note
              that this will not remove existing history however, so
              plots of ``self.loss`` or examinations of
              ``self.saved_signal`` may look weird
            - ``'continue'``: continue from where it left off
        initial_noise : `float`, optional
            standard deviation of the Gaussian noise used to create the
            initial image from the target image
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
        norm_loss : bool, optional
            Whether to normalize the loss of each model. You probably
            want them to be normalized so that they are of the same
            magnitude and thus their gradients are also of the same
            magnitude. However, you can turn it off and see how that
            affects performance. It's also useful for debugging
            purposes.
        seed : int or None, optional
            Number with which to seed pytorch and numy's random number
            generators. If None, won't set the seed; general use case
            for this is to avoid resetting the seed when resuming
            synthesis
        max_iter : int, optional
            The maximum number of iterations to run before we end
        learning_rate : float, optional
            The learning rate for our optimizer
        scheduler : bool, optional
            whether to initialize the scheduler or not. If False, the
            learning rate will never decrease. Setting this to True
            seems to improve performance, but it might be useful to turn
            it off in order to better work through what's happening
        optimizer: {'GD', 'Adam', 'SGD', 'LBFGS'}
            The choice of optimization algorithm. 'GD' is regular
            gradient descent, as decribed in [1]_
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
            same as True). If True or int>0, ``self.saved_signal``
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
        loss_thresh : float, optional
            If the loss over the past ``loss_change_iter`` has changed
            less than ``loss_thresh``, we stop.
        loss_change_iter : int, optional
            How many iterations back to check in order to see if the
            loss has stopped decreasing in order to determine whether we
            should only calculate the gradient with respect to the
            ``loss_change_fraction`` fraction of statistics with
            the highest error.
        fraction_removed: float, optional
            The fraction of the representation that will be ignored
            when computing the loss. At every step the loss is computed
            using the remaining fraction of the representation only.
            A new sample is drawn a every step. This gives a stochastic
            estimate of the gradient and might help optimization.
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
        coarse_to_fine : { 'together', 'separate', False}, optional
            If False, don't do coarse-to-fine optimization. Else, there
            are two options for how to do it:
            - 'together': start with the coarsest scale, then gradually
              add each finer scale. this is like blurring the objective
              function and then gradually adding details and is probably
              what you want.
            - 'separate': compute the gradient with respect to each
              scale separately (ignoring the others), then with respect
              to all of them at the end.
            (see above for more details on what's required of the model
            for this to work).
        clip_grad_norm : bool or float, optional
            If the gradient norm gets too large, the optimization can
            run into problems with numerical overflow. In order to avoid
            that, you can clip the gradient norm to a certain maximum by
            setting this to True or a float (if you set this to False,
            we don't clip the gradient norm). If True, then we use 1,
            which seems reasonable. Otherwise, we use the value set
            here.
        optimizer_kwargs :
            Dictionary of keyword arguments to pass to the optimizer (in
            addition to learning_rate). What these should be depend on
            the specific optimizer you're using

        """
        initial_noise_orig = initial_noise
        learning_rate_orig = learning_rate
        for target in ['model_1_min', 'model_1_max', 'model_2_min', 'model_2_max']:
            initial_noise = initial_noise_orig
            learning_rate = learning_rate_orig
            run = True
            s = f"Synthesizing {target}"
            if self.synthesized_signal_all[target] is not None:
                s = f"Synthesis with target {target} has been run before, "
                if if_existing == 'skip':
                    run = False
                    s += "skipping"
                elif if_existing == 're-run':
                    s += 're-running from scratch'
                elif if_existing == 'continue':
                    s += 'continuing from where it left off'
                    initial_noise = None
                    learning_rate = None
                else:
                    raise Exception(f"Don't know how to handle if_existing option {if_existing}!")
            if save_path is not None and save_progress is not False:
                if '}' in save_path:
                    save_path_tmp = save_path.format(target)
                else:
                    save_path_tmp = save_path
                s += f", saving at {save_path_tmp}"
            print(s)
            if run:
                self.synthesize(target, initial_noise, fix_step_n_iter, norm_loss, seed, max_iter,
                                learning_rate, scheduler, optimizer, clamper, clamp_each_iter,
                                store_progress, save_progress, save_path, loss_thresh,
                                loss_change_iter, fraction_removed, loss_change_thresh,
                                loss_change_fraction, coarse_to_fine, clip_grad_norm,
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
        # the first two lines here make sure that we have both the _1 and _2
        # versions, regardless of which is currently found in .values()
        attrs = ([k.replace('_1', '_2') for k in self._names.values()] +
                 [k.replace('_2', '_1') for k in self._names.values()] +
                 [k + '_all' for k in self._attrs_all])
        # Removes duplicates
        attrs = list(set(attrs))
        # add the attributes not included above
        attrs += ['seed', 'scales', 'scales_timing', 'scales_loss', 'scales_finished',
                  'store_progress', 'save_progress', 'save_path', 'synthesis_target']
        super().save(file_path, save_model_reduced, attrs, ['model_1', 'model_2'])

    @classmethod
    def load(cls, file_path, model_constructor=[None, None], map_location='cpu',
             **state_dict_kwargs):
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
        model_constructor : list, optional
            When saving the synthesis object, we have the option to only
            save the ``state_dict_reduced`` (in order to save space). If
            we do that, then we need some way to construct that model
            again and, not knowing its class or anything, this object
            doesn't know how. Therefore, a user must pass a constructor
            for the model that takes in the ``state_dict_reduced``
            dictionary and returns the initialized model. See the
            VentralModel class for an example of this. Since
            MADCompetition has two models, this must be a list with two
            elements, the first corresponding to model_1, the second to
            model_2
        map_location : str, optional
            map_location argument to pass to ``torch.load``. If you save
            stuff that was being run on a GPU and are loading onto a
            CPU, you'll need this to make sure everything lines up
            properly. This should be structured like the str you would
            pass to ``torch.device``
        state_dict_kwargs :
            any additional kwargs will be added to the model's
            state_dict before construction (this only applies if the
            model is a dict, see above for more description of that)

        Returns
        -------
        mad : plenoptic.synth.MADCompetition
            The loaded MADCompetition object


        Examples
        --------
        >>> mad = po.synth.MADCompetition(img, model1, model2)
        >>> mad.synthesize(max_iter=10, store_progress=True)
        >>> mad.save('mad.pt')
        >>> mad_copy = po.synth.MADCompetition.load('mad.pt')

        Things are slightly more complicated if you saved a reduced
        representation of the model by setting the
        ``save_model_reduced`` flag to ``True``. In that case, you also
        need to pass a model constructor argument, like so:

        >>> model1 = po.simul.RetinalGanglionCells(1)
        >>> model2 = po.metric.nlpd
        >>> mad = po.synth.MADCompetition(img, model1, model2)
        >>> mad.synthesize(max_iter=10, store_progress=True)
        >>> mad.save('mad.pt', save_model_reduced=True)
        >>> mad_copy = po.synth.MADCompetition.load('mad.pt',
                                                    [po.simul.RetinalGanglionCells.from_state_dict_reduced,
                                                     None])

        You may want to update one or more of the arguments used to
        initialize the model. The example I have in mind is where you
        run the metamer synthesis on a cluster but then load it on your
        local machine. The VentralModel classes have a ``cache_dir``
        attribute which you will want to change so it finds the
        appropriate location:

        >>> model1 = po.simul.RetinalGanglionCells(1)
        >>> model2 = po.metric.nlpd
        >>> mad = po.synth.MADCompetition(img, model1, model2)
        >>> mad.synthesize(max_iter=10, store_progress=True)
        >>> mad.save('mad.pt', save_model_reduced=True)
        >>> mad_copy = po.synth.MADCompetition.load('mad.pt',
                                                    [po.simul.RetinalGanglionCells.from_state_dict_reduced,
                                                     None],
                                                    cache_dir="/home/user/Desktop/metamers/windows_cache")

        """
        tmp = super().load(file_path, ['model_1', 'model_2'], model_constructor, map_location,
                           **state_dict_kwargs)
        synth_target = tmp.synthesis_target
        if '1' in synth_target:
            tmp_target = synth_target.replace('1', '2')
        else:
            tmp_target = synth_target.replace('2', '1')
        tmp.update_target(tmp_target, 'main')
        tmp.update_target(synth_target, 'main')
        return tmp

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
        attrs = ['base_signal', 'base_representation_1', 'base_representation_2',
                 'synthesized_signal', 'synthesized_representation_1', 'synthesized_representation_2',
                 'saved_signal', 'saved_representation_1', 'saved_representation_2',
                 'saved_signal_gradient', 'saved_representation_1_gradient',
                 'saved_representation_2_gradient', 'model_1', 'model_2']
        return super().to(*args, attrs=attrs, **kwargs)

    def _check_state(self, synthesis_target, model):
        """check which synthesis target/model to investigate and update if necessary

        since we have many possible states, the functions that we use to
        investigate the synthesis history can get a bit messy. to help
        with that, we use this helper. user specifies which synthesis
        target and model they want the attributes to work for and we
        call update_target appropriately.

        Importantly, both of those can be None, in which case we update
        nothing

        Since a single MADCompetition instance can be used for
        synthesizing multiple targets and has two models with different
        errors, you can specify the target and the model as well. If
        both are None, we use the current target of the synthesis. If
        synthesis_target is not None, but model is, we use the model
        that's the main target (e.g., if
        ``synthesis_target=='model_1_min'``, the we'd use `'model_1'`)

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

        Returns
        -------
        last_state : list
            The ``[synthesis_target, model]`` from before we updated (or
            None if no update was performed). call
            ``self.update_target(*last_state)`` to return to this state

        """
        # if both are None, then we don't update the target at all
        last_state = None
        if synthesis_target is not None or model is not None:
            if synthesis_target is None:
                synthesis_target = self.synthesis_target
            if model not in ['model_1', 'model_2', None]:
                raise Exception(f"Can't handle model {model}, must be one of 'model_1', 'model_2',"
                                " or None")
            if model is None:
                step = 'main'
            elif model.split('_') == synthesis_target.split('_')[:-1]:
                step = 'main'
            else:
                step = 'fix'
            last_state = self._last_update_target_args
            self.update_target(synthesis_target, step)
        return last_state

    def representation_error(self, iteration=None, synthesis_target=None, model=None, **kwargs):
        r"""Get the representation error

        This is (synthesized_representation - base_representation). If
        ``iteration`` is not None, we use
        ``self.saved_representation[iteration]`` for
        synthesized_representation..

        Since a single MADCompetition instance can be used for
        synthesizing multiple targets and has two models with different
        errors, you can specify the target and the model as well. If
        both are None, we use the current target of the synthesis. If
        synthesis_target is not None, but model is, we use the model
        that's the main target (e.g., if
        ``synthesis_target=='model_1_min'``, the we'd use
        `'model_1'`). If ``model=='both'``, we reutrn a dictionary
        containing both errors

        Regardless, we always reset the target state to what it was
        before this was called

        Any kwargs are passed through to self.analyze when computing the
        synthesized/target representation.

        Parameters
        ----------
        iteration: int or None, optional
            Which iteration to create the representation ratio for. If
            None, we use the current ``synthesized_representation``
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).
        model : {None, 'model_1', 'model_2', 'both'}, optional
            which model's representation to get the error for. If None
            and ``synthesis_targe`` is not None, we use the model that's
            the main target for synthesis_target (so if
            synthesis_target=='model_1_min', then we'd use
            'model_1'). If both are None, we use the current target. If
            'both', we return a dictionary of tensors (with keys
            'model_1' and 'model_2'), which contain both representation
            errors
        kwargs :
            passed through to self.analyze()

        Returns
        -------
        torch.Tensor

        """
        if model == 'both':
            last_state = self._check_state(synthesis_target, None)
            rep_error = {}
            rep_error['model_1'] = self.representation_error(iteration, synthesis_target,
                                                             'model_1')
            rep_error['model_2'] = self.representation_error(iteration, synthesis_target,
                                                             'model_2')
        else:
            last_state = self._check_state(synthesis_target, model)
            rep_error = super().representation_error(iteration, **kwargs)
            # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return rep_error

    def plot_representation_error(self, batch_idx=0, iteration=None, figsize=(12, 5), ylim=None,
                                  ax=None, title='', synthesis_target=None):
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

        If ``iteration`` is not None, we use
        ``self.saved_representation[iteration]`` for
        synthesized_representation..

        Since a single MADCompetition instance can be used for
        synthesizing multiple targets, you can specify the target as
        well. If None, we use the current target of the synthesis.

        MADCompetition also has two models, and we will plot the
        representation error for both of them, on separate subplots
        (titling them appropriately).

        Regardless, we always reset the target state to what it was
        before this was called

        Any kwargs are passed through to self.analyze when computing the
        synthesized/target representation.

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension (the first one)
        iteration: int or None, optional
            Which iteration to create the representation ratio for. If
            None, we use the current ``synthesized_representation``
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
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).


        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot

        """
        last_state = self._check_state(synthesis_target, None)
        rep_error = self.representation_error(iteration, synthesis_target, 'both')
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        else:
            warnings.warn("ax is not None, so we're ignoring figsize...")
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
            fig = ax.figure
            gs = ax.get_subplotspec().subgridspec(1, 2)
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        for i, (model, error) in enumerate(rep_error.items()):
            plot_representation(getattr(self, model), error, axes[i], figsize, ylim, batch_idx,
                                f'Model {i+1}: {self._get_model_name(model)} {title}')
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return fig

    def plot_synthesized_image(self, batch_idx=0, channel_idx=0, iteration=None, title=None,
                               figsize=(5, 5), ax=None, imshow_zoom=None, vrange=(0, 1),
                               synthesis_target=None):
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

        Since a single MADCompetition instance can be used for
        synthesizing multiple targets, you can specify the target as
        well. If None, we use the current target of the synthesis.

        Regardless, we always reset the target state to what it was
        before this was called

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
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        last_state = self._check_state(synthesis_target, None)
        if title is None:
            title = self.synthesis_target
        fig = super().plot_synthesized_image(batch_idx, channel_idx, iteration, title, figsize,
                                             ax, imshow_zoom, vrange)
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return fig

    def plot_synthesized_image_all(self, batch_idx=0, channel_idx=0, iteration=None, title=None,
                                   figsize=(10, 10), ax=None, imshow_zoom=None, vrange=(0, 1)):
        """show all synthesized images

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

        We show all synthesized images, as separate subplots.

        We always reset the target state to what it was before this was
        called

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
        if ax is None:
            if imshow_zoom is None:
                imshow_zoom = 1
            fig = pt.tools.display.make_figure(2, 2, [imshow_zoom * i for i in
                                                      self.base_signal.shape[2:]])
            axes = fig.axes
            axes = [clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
                    for ax in axes]
        else:
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
            fig = ax.figure
            gs = ax.get_subplotspec().subgridspec(12, 2)
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                    fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        for ax, target in zip(axes, ['model_1_min', 'model_1_max', 'model_2_min', 'model_2_max']):
            if self.synthesized_signal_all[target] is not None:
                self.plot_synthesized_image(batch_idx, channel_idx, iteration, title, None, ax,
                                            imshow_zoom, vrange, target)
        return fig

    def plot_loss(self, iteration=None, figsize=(5, 5), ax=None, synthesis_target=None, **kwargs):
        """Plot the synthesis loss

        We plot ``self.loss`` over all iterations. We also plot a red
        dot at ``iteration``, to highlight the loss there. If
        ``iteration=None``, then the dot will be at the final iteration.

        Since a single MADCompetition instance can be used for
        synthesizing multiple targets, you can specify the target as
        well. If None, we use the current target of the synthesis.

        MADCompetition also has two models, and we will plot the loss
        for both of them, on the same subplot (labelling them
        appropriately).

        Regardless, we always reset the target state to what it was
        before this was called

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
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).
        kwargs :
            passed to plt.semilogy

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        last_state = self._check_state(synthesis_target, 'model_1')
        model_1_name = self._get_model_name('model_1')
        super().plot_loss(iteration, ax=ax, label=f'Model 1: {model_1_name}',
                          title=f'{self.synthesis_target} loss', **kwargs)
        self._check_state(synthesis_target, 'model_2')
        model_2_name = self._get_model_name('model_2')
        super().plot_loss(iteration, ax=ax, label=f'Model 2: {model_2_name}',
                          title=f'{self.synthesis_target} loss', **kwargs)
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        ax.legend()
        return fig

    def plot_loss_all(self, iteration=None, figsize=(10, 10), ax=None, **kwargs):
        """Plot loss for all synthesis calls

        We plot ``self.loss`` over all iterations. We also plot a red
        dot at ``iteration``, to highlight the loss there. If
        ``iteration=None``, then the dot will be at the final iteration.

        We will plot the loss for each synthesis target, as a separate
        subplot.

        MADCompetition also has two models, and we will plot the loss
        for both of them, on the same subplot (labelling them
        appropriately).

        We always reset the target state to what it was before this was
        called

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
        if ax is None:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
        else:
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
            fig = ax.figure
            gs = ax.get_subplotspec().subgridspec(12, 2)
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                    fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        for ax, target in zip(axes, ['model_1_min', 'model_1_max', 'model_2_min', 'model_2_max']):
            self.plot_loss(iteration, ax=ax, synthesis_target=target)
        return fig

    def plot_synthesis_status(self, batch_idx=0, channel_idx=0, iteration=None, figsize=(23, 5),
                              ylim=None, plot_representation_error=True, imshow_zoom=None,
                              vrange=(0, 1), fig=None, synthesis_target=None):
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

        Since a single MADCompetition instance can be used for
        synthesizing multiple targets, you can specify the target as
        well. If None, we use the current target of the synthesis.

        MADCompetition also has two models, and we will plot the loss
        for both of them, on the same subplot (labelling them
        appropriately).

        Regardless, we always reset the target state to what it was
        before this was called

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
            of playing around to find a reasonable value.
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
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        last_state = self._check_state(synthesis_target, None)
        if fig is None:
            if plot_representation_error:
                n_subplots = 3
                width_ratios = [.25, .25, .5]
            else:
                n_subplots = 2
                width_ratios = [.5, .5]
            fig, axes = plt.subplots(1, n_subplots, figsize=figsize,
                                     gridspec_kw={'width_ratios': width_ratios})
        super().plot_synthesis_status(batch_idx, channel_idx, iteration, figsize, ylim,
                                      plot_representation_error, imshow_zoom, vrange, fig)
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return fig

    def animate(self, batch_idx=0, channel_idx=0, figsize=(23, 5), framerate=10, ylim=None,
                plot_representation_error=True, imshow_zoom=None, synthesis_target=None):
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

        Since a single MADCompetition instance can be used for
        synthesizing multiple targets, you can specify the target as
        well. If None, we use the current target of the synthesis.

        MADCompetition also has two models, and we will plot the loss
        for both of them, on the same subplot (labelling them
        appropriately).

        Regardless, we always reset the target state to what it was
        before this was called

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
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object. In order to view, must convert to HTML
            or save.

        """
        last_state = self._check_state(synthesis_target, None)
        if isinstance(ylim, str):
            warnings.warn("Be careful with rescaling the ylims, this can mess up any image that's"
                          " being shown (for example, the representation error of one of the "
                          "models) and, because of the way we handle having two models, the "
                          "animate() method is not as able to determine whether rescaling is "
                          "appropriate.")
        anim = super().animate(batch_idx, channel_idx, figsize, framerate, ylim,
                               plot_representation_error, imshow_zoom, ['loss_1', 'loss_2'],
                               {'model': 'both'})
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return anim
