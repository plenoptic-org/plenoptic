import torch
import warnings
from tqdm.auto import tqdm
import dill
import pyrtools as pt
from .synthesis import Synthesis
import matplotlib.pyplot as plt
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

    Note that for many attributes (e.g., `loss`, `synthesized_signal`), there are
    two versions, on with `_1` as a suffix and one with `_2`. This is because
    we need to store those attributes for each model: the version that ends in
    `_1` corresponds to `model_1` and the one that ends in `_2` corresponds to
    `model_2`. Similarly, many of these have a version that ends in `_all`,
    which is a dictionary containing that attribute for each synthesis target
    (some, such as `loss`, have both of these). See `MAD_Competition` notebook
    for more details.

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
    model_1, model_2 : torch.nn.Module or function
        The two visual models or metrics to compare, see `MAD_Competition`
        notebook for more details
    loss_function : callable or None, optional
        the loss function to use to compare the representations of the
        models in order to determine their loss. Only used for the
        Module models, ignored otherwise. If None, we use the default:
        the element-wise 2-norm. See `MAD_Competition` notebook for more
        details
    model_1_kwargs, model_2_kwargs : dict
        if model_1 or model_2 are functions (that is, you're using a
        metric instead of a model), then there might be additional
        arguments you want to pass it at run-time. Those should be
        included in a dictionary as ``key: value`` pairs. Note that this
        means they will be passed on every call.

    Attributes
    ----------
    base_signal : torch.Tensor
        A 2d tensor, this is the image whose representation we wish to
        match.
    model_1, model_2 : torch.nn.Module
        Two differentiable model that takes an image as an input and
        transforms it into a representation of some sort. We only
        require that they have a forward method, which returns the
        representation to match.
    base_representation_1, base_representation_2 : torch.Tensor
        Whatever is returned by ``model_1`` and ``model_2``
        ``forward(base_signal)`` methods, respectively. This is the
        representation we're trying to get as close or far away from as
        possible when targeting a given model.
    initial_image : torch.Tensor
        base_signal with white noise added to it (and clamped, if
        applicable), this is the starting point of our synthesis
    synthesized_signal : torch.Tensor
        The synthesized image from the last call to
        ``synthesis()``. This may be unfinished depending on how many
        iterations we've run for.
    synthesized_represetation_1, synthesized_representation_2: torch.Tensor
        Whatever is returned by ``model_1`` and ``model_2``
        ``forward(synthesized_signal)``, respectively.
    seed : int
        Number with which to seed pytorch and numy's random number
        generators
    loss_1, loss_2 : list
        list of the loss with respect to model_1, model_2 over
        iterations.
    gradient : list
        list containing the gradient over iterations
    learning_rate : list
        list containing the learning_rate over iterations. We use a
        scheduler that gradually reduces this over time, so it won't be
        constant.
    pixel_change : list
        A list containing the max pixel change over iterations
        (``pixel_change[i]`` is the max pixel change in
        ``synthesized_signal`` between iterations ``i`` and ``i-1``). note
        this is calculated before any clamping, so may have some very
        large numbers in the beginning
    nu : list
        list containing the nu parameter over iterations. Nu is the
        parameter used to correct the image so that the other model's
        representation will not change; see docstring of
        ``self._find_nu()`` for more details
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
        self.loss_1_norm = 1
        self.loss_2_norm = 1

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
                           'synthesized_representation_2', 'pixel_change']

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

    def _get_formatted_synthesis_target(self, synthesis_target=None):
        """Format synthesis_target for use as title."""
        if synthesis_target is None:
            synthesis_target = self.synthesis_target
        verb = synthesis_target.split('_')[-1].capitalize()
        model_name = self._get_model_name('_'.join(synthesis_target.split('_')[:-1]))
        return f"{verb}imize {model_name}"

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
        grad : torch.Tensor
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
        synth_rep : torch.Tensor
            model representation of the synthesized image
        ref_rep : torch.Tensor
            model representation of the reference image
        synth_img : torch.Tensor
            the synthesized image.
        ref_img : torch.Tensor
            the reference image

        Returns
        -------
        loss : torch.Tensor
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
            self.initial_image = (self.base_signal + initial_noise *
                                  torch.randn_like(self.base_signal))
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

    def _init_ctf_and_randomizer(self, loss_thresh=1e-4, coarse_to_fine=False,
                                 loss_change_thresh=1e-2, loss_change_iter=50):
        """initialize stuff related to randomization and coarse-to-fine

        we always make the stable model's coarse to fine False

        Parameters
        ----------
        loss_thresh : float, optional
            If the loss over the past ``loss_change_iter`` is less than
            ``loss_thresh``, we stop.
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
        loss_change_thresh : float, optional
            The threshold below which we consider the loss as unchanging and so
            should switch scales if `coarse_to_fine is not False`. Ignored
            otherwise.
        loss_change_iter : int, optional
            How many iterations back to check in order to see if the
            loss has stopped decreasing (for loss_change_thresh).

        """
        self.update_target(self.synthesis_target, 'main')
        super()._init_ctf_and_randomizer(loss_thresh, coarse_to_fine,
                                         loss_change_thresh, loss_change_iter)
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
                   optimizer_kwargs={}, clamper=RangeClamper((0, 1)),
                   clamp_each_iter=True, store_progress=False,
                   save_progress=False, save_path='mad.pt', loss_thresh=1e-4, loss_change_iter=50,
                   loss_change_thresh=1e-2, coarse_to_fine=False, clip_grad_norm=False):
        r"""Synthesize one maximally-differentiating image

        This synthesizes a single image, minimizing or maximizing either
        model 1 or model 2 while holding the other constant. By setting
        ``synthesis_target``, you can determine which of these you wish
        to synthesize.

        We run this until either we reach ``max_iter`` or the change
        over the past ``loss_change_iter`` iterations is less than
        ``loss_thresh``, whichever comes first

        Parameters
        ----------
        synthesis_target : {'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which image to synthesize
        initial_noise : `float` or None, optional
            standard deviation of the Gaussian noise used to create the initial
            image from the target image. Can only be None if
            ``self.saved_signal`` is not empty (i.e., this has been called at
            least once before with ``store_progress!=False``). In that case,
            the initial image is the last value of ``self.saved_signal``
        fix_step_n_iter : int, optional
            how many iterations we should use in the loop to determine the step
            size for re-adjusting the image so that the other model's loss
            doesn't change.
        norm_loss : bool, optional
            Whether to normalize the loss of each model, so that their losses
            (and thus gradients) are of the same magnitude.
        seed : int or None, optional
            Number with which to seed pytorch and numy's random number
            generators. If None, won't set the seed.
        max_iter : int, optional
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
            The MAD competition image we've created
        synthesized_representation_1 : torch.Tensor
            model_1's representation of this image
        synthesized_representation_2 : torch.Tensor
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
        self._init_ctf_and_randomizer(loss_thresh, coarse_to_fine,
                                      loss_change_thresh, loss_change_iter)

        # initialize the optimizer
        self._init_optimizer(optimizer, learning_rate, scheduler, clip_grad_norm,
                             optimizer_kwargs)

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
            loss, g, lr, pixel_change = self._optimizer_step(pbar, stable_loss="%.4e" % loss_2)
            self.loss.append(abs(loss.item()))
            self.gradient.append(g.item())
            self.pixel_change.append(pixel_change.item())
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

    def synthesize_all(self, if_existing='skip', **synthesize_kwargs):
        r"""Synthesize two pairs of maximally-differentiating images

        MAD Competition consists of two pairs of
        maximally-differentiating images: one pair minimizes and
        maximizes model 1, while holding model 2 constant, and the other
        minimizes and maximizes model 2, while holding model 1
        constant. This creates all four images. We return nothing, but
        all the outputs are stored in attributes.

        All additional parameters are passed directly through to
        ``synthesis()`` so if you want to synthesize the four images with
        different arguments, you should call ``synthesis()`` directly. The
        exception is ``save_path`` -- if it contains ``'{}'``, we format it to
        include the target name.

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

        """
        initial_noise_orig = synthesize_kwargs.pop('initial_noise', .1)
        learning_rate_orig = synthesize_kwargs.pop('learning_rate', 1)
        save_path = synthesize_kwargs.pop('save_path', None)
        save_progress = synthesize_kwargs.pop('save_progress', False)
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
            else:
                save_path_tmp = None
            print(s)
            if run:
                self.synthesize(target, initial_noise=initial_noise,
                                learning_rate=learning_rate,
                                save_path=save_path_tmp,
                                save_progress=save_progress,
                                **synthesize_kwargs)

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
        # the first two lines here make sure that we have both the _1 and _2
        # versions, regardless of which is currently found in .values()
        attrs = ([k.replace('_1', '_2') for k in self._names.values()] +
                 [k.replace('_2', '_1') for k in self._names.values()] +
                 [k + '_all' for k in self._attrs_all])
        # Removes duplicates
        attrs = list(set(attrs))
        # Remove the models, don't want to save those.
        attrs.remove('model_1')
        attrs.remove('model_2')
        # add the attributes not included above
        attrs += ['seed', 'scales', 'scales_timing', 'scales_loss', 'scales_finished',
                  'store_progress', 'save_progress', 'save_path', 'synthesis_target',
                  'coarse_to_fine']
        super().save(file_path, attrs)

    def load(self, file_path, map_location='cpu', **pickle_load_args):
        r"""Load all relevant stuff from a .pt file.

        This should be called by an initialized ``MADComptetion`` object -- we
        will ensure that ``base_signal``, ``base_representation_1`` (and thus
        ``model_1``), ``base_representation_2`` (and thus ``model_2``), and
        ``loss_function`` are all identical.

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
        >>> mad = po.synth.MADCompetition(img, model1, model2)
        >>> mad.synthesize(max_iter=10, store_progress=True)
        >>> mad.save('mad.pt')
        >>> mad_copy = po.synth.MADCompetition(img, model1, model2)
        >>> mad_copy = mad_copy.load('mad.pt')

        """
        # we have to check the loss functions ourself, because they're a bit
        # finicky
        tmp_dict = torch.load(file_path, map_location=map_location, pickle_module=dill)
        img = torch.rand_like(self.base_signal)
        rep = torch.rand_like(self.base_representation_1)
        saved_loss = tmp_dict['loss_function_1'](rep, self.base_representation_1, img,
                                                 self.base_signal)
        init_loss = self.loss_function_1(rep, self.base_representation_1, img,
                                         self.base_signal)
        if not torch.allclose(saved_loss, init_loss):
            raise Exception("Saved and initialized loss_function_1 are different! On base and random "
                            f"representation got: Initialized: {init_loss}"
                            f", Saved: {saved_loss}, difference: {init_loss-saved_loss}")
        rep = torch.rand_like(self.base_representation_2)
        saved_loss = tmp_dict['loss_function_2'](rep, self.base_representation_2, img,
                                                 self.base_signal)
        init_loss = self.loss_function_2(rep, self.base_representation_2, img,
                                         self.base_signal)
        if not torch.allclose(saved_loss, init_loss):
            raise Exception("Saved and initialized loss_function_2 are different! On base and random "
                            f"representation got: Initialized: {init_loss}"
                            f", Saved: {saved_loss}, difference: {init_loss-saved_loss}")
        super().load(file_path, map_location,
                     ['base_signal', 'base_representation_1',
                      'base_representation_2'],
                     **pickle_load_args)
        synth_target = self.synthesis_target
        if '1' in synth_target:
            tmp_target = synth_target.replace('1', '2')
        else:
            tmp_target = synth_target.replace('2', '1')
        self.update_target(tmp_target, 'main')
        self.update_target(synth_target, 'main')

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

    def representation_error(self, synthesis_target=None, iteration=None,
                             model=None, **kwargs):
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
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).
        iteration: int or None, optional
            Which iteration to create the representation ratio for. If
            None, we use the current ``synthesized_representation``
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
            rep_error['model_1'] = self.representation_error(iteration=iteration,
                                                             synthesis_target=synthesis_target,
                                                             model='model_1')
            rep_error['model_2'] = self.representation_error(iteration=iteration,
                                                             synthesis_target=synthesis_target,
                                                             model='model_2')
        else:
            last_state = self._check_state(synthesis_target, model)
            rep_error = super().representation_error(iteration, **kwargs)
            # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return rep_error

    def plot_representation_error(self,synthesis_target=None, batch_idx=0,
                                  iteration=None, figsize=(12, 5), ylim=None,
                                  ax=None, title='', as_rgb=False):
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
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).
        batch_idx : int, optional
            Which index to take from the batch dimension
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
            None, we create our own 2 subplot figure to hold it
        title : str, optional
            The title to put above this axis. If you want no title, pass
            the empty string (``''``)
        as_rgb : bool, optional
            The representation can be image-like with multiple channels, and we
            have no way to determine whether it should be represented as an RGB
            image or not, so the user must set this flag to tell us. It will be
            ignored if the representation doesn't look image-like or if the
            model has its own plot_representation_error() method. Else, it will
            be passed to `po.imshow()`, see that methods docstring for details.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot

        """
        last_state = self._check_state(synthesis_target, None)
        rep_error = self.representation_error(iteration=iteration,
                                              synthesis_target=synthesis_target,
                                              model='both')
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
                                f'Model {i+1}: {self._get_model_name(model)} {title}', as_rgb)
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return fig

    def plot_synthesized_image(self, synthesis_target=None, batch_idx=0,
                               channel_idx=None, iteration=None, title=None,
                               figsize=(5, 5), ax=None, imshow_zoom=None,
                               vrange=(0, 1)):
        """Show the synthesized image.

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
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).
        batch_idx : int, optional
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we use all
            channels (assumed use-case is RGB(A) image).
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
        last_state = self._check_state(synthesis_target, None)
        if title is None:
            title = self._get_formatted_synthesis_target(synthesis_target)
        fig = super().plot_synthesized_image(batch_idx, channel_idx, iteration, title, figsize,
                                             ax, imshow_zoom, vrange)
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return fig

    def plot_synthesized_image_all(self, batch_idx=0, channel_idx=None, iteration=None, title=None,
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
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we use all
            channels (assumed use-case is RGB(A) image).
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
                self.plot_synthesized_image(target, batch_idx, channel_idx,
                                            iteration, title, None, ax,
                                            imshow_zoom, vrange)
        return fig

    def plot_loss(self, synthesis_target=None, iteration=None, figsize=(5, 5), ax=None, **kwargs):
        """Plot the synthesis loss.

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
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).
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
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        last_state = self._check_state(synthesis_target, 'model_1')
        model_1_name = self._get_model_name('model_1')
        super().plot_loss(iteration, ax=ax, label=f'Model 1: {model_1_name}',
                          title=f'{self._get_formatted_synthesis_target()} loss',
                          **kwargs)
        self._check_state(synthesis_target, 'model_2')
        model_2_name = self._get_model_name('model_2')
        super().plot_loss(iteration, ax=ax, label=f'Model 2: {model_2_name}',
                          title=f'{self._get_formatted_synthesis_target()} loss', **kwargs)
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        ax.legend()
        return fig

    def plot_loss_all(self, iteration=None, figsize=(10, 10), ax=None, **kwargs):
        """Plot loss for all synthesis calls.

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
            self.plot_loss(iteration=iteration, ax=ax, synthesis_target=target)
        return fig

    def _grab_value_for_comparison(self, value, batch_idx=0, channel_idx=None,
                                   iteration=None, scatter_subsample=1,
                                   synthesis_target=None, model=None, **kwargs):
        """Grab and shape values for comparison plot.

        This grabs the appropriate batch_idx, channel_idx, and iteration from
        the saved representation or signal, respectively, and subsamples it if
        necessary.

        We then concatenate thema long the last dimension.

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
        scatter_subsample : float, optional
            What percentage of points to plot. If less than 1, will select that
            proportion of the points to plot. Done to make visualization
            clearer. Note we don't do this randomly (so that animate looks
            reasonable).
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
            passed to self.analyze

        Returns
        -------
        plot_vals : torch.Tensor
            2d tensor containing the base and synthesized value (indexed along
            last dimension)

        """
        if model == 'both':
            last_state = self._check_state(synthesis_target, None)
            plot_vals = {}
            plot_vals['model_1'] = self._grab_value_for_comparison(value=value,
                                                                   batch_idx=batch_idx,
                                                                   channel_idx=channel_idx,
                                                                   iteration=iteration,
                                                                   synthesis_target=synthesis_target,
                                                                   scatter_subsample=scatter_subsample,
                                                                   model='model_1',
                                                                   **kwargs)
            plot_vals['model_2'] = self._grab_value_for_comparison(value=value,
                                                                   batch_idx=batch_idx,
                                                                   channel_idx=channel_idx,
                                                                   iteration=iteration,
                                                                   synthesis_target=synthesis_target,
                                                                   scatter_subsample=scatter_subsample,
                                                                   model='model_2', **kwargs)
        else:
            last_state = self._check_state(synthesis_target, model)
            plot_vals = super()._grab_value_for_comparison(value=value,
                                                           batch_idx=batch_idx,
                                                           channel_idx=channel_idx,
                                                           iteration=iteration,
                                                           scatter_subsample=scatter_subsample,
                                                           **kwargs)
            # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return plot_vals

    def plot_value_comparison(self, synthesis_target=None,
                              value='representation', batch_idx=0,
                              channel_idx=None, iteration=None, figsize=(10, 5),
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
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).
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
            If not None, the axis to plot this representation on. If None, we
            create our own 1 (if value='signal') or 2 (if
            value='representatin') subplot figure to hold it
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
        last_state = self._check_state(synthesis_target, None)
        if ax is None:
            if value == 'representation':
                fig, axes = plt.subplots(1, 2, figsize=figsize)
            else:
                fig, axes = plt.subplots(1, 1, figsize=figsize)
                axes = [axes]
        else:
            if value == 'representation':
                warnings.warn("ax is not None, so we're ignoring figsize...")
                ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
                fig = ax.figure
                gs = ax.get_subplotspec().subgridspec(1, 2)
                axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
            else:
                fig = ax.figure
                axes = [ax]
        for ax, model in zip(axes, ['model_1', 'model_2']):
            super().plot_value_comparison(value, batch_idx, channel_idx,
                                          iteration, ax=ax, func=func,
                                          hist2d_nbins=hist2d_nbins,
                                          hist2d_cmap=hist2d_cmap,
                                          scatter_subsample=scatter_subsample,
                                          synthesis_target=synthesis_target,
                                          model=model,
                                          **kwargs)
            model_name = self._get_model_name(model)
            if value == 'representation':
                ax.set(ylabel=f'{model_name} synthesized {value}',
                       xlabel=f'{model_name} base {value}')
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return fig

    def plot_synthesis_status(self, synthesis_target=None, batch_idx=0,
                              channel_idx=None, iteration=None, figsize=None,
                              ylim=None, plot_synthesized_image=True,
                              plot_loss=True, plot_representation_error=True,
                              imshow_zoom=None, vrange=(0, 1), fig=None,
                              plot_image_hist=False, plot_rep_comparison=False,
                              plot_signal_comparison=False,
                              signal_comp_func='scatter',
                              signal_comp_subsample=.01, axes_idx={},
                              plot_representation_error_as_rgb=False,
                              width_ratios={}):
        r"""Make a plot showing synthesized image, loss, and (optionally) representation ratio.

        We create several subplots to analyze this. By default, we create three
        subplots on a new figure: the first one contains the synthesized image,
        the second contains the loss, and the third contains the representation
        error.

        There are several optional additional plots: image_hist, rep_comparison, and
        signal_comparison:

        - image_hist contains a histogram of pixel values of the synthesized
          and base images.

        - rep_comparison is a scatter plot comparing the representation of the
          synthesized and base images.

        - signal_comparison is a scatter plot (by default) or 2d histogram (if
          signal_comp_func='hist2d') of the pixel values in the synthesized and
          base images.

        All of these (including the default plots) can be toggled using their
        corresponding boolean flags, and can be created separately using the
        method with the same name as the flag.

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
        before this was called.

        MADCompetition also has two models, and we will plot the loss for both
        of them, on the same subplot (labelling them appropriately). The
        rep_comparison and representation_error plots will take up 2 subplots
        (one for each model), if created.

        Parameters
        ----------
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).
        batch_idx : int, optional
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we use all
            channels (assumed use-case is RGB(A) image).
        iteration : int or None, optional
            Which iteration to display. If None, the default, we show
            the most recent one. Negative values are also allowed.
        figsize : tuple or None, optional
            The size of the figure to create. It may take a little bit of
            playing around to find a reasonable value. If None, we attempt to
            make our best guess, aiming to have each axis be of size (5, 5)
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
        plot_image_hist : bool, optional
            Whether to plot the histograms of image pixel intensities or
            not.
        plot_rep_comparison : bool, optional
            Whether to plot a scatter plot comparing the synthesized and base
            representation.
        plot_signal_comparison : bool, optional
            Whether to plot the comparison of the synthesized and base
            signal.
        signal_comp_func : {'scatter', 'hist2d'}, optional
            Whether to use a scatter plot or 2d histogram to plot this signal
            comparison. When there are many values (as often happens), then
            hist2d will be clearer
        signal_comp_subsample : float, optional
            What percentage of signal points to plot. If less than 1, will
            randomly select that proportion of the points to plot. Done to make
            visualization clearer.
        axes_idx : dict, optional
            Dictionary specifying which axes contains which type of plot,
            allows for more fine-grained control of the resulting figure.
            Probably only helpful if fig is also defined. Possible keys: image,
            loss, rep_error, hist, rep_comp, signal_comp, misc. Values should
            all be ints. If you tell this function to create a plot that doesn't
            have a corresponding key, we find the lowest int that is not
            already in the dict, so if you have axes that you want unchanged,
            place their idx in misc.
        plot_representation_error_as_rgb : bool, optional
            The representation can be image-like with multiple channels, and we
            have no way to determine whether it should be represented as an RGB
            image or not, so the user must set this flag to tell us. It will be
            ignored if the representation doesn't look image-like or if the
            model has its own plot_representation_error() method. Else, it will
            be passed to `po.imshow()`, see that methods docstring for details.
        width_ratios : dict, optional
            By defualt, all plots axes will have the same width. To change
            that, specify their relative widths using keys of the format
            "{x}_width", where `x` in ['synthesized_image', 'loss',
            'representation_error', 'image_hist', 'rep_comparison',
            'signal_comparison']

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        last_state = self._check_state(synthesis_target, None)
        if fig is None:
            fig = self._setup_synthesis_fig(fig, axes_idx, figsize,
                                            plot_synthesized_image,
                                            plot_loss,
                                            plot_representation_error,
                                            plot_image_hist,
                                            plot_rep_comparison,
                                            plot_signal_comparison,
                                            representation_error_width=2,
                                            rep_comparison_width=2)[0]
        super().plot_synthesis_status(batch_idx, channel_idx, iteration,
                                      figsize, ylim, plot_synthesized_image,
                                      plot_loss, plot_representation_error,
                                      imshow_zoom, vrange, fig,
                                      plot_image_hist, plot_rep_comparison,
                                      plot_signal_comparison, signal_comp_func,
                                      signal_comp_subsample, axes_idx,
                                      plot_representation_error_as_rgb,
                                      width_ratios)
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return fig

    def animate(self, synthesis_target=None, batch_idx=0, channel_idx=None,
                figsize=None, framerate=10, ylim=None,
                plot_synthesized_image=True, plot_loss=True,
                plot_representation_error=True, imshow_zoom=None,
                plot_image_hist=False, plot_rep_comparison=False,
                plot_signal_comparison=False,
                fig=None, signal_comp_func='scatter', signal_comp_subsample=.01,
                axes_idx={}, init_figure=True,
                plot_representation_error_as_rgb=False,
                width_ratios={}):
        r"""Animate synthesis progress.

        This is essentially the figure produced by
        ``self.plot_synthesis_status`` animated over time, for each stored
        iteration.

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

        Regardless, we always reset the target state to what it was
        before this was called.

        MADCompetition also has two models, and we will plot the loss for both
        of them, on the same subplot (labelling them appropriately). The
        rep_comparison and representation_error plots will each take up 2
        subplots (one for each model), if created.

        Parameters
        ----------
        synthesis_target : {None, 'model_1_min', 'model_1_max', 'model_2_min', 'model_2_max'}
            which synthesis target to grab the representation for. If
            None, we use the most recent synthesis_target (i.e.,
            ``self.synthesis_target``).
        batch_idx : int, optional
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we use all
            channels (assumed use-case is RGB(A) image).
        figsize : tuple or None, optional
            The size of the figure to create. It may take a little bit of
            playing around to find a reasonable value. If None, we attempt to
            make our best guess, aiming to have each axis be of size (5, 5)
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
        plot_image_hist : bool, optional
            Whether to plot the histograms of image pixel intensities or
            not. Note that we update this in the most naive way possible
            (by clearing and replotting the values), so it might not
            look as good as the others and may take some time.
        plot_rep_comparison : bool, optional
            Whether to plot a scatter plot comparing the synthesized and base
            representation.
        plot_signal_comparison : bool, optional
            Whether to plot a 2d histogram comparing the synthesized and base
            representation. Note that we update this in the most naive way
            possible (by clearing and replotting the values), so it might not
            look as good as the others and may take some time.
        fig : plt.Figure or None, optional
            If None, create the figure from scratch. Else, should be an empty
            figure with enough axes (the expected use here is have same-size
            movies with different plots).
        signal_comp_func : {'scatter', 'hist2d'}, optional
            Whether to use a scatter plot or 2d histogram to plot this signal
            comparison. When there are many values (as often happens), then
            hist2d will be clearer
        signal_comp_subsample : float, optional
            What percentage of signal points to plot. If less than 1, will
            randomly select that proportion of the points to plot. Done to make
            visualization clearer.
        axes_idx : dict, optional
            Dictionary specifying which axes contains which type of plot,
            allows for more fine-grained control of the resulting figure.
            Probably only helpful if fig is also defined. Possible keys: image,
            loss, rep_error, hist, rep_comp, signal_comp, misc. Values should
            all be ints. If you tell this function to create a plot that doesn't
            have a corresponding key, we find the lowest int that is not
            already in the dict, so if you have axes that you want unchanged,
            place their idx in misc.
        init_figure : bool, optional
            If True, we call plot_synthesis_status to initialize the figure. If
            False, we assume fig has already been intialized with the proper
            plots (e.g., you already called plot_synthesis_status and are
            passing that figure as the fig argument). In this case, axes_idx
            must also be set and include keys for each of the included plots,
        plot_representation_error_as_rgb : bool, optional
            The representation can be image-like with multiple channels, and we
            have no way to determine whether it should be represented as an RGB
            image or not, so the user must set this flag to tell us. It will be
            ignored if the representation doesn't look image-like or if the
            model has its own plot_representation_error() method. Else, it will
            be passed to `po.imshow()`, see that methods docstring for details.
            since plot_synthesis_status normally sets it up for us
        width_ratios : dict, optional
            By defualt, all plots axes will have the same width. To change
            that, specify their relative widths using keys of the format
            "{x}_width", where `x` in ['synthesized_image', 'loss',
            'representation_error', 'image_hist', 'rep_comparison',
            'signal_comparison']

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
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
        last_state = self._check_state(synthesis_target, None)
        if isinstance(ylim, str):
            warnings.warn("Be careful with rescaling the ylims, this can mess up any image that's"
                          " being shown (for example, the representation error of one of the "
                          "models) and, because of the way we handle having two models, the "
                          "animate() method is not as able to determine whether rescaling is "
                          "appropriate.")
        anim = super().animate(batch_idx, channel_idx, figsize, framerate,
                               ylim, plot_synthesized_image, plot_loss,
                               plot_representation_error, imshow_zoom,
                               ['loss_1', 'loss_2'], {'model': 'both'},
                               plot_image_hist, plot_rep_comparison,
                               plot_signal_comparison, fig, signal_comp_func,
                               signal_comp_subsample, axes_idx, init_figure,
                               plot_representation_error_as_rgb,
                               width_ratios={})
        # reset to state before calling this function
        if last_state is not None:
            self.update_target(*last_state)
        return anim
