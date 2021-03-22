"""abstract synthesis super-class
"""
import abc
import re
import torch
from torch import optim
import torchcontrib
import numpy as np
import warnings
from ..tools.data import to_numpy, _find_min_int
from ..tools.optim import l2_norm
import matplotlib.pyplot as plt
import pyrtools as pt
from ..tools.display import rescale_ylim, plot_representation, update_plot, imshow
from matplotlib import animation
from ..simulate.models.naive import Identity
from tqdm import tqdm
import dill
from ..tools.metamer_utils import RangeClamper


class Synthesis(metaclass=abc.ABCMeta):
    r"""Abstract super-class for synthesis methods

    All synthesis methods share a variety of similarities and thus need
    to have similar methods. Some of these can be implemented here and
    simply inherited, some of them will need to be different for each
    sub-class and thus are marked as abstract methods here

    Parameters
    ----------
    base_signal : torch.Tensor or array_like
        A 4d tensor, this is the image whose representation we wish to
        match. If this is not a tensor, we try to cast it as one.
    model : torch.nn.Module or function
        The visual model or metric to synthesize with. See `MAD_Competition`
        for details.
    loss_function : callable or None, optional
        the loss function to use to compare the representations of the
        models in order to determine their loss. Only used for the
        Module models, ignored otherwise. If None, we use the default:
        the element-wise 2-norm. See `MAD_Competition` notebook for more
        details
    model_kwargs : dict
        if model is a function (that is, you're using a metric instead
        of a model), then there might be additional arguments you want
        to pass it at run-time. Note that this means they will be passed
        on every call.

    """

    def __init__(self, base_signal, model, loss_function, model_kwargs={}, loss_function_kwargs={}):
        # this initializes all the attributes that are shared, though
        # they can be overwritten in the individual __init__() if
        # necessary
        self._use_subset_for_gradient = False

        if not isinstance(base_signal, torch.Tensor):
            base_signal = torch.tensor(base_signal, dtype=torch.float32)
        if base_signal.ndim != 4:
            raise ValueError("Synthesis expect base_signal to be 4d, but it is of shape"
                             f" {base_signal.shape} instead!")
        self.base_signal = base_signal
        self.seed = None
        self._rep_warning = False

        if loss_function is None:
            loss_function = l2_norm
        else:
            if not isinstance(model, torch.nn.Module):
                warnings.warn("Ignoring custom loss_function for model since it's a metric")

        # we handle models a little differently, so this is here
        if isinstance(model, torch.nn.Module):
            self.model = model

            def wrapped_loss_func(synth_rep, ref_rep, synth_img, ref_img):
                return loss_function(ref_rep=ref_rep, synth_rep=synth_rep, ref_img=ref_img,
                                     synth_img=synth_img, **loss_function_kwargs)
            self.loss_function = wrapped_loss_func
        else:
            self.model = Identity(model.__name__).to(base_signal.device)

            def wrapped_model(synth_rep, ref_rep, synth_img, ref_img):
                return model(synth_rep, ref_rep, **model_kwargs)
            self.loss_function = wrapped_model
            self._rep_warning = True

        self.base_representation = self.analyze(self.base_signal)
        self.synthesized_signal = None
        self.synthesized_representation = None
        self._optimizer = None
        self._scheduler = None

        self.loss = []
        self.gradient = []
        self.learning_rate = []
        self.pixel_change = []
        self._last_iter_synthesized_signal = None
        self.saved_representation = []
        self.saved_signal = []
        self.saved_signal_gradient = []
        self.saved_representation_gradient = []
        self.scales_loss = None
        self.scales = None
        self.scales_timing = None
        self.scales_finished = None
        self.coarse_to_fine = False
        self.store_progress = None

    def _set_seed(self, seed):
        """set the seed

        we call both ``torch.manual_seed()`` and ``np.random.seed()``

        we also set the ``self.seed`` attribute

        Parameters
        ----------
        seed : int
            the seed to set
        """
        self.seed = seed
        if seed is not None:
            # random initialization
            torch.manual_seed(seed)
            np.random.seed(seed)

    def _init_synthesized_signal(self, synthesized_signal_data, clamper=RangeClamper((0, 1)),
                                 clamp_each_iter=True):
        """initialize the synthesized image

        set the ``self.synthesized_signal`` attribute to be a parameter with
        the user-supplied data, making sure it's the right shape and
        calling clamper on it, if set

        also initialize the ``self.synthesized_representation`` attribute

        Parameters
        ----------
        synthesized_signal_data : torch.Tensor or array_like
            the data to use as the first synthesized_signal
        clamper : Clamper or None, optional
            will set ``self.clamper`` attribute to this, and if not
            None, will call ``clamper.clamp`` on synthesized_signal
        clamp_each_iter : bool, optional
            If True (and ``clamper`` is not ``None``), we clamp every
            iteration. If False, we only clamp at the very end, after
            the last iteration
        """
        self.synthesized_signal = torch.nn.Parameter(synthesized_signal_data, requires_grad=True)
        while self.synthesized_signal.ndimension() < 4:
            self.synthesized_signal.data = self.synthesized_signal.data.unsqueeze(0)
        self.clamper = clamper
        if self.clamper is not None:
            self.synthesized_signal.data = self.clamper.clamp(self.synthesized_signal.data)
        self.synthesized_representation = self.analyze(self.synthesized_signal)
        self.clamp_each_iter = clamp_each_iter

    def _init_ctf_and_randomizer(self, loss_thresh=1e-4, fraction_removed=0, coarse_to_fine=False,
                                 loss_change_fraction=1, loss_change_thresh=1e-2,
                                 loss_change_iter=50):
        """initialize stuff related to randomization and coarse-to-fine

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
        if fraction_removed > 0 or loss_change_fraction < 1:
            self._use_subset_for_gradient = True
            if isinstance(self.model, Identity):
                raise Exception("Can't use fraction_removed or loss_change_fraction with metrics!"
                                " Since most of the metrics rely on the image being correctly "
                                "structured (and thus not randomized) when passed to them")
        self.fraction_removed = fraction_removed
        self.loss_thresh = loss_thresh
        self.loss_change_thresh = loss_change_thresh
        self.loss_change_iter = int(loss_change_iter)
        self.loss_change_fraction = loss_change_fraction
        self.coarse_to_fine = coarse_to_fine
        if coarse_to_fine not in [False, 'separate', 'together']:
            raise Exception(f"Don't know how to handle value {coarse_to_fine}! Must be one of: "
                            "False, 'separate', 'together'")
        if coarse_to_fine:
            if self.scales is None:
                # this creates a new object, so we don't modify model.scales
                self.scales = [i for i in self.model.scales[:-1]]
                if coarse_to_fine == 'separate':
                    self.scales += [self.model.scales[-1]]
                self.scales += ['all']
                self.scales_timing = dict((k, []) for k in self.scales)
                self.scales_timing[self.scales[0]].append(0)
                self.scales_finished = []
                self.scales_loss = []
            # else, we're continuing a previous version and want to continue
        if (loss_change_thresh is not None) and (loss_thresh >= loss_change_thresh):
            raise Exception("loss_thresh must be strictly less than loss_change_thresh, or things"
                            " get weird!")

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
        # python's implicit boolean-ness means we can do this! it will evaluate to False for False
        # and 0, and True for True and every int >= 1
        if store_progress is None and self.store_progress is not None:
            store_progress = self.store_progress
        if store_progress:
            if store_progress is True:
                store_progress = 1
            # if this is not the first time synthesize is being run for
            # this metamer object,
            # saved_signal/saved_representation(_gradient) will be
            # tensors instead of lists. This converts them back to lists
            # so we can use append. If it's the first time, they'll be
            # empty lists and this does nothing
            self.saved_signal = list(self.saved_signal)
            self.saved_representation = list(self.saved_representation)
            self.saved_signal_gradient = list(self.saved_signal_gradient)
            self.saved_representation_gradient = list(self.saved_representation_gradient)
            self.saved_signal.append(self.synthesized_signal.clone().to('cpu'))
            self.saved_representation.append(self.analyze(self.synthesized_signal).to('cpu'))
        else:
            if save_progress:
                raise Exception("Can't save progress if we're not storing it! If save_progress is"
                                " True, store_progress must be not False")
        if self.store_progress is not None and store_progress != self.store_progress:
            # we require store_progress to be the same because otherwise
            # the subsampling relationship between attrs that are stored
            # every iteration (loss, gradient, etc) and those that are
            # stored every store_progress iteration (e.g., saved_signal,
            # saved_representation) changes partway through and that's
            # annoying
            raise Exception("If you've already run synthesize() before, must re-run it with same"
                            f" store_progress arg. You passed {store_progress} instead of"
                            f" {self.store_progress} (True is equivalent to 1)")
        self.store_progress = store_progress
        self.save_progress = save_progress
        self.save_path = save_path

    def _check_nan_loss(self, loss):
        """check if loss is nan and, if so, return True

        This checks if loss is NaN and, if so, updates
        synthesized_signal/representation to be several iterations ago (so
        they're meaningful) and then returns True

        Parameters
        ----------
        loss : torch.Tensor
            the loss from the most recent iteration

        Returns
        -------
        is_nan : bool
            True if loss was nan, False otherwise

        """
        if np.isnan(loss.item()):
            warnings.warn("Loss is NaN, quitting out! We revert synthesized_signal / synthesized_"
                          "representation to our last saved values (which means this will "
                          "throw an IndexError if you're not saving anything)!")
            # need to use the -2 index because the last one will be
            # the one full of NaNs. this happens because the loss is
            # computed before calculating the gradient and updating
            # synthesized_signal; therefore the iteration where loss is
            # NaN is the one *after* the iteration where
            # synthesized_signal (and thus synthesized_representation)
            # started to have NaN values. this will fail if it hits
            # a nan before store_progress iterations (because then
            # saved_signal/saved_representation only has a length of
            # 1) but in that case, you have more severe problems
            self.synthesized_signal = torch.nn.Parameter(self.saved_signal[-2])
            self.synthesized_representation = self.saved_representation[-2]
            return True
        return False

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

        Returns
        -------
        stored : bool
            True if we stored this iteration, False if not. Note that
            storing and saving can be separated (if both
            ``store_progress`` and ``save_progress`` are different
            integers, for example). This only reflects *storing*, not
            saving

        """
        stored = False
        with torch.no_grad():
            if self.clamper is not None and self.clamp_each_iter:
                self.synthesized_signal.data = self.clamper.clamp(self.synthesized_signal.data)

            # i is 0-indexed but in order for the math to work out we want to be checking a
            # 1-indexed thing against the modulo (e.g., if max_iter=10 and
            # store_progress=3, then if it's 0-indexed, we'll try to save this four times,
            # at 0, 3, 6, 9; but we just want to save it three times, at 3, 6, 9)
            if self.store_progress and ((i+1) % self.store_progress == 0):
                # want these to always be on cpu, to reduce memory use for GPUs
                self.saved_signal.append(self.synthesized_signal.clone().to('cpu'))
                # we do this instead of using
                # self.synthesized_representation because its size might
                # change over time (if we're doing coarse-to-fine), and
                # we want to be able to stack this
                self.saved_representation.append(self.analyze(self.synthesized_signal).to('cpu'))
                self.saved_signal_gradient.append(self.synthesized_signal.grad.clone().to('cpu'))
                self.saved_representation_gradient.append(self.synthesized_representation.grad.clone().to('cpu'))
                if self.save_progress is True:
                    self.save(self.save_path, True)
                stored = True
            if type(self.save_progress) == int and ((i+1) % self.save_progress == 0):
                self.save(self.save_path, True)
        return stored

    def _check_for_stabilization(self, i):
        r"""Check whether the loss has stabilized and, if so, return True

        We check whether the loss has stopped decreasing and return True
        if so.

        We rely on a handful of attributes to do this, and take the
        following steps:

        1. Check if we've been synthesizing for at least
           ``self.loss_change_iter`` iterations.

        2a. If so, check whether the absolute difference between the most
           recent loss and the loss ``self.loss_change_iter`` iterations
           ago is less than ``self.loss_thresh``.

        2b. If not, return False

        3a. If so, check whether coarse_to_fine is not False.

        3b. If not, return False

        4a. If so, check whether we're synthesizing with respect to all
           scales and have been doing so for at least
           ``self.loss_change_iter`` iterations.

        4b. If not, return True

        5a. If so, return True

        5b. If not, return False

        Parameters
        ----------
        i : int
            the current iteration (0-indexed)

        """
        if len(self.loss) >= self.loss_change_iter:
            if self.loss_thresh is None or abs(self.loss[-self.loss_change_iter] - self.loss[-1]) < self.loss_thresh:
                if self.coarse_to_fine:
                    # only break out if we've been doing for long enough
                    if self.scales[0] == 'all' and i - self.scales_timing['all'][0] >= self.loss_change_iter:
                        return True
                else:
                    return True
        return False

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
        if self.clamper is not None:
            try:
                # setting the data directly avoids the issue of setting
                # a non-Parameter tensor where a tensor should be
                self.synthesized_signal.data = self.clamper.clamp(self.synthesized_signal.data)
                self.synthesized_representation.data = self.analyze(self.synthesized_signal).data
            except RuntimeError:
                # this means that we hit a NaN during optimization and
                # so self.synthesized_signal is on the cpu (since we're
                # copying from self.saved_imgae, which is always on the
                # cpu), whereas the model is on a different device. this
                # should be the same as self.base_signal.device
                # (unfortunatley we can't trust that self.model has a
                # device attribute), and so the following should hopefully work
                self.synthesized_signal.data = self.clamper.clamp(self.synthesized_signal.data.to(self.base_signal.device))
                self.synthesized_representation.data = self.analyze(self.synthesized_signal).data

        if self.store_progress:
            self.saved_representation = torch.stack(self.saved_representation)
            self.saved_signal = torch.stack(self.saved_signal)
            self.saved_signal_gradient = torch.stack(self.saved_signal_gradient)
            # we can't stack the gradients if we used coarse-to-fine
            # optimization, because then they'll be different shapes, so
            # we have to keep them as a list
            try:
                self.saved_representation_gradient = torch.stack(self.saved_representation_gradient)
            except RuntimeError:
                pass

    @abc.abstractmethod
    def synthesize(self, seed=0, max_iter=100, learning_rate=1, scheduler=True, optimizer='Adam',
                   optimizer_kwargs={}, swa=False, swa_kwargs={}, clamper=RangeClamper((0, 1)),
                   clamp_each_iter=True, store_progress=False,
                   save_progress=False, save_path='synthesis.pt', loss_thresh=1e-4,
                   loss_change_iter=50, fraction_removed=0., loss_change_thresh=1e-2,
                   loss_change_fraction=1., coarse_to_fine=False, clip_grad_norm=False):
        r"""synthesize an image

        this is a skeleton of how synthesize() works, just to serve as a
        guide -- YOU SHOULD NOT CALL THIS FUNCTION.

        You should, however, copy the call signature and then add any
        extra arguments specific to the given synthesis method at the
        beginning

        FOLLOWING DOCUMENTATION APPLIES AS LONG AS YOU USE ALL THE ABOVE
        ARGUMENTS AND FOLLOW THE GENERAL STRUCTURE OUTLINED IN THIS
        FUNCTION

        Parameters
        ----------
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
        swa : bool, optional
            whether to use stochastic weight averaging or not
        swa_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the SWA object. See
            torchcontrib.optim.SWA docs for more info.
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
            loss has stopped decreasing in order to determine whether we
            should only calculate the gradient with respect to the
            ``loss_change_fraction`` fraction of statistics with
            the highest error.
        fraction_removed: float, optional
            The fraction of the representation that will be ignored
            when computing the loss. At every step the loss is computed
            using the remaining fraction of the representation only.
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
            The synthesized image we've created
        synthesized_representation : torch.Tensor
            model's representation of this image

        """
        raise NotImplementedError("Synthesis.synthesize() should not be called!")
        # set the seed
        self._set_seed(seed)
        # initialize synthesized_signal -- how exactly you do this will
        # depend on the synthesis method
        self._init_synthesized_signal(synthesized_signal_data, clamper, clamp_each_iter)
        # initialize stuff related to coarse-to-fine and randomization
        self._init_ctf_and_randomizer(loss_thresh, fraction_removed, coarse_to_fine,
                                      loss_change_fraction, loss_change_thresh, loss_change_iter)
        # initialize the optimizer
        self._init_optimizer(optimizer, learning_rate, scheduler, clip_grad_norm,
                             optimizer_kwargs, swa, swa_kwargs)
        # get ready to store progress
        self._init_store_progress(store_progress, save_progress)

        # initialize the progress bar...
        pbar = tqdm(range(max_iter))

        # and start synthesizing.
        for i in pbar:
            # this is an example, because this is the section that will
            # vary the most amongst synthesis methods
            loss, g, lr, pixel_change = self._optimizer_step(pbar)
            self.loss.append(loss.item())
            self.pixel_change.append(pixel_change.item())
            self.gradient.append(g.item())
            self.learning_rate.append(lr)

            # check if loss is nan
            if self._check_nan_loss(loss):
                break

            # clamp and update saved_* attrs
            self._clamp_and_store(i)

            if self._check_for_stabilization(i):
                break

        pbar.close()

        # finally, stack the saved_* attributes
        self._finalize_store_progress()

        # and return
        return self.synthesized_signal.data, self.synthesized_representation.data

    def analyze(self, x, **kwargs):
        r"""Analyze the image, that is, obtain the model's representation of it

        Any kwargs are passed to the model's forward method

        Parameters
        ----------
        x : torch.Tensor
            The image to analyze

        Returns
        -------
        y : torch.Tensor
            The model's representation of x
        """
        y = self.model(x, **kwargs)
        if isinstance(y, list):
            return torch.cat([s.squeeze().view(-1) for s in y]).unsqueeze(1)
        else:
            return y

    def objective_function(self, synth_rep, ref_rep, synth_img, ref_img):
        r"""Calculate the loss

        This is what we minimize. We call
        ``self.loss_function(ref_rep=ref_rep, synth_rep=synth_rep,
        ref_img=ref_img, synth_img=synth_img)`` -- by default, this is
        the L2-norm of the difference between the two representations:
        ``torch.norm(ref_rep - synth_rep, p=2)``.

        We have this as a separate method, instead of just using the
        attribute, in order to allow sub-classes to overwrite. For
        example, if you want to take this output and then do something
        else to it (like flip its sign or normalize it)

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
        return self.loss_function(ref_rep=ref_rep, synth_rep=synth_rep, ref_img=ref_img,
                                  synth_img=synth_img)

    def representation_error(self, iteration=None, **kwargs):
        r"""Get the representation error

        This is (synthesized_representation - base_representation). If
        ``iteration`` is not None, we use
        ``self.saved_representation[iteration]`` for
        synthesized_representation

        Any kwargs are passed through to self.analyze when computing the
        synthesized/base representation.

        Parameters
        ----------
        iteration: int or None, optional
            Which iteration to create the representation ratio for. If
            None, we use the current ``synthesized_representation``

        Returns
        -------
        torch.Tensor

        """
        if self._rep_warning:
            warnings.warn("Since at least one of your models is a metric, its representation_error"
                          " will be meaningless -- it will just show the pixel-by-pixel difference"
                          ". (Your loss is still meaningful, however, since it's the actual "
                          "metric)")
        if iteration is not None:
            synthesized_rep = self.saved_representation[iteration].to(self.base_representation.device)
        else:
            synthesized_rep = self.analyze(self.synthesized_signal, **kwargs)
        try:
            rep_error = synthesized_rep - self.base_representation
        except RuntimeError:
            # try to use the last scale (if the above failed, it's
            # because they were different shapes), but only if the user
            # didn't give us another scale to use
            if 'scales' not in kwargs.keys():
                kwargs['scales'] = [self.scales[-1]]
            rep_error = synthesized_rep - self.analyze(self.base_signal, **kwargs)
        return rep_error

    def _init_optimizer(self, optimizer, lr, scheduler=True, clip_grad_norm=False,
                        optimizer_kwargs={}, swa=False, swa_kwargs={}):
        """Initialize the optimzer and learning rate scheduler

        This gets called at the beginning of synthesize() and can also
        be called at other moments to make sure we're using the original
        learning rate (e.g., when moving to a different scale for
        coarse-to-fine optimization).

        we also (optionally) initialize a learning rate scheduler which
        will reduce the LR on plateau by a factor of .5. To turn this
        behavior off, pass ``scheduler=False``

        optimizer options. each has some default arguments which are
        explained below. with the exception of ``'GD'``, each of these
        can be overwritten by values passed as ``optimizer_kwargs``:
        - 'GD': gradient descent, ``optim.SGD(nesterov=False,
          momentum=0, weight_decay=0)`` (these cannot be modified)
        - 'SGD': stochastic gradient descent, ``optim.SGD(nesterov=True,
          momentum=.8)``
        - 'LBFGS': limited-memory BFGS , ``optim.LBFGS(history_size=10,
          max_iter=4)``
        - 'Adam': Adam, ``optim.Adam(amsgrad=True)``
        - 'AdamW': AdamW, ``optim.AdamW(amsgrad=True)``

        Note that if you modify this function to take extra arguments, make
        sure to modify the line that creates _init_optimizer_kwargs and add it
        there. If you over-write this in a subclass, will also need to update
        _init_optimizer_kwargs to include additional arguments

        Parameters
        ----------
        optimizer : {'GD', 'SGD', 'LBFGS', 'Adam', 'AdamW'}
            the optimizer to initialize.
        lr : float or None
            The learning rate for our optimizer. None is only accepted
            if we're resuming synthesis, in which case we use the last
            learning rate from the previous instance.
        scheduler : bool, optional
            whether to initialize the scheduler or not. If False, the
            learning rate will never decrease. Setting this to True
            seems to improve performance, but it might be useful to turn
            it off in order to better work through what's happening
        clip_grad_norm : bool or float, optional
            If the gradient norm gets too large, the optimization can
            run into problems with numerical overflow. In order to avoid
            that, you can clip the gradient norm to a certain maximum by
            setting this to True or a float (if you set this to False,
            we don't clip the gradient norm). If True, then we use 1,
            which seems reasonable. Otherwise, we use the value set
            here.
        optimizer_kwargs :
            passed to the optimizer's initializer
        swa : bool, optional
            whether to use stochastic weight averaging or not
        swa_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the SWA object.

        """
        # there's a weird scoping issue that happens if we don't copy the
        # dictionary, where it can accidentally persist across instances of the
        # object, which messes all sorts of things up
        optimizer_kwargs = optimizer_kwargs.copy()
        swa_kwargs = swa_kwargs.copy()
        # if lr is None, we're resuming synthesis from earlier, and we
        # want to start with the last learning rate. however, we also
        # want to keep track of the initial learning rate, since we use
        # this for resetting the optimizer during coarse-to-fine
        # optimization. we thus also track the initial_learnig_rate...
        if lr is None:
            lr = self.learning_rate[-1]
            initial_lr = self.learning_rate[0]
        else:
            initial_lr = lr
        if optimizer == 'GD':
            # std gradient descent
            self._optimizer = optim.SGD([self.synthesized_signal], lr=lr, nesterov=False,
                                        momentum=0, weight_decay=0, **optimizer_kwargs)
        elif optimizer == 'SGD':
            for k, v in zip(['nesterov', 'momentum'], [True, .8]):
                if k not in optimizer_kwargs:
                    optimizer_kwargs[k] = v
            self._optimizer = optim.SGD([self.synthesized_signal], lr=lr, **optimizer_kwargs)
        elif optimizer == 'LBFGS':
            for k, v in zip(['history_size', 'max_iter'], [10, 4]):
                if k not in optimizer_kwargs:
                    optimizer_kwargs[k] = v
            self._optimizer = optim.LBFGS([self.synthesized_signal], lr=lr, **optimizer_kwargs)
            warnings.warn('This second order optimization method is more intensive')
            if hasattr(self, 'fraction_removed') and self.fraction_removed > 0:
                warnings.warn('For now the code is not designed to handle LBFGS and random'
                              ' subsampling of coeffs')
        elif optimizer == 'Adam':
            if 'amsgrad' not in optimizer_kwargs:
                optimizer_kwargs['amsgrad'] = True
            self._optimizer = optim.Adam([self.synthesized_signal], lr=lr, **optimizer_kwargs)
        elif optimizer == 'AdamW':
            if 'amsgrad' not in optimizer_kwargs:
                optimizer_kwargs['amsgrad'] = True
            self._optimizer = optim.AdamW([self.synthesized_signal], lr=lr, **optimizer_kwargs)
        else:
            raise Exception("Don't know how to handle optimizer %s!" % optimizer)
        self._swa = swa
        if swa:
            self._optimizer = torchcontrib.optim.SWA(self._optimizer, **swa_kwargs)
            warnings.warn("When using SWA, can't also use LR scheduler")
        else:
            if scheduler:
                self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, 'min', factor=.5)
            else:
                self._scheduler = None
        if not hasattr(self, '_init_optimizer_kwargs'):
            # this will only happen the first time _init_optimizer gets
            # called, and ensures that we can always re-initilize the
            # optimizer to the same state (mainly used to make sure that
            # the learning rate gets reset when we change target during
            # coarse-to-fine optimization). note that we use the
            # initial_lr here
            init_optimizer_kwargs = {'optimizer': optimizer, 'lr': initial_lr,
                                     'scheduler': scheduler, 'swa': swa,
                                     'swa_kwargs': swa_kwargs,
                                     'optimizer_kwargs': optimizer_kwargs}
            self._init_optimizer_kwargs = init_optimizer_kwargs
        if clip_grad_norm is True:
            self.clip_grad_norm = 1
        else:
            self.clip_grad_norm = clip_grad_norm

    def _closure(self):
        r"""An abstraction of the gradient calculation, before the optimization step.

        This enables optimization algorithms that perform several
        evaluations of the gradient before taking a step (ie. second
        order methods like LBFGS).

        Note that the fraction removed also happens here, and for now a
        fresh sample of noise is drawn at each iteration.
            1) that means for now we do not support LBFGS with a random
               fraction removed.
            2) beyond removing random fraction of the coefficients, one
               could schedule the optimization (eg. coarse to fine)

        Additionally, this is where:
        - ``synthesized_representation`` is updated
        - ``loss.backward()`` is called

        """
        self._optimizer.zero_grad()
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
        self.synthesized_representation = self.analyze(self.synthesized_signal, **analyze_kwargs)
        base_rep = self.analyze(self.base_signal, **analyze_kwargs)
        if self.store_progress:
            self.synthesized_representation.retain_grad()

        if self._use_subset_for_gradient:
            # here we get a boolean mask (bunch of ones and zeroes) for all
            # the statistics we want to include. We only do this if the loss
            # appears to be roughly unchanging for some number of iterations
            if (len(self.loss) >= self.loss_change_iter and
                (self.loss_change_iter is None or self.loss[-self.loss_change_iter] - self.loss[-1] < self.loss_change_thresh)):
                error_idx = self.representation_error(**analyze_kwargs).flatten().abs().argsort(descending=True)
                error_idx = error_idx[:int(self.loss_change_fraction * error_idx.numel())]
            # else, we use all of the statistics
            else:
                error_idx = torch.nonzero(torch.ones_like(self.synthesized_representation.flatten()))
            # for some reason, pytorch doesn't have the equivalent of
            # np.random.permutation, something that returns a shuffled copy
            # of a tensor, so we use numpy's version
            idx_shuffled = torch.LongTensor(np.random.permutation(to_numpy(error_idx)))
            # then we optionally randomly select some subset of those.
            idx_sub = idx_shuffled[:int((1 - self.fraction_removed) * idx_shuffled.numel())]
            synthesized_rep = self.synthesized_representation.flatten()[idx_sub]
            base_rep = base_rep.flatten()[idx_sub]
        else:
            synthesized_rep = self.synthesized_representation

        loss = self.objective_function(synthesized_rep, base_rep, self.synthesized_signal,
                                       self.base_signal)
        loss.backward(retain_graph=True)

        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_([self.synthesized_signal], self.clip_grad_norm)

        return loss

    def _optimizer_step(self, pbar=None, **kwargs):
        r"""Compute and propagate gradients, then step the optimizer to update synthesized_signal

        Parameters
        ----------
        pbar : tqdm.tqdm or None, optional
            A tqdm progress-bar, which we update with a postfix
            describing the current loss, gradient norm, and learning
            rate (it already tells us which iteration and the time
            elapsed). If None, then we don't display any progress
        kwargs :
            will also display in the progress bar's postfix

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
        self._last_iter_synthesized_signal = self.synthesized_signal.clone()
        postfix_dict = {}
        if self.coarse_to_fine:
            # the last scale will be 'all', and we never remove
            # it. Otherwise, check to see if it looks like loss has
            # stopped declining and, if so, switch to the next scale
            if (len(self.scales) > 1 and len(self.scales_loss) >= self.loss_change_iter and
                ((self.loss_change_thresh is None) or abs(self.scales_loss[-1] - self.scales_loss[-self.loss_change_iter]) < self.loss_change_thresh) and
                len(self.loss) - self.scales_timing[self.scales[0]][0] >= self.loss_change_iter):
                self.scales_timing[self.scales[0]].append(len(self.loss)-1)
                self.scales_finished.append(self.scales.pop(0))
                self.scales_timing[self.scales[0]].append(len(self.loss))
                # reset scheduler and optimizer.
                self._init_optimizer(**self._init_optimizer_kwargs)
            # we have some extra info to include in the progress bar if
            # we're doing coarse-to-fine
            postfix_dict['current_scale'] = self.scales[0]
        loss = self._optimizer.step(self._closure)
        # we have this here because we want to do the above checking at
        # the beginning of each step, before computing the loss
        # (otherwise there's an error thrown because self.scales[-1] is
        # not the same scale we computed synthesized_representation using)
        if self.coarse_to_fine:
            postfix_dict['current_scale_loss'] = loss.item()
            # and we also want to keep track of this
            self.scales_loss.append(loss.item())
        g = self.synthesized_signal.grad.detach()
        # optionally step the scheduler
        if self._scheduler is not None:
            self._scheduler.step(loss.item())

        if self.coarse_to_fine and self.scales[0] != 'all':
            with torch.no_grad():
                tmp_im = self.synthesized_signal.detach().clone()
                full_synthesized_rep = self.analyze(tmp_im)
                loss = self.objective_function(full_synthesized_rep, self.base_representation,
                                               self.synthesized_signal, self.base_signal)
        else:
            loss = self.objective_function(self.synthesized_representation, self.base_representation,
                                           self.synthesized_signal, self.base_signal)

        pixel_change = torch.max(torch.abs(self.synthesized_signal - self._last_iter_synthesized_signal))
        # for display purposes, always want loss to be positive
        postfix_dict.update(dict(loss="%.4e" % abs(loss.item()),
                                 gradient_norm="%.4e" % g.norm().item(),
                                 learning_rate=self._optimizer.param_groups[0]['lr'],
                                 pixel_change=f"{pixel_change:.04e}", **kwargs))
        # add extra info here if you want it to show up in progress bar
        if pbar is not None:
            pbar.set_postfix(**postfix_dict)
        return loss, g.norm(), self._optimizer.param_groups[0]['lr'], pixel_change

    @abc.abstractmethod
    def save(self, file_path, save_model_reduced=False, attrs=['model'],
             model_attr_names=['model']):
        r"""save all relevant variables in .pt file

        This is an abstractmethod only because you need to specify which
        attributes to save. See ``metamer.save`` as an example, but the
        save method in your synthesis object should probably should have
        a line defining the attributes to save and then a call to
        ``super().save(file_path, save_model_reduced, attrs)``

        Note that if store_progress is True, this will probably be very
        large

        Parameters
        ----------
        file_path : str
            The path to save the synthesis object to
        save_model_reduced : bool
            Whether we save the full model or just its attribute
            ``state_dict_reduced`` (this is a custom attribute of ours,
            the basic idea being that it only contains the attributes
            necessary to initialize the model, none of the (probably
            much larger) ones it gets during run-time).
        attrs : list
            List of strs containing the names of the attributes of this
            object to save.
        model_attr_names : list, optional
            The attribute that gives the model(s) names. Must be a list
            of strs. These are the attributes we try to save in reduced
            form if ``save_model_reduced`` is True.

        """
        save_dict = {}
        for name in model_attr_names:
            if name in attrs:
                model = getattr(self, name)
                try:
                    if save_model_reduced:
                        model = model.state_dict_reduced
                except AttributeError:
                    warnings.warn("self.model doesn't have a state_dict_reduced attribute, will pickle "
                                  "the whole model object")
                save_dict[name] = model
                attrs.remove(name)
        for k in attrs:
            attr = getattr(self, k)
            # detaching the tensors avoids some headaches like the
            # tensors having extra hooks or the like
            if isinstance(attr, torch.Tensor):
                attr = attr.detach()
            save_dict[k] = attr
        torch.save(save_dict, file_path, pickle_module=dill)

    @classmethod
    @abc.abstractmethod
    def load(cls, file_path, model_attr_name='model', model_constructor=None, map_location='cpu',
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
        model_attr_name : str or list, optional
            The attribute that gives the model(s) names. Can be a str or
            a list of strs. If a list and the reduced version of the
            model was saved, ``model_constructor`` should be a list of
            the same length.
        model_constructor : callable, list, or None, optional
            When saving the synthesis object, we have the option to only
            save the ``state_dict_reduced`` (in order to save space). If
            we do that, then we need some way to construct that model
            again and, not knowing its class or anything, this object
            doesn't know how. Therefore, a user must pass a constructor
            for the model that takes in the ``state_dict_reduced``
            dictionary and returns the initialized model. See the
            VentralModel class for an example of this. If a list, should
            be a list of the above and the same length as
            ``model_attr_name``
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
        synthesis : plenoptic.synth.Synthesis
            The loaded synthesis object


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

        >>> model = po.simul.PooledRGC(1)
        >>> metamer = po.synth.Metamer(img, model)
        >>> metamer.synthesize(max_iter=10, store_progress=True)
        >>> metamer.save('metamers.pt', save_model_reduced=True)
        >>> metamer_copy = po.synth.Metamer.load('metamers.pt',
                                                 model_constructor=po.simul.PooledRGC.from_state_dict_reduced)

        You may want to update one or more of the arguments used to
        initialize the model. The example I have in mind is where you
        run the metamer synthesis on a cluster but then load it on your
        local machine. The VentralModel classes have a ``cache_dir``
        attribute which you will want to change so it finds the
        appropriate location:

        >>> model = po.simul.PooledRGC(1)
        >>> metamer = po.synth.Metamer(img, model)
        >>> metamer.synthesize(max_iter=10, store_progress=True)
        >>> metamer.save('metamers.pt', save_model_reduced=True)
        >>> metamer_copy = po.synth.Metamer.load('metamers.pt',
                                                 model_constructor=po.simul.PooledRGC.from_state_dict_reduced,
                                                 cache_dir="/home/user/Desktop/metamers/windows_cache")

        """
        tmp_dict = torch.load(file_path, map_location=map_location, pickle_module=dill)
        device = torch.device(map_location)
        if not isinstance(model_attr_name, list):
            model_attr_name = [model_attr_name]
        if not isinstance(model_constructor, list):
            model_constructor = [model_constructor]
        base_signal = tmp_dict.pop('base_signal').to(device)
        models = {}
        for attr, constructor in zip(model_attr_name, model_constructor):
            model = tmp_dict.pop(attr)
            if isinstance(model, dict):
                for k, v in state_dict_kwargs.items():
                    warnings.warn("Replacing state_dict key %s, value %s with kwarg value %s" %
                                  (k, model.pop(k, None), v))
                    model[k] = v
                # then we've got a state_dict_reduced and we need the model_constructor
                model = constructor(model)
                # want to make sure the dtypes match up as well
                model = model.to(device, base_signal.dtype)
            models[attr] = model
        loss_function = tmp_dict.pop('loss_function', None)
        loss_function_kwargs = tmp_dict.pop('loss_function_kwargs', {})
        synth = cls(base_signal, loss_function=loss_function,
                    loss_function_kwargs=loss_function_kwargs, **models)
        for k, v in tmp_dict.items():
            setattr(synth, k, v)
        return synth

    @abc.abstractmethod
    def to(self, *args, attrs=[], **kwargs):
        r"""Moves and/or casts the parameters and buffers.

        Similar to ``save``, this is an abstract method only because you
        need to define the attributes to call to on.

        NOTE: We always call ``model.to(*args, **kwargs)`` (and thus
        ``'model'`` does not need to be in the ``attrs`` argument), but
        we only raise a Warning (not an Exception) if ``model`` does not
        have a ``to()`` method

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

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            attrs (:class:`list`): list of strs containing the attributes of
                this object to move to the specified device/dtype

        Returns:
            Module: self

        """
        try:
            self.model = self.model.to(*args, **kwargs)
        except AttributeError:
            warnings.warn("model has no `to` method, so we leave it as is...")
        for k in attrs:
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

    def plot_representation_error(self, batch_idx=0, iteration=None, figsize=(5, 5), ylim=None,
                                  ax=None, title=None, as_rgb=False):
        r"""Plot distance ratio showing how close we are to convergence.

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
            None, we create our own 1 subplot figure to hold it
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
        representation_error = self.representation_error(iteration=iteration)
        return plot_representation(self.model, representation_error, ax, figsize, ylim,
                                   batch_idx, title, as_rgb)

    def plot_loss(self, iteration=None, figsize=(5, 5), ax=None, title='Loss', **kwargs):
        """Plot the synthesis loss.

        We plot ``self.loss`` over all iterations. We also plot a red
        dot at ``iteration``, to highlight the loss there. If
        ``iteration=None``, then the dot will be at the final iteration.

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
        if iteration is None:
            loss_idx = len(self.loss) - 1
        else:
            if iteration < 0:
                # in order to get the x-value of the dot to line up,
                # need to use this work-around
                loss_idx = len(self.loss) + iteration
            else:
                loss_idx = iteration
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        ax.semilogy(self.loss, **kwargs)
        try:
            ax.scatter(loss_idx, self.loss[loss_idx], c='r')
        except IndexError:
            # then there's no loss here
            pass
        ax.set_title(title)
        return fig

    def plot_synthesized_image(self, batch_idx=0, channel_idx=None, iteration=None, title=None,
                               figsize=(5, 5), ax=None, imshow_zoom=None, vrange=(0, 1)):
        """Show the synthesized image.

        You can specify what iteration to view by using the ``iteration`` arg.
        The default, ``None``, shows the final one.

        We use ``plenoptic.imshow`` to display the synthesized image and
        attempt to automatically find the most reasonable zoom value. You can
        override this value using the imshow_zoom arg, but remember that
        ``plenoptic.imshow`` is opinionated about the size of the resulting
        image and will throw an Exception if the axis created is not big enough
        for the selected zoom.

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we assume
            image is RGB(A) and show all channels.
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
            attempt to find the best value ourselves, but we cannot find a
            value <1. Else, if >1, must be an integer. If <1, must be 1/d where
            d is a a divisor of the size of the largest image.
        vrange : tuple or str, optional
            The vrange option to pass to ``pyrtools.imshow``. See that
            function for details

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        if iteration is None:
            image = self.synthesized_signal
        else:
            image = self.saved_signal[iteration]
        if batch_idx is None:
            raise Exception("batch_idx must be an integer!")
        # we're only plotting one image here, so if the user wants multiple
        # channels, they must be RGB
        if channel_idx is None and image.shape[1] > 1:
            as_rgb = True
        else:
            as_rgb = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        if imshow_zoom is None:
            # image.shape[-2] is the height of the image
            imshow_zoom = ax.bbox.height // image.shape[-2]
            if imshow_zoom == 0:
                raise Exception("imshow_zoom would be 0, cannot display synthesized image! Enlarge"
                                " your figure")
        if title is None:
            title = self.__class__.__name__
        fig = imshow(image, ax=ax, title=title, zoom=imshow_zoom,
                     batch_idx=batch_idx, channel_idx=channel_idx,
                     vrange=vrange, as_rgb=as_rgb)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return fig

    def plot_image_hist(self, batch_idx=0, channel_idx=None, iteration=None, figsize=(5, 5),
                        ylim=None, ax=None, **kwargs):
        r"""Plot histogram of target and matched image.

        As a way to check the distributions of pixel intensities and see
        if there's any values outside the allowed range

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension
        channel_idx : int or None, optional
            Which index to take from the channel dimension. If None, we use all
            channels (assumed use-case is RGB(A) images).
        iteration : int or None, optional
            Which iteration to display. If None, the default, we show
            the most recent one. Negative values are also allowed.
        figsize : tuple, optional
            The size of the figure to create. Ignored if ax is not None
        ylim : tuple or None, optional
            if tuple, the ylimit to set for this axis. If None, we leave
            it untouched
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        kwargs :
            passed to plt.hist

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        def _freedman_diaconis_bins(a):
            """Calculate number of hist bins using Freedman-Diaconis rule. copied from seaborn"""
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
            image = self.synthesized_signal[batch_idx]
        else:
            image = self.saved_signal[iteration, batch_idx]
        base_signal = self.base_signal[batch_idx]
        if channel_idx is not None:
            image = image[channel_idx]
            base_signal = base_signal[channel_idx]
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        image = to_numpy(image).flatten()
        base_signal = to_numpy(base_signal).flatten()
        ax.hist(image, bins=min(_freedman_diaconis_bins(image), 50),
                label='synthesized image', **kwargs)
        ax.hist(base_signal, bins=min(_freedman_diaconis_bins(image), 50),
                label='base image', **kwargs)
        ax.legend()
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title("Histogram of pixel values")
        return fig

    def _grab_value_for_comparison(self, value, batch_idx=0, channel_idx=None,
                                   iteration=None, scatter_subsample=1,
                                   **kwargs):
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
        kwargs :
            passed to self.analyze

        Returns
        -------
        plot_vals : torch.Tensor
            4d tensor containing the base and synthesized value (indexed along
            last dimension). First two dims are dummy dimensions and will
            always have value 1 (update_plot needs them)

        """
        if value == 'representation':
            if iteration is not None:
                synthesized_val = self.saved_representation[iteration]
            else:
                synthesized_val = self.analyze(self.synthesized_signal, **kwargs)
            base_val = self.base_representation
        elif value == 'signal':
            if iteration is not None:
                synthesized_val = self.saved_signal[iteration]
            else:
                synthesized_val = self.synthesized_signal
            base_val = self.base_signal
        else:
            raise Exception(f"Don't know how to handle value {value}!")
        # if this is 4d, this will convert it to 3d (if it's 3d, nothing
        # changes)
        base_val = base_val.flatten(2, -1)
        synthesized_val = synthesized_val.flatten(2, -1)
        plot_vals = torch.stack((base_val, synthesized_val), -1)
        if scatter_subsample < 1:
            plot_vals = plot_vals[:, :, ::int(1/scatter_subsample)]
        plot_vals = plot_vals[batch_idx]
        if channel_idx is not None:
            plot_vals = plot_vals[channel_idx]
        else:
            plot_vals = plot_vals.flatten(0, 1)
        return plot_vals.unsqueeze(0).unsqueeze(0)


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
        if self._rep_warning and value=='representation':
            warnings.warn("Since at least one of your models is a metric, its representation"
                          " will be meaningless -- it will just show the pixel values"
                          ". (Your loss is still meaningful, however, since it's the actual "
                          "metric)")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'aspect': 1})
        else:
            fig = ax.figure
        plot_vals = to_numpy(self._grab_value_for_comparison(value, batch_idx,
                                                             channel_idx, iteration,
                                                             scatter_subsample,
                                                             **kwargs)).squeeze()
        if func == 'scatter':
            ax.scatter(plot_vals[..., 0], plot_vals[..., 1])
            ax.set(xlim=ax.get_ylim())
        elif func == 'hist2d':
            ax.hist2d(plot_vals[..., 0].flatten(), plot_vals[..., 1].flatten(),
                      bins=np.linspace(0, 1, hist2d_nbins),
                      cmap=hist2d_cmap, cmin=0)
        ax.set(ylabel=f'Synthesized {value}', xlabel=f'Base {value}')
        return fig

    def _setup_synthesis_fig(self, fig, axes_idx, figsize,
                             plot_synthesized_image=True, plot_loss=True,
                             plot_representation_error=True,
                             plot_image_hist=False, plot_rep_comparison=False,
                             plot_signal_comparison=False,
                             synthesized_image_width=1, loss_width=1,
                             representation_error_width=1, image_hist_width=1,
                             rep_comparison_width=1, signal_comparison_width=1):
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
        fig : matplotlib.pyplot.Figure or None
            The figure to plot on or None. If None, we create a new figure
        axes_idx : dict
            Dictionary specifying which axes contains which type of plot,
            allows for more fine-grained control of the resulting figure.
            Probably only helpful if fig is also defined. Possible keys: image,
            loss, rep_error, hist, rep_comp, signal_comp, misc. Values should
            all be ints. If you tell this function to create a plot that doesn't
            have a corresponding key, we find the lowest int that is not
            already in the dict, so if you have axes that you want unchanged,
            place their idx in misc.
        figsize : tuple or None, optional
            The size of the figure to create. It may take a little bit of
            playing around to find a reasonable value. If None, we attempt to
            make our best guess, aiming to have relative width=1 correspond to 5
        plot_synthesized_image : bool, optional
            Whether to include axis for plot of the synthesized image or not.
        plot_loss : bool, optional
            Whether to include axis for plot of the loss or not.
        plot_representation_error : bool, optional
            Whether to include axis for plot of the representation ratio or not.
        plot_image_hist : bool, optional
            Whether to include axis for plot of the histograms of image pixel
            intensities or not.
        plot_rep_comparison : bool, optional
            Whether to include axis for plot of a scatter plot comparing the
            synthesized and base representation.
        plot_signal_comparison : bool, optional
            Whether to include axis for plot of the comparison of the
            synthesized and base signal.
        synthesized_image_width : float, optional
            Relative width of the axis for the synthesized image.
        loss_width : float, optional
            Relative width of the axis for loss plot.
        representation_error_width : float, optional
            Relative width of the axis for representation error plot.
        image_hist_width : float, optional
            Relative width of the axis for image pixel intensities histograms.
        rep_comparison_width : float, optional
            Relative width of the axis for representation comparison plot.
        signal_comparison_width : float, optional
            Relative width of the axis for signal comparison plot.

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure to plot on
        axes : array_like
            List or array of axes contained in fig
        axes_idx : dict
            Dictionary identifying the idx for each plot type

        """
        n_subplots = 0
        axes_idx = axes_idx.copy()
        width_ratios = []
        if plot_synthesized_image:
            n_subplots += 1
            width_ratios.append(synthesized_image_width)
            if 'image' not in axes_idx.keys():
                axes_idx['image'] = _find_min_int(axes_idx.values())
        if plot_loss:
            n_subplots += 1
            width_ratios.append(loss_width)
            if 'loss' not in axes_idx.keys():
                axes_idx['loss'] = _find_min_int(axes_idx.values())
        if plot_representation_error:
            n_subplots += 1
            width_ratios.append(representation_error_width)
            if 'rep_error' not in axes_idx.keys():
                axes_idx['rep_error'] = _find_min_int(axes_idx.values())
        if plot_image_hist:
            n_subplots += 1
            width_ratios.append(image_hist_width)
            if 'hist' not in axes_idx.keys():
                axes_idx['hist'] = _find_min_int(axes_idx.values())
        if plot_rep_comparison:
            n_subplots += 1
            width_ratios.append(rep_comparison_width)
            if 'rep_comp' not in axes_idx.keys():
                axes_idx['rep_comp'] = _find_min_int(axes_idx.values())
        if plot_signal_comparison:
            n_subplots += 1
            width_ratios.append(signal_comparison_width)
            if 'signal_comp' not in axes_idx.keys():
                axes_idx['signal_comp'] = _find_min_int(axes_idx.values())
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

    def plot_synthesis_status(self, batch_idx=0, channel_idx=None, iteration=None,
                              figsize=None, ylim=None,
                              plot_synthesized_image=True, plot_loss=True,
                              plot_representation_error=True, imshow_zoom=None,
                              vrange=(0, 1), fig=None, plot_image_hist=False,
                              plot_rep_comparison=False,
                              plot_signal_comparison=False,
                              signal_comp_func='scatter',
                              signal_comp_subsample=.01, axes_idx={},
                              plot_representation_error_as_rgb=False):
        r"""Make a plot showing synthesis status.

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
        figsize : tuple or None, optional
            The size of the figure to create. It may take a little bit of
            playing around to find a reasonable value. If None, we attempt to
            make our best guess, aiming to have each axis be of size (5, 5)
        ylim : tuple or None, optional
            The ylimit to use for the representation_error plot. We pass
            this value directly to ``self.plot_representation_error``
        plot_synthesized_image : bool, optional
            Whether to plot the synthesized image or not.
        plot_loss : bool, optional
            Whether to plot the loss or not.
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

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        if iteration is not None and not self.store_progress:
            raise Exception("synthesis() was run with store_progress=False, "
                            "cannot specify which iteration to plot (only"
                            " last one, with iteration=None)")
        if self.synthesized_signal.ndim not in [3, 4]:
            raise Exception("plot_synthesis_status() expects 3 or 4d data;"
                            "unexpected behavior will result otherwise!")
        fig, axes, axes_idx = self._setup_synthesis_fig(fig, axes_idx, figsize,
                                                        plot_synthesized_image,
                                                        plot_loss,
                                                        plot_representation_error,
                                                        plot_image_hist,
                                                        plot_rep_comparison,
                                                        plot_signal_comparison)

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

        if plot_synthesized_image:
            self.plot_synthesized_image(batch_idx=batch_idx,
                                        channel_idx=channel_idx,
                                        iteration=iteration, title=None,
                                        ax=axes[axes_idx['image']],
                                        imshow_zoom=imshow_zoom, vrange=vrange)
        if plot_loss:
            self.plot_loss(iteration=iteration, ax=axes[axes_idx['loss']])
        if plot_representation_error:
            fig = self.plot_representation_error(batch_idx=batch_idx,
                                                 iteration=iteration,
                                                 ax=axes[axes_idx['rep_error']],
                                                 ylim=ylim,
                                                 as_rgb=plot_representation_error_as_rgb)
            # this can add a bunch of axes, so this will try and figure
            # them out
            new_axes = [i for i, _ in enumerate(fig.axes) if not
                        check_iterables(i, axes_idx.values())] + [axes_idx['rep_error']]
            axes_idx['rep_error'] = new_axes
        if plot_image_hist:
            fig = self.plot_image_hist(batch_idx=batch_idx,
                                       channel_idx=channel_idx,
                                       iteration=iteration,
                                       ax=axes[axes_idx['hist']])
        if plot_rep_comparison:
            fig = self.plot_value_comparison(value='representation',
                                             batch_idx=batch_idx,
                                             channel_idx=channel_idx,
                                             iteration=iteration,
                                             ax=axes[axes_idx['rep_comp']])
            # this can add some axes, so this will try and figure them out
            new_axes = [i for i, _ in enumerate(fig.axes) if not
                        check_iterables(i, axes_idx.values())] + [axes_idx['rep_comp']]
            axes_idx['rep_comp'] = new_axes
        if plot_signal_comparison:
            fig = self.plot_value_comparison(value='signal',
                                             batch_idx=batch_idx,
                                             channel_idx=channel_idx,
                                             iteration=iteration,
                                             ax=axes[axes_idx['signal_comp']],
                                             func=signal_comp_func,
                                             scatter_subsample=signal_comp_subsample)
        self._axes_idx = axes_idx
        return fig

    def animate(self, batch_idx=0, channel_idx=None, figsize=None,
                framerate=10, ylim='rescale', plot_synthesized_image=True,
                plot_loss=True, plot_representation_error=True,
                imshow_zoom=None, plot_data_attr=['loss'], rep_func_kwargs={},
                plot_image_hist=False, plot_rep_comparison=False,
                plot_signal_comparison=False, fig=None,
                signal_comp_func='scatter', signal_comp_subsample=.01,
                axes_idx={}, init_figure=True,
                plot_representation_error_as_rgb=False):
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

        Parameters
        ----------
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
        plot_data_attr : list, optional
            list of strs giving the names of the attributes with data
            plotted on the second subplot. this allows us to update
            whatever is in there if your plot_synthesis_status() plots
            something other than loss or if you plotted more than one
            attribute (e.g., MADCompetition plots two losses)
        rep_func_kwargs : dict, optional
            a dictionary of additional kwargs to pass through to the repeated
            calls to representation_error() or _grab_value_for_comparison()
            (for plotting representation error and representation comparison,
            respectively)
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

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object. In order to view, must convert to HTML
            or save.

        """
        if not self.store_progress:
            raise Exception("synthesize() was run with store_progress=False,"
                            " cannot animate!")
        if self.saved_representation is not None and len(self.saved_signal) != len(self.saved_representation):
            raise Exception("saved_signal and saved_representation need to be the same length in "
                            "order for this to work!")
        if self.synthesized_signal.ndim not in [3, 4]:
            raise Exception("animate() expects 3 or 4d data; unexpected"
                            " behavior will result otherwise!")
        # every time we call synthesize(), store_progress gets one extra
        # element compared to loss. this uses that fact to figure out
        # how many times we've called sythesize())
        times_called = ((self.saved_signal.shape[0] * self.store_progress - len(self.loss)) //
                        self.store_progress)
        # which we use in order to pad out the end of plot_data so that
        # the lengths work out correctly (technically, should be
        # inserting this at the moments synthesize() was called, but I
        # don't know how to figure that out and the difference shouldn't
        # be noticeable except in extreme circumstances, e.g., you
        # called synthesize(max_iter=5) 100 times).

        plot_data = [getattr(self, d) + self.store_progress*times_called*[getattr(self, d)[-1]]
                     for d in plot_data_attr]
        if self.base_representation.ndimension() == 4:
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
                    ylim_rescale_interval = int((self.saved_representation.shape[0] - 1) // 10)
                    if ylim_rescale_interval == 0:
                        ylim_rescale_interval = int(self.saved_representation.shape[0] - 1)
                ylim = None
            else:
                raise Exception("Don't know how to handle ylim %s!" % ylim)
        except AttributeError:
            # this way we'll never rescale
            ylim_rescale_interval = len(self.saved_signal)+1
        if init_figure:
            # initialize the figure
            fig = self.plot_synthesis_status(batch_idx=batch_idx, channel_idx=channel_idx,
                                             iteration=0, figsize=figsize, ylim=ylim,
                                             plot_loss=plot_loss,
                                             plot_representation_error=plot_representation_error,
                                             imshow_zoom=imshow_zoom, fig=fig,
                                             plot_synthesized_image=plot_synthesized_image,
                                             plot_image_hist=plot_image_hist,
                                             plot_signal_comparison=plot_signal_comparison,
                                             plot_rep_comparison=plot_rep_comparison,
                                             signal_comp_func=signal_comp_func,
                                             signal_comp_subsample=signal_comp_subsample,
                                             axes_idx=axes_idx,
                                             plot_representation_error_as_rgb=plot_representation_error_as_rgb)
            # plot_synthesis_status creates a hidden attribute, _axes_idx, a dict
            # which tells us which axes contains which plot
            axes_idx = self._axes_idx
        # grab the artists for the second plot (we don't need to do this
        # for the synthesized image or representation plot, because we
        # use the update_plot function for that)
        if plot_loss:
            scat = fig.axes[axes_idx['loss']].collections
        # can have multiple plots
        if plot_representation_error:
            try:
                rep_error_axes = [fig.axes[i] for i in axes_idx['rep_error']]
            except TypeError:
                # in this case, axes_idx['rep_error'] is not iterable and so is
                # a single value
                rep_error_axes = [fig.axes[axes_idx['rep_error']]]
        else:
            rep_error_axes = []
        # can also have multiple plots
        if plot_rep_comparison:
            try:
                rep_comp_axes = [fig.axes[i] for i in axes_idx['rep_comp']]
            except TypeError:
                # in this case, axes_idx['rep_comp'] is not iterable and so is
                # a single value
                rep_comp_axes = [fig.axes[axes_idx['rep_comp']]]
        else:
            rep_comp_axes = []

        if self.base_representation.ndimension() == 4:
            warnings.warn("Looks like representation is image-like, haven't fully thought out how"
                          " to best handle rescaling color ranges yet!")
            # replace the bit of the title that specifies the range,
            # since we don't make any promises about that. we have to do
            # this here because we need the figure to have been created
            for ax in rep_error_axes:
                ax.set_title(re.sub(r'\n range: .* \n', '\n\n', ax.get_title()))

        def movie_plot(i):
            artists = []
            if plot_synthesized_image:
                artists.extend(update_plot(fig.axes[axes_idx['image']],
                                           data=self.saved_signal[i],
                                           batch_idx=batch_idx))
            if plot_representation_error:
                representation_error = self.representation_error(iteration=i,
                                                                 **rep_func_kwargs)
                # we pass rep_error_axes to update, and we've grabbed
                # the right things above
                artists.extend(update_plot(rep_error_axes,
                                           batch_idx=batch_idx,
                                           model=self.model,
                                           data=representation_error))
                # again, we know that rep_error_axes contains all the axes
                # with the representation ratio info
                if ((i+1) % ylim_rescale_interval) == 0:
                    if self.base_representation.ndimension() == 3:
                        rescale_ylim(rep_error_axes,
                                     representation_error)
            if plot_image_hist:
                # this is the dumbest way to do this, but it's simple --
                # clearing the axes can cause problems if the user has, for
                # example, changed the tick locator or formatter. not sure how
                # to handle this best right now
                fig.axes[axes_idx['hist']].clear()
                self.plot_image_hist(batch_idx=batch_idx,
                                     channel_idx=channel_idx, iteration=i,
                                     ax=fig.axes[axes_idx['hist']])
            if plot_signal_comparison:
                if signal_comp_func == 'hist2d':
                    # this is the dumbest way to do this, but it's simple --
                    # clearing the axes can cause problems if the user has, for
                    # example, changed the tick locator or formatter. not sure how
                    # to handle this best right now
                    fig.axes[axes_idx['signal_comp']].clear()
                    self.plot_value_comparison(value='signal', batch_idx=batch_idx,
                                               channel_idx=channel_idx, iteration=i,
                                               ax=fig.axes[axes_idx['signal_comp']],
                                               func=signal_comp_func)
                else:
                    plot_vals = self._grab_value_for_comparison('signal',
                                                                batch_idx,
                                                                channel_idx, i,
                                                                signal_comp_subsample)
                    artists.extend(update_plot(fig.axes[axes_idx['signal_comp']],
                                               plot_vals))
            if plot_loss:
                # loss always contains values from every iteration, but
                # everything else will be subsampled
                for s, d in zip(scat, plot_data):
                    s.set_offsets((i*self.store_progress, d[i*self.store_progress]))
                artists.extend(scat)
            if plot_rep_comparison:
                plot_vals = self._grab_value_for_comparison('representation',
                                                            batch_idx, channel_idx,
                                                            i, **rep_func_kwargs)
                artists.extend(update_plot(rep_comp_axes, plot_vals))
            # as long as blitting is True, need to return a sequence of artists
            return artists


        # don't need an init_func, since we handle initialization ourselves
        anim = animation.FuncAnimation(fig, movie_plot, frames=len(self.saved_signal),
                                       blit=True, interval=1000./framerate, repeat=False)
        plt.close(fig)
        return anim
