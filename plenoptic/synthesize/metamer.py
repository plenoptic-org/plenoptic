import torch
import warnings
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import pyrtools as pt
from ..tools.display import rescale_ylim
from ..tools.data import to_numpy
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
        A pytorch optimization method. Currently, user cannot specify
        the method they want, and we use SGD (stochastic gradient
        descent).
    scheduler : torch.optim.lr_scheduler._LRScheduler
        A pytorch scheduler, which tells us how to change the learning
        rate over iterations. Currently, user cannot set and we use
        ReduceLROnPlateau (so that the learning rate gets reduced if it
        seems like we're on a plateau i.e., the loss isn't changing
        much)
    loss : list
        A list of our loss over iterations.
    saved_representation : torch.tensor
        If the ``store_progress`` arg in ``synthesize`` is set to
        True or an int>0, we will save ``self.matched_representation``
        at each iteration, for later examination.
    saved_image : torch.tensor
        If the ``store_progress`` arg in ``synthesize`` is set to True
        or an int>0, we will save ``self.matched_image`` at each
        iteration, for later examination.  seed : int Number with which
        to seed pytorch and numy's random number generators

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

        self.loss = []
        self.saved_representation = []
        self.saved_image = []

    def analyze(self, x):
        r"""Analyze the image, that is, obtain the model's representation of it

        """
        y = self.model(x)
        if isinstance(y, list):
            return torch.cat([s.squeeze().view(-1) for s in y]).unsqueeze(1)
        else:
            return y

    def objective_function(self, x, y):
        r"""Calculate the loss between x and y

        This is what we minimize. Currently it's the L2-norm
        """
        return torch.norm(x - y, p=2)

    def _optimizer_step(self, pbar):
        r"""step the optimizer, propagating the gradients, and updating our matched_image

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

        """
        self.optimizer.zero_grad()
        self.matched_representation = self.analyze(self.matched_image)
        # TODO randomness

        loss = self.objective_function(self.matched_representation, self.target_representation)
        loss.backward(retain_graph=True)
        g = self.matched_image.grad.data
        self.optimizer.step()
        self.scheduler.step(loss.item())

        # add extra info here if you want it to show up in progress bar
        pbar.set_postfix(loss="%.4e" % loss.item(), gradient_norm="%.4e" % g.norm().item(),
                         learning_rate=self.optimizer.param_groups[0]['lr'])
        return loss

    def synthesize(self, seed=0, learning_rate=.01, max_iter=100, initial_image=None,
                   clamper=None, store_progress=False, loss_thresh=1e-4, save_progress=False,
                   save_path='metamer.pt'):
        r"""synthesize a metamer

        This is the main method, trying to update the ``initial_image``
        until its representation matches that of ``target_image``. If
        ``initial_image`` is not set, we initialize with
        uniformly-distributed random noise between 0 and 1. NOTE: This
        means that the value of ``target_image`` should probably lie
        between 0 and 1. If that's not the case, you might want to pass
        something to act as the initial image.

        We run this until either we reach ``max_iter`` or loss is below
        ``loss_thresh``, whichever comes first

        Note that you can run this several times in sequence by setting
        ``initial_image`` to the ``matched_image`` we return. Everything
        that stores the progress of the optimization (``loss``,
        ``saved_representation``, ``saved_image``) will persist between
        calls and so potentially get very large.

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
        store_progress : bool or int, optional
            Whether we should store the representation of the metamer
            and the metamer image in progress on every iteration. If
            False, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress``
            iterations (note then that 0 is the same as False and 1 the
            same as True). If True or int>0, ``self.saved_image``
            contains the stored images, and ``self.saved_representation
            contains the stored representations.
        loss_thresh : float, optional
            The value of the loss function that we consider "good
            enough", at which point we stop optimizing
        save_progress : bool, optional
            Whether to save the metamer as we go (so that you can check
            it periodically and so you don't lose everything if you have
            to kill the job / it dies before it finishes running). If
            True, we save to ``save_path`` every time we update the
            saved_representation. We attempt to save with the
            ``save_model_reduced`` flag set to True
        save_path : str, optional
            The path to save the synthesis-in-progress to (ignored if
            ``save_progress`` is False)

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
        # self.optimizer = optim.Adam([self.matched_image], lr=learning_rate, amsgrad=True)
        self.optimizer = optim.SGD([self.matched_image], lr=learning_rate, momentum=0.8)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=.2)

        # python's implicit boolean-ness means we can do this! it will evaluate to False for False
        # and 0, and True for True and every int >= 1
        if store_progress:
            if store_progress is True:
                store_progress = 1
            self.saved_image.append(self.matched_image.clone())
            self.saved_representation.append(self.analyze(self.matched_image))
        else:
            if save_progress:
                raise Exception("Can't save progress if we're not storing it! If save_progress is"
                                " True, store_progress must be not False")

        pbar = tqdm(range(max_iter))

        for i in pbar:
            loss = self._optimizer_step(pbar)
            self.loss.append(loss.item())
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
                # started to have NaN values
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
                    self.saved_image.append(self.matched_image.clone())
                    self.saved_representation.append(self.analyze(self.matched_image))
                    if save_progress:
                        self.save(save_path, True)

            if loss.item() < loss_thresh:
                break

        pbar.close()

        if store_progress:
            self.saved_representation = torch.stack(self.saved_representation)
            self.saved_image = torch.stack(self.saved_image)
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
        torch.save({'matched_image': self.matched_image, 'target_image': self.target_image,
                    'model': model, 'seed': self.seed, 'loss': self.loss,
                    'target_representation': self.target_representation,
                    'matched_representation': self.matched_representation,
                    'saved_representation': self.saved_representation,
                    'saved_image': self.saved_image}, file_path)

    @classmethod
    def load(cls, file_path, model_constructor=None):
        r"""load all relevant stuff from a .pt file

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

        """
        tmp_dict = torch.load(file_path)
        model = tmp_dict.pop('model')
        if isinstance(model, dict):
            # then we've got a state_dict_reduced and we need the model_constructor
            model = model_constructor(model)
        metamer = cls(tmp_dict.pop('target_image'), model)
        for k, v in tmp_dict.items():
            setattr(metamer, k, v)
        return metamer

    def representation_ratio(self, iteration=None):
        r"""Get the representation ratio

        This is (matched_representation - target_representation) /
        target_representation. If ``iteration`` is not None, we use
        ``self.saved_representation[iteration]`` for
        matched_representation

        Parameters
        ----------
        iteration: int or None, optional
            Which iteration to create the representation ratio for. If
            None, we use the current ``matched_representation``

        Returns
        -------
        np.array

        """
        if iteration is not None:
            matched_rep = self.saved_representation[iteration]
        else:
            matched_rep = self.matched_representation
        representation_ratio = to_numpy(((matched_rep - self.target_representation) /
                                         self.target_representation))
        representation_ratio[np.isnan(representation_ratio)] = 0
        return representation_ratio

    def plot_representation_ratio(self, batch_idx=0, iteration=None, figsize=(5, 5), ylim=None,
                                  ax=None, title=None):
        r"""Plot distance ratio showing how close we are to convergence

        We plot ``self.representation_ratio(iteration)``

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
            limit of ``np.abs(representation_ratio).max()``
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
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            warnings.warn("ax is not None, so we're ignoring figsize...")
        representation_ratio = self.representation_ratio(iteration)
        try:
            fig, axes = self.model.plot_representation(figsize, ylim, ax, title, batch_idx,
                                                       data=representation_ratio)
        except AttributeError:
            ax.plot(representation_ratio)
            fig = ax.figure
            axes = [ax]
        if ylim is None:
            rescale_ylim(axes, representation_ratio)
        return fig

    def plot_metamer_status(self, batch_idx=0, channel_idx=0, iteration=None, figsize=(17, 5),
                            ylim=None, plot_representation_ratio=True, imshow_zoom=None):
        r"""Make a plot showing metamer, loss, and (optionally) representation ratio

        We create two or three subplots on a new figure. The first one
        contains the metamer, the second contains the loss, and the
        (optional) third contains the representation ratio, as plotted
        by ``self.plot_representation_ratio``.

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
            The ylimit to use for the representation_ratio plot. We pass
            this value directly to ``self.plot_representation_ratio``
        plot_representation_ratio : bool, optional
            Whether to plot the representation ratio or not.
        imshow_zoom : None or float, optional
            How much to zoom in / enlarge the metamer image, the ratio
            of display pixels to image pixels. If None (the default), we
            attempt to find the best value ourselves. Else, if >1, must
            be an integer.  If <1, must be 1/d where d is a a divisor of
            the size of the largest image.

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure containing this plot

        """
        if plot_representation_ratio:
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
            imshow_zoom = axes[0].bbox.width // image.shape[0]
            if imshow_zoom == 0:
                raise Exception("imshow_zoom would be 0, cannot display metamer image! Enlarge "
                                "your figure")
        fig = pt.imshow(to_numpy(image), ax=axes[0], title='Metamer', zoom=imshow_zoom)
        axes[0].xaxis.set_visible(False)
        axes[0].yaxis.set_visible(False)
        axes[1].semilogy(self.loss)
        axes[1].scatter(loss_idx, self.loss[loss_idx], c='r')
        axes[1].set_title('Loss')
        if plot_representation_ratio:
            fig = self.plot_representation_ratio(batch_idx, iteration, ax=axes[2], ylim=ylim)
        return fig

    def animate(self, batch_idx=0, channel_idx=0, figsize=(17, 5), framerate=10, ylim='rescale',
                plot_representation_ratio=True):
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

        NOTE: This requires that the model has ``_update_plot``, and
        ``plot_representation`` functions in order to work nicely. It
        will work otherwise, but we'll just create a simple line plot

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
            The y-limits of the representation_ratio plot (ignored if
            ``plot_representation_ratio`` arg is False).

            * If a tuple, then this is the ylim of all plots

            * If None, then all plots have the same limits, all
              symmetric about 0 with a limit of
              ``np.abs(representation_ratio).max()`` (for the initial
              representation_ratio)

            * If a string, must be 'rescale' or of the form 'rescaleN',
              where N can be any integer. If 'rescaleN', we rescale the
              limits every N frames (we rescale as if ylim = None). If
              'rescale', then we do this 10 times over the course of the
              animation

        plot_representation_ratio : bool, optional
            Whether to plot the representation ratio or not.

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
        images = self.saved_image[:, batch_idx, channel_idx]
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
            ylim_rescale_interval = len(images)+1
        # initialize the figure
        fig = self.plot_metamer_status(batch_idx, channel_idx, 0, figsize, ylim,
                                       plot_representation_ratio)
        # grab the artists for the first two plots (we don't need to do
        # this for the representation plot, because the model has an
        # _update_plot method that handles this for us)
        image_artist = fig.axes[0].images[0]
        scat = fig.axes[1].collections[0]

        def movie_plot(i):
            artists = []
            image_artist.set_data(to_numpy(images[i]))
            artists.append(image_artist)
            if plot_representation_ratio:
                representation_ratio = self.representation_ratio(i)
                try:
                    # we know that the first two axes are the image and
                    # loss, so we pass everything after that to update
                    rep_artists = self.model._update_plot(fig.axes[2:], batch_idx,
                                                          data=representation_ratio)
                    try:
                        # if this is a list, we just want to include its
                        # members (not include a list of its members)...
                        artists.extend(rep_artists)
                    except TypeError:
                        # but if it's not a list, we just want the one
                        # artist
                        artists.append(rep_artists)
                except AttributeError:
                    artists.append(fig.axes[2].lines[0])
                    artists[-1].set_ydata(representation_ratio)
                # again, we know that fig.axes[2:] contains all the axes
                # with the representation ratio info
                if ((i+1) % ylim_rescale_interval) == 0:
                    rescale_ylim(fig.axes[2:], representation_ratio)
            # loss always contains values from every iteration, but
            # everything else will be subsampled
            scat.set_offsets((i*saved_subsample, loss[i*saved_subsample]))
            artists.append(scat)
            # as long as blitting is True, need to return a sequence of artists
            return artists

        # don't need an init_func, since we handle initialization ourselves
        anim = animation.FuncAnimation(fig, movie_plot, frames=len(images),
                                       blit=True, interval=1000./framerate, repeat=False)
        plt.close(fig)
        return anim
