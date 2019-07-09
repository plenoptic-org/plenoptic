import torch
import warnings
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import numpy as np
import time
from torch.optim import lr_scheduler


class Metamer(nn.Module):
    """Synthesize metamers for image-computable differentiable models!

    Following the basic idea in [1]_, this module creates a metamer for a given model on a given
    image. We start with some random noise (typically, though users can choose to start with
    something else) and iterative adjust the pixel values so as to match the representation of this
    metamer-to-be and the ``target_image``. This is optimization though, so you'll probably need to
    experiment with the optimization hyper-parameters before you find a good solution.

    Currently we do not: support batch creation of images.

    Parameters
    ----------
    target_image : torch.tensor or array_like
        A 2d tensor, this is the image whose representation we wish to match. If this is not a
        tensor, we try to cast it as one.
    model : torch.nn.Module
        A differentiable model that takes an image as an input and transforms it into a
        representation of some sort. We only require that it has a forward method, which returns
        the representation to match.

    Attributes
    ----------
    target_image : torch.tensor
        A 2d tensor, this is the image whose representation we wish to match.
    model : torch.nn.Module
        A differentiable model that takes an image as an input and transforms it into a
        representation of some sort. We only require that it has a forward method, which returns
        the representation to match.
    target_representation : torch.tensor
        Whatever is returned by ``model.foward(target_image)``, this is what we match in order to
        create a metamer
    matched_image : torch.tensor
        The metamer. This may be unfinished depending on how many iterations we've run for.
    matched_represetation: torch.tensor
        Whatever is returned by ``model.forward(matched_image)``; we're trying to make this
        identical to ``self.target_representation``
    optimizer : torch.optim.Optimizer
        A pytorch optimization method. Currently, user cannot specify the method they want, and we
        use SGD (stochastic gradient descent).
    scheduler : torch.optim.lr_scheduler._LRScheduler
        A pytorch scheduler, which tells us how to change the learning rate over
        iterations. Currently, user cannot set and we use ReduceLROnPlateau (so that the learning
        rate gets reduced if it seems like we're on a plateau i.e., the loss isn't changing much)
    loss : list
        A list of our loss over time.
    saved_representation : torch.tensor
        If the ``save_representation`` arg in ``synthesize`` is set to True, we will save
        ``self.matched_representation`` at each iteration, for later examination.
    saved_image : torch.tensor
        If the ``save_image`` arg in ``synthesize`` is set to True, we will save
        ``self.matched_image`` at each iteration, for later examination.
    time : list
        A list of time, in seconds, relative to the most recent call to ``synthesize``.
    seed : int
        Number with which to seed pytorch and numy's random number generators

    References
    -----
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model based on Joint Statistics of
       Complex Wavelet Coefficients. Int'l Journal of Computer Vision. 40(1):49-71, October, 2000.
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
    - [ ] add example for load method
    - [ ] add animate method, which creates a three-subplot animation: the metamer over time, the
          plot of differences in representation over time, and the loss over time (as a red point
          on the loss curve) -- some models' representation might not be practical to plot, add the
          ability to take a function for the plot representation and if it's set to None, don't
          plot anything; make this a separate class or whatever because we'll want to be able to do
          this for eigendistortions, etc (this will require standardizing our API, which we want to
          do anyway)
    - [x] how to handle device? -- get rid of device in here, expect the user to set .to(device)
          (and then check self.target_image.device when initializing any tensors)
    - [ ] how do we handle continuation? right now the way to do it is to just pass matched_im
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
        self.target_representation = self.analyze(self.target_image)
        self.matched_image = None
        self.matched_representation = None
        self.optimizer = None
        self.scheduler = None
        self.loss = []
        self.saved_representation = torch.empty((0, *self.target_representation.shape))
        self.saved_image = torch.empty((0, *self.target_image.shape))
        self.time = []
        self.seed = None

    def analyze(self, x):
        """Analyze the image, that is, obtain the model's representation of it

        Note: analysis is applied on the squished input, so as to softly enforce the desired range
        during the optimization, (as a consequence, there is a discrepency to corresponding
        statistics during the optimization- which is fixed at the moment of returning an output in
        the synthesize method)

        """
        y = self.model(x)
        if isinstance(y, list):
            return torch.cat([s.squeeze().view(-1) for s in y]).unsqueeze(1)
        else:
            return y

    def objective_function(self, x, y):
        """Calculate the loss between x and y

        This is what we minimize. Currently it's the L2-norm
        """
        return torch.norm(x - y, p=2)

    def _optimizer_step(self, pbar):
        """step the optimizer, propagating the gradients, and updating our matched_image

        Parameters
        ----------
        pbar : tqdm.tqdm
            A tqdm progress-bar, which we update with a postfix describing the current loss,
            gradient norm, and learning rate (it already tells us which iteration and the time
            elapsed)

        Returns
        -------
        loss : torch.tensor
            1-element tensor containing the loss on this step

        """
        self.optimizer.zero_grad()
        self.matched_representation = self.analyze(self.matched_image)
        loss = self.objective_function(self.matched_representation, self.target_representation)
        loss.backward(retain_graph=True)
        g = self.matched_image.grad.data
        self.optimizer.step()
        # add extra info here if you want it to show up in progress bar
        pbar.set_postfix(loss="%.4e" % loss.item(), gradient_norm="%.4e" % g.norm().item(),
                         learning_rate=self.optimizer.param_groups[0]['lr'])
        return loss

    def synthesize(self, seed=0, learning_rate=.01, max_iter=100, initial_image=None,
                   clamper=None, save_representation=False, save_image=False, loss_thresh=1e-4):
        """synthesize a metamer

        This is the main method, trying to update the ``initial_image`` until its representation
        matches that of ``target_image``. If ``initial_image`` is not set, we initialize with
        uniformly-distributed random noise between 0 and 1. NOTE: This means that the value of
        ``target_image`` should probably lie between 0 and 1. If that's not the case, you might
        want to pass something to act as the initial image.

        We run this until either we reach ``max_iter`` or loss is below ``loss_thresh``, whichever
        comes first

        Note that you can run this several times in sequence by setting ``initial_image`` to the
        ``matched_image`` we return

        Parameters
        ----------
        seed : int, optional
            Number with which to seed pytorch and numy's random number generators
        learning_rate : float, optional
            The learning rate for our optimizer
        max_iter : int, optinal
            The maximum number of iterations to run before we end
        initial_image : torch.tensor, array_like, or None, optional
            The 2d tensor we use to initialize the metamer. If None (the default), we initialize
            with uniformly-distributed random noise lying between 0 and 1. If this is not a tensor
            or None, we try to cast it as a tensor.
        clamper : plenoptic.Clamper or None, optional
            Clamper makes a change to the image in order to ensure that it stays reasonable. The
            classic example is making sure the range lies between 0 and 1, see
            plenoptic.RangeClamper for an example.
        save_representation : bool, optional
            Whether we should save the representation of the metamer in progress on every
            iteration. If yes, ``self.saved_representation`` contains the saved representations.
        save_image : bool, optional
            Whether we should save the metamer in progress on every iteration. If yes,
            ``self.saved_image`` contains the saved images.
        loss_thresh : float, optional
            The value of the loss function that we consider "good enough", at which point we stop
            optimizing

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

        start_time = time.time()

        if initial_image is None:
            self.matched_image = torch.rand_like(self.target_image, dtype=torch.float32,
                                                 device=self.target_image.device)
            self.matched_image.requires_grad = True
        else:
            if not isinstance(initial_image, torch.Tensor):
                initial_image = torch.tensor(initial_image, dtype=torch.float32,
                                             device=self.target_image.device)
            self.matched_image = torch.nn.Parameter(initial_image, requires_grad=True)

        # self.optimizer = optim.Adam([self.matched_image], lr=learning_rate, amsgrad=True)
        self.optimizer = optim.SGD([self.matched_image], lr=learning_rate, momentum=0.8)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=.2)

        self.matched_representation = self.analyze(self.matched_image)
        # there's a +1 in each of thes because we want to save the initial state as well
        if save_representation:
            self.saved_representation = torch.empty((max_iter+1,
                                                     *self.target_representation.shape))
            self.saved_representation[0, :] = self.analyze(self.matched_image)
        if save_image:
            self.saved_image = torch.empty((max_iter+1, *self.target_image.shape))
            self.saved_image[0, :] = self.matched_image

        with torch.no_grad():
            self.loss.append(self.objective_function(self.matched_representation,
                                                     self.target_representation).item())
        self.time.append(time.time() - start_time)

        pbar = tqdm(range(max_iter))

        for i in pbar:
            loss = self._optimizer_step(pbar)
            if np.isnan(loss.item()):
                warnings.warn("Loss is NaN, quitting out!")
                break
            self.loss.append(loss.item())
            self.time.append(time.time() - start_time)

            self.scheduler.step(loss.item())

            with torch.no_grad():
                if clamper is not None:
                    self.matched_image.data = clamper.clamp(self.matched_image.data)

                if save_representation:
                    # save stats from this step
                    self.saved_representation[i+1, :] = self.analyze(self.matched_image)

                if save_image:
                    # save stats from this step
                    self.saved_image[i+1, :] = self.matched_image

            if loss.item() < loss_thresh:
                break

        pbar.close()
        # drop any empty columns (that is, if we don't reach the max iterations, don't want to hold
        # onto these zeroes). we go to i+2 so we include the first entry (which is the initial
        # state of things) and the last one of interest (in python, a[:i] gives you from 0 to i-1)
        if save_representation:
            self.saved_representation = self.saved_representation[:i+2, :]
        if save_image:
            self.saved_image = self.saved_image[:i+2, :]
        return self.matched_image.data.squeeze(), self.matched_representation.data.squeeze()

    def save(self, file_path):
        """save all relevant variables in .pt file

        Note that if save_representation and save_image are True, this will probably be very large

        Parameters
        ----------
        file_path : str
            The path to save the metamer object to
        """
        torch.save({'matched_image': self.matched_image, 'target_image': self.target_image,
                    'model': self.model, 'seed': self.seed, 'time': self.time, 'loss': self.loss,
                    'target_representation': self.target_representation,
                    'matched_representation': self.matched_representation,
                    'saved_representation': self.saved_representation,
                    'saved_image': self.saved_image}, file_path)

    @classmethod
    def load(cls, file_path):
        """load all relevant stuff from a .pt file

        Parameters
        ----------
        file_path : str
            The path to load the metamer object from

        Returns
        -------
        metamer : plenoptic.synth.Metamer
            The loaded metamer object

        """
        tmp_dict = torch.load(file_path)
        metamer = cls(tmp_dict.pop('target_image'), tmp_dict.pop('model'))
        for k, v in tmp_dict.items():
            setattr(metamer, k, v)
        return metamer
