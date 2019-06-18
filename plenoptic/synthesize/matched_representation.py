import torch
import torch.nn as nn
from torch import optim
from ..tools.signal import rescale
from ..tools.fit import pretty_print, stretch
import numpy as np
import time


class Matched_representation(nn.Module):
    """
    TODO one sentence description

    Parameters
    ----------
    image:


    model:


    Notes
    -----

    J Portilla and E P Simoncelli. A Parametric Texture Model based on
    Joint Statistics of Complex Wavelet Coefficients. Int'l Journal
    of Computer Vision. 40(1):49-71, October, 2000.
    http://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
    http://www.cns.nyu.edu/~lcv/texture/

    TODO
    ----
    batch
    return multiple samples
    drift duffusion instead of Browninan motion (a la HMC)

    """

    def __init__(self, target_image, model):
        super().__init__()

        self.target_image = torch.tensor(target_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.target_image = stretch(rescale(self.target_image, -1, 1))

        self.squish = nn.Tanh()
        # TODO nn.Hardtanh

        self.model = model
        self.target_representation = self.analyze(self.target_image)

    def analyze(self, x):
        # Note: analysis is applied on the squished input, so as to softly enforce the desired range during the optimization, (as a consequence, there is a discrepency to corresponding statistics during the optimization- which is fixed at the moment of returning an output in the synthesize method)
        # return self.model(self.squish(x))

        y = self.model(self.squish(x))
        if isinstance(y, list):
            return torch.cat([s.squeeze().view(-1) for s in y]).unsqueeze(1)
        else:
            return y

    def objective_function(self, x, y):
        return torch.norm(x - y, p=2)

    def _optimizer_step(self, i, max_iter,verbose):

        self.optimizer.zero_grad()
        self.matched_representation = self.analyze(self.matched_image)
        loss = self.objective_function(self.matched_representation, self.target_representation)
        loss.backward(retain_graph=True)
        g = self.matched_image.grad.data
        self.optimizer.step()

        self.t0 = self.t1
        self.t1 = time.time()
        if verbose:
            pretty_print(i, max_iter, (self.t1 - self.t0), loss.item(), g.norm().item())

        return loss

    def synthesize(self, init=None, seed=0, learning_rate=.01, max_iter=100, verbose=True):

        if init is None:
            # random initialization
            torch.manual_seed(seed)
            np.random.seed(seed)

            # making sure we don't break the graph
            self.matched_image = torch.tensor(np.random.uniform(low=-1, size=tuple(self.target_image.size())), requires_grad=True, dtype=torch.float32)
        else:
            self.matched_image = torch.tensor(init.view_as(self.target_image), requires_grad=True, dtype=torch.float32)

        self.optimizer = optim.Adam([self.matched_image], lr=learning_rate, amsgrad=True)

        self.loss = []
        self.t1 = time.time()

        for i in range(max_iter):
            loss = self._optimizer_step(i, max_iter, verbose)
            self.loss.append(loss.item())

        # correcting for the discrepency between matched_image and matched_representation
        # that results from the squishing during optimization
        self.matched_image = self.squish(self.matched_image)

        return self.matched_image.data.squeeze().add_(0.5).mul_(255),\
        self.matched_representation.data.squeeze().add_(0.5).mul_(255)
