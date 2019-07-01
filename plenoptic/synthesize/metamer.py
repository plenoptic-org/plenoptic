import torch
import torch.nn as nn
from torch import optim
from ..tools.fit import pretty_print
import numpy as np
import time
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..simulate import Steerable_Pyramid_Freq
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Metamer(nn.Module):
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
    (musts)
    synthesize an image of a different size than the target image
    flexible objective function
    flexibility on the optimizer / scheduler (or at least parameterize the stuff)

    (other)
    batch
    return multiple samples


    """

    def __init__(self, target_image, model):
        super().__init__()

        self.target_image = target_image
        self.model = model
        self.target_representation = self.analyze(self.target_image)

    def analyze(self, x):
        # Note: analysis is applied on the squished input, so as to softly enforce the desired range during the optimization, (as a consequence, there is a discrepency to corresponding statistics during the optimization- which is fixed at the moment of returning an output in the synthesize method)
        # return self.model(self.squish(x))
        
        y = self.model(x)
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
            pretty_print(i, max_iter, (self.t1 - self.t0), loss.item(), g.norm().item(),lr=self.optimizer.param_groups[0]['lr'])

        return loss

    def synthesize(self, seed=0, learning_rate=.01, max_iter=100,verbose=True,
        initial_image=torch.empty(0), clamper=None, save_iter=False, save_iter_weights=False):
        # random initialization
        torch.manual_seed(seed)
        np.random.seed(seed)
        # making sure we don't break the graph
        
        if initial_image.shape[0] == 0:
            self.matched_image = torch.randn_like(self.target_image, dtype=torch.float32)*.5+.5
            self.matched_image.requires_grad = True
        else:
            self.matched_image = initial_image

        # self.optimizer = optim.Adam([self.matched_image], lr=learning_rate, amsgrad=True)
        self.optimizer = optim.SGD([self.matched_image], lr = learning_rate, momentum=0.8)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=.2) 

        self.loss = []
        if save_iter:
            self.save = torch.empty((self.target_representation.numel(),max_iter))
        if save_iter_weights:
            self.save_weights = torch.empty((self.matched_image.numel(),max_iter))
        
        self.t1 = time.time()
        
        for i in range(max_iter):

            loss = self._optimizer_step(i, max_iter,verbose,)
            self.loss.append(loss.item())
            

            if loss.item() < 1e-4:
                break


            self.scheduler.step(loss.item())

            with torch.no_grad():
                if clamper is not None:
                    self.matched_image.data = clamper.clamp(self.matched_image.data)

                if save_iter:
                    # save stats from this step
                    self.save[:,i] = self.analyze(self.matched_image).flatten()

                if save_iter_weights:
                    # save stats from this step
                    self.save_weights[:,i] = self.matched_image.flatten()

        return self.matched_image.data.squeeze(), self.matched_representation.data.squeeze()


    
