import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ..tools.signal import make_disk, rescale
from ..tools.fit import pretty_print, stretch


class Geodesic(nn.Module):
    '''Synthesize a geodesic between two images according to a model.

    Parameters
    ----------
    imgA:

    imgB:

    model:

    Returns
    -------
    gamma:
        the calculated geodesic

    Notes
    -----
    Method for visualizing and refining the invariances of learned representations

    http://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Henaff16b
    this script is based on an earlier version by Olivier Henaff

    TODO
    ----
    finish clean-up / reorganization
    in particular put much of the constructor in a class method
    compare losses
        multiscale
        angle vs. distance
    compute the argmin and substract it from loss so that 0 is meaningful
        argmin = nsmpl * ((output[-1] - output[0]) / nsmpl) ** 2
    accelerate
        with torch.autograd.profiler.profile() as prof:
            print(prof)
    conditional geodesics (project out)
    '''

    def __init__(self, imgA, imgB, n_steps, model):
        super().__init__()

        self.n_steps = n_steps
        image_size = imgA.size(1)

        # TODO explicitely asserts that A and B are square and of same size, no batch no channel for now
        # accomodate numpy and torch
        # self.xA = torch.tensor(imgA, dtype=torch.float32).unsqueeze(0)

        # rescale image to [-1,1]
        # rescale(imgA)
        self.xA = (2*imgA/255 - 1).view(1, image_size, image_size)
        self.xB = (2*imgB/255 - 1).view(1, image_size, image_size)

        # compute pixel linear interpolation for initialization
        self.x = torch.Tensor(self.n_steps-2, image_size, image_size)
        for i in range(self.n_steps-2):
            t = (i+1)/(self.n_steps-1)
            self.x[i].copy_(self.xA[0] * (1 - t)+(t * self.xB[0]))

        # disk mask, note that it divides amplitude by two
        self.mask = torch.Tensor(image_size, image_size)
        self.mask.copy_(make_disk(image_size)).div_(2)

        # keep rescaled pixel linear interpolation for reference
        self.pixelfade = torch.Tensor(self.n_steps, image_size, image_size)
        self.pixelfade[0].copy_(self.xA.squeeze())
        self.pixelfade[1:-1].copy_(self.x)
        self.pixelfade[-1].copy_(self.xB.squeeze())
        self.pixelfade = (self.mask * self.pixelfade).add_(0.5).mul_(255)

        # initialization
        self.gamma = torch.Tensor(self.n_steps, image_size, image_size)

        # soft rescaling from (-inf,inf) to (-1, 1)
        # used before each optimization step
        self.squish = nn.Tanh()

        self.xA = stretch(self.xA)
        self.xB = stretch(self.xB)
        self.x = stretch(self.x)

        self.x = self.x.requires_grad_()

        self.model = model

        # multi-scale tools
        self.n_scales = int(np.log2(self.n_steps))
        self.diff = nn.Conv3d(1, 1, (2, 1, 1), bias=False)
        self.blur = nn.Conv3d(1, 1, (2, 1, 1), bias=False, stride=(2, 1, 1))
        self.diff.weight.requires_grad = False
        self.blur.weight.requires_grad = False
        self.diff.weight = nn.Parameter(torch.tensor([[[[[-1.]], [[1.]]]]]))
        self.blur.weight = nn.Parameter(torch.ones_like(self.diff.weight))
        self.diff.weight.requires_grad = False
        self.blur.weight.requires_grad = False

    def analyze(self, x):

        x = torch.cat((self.xA, x, self.xB), 0)
        x = self.squish(x)
        self.img = self.mask * x
        y = self.model(self.img.unsqueeze(1))

        # TODO reshape n_steps, C, Y, X -> n_steps, -1
        if isinstance(y, dict):
            return torch.cat([s.squeeze().view(-1) for s in y.values()]).unsqueeze(1)
        else:
            return y

    def objective_function(self, x):

        z = x.permute(1, 0, 2, 3).unsqueeze(1)
        loss = 0
        for s in range(self.n_scales):
            loss = loss + torch.sum(self.diff(z) ** 2)
            z = self.blur(z)

        return loss

    def _optimizer_step(self, i, max_iter):

        self.optimizer.zero_grad()
        output = self.analyze(self.x)
        loss = self.objective_function(output)
        loss.backward()
        g = self.x.grad.data
        if math.isnan(loss.item()):
            raise Exception('found a NaN during optimization')
        self.optimizer.step()

        self.t0 = self.t1
        self.t1 = time.time()
        pretty_print(i, max_iter, (self.t1 - self.t0), loss.item(), g.norm().item())

        return loss

    def synthesize(self, max_iter=20, learning_rate=.01, objective='multiscale', seed=0):

        torch.manual_seed(seed)
        self.optimizer = optim.Adam([self.x], lr=learning_rate, amsgrad=True)

        self.loss = []
        self.t1 = time.time()

        for i in range(max_iter):
            loss = self._optimizer_step(i, max_iter)
            self.loss.append(loss.item())

        self.gamma.copy_(self.img.data.squeeze()).add_(0.5).mul_(255)

        return self.gamma
