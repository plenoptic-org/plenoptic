from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn

import torch.optim as optim
from ..tools.straightness import sample_brownian_bridge, make_straight_line
from ..tools.fit import penalize_range
from ..tools.data import to_numpy


class Geodesic(nn.Module):
    r'''Synthesize a geodesic between two images according to a model [1]_.

    Parameters
    ----------
    imgA (resp. imgB): 'torch.FloatTensor'
        Start (resp. stop) anchor of the geodesic,
        of shape [1, C, H, W] in range [0, 1].

    model: nn.Module
        an analysis model that computes image representations

    n_steps: int
        the number of steps in the trajectory between the two anchor points

    lmbda: float, optional
        strength of the regularizer that enforces the image range,
        default value is .1

    init: string in ['straight', 'bridge'], optional
        initialize the geodesic with pixel linear interpolation (default),
        or with a brownian bridge between the two anchors

    Attributes
    -------
    geodesic:
        synthesized sequence of images between the two anchor points that
        minimizes distance in representation space

    pixelfade:
        straight interpolation between the two anchor points for reference
        
    reference_length:
        step length of representation strainght line, used as a reference
        when computing loss

    step_lengths:
        step lengths in representation space, stored along the optimization
        process

    dist_from_line:
        l2 distance of the geodesic's representation to the straight line in
        representation space, stored along the optimization process

    Notes
    -----
    Method for visualizing and refining the invariances of a model's
    representations

    References
    ----------
    .. [1] Geodesics of learned representations
        O J Hénaff and E P Simoncelli
        Published in Int'l Conf on Learning Representations (ICLR), May 2016.
        http://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Henaff16b

    '''

    def __init__(self, imgA, imgB, model, n_steps=11, init='straight',
                 lmbda=.1):
        super().__init__()

        self.xA = imgA.clone().detach()
        self.xB = imgB.clone().detach()
        self.model = model
        self.n_steps = n_steps
        self.lmbda = lmbda
        self.image_size = imgA.shape

        self.pixelfade = self.initialize(init='straight')
        self.x = self.initialize(init=init)[1:-1]
        self.x = nn.Parameter(self.x)

        self.loss = []
        self.dist_from_line = []
        self.step_lengths = []

        with torch.no_grad():
            self.yA = self.model(self.xA)
            self.yB = self.model(self.xB)

        n = self.n_steps - 1
        step = (n-1)/n * self.yB + 1/n * self.yA
        self.reference_length = self.metric(self.yB - step) * n

    def initialize(self, init):
        if init == 'straight':
            x = make_straight_line(self.xA, self.xB, self.n_steps)
        elif init == 'bridge':
            x = sample_brownian_bridge(self.xA, self.xB, self.n_steps)
        return x

    def analyze(self):
        y = self.model(self.x)
        return y

    def metric(self, x, p=2):
        return torch.norm(x, p=p) ** p

    def objective_function(self, y):
        """relative to straight interpolation
        """

        step_lengths = torch.empty(1, self.n_steps - 1)

        step_lengths[:, 0] = self.metric(self.yA - y[0])
        for i in range(1, self.n_steps-2):
            step_lengths[:, i] = self.metric(y[i] - y[i-1])
        step_lengths[:, -1] = self.metric(self.yB - y[-1])

        loss = torch.sum(step_lengths)
        self.step_lengths.append(step_lengths.detach())

        return loss / self.reference_length - 1

    def _optimizer_step(self, i, pbar, noise):

        self.optimizer.zero_grad()
        y = self.analyze()
        loss = self.objective_function(y)
        if self.lmbda >= 0:
            loss = loss + self.lmbda * penalize_range(self.x, (0, 1))
        if loss.item() != loss.item():
            raise Exception('found a NaN in the loss during optimization')

        loss.backward()
        self.optimizer.step()
        grad_norm = torch.norm(self.x.grad.data)
        pbar.set_postfix(OrderedDict([('loss', f'{loss.item():.4e}'),
                        ('gradient norm', f'{grad_norm:.4e}'),
                        ('lr', self.optimizer.param_groups[0]['lr'])]))
        if grad_norm.item() != grad_norm.item():
            raise Exception('found a NaN in the gradients during optimization')

        return loss

    def synthesize(self, max_iter=1000, learning_rate=.001, optimizer='adam',
                   objective='multiscale', noise=None, seed=0):
        """
        objective:

        noise:
        """

        torch.manual_seed(seed)
        if optimizer == 'adam':
            self.optimizer = optim.Adam([self.x],
                                        lr=learning_rate, amsgrad=True)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD([self.x],
                                       lr=learning_rate, momentum=0.9)
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer

        pbar = tqdm(range(max_iter))
        for i in pbar:
            loss = self._optimizer_step(i, pbar, noise)

            # storing some information
            self.loss.append(loss.item())
            self.geodesic = torch.cat((self.xA, self.x.data, self.xB), 0)
            self.dist_from_line.append(self.distance_from_line(self.geodesic).unsqueeze(0))

            if loss.item() < 1e-6:
                raise Exception("""the geodesic matches the representation straight line up to floating point precision""")

    def distance_from_line(self, x):
        """l2 distance of x's representation to its projection onto the representation line

        x: torch.FloatTensor
            a sequence of images, preferably with anchor images as endpoints
        """

        y = self.model(x)
        l = (self.yB - self.yA).flatten()
        l /= torch.norm(l)
        y_ = (y - self.yA).view(self.n_steps, -1)

        return torch.norm(y_ - (y_ @ l)[:, None]*l[None, :], dim=1)

    def plot_distance_from_line(self, vid=None):
        import matplotlib.pyplot as plt
        if vid is not None:
            plt.plot(to_numpy(self.distance_from_line(vid)), 'b-o', label='video')
        plt.plot(to_numpy(self.distance_from_line(self.pixelfade)), 'g-o', label='pixelfade')
        plt.plot(to_numpy(self.distance_from_line(self.geodesic)), 'r-o', label='geodesic')
        plt.legend(loc=1)
        plt.ylabel('distance from representation line')
        plt.xlabel('projection on representation line')
        # plt.yscale('log')
        plt.show()
