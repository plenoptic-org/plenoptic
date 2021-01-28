from collections import OrderedDict
from plenoptic.tools.display import update_plot

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

from ..tools.data import to_numpy
from ..tools.fit import penalize_range
from ..tools.straightness import (distance_from_line, make_straight_line,
                                  sample_brownian_bridge)


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
        step length of representation strainght line. It is the shortest
        distance that could possibly be achieved and is used as a floor
        relative to which loss is calculated.

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

    def __init__(self, imgA, imgB, model, n_steps=10, init='straight'):
        super().__init__()

        self.xA = imgA.clone().detach()
        self.xB = imgB.clone().detach()
        self.model = model
        self.n_steps = n_steps
        self.image_size = imgA.shape

        self.pixelfade = self.initialize(init='straight')
        self.x = self.initialize(init=init)[1:-1]
        self.x = nn.Parameter(self.x)

        self.geodesic = torch.cat((self.xA, self.x.data, self.xB), 0)
        with torch.no_grad():
            self.yA = self.model(self.xA)
            self.yB = self.model(self.xB)

        self.loss = []
        self.dist_from_line = [distance_from_line(
                               self.geodesic)]
        self.step_energy = []

        n = self.n_steps
        # step = (n-1)/n * self.yB + 1/n * self.yA
        # self.reference_length = self.metric(self.yB - step) * n
        self.repres_unit = self.metric(self.yB - self.yA) / n ** 2
        self.signal_unit = self.metric(self.xB - self.xA) / n ** 2

    def initialize(self, init):
        """initialize the geodesic

        Parameters
        ----------
        init : 'bridge', 'straight' or Tensor
            if a tensor is passed it must match the shape of the
            desired geodesic.
        """
        if init == 'bridge':
            x = sample_brownian_bridge(self.xA, self.xB, self.n_steps)
        elif init == 'straight':
            x = make_straight_line(self.xA, self.xB, self.n_steps)
        else:
            assert init.shape == (self.n_steps, *self.xA.shape[1:])
            x = init

        return x

    def analyze(self):
        """run the model on the current iterate of the geodesic
        """
        y = self.model(self.x)
        return y

    def metric(self, x, p=2):
        """distance function"""
        return torch.norm(x, p=p) ** p

    def path_energy(self, z, zA, zB, unit=None):
        """
        step_energy: sqaured length of each step
        """

        step_energy = torch.empty(1, self.n_steps)

        step_energy[:, 0] = self.metric(zA - z[0])
        for i in range(1, self.n_steps-1):
            step_energy[:, i] = self.metric(z[i] - z[i-1])
        step_energy[:, -1] = self.metric(zB - z[-1])
        self.step_energy.append(step_energy.detach())

        total_energy = torch.sum(step_energy)
        if unit is not None:
            total_energy = (total_energy / unit) - 1
        return total_energy

    def _optimizer_step(self, i, pbar):

        self.optimizer.zero_grad()
        y = self.analyze()
        repres_path_energy = self.path_energy(y, self.yA, self.yB)
        loss = repres_path_energy
        if self.lmbda >= 0:
            loss = loss + self.lmbda * penalize_range(self.x, (0, 1))

        if loss.item() != loss.item():
            self.step_energy.pop()
            raise Exception('found a NaN in the loss during optimization')

        loss.backward()

        # TODO undercomplete case
        # repres_grad = x.grad

        # self.optimizer.zero_grad()
        # signal_path_energy = self.path_energy(self.x, self.xA, self.xB)
        # signal_grad = x.grad
        # x.grad = signal_grad - (
        #           signal_grad @ repres_grad
        #                         ) / torch.norm(repres_grad) * repres_grad

        self.optimizer.step()
        grad_norm = torch.norm(self.x.grad.data)
        pbar.set_postfix(OrderedDict([('loss', f'{loss.item():.4e}'),
                         ('gradient norm', f'{grad_norm:.4e}'),
                         ('lr', self.optimizer.param_groups[0]['lr'])]))
        if grad_norm.item() != grad_norm.item():
            raise Exception('found a NaN in the gradients during optimization')

        return loss

    def synthesize(self, max_iter=1000, learning_rate=.001, optimizer='adam',
                   lmbda=.1, objective='multiscale', seed=0):
        """ synthesize a geodesic

        Parameters
        ----------
        max_iter: int, optional
            maximum number of steps taken by the optimization a

        learning_rate: float, optional
            controls the step sizes of the search algorithm

        optimizer: str or torch.optim.Optimizer, optional
            algorithm that will perform the search

        lmbda: float, optional
            strength of the regularizer that enforces the image range in [0, 1]

        objective: str, optional
            'default'
            'multiscale'

        seed: int
            set the random number generator
        """
        self.lmbda = lmbda

        torch.manual_seed(seed)
        if optimizer == 'adam':
            self.optimizer = optim.Adam([self.x],
                                        lr=learning_rate, amsgrad=True)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD([self.x],
                                       lr=learning_rate, momentum=0.9)
        elif isinstance(optimizer, optim.Optimizer):
            self.optimizer = optimizer

        pbar = tqdm(range(max_iter))
        for i in pbar:
            loss = self._optimizer_step(i, pbar)

            # storing some information
            self.loss.append(loss.item())
            self.geodesic = torch.cat((self.xA, self.x.data, self.xB), 0)
            # TODO flag to store progress or not
            self.dist_from_line.append(distance_from_line(
                                        self.geodesic))

            if loss.item() < 1e-6:
                raise Exception("""the geodesic matches the representation
                                straight line up to floating point
                                precision""")

    def plot_loss(self):
        plt.semilogy(self.loss)
        plt.xlabel('iter step')
        plt.ylabel('loss value')
        plt.show()

    def plot_distance_from_line(self, vid=None, iteration=None,
                                figsize=(7, 5)):
        """visual diagnostic of geodesic linearity in representation space.

        Parameters
        ----------
        vid : torch.Tensor, optional
            natural video that bridges the anchor points
        iteration : int, optional
            plot the geodesic at a given step number of the optimization
        figsize : tuple, optional
            set the dimension of the figure

        Returns
        -------
        fig: matplotilb figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(to_numpy(distance_from_line(self.pixelfade)),
                'g-o', label='pixelfade')

        if iteration is None:
            distance = distance_from_line(self.geodesic)
        else:
            distance = self.dist_from_line[iteration]
        ax.plot(to_numpy(distance), 'r-o', label='geodesic')

        if vid is not None:
            ax.plot(to_numpy(distance_from_line(vid)),
                    'b-o', label='video')
        ax.set(xlabel='projection on representation line',
               ylabel='distance from representation line')
        ax.legend(loc=1)

        return fig

    def animate_distance_from_line(self, vid=None, framerate=25):
        """dynamic visualisation of geodesic linearity along the optimization process

        Parameters
        ----------
        vid : torch.Tensor, optional
            natural video that bridges the anchor points
        framerate : int, optional
            set the number of frames per second in the animation

        Returns
        -------
        anim: matplotlib animation object (can call anim.save(target_location.mp4))
        """

        fig = self.plot_distance_from_line(vid=vid, iteration=0)

        def animate(i):
            # update_plot requires 3d data for lines
            data = self.dist_from_line[i].unsqueeze(0).unsqueeze(0)
            artist = update_plot(fig.axes[0], {'geodesic': data})
            return artist

        anim = FuncAnimation(fig, animate,
                             frames=len(self.dist_from_line),
                             interval=1000./framerate, blit=True, repeat=False)
        plt.close(fig)
        return anim
