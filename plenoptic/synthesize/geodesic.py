from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..tools.data import to_numpy
from ..tools.fit import penalize_range
from ..tools.straightness import make_straight_line, sample_brownian_bridge


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

    TODO
    ----
    OFF BY ONE ERROR IN n_steps
    should be 10 vertices for 11 edges

    fix animate

    compare stability relative loss

    projected version for surjective transform (eg unercomplete, low rank) 
    '''

    def __init__(self, imgA, imgB, model, n_steps=11, init='straight'):
        super().__init__()

        self.xA = imgA.clone().detach()
        self.xB = imgB.clone().detach()
        self.model = model
        self.n_steps = n_steps
        self.image_size = imgA.shape

        self.pixelfade = self.initialize(init='straight')
        self.x = self.initialize(init=init)[1:-1]
        self.x = nn.Parameter(self.x)

        self.loss = []
        self.dist_from_line = []
        self.step_energy = []

        with torch.no_grad():
            self.yA = self.model(self.xA)
            self.yB = self.model(self.xB)

        n = self.n_steps - 1
        # step = (n-1)/n * self.yB + 1/n * self.yA
        # self.reference_length = self.metric(self.yB - step) * n
        self.repres_unit = self.metric(self.yB - self.yA) / n ** 2
        self.signal_unit = self.metric(self.xB - self.xA) / n ** 2

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

    def path_energy(self, z, zA, zB, unit=None):
        """
        step_energy: sqaured length of each step
        """

        step_energy = torch.empty(1, self.n_steps - 1)

        step_energy[:, 0] = self.metric(zA - z[0])
        for i in range(1, self.n_steps-2):
            step_energy[:, i] = self.metric(z[i] - z[i-1])
        step_energy[:, -1] = self.metric(zB - z[-1])
        self.step_energy.append(step_energy.detach())

        total_energy = torch.sum(step_energy)
        if unit is None:
            return total_energy
        else:
            return (total_energy / unit) - 1

    def _optimizer_step(self, i, pbar, noise):

        self.optimizer.zero_grad()
        y = self.analyze()
        repres_path_energy = self.path_energy(y, self.yA, self.yB)
        if self.lmbda >= 0:
            loss = repres_path_energy \
                   + self.lmbda * penalize_range(self.x, (0, 1))

        if loss.item() != loss.item():
            self.step_energy.pop()
            raise Exception('found a NaN in the loss during optimization')

        loss.backward()
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
                   lmbda=.1, objective='multiscale', noise=None, seed=0):
        """
        objective:

        noise:
        """
        self.lmbda = lmbda

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
            # TODO flag to store progress or not
            self.dist_from_line.append(self.distance_from_line(
                                        self.geodesic).unsqueeze(0))

            if loss.item() < 1e-6:
                raise Exception("""the geodesic matches the representation
                                straight line up to floating point
                                precision""")

    def plot_loss(self):
        plt.semilogy(self.loss)
        plt.xlabel('iter step')
        plt.ylabel('loss value')
        plt.show()

    def distance_from_line(self, x):
        """l2 distance of x's representation to its projection onto the
        representation line

        x: torch.FloatTensor
            a sequence of images, preferably with anchor images as endpoints
        """

        y = self.model(x)
        line = (self.yB - self.yA).flatten()
        u = line / torch.norm(line)
        # center
        y = (y - self.yA).view(self.n_steps, -1)

        return torch.norm(y - (y @ u)[:, None]*u[None, :], dim=1)

    def plot_distance_from_line(self, vid=None):

        fig, ax = plt.subplots()

        if vid is not None:
            ax.plot(to_numpy(self.distance_from_line(vid)),
                    'b-o', label='video')
        ax.plot(to_numpy(self.distance_from_line(self.pixelfade)),
                'g-o', label='pixelfade')
        ax.plot(to_numpy(self.distance_from_line(self.geodesic)),
                'r-o', label='geodesic')
        plt.legend(loc=1)
        plt.ylabel('distance from representation line')
        plt.xlabel('projection on representation line')
        # plt.yscale('log')

        return fig, ax

    def animate_distance_from_line(self, vid):
        """
        TODO remove from gedesic from figure initialization
        """
        from IPython.display import HTML
        from matplotlib import animation

        fig, ax = self.plot_distance_from_line(vid=vid)

        artist, = ax.plot(to_numpy(self.distance_from_line(self.geodesic)),
                          'r-o', label='geodesic')

        def animate(i):

            artist.set_data(range(11), to_numpy(self.dist_from_line[i],
                                                squeeze=True))
        #     artist.
            return (artist,)

        anim = animation.FuncAnimation(fig, animate,
                                       frames=100, interval=20, blit=True,
                                       repeat=False)
        anim = HTML(anim.to_html5_video())
        plt.close()

        return anim
