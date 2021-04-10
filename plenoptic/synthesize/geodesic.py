from collections import OrderedDict
from plenoptic.tools.display import update_plot

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from tqdm import tqdm
from matplotlib.animation import FuncAnimation

from ..tools.data import to_numpy
from ..tools.optim import penalize_range
from ..tools.straightness import (distance_from_line, make_straight_line,
                                  sample_brownian_bridge)


class Geodesic(nn.Module):
    r'''Synthesize a geodesic between two images according to a model [1]_.

    This method can be used to visualize and refine the invariances of a
    model's representation.

    Parameters
    ----------
    imgA (resp. imgB): torch.FloatTensor
        Start (resp. stop) anchor of the geodesic,
        of shape [1, C, H, W] in range [0, 1].

    model: nn.Module
        an analysis model that computes image representations

    n_steps: int, optional
        the number of steps in the trajectory between the two anchor points

    init: {'straight', 'bridge'}, optional
        initialize the geodesic with pixel linear interpolation (default),
        or with a brownian bridge between the two anchors

    Attributes
    ----------
    x:
        optimization variable, shape: [n_steps-1, dx], where dx = C x H x W

    geodesic:
        synthesized sequence of images between the two anchor points that
        minimizes distance in representation space

    pixelfade:
        straight interpolation between the two anchor points for reference

    step_energy:
        step lengths in representation space, stored along the optimization
        process

    step_jerkiness:
        alignment of representation's acceleration with local model curvature,
        stored along the optimization process

    dist_from_line:
        l2 distance of the geodesic's representation to the straight line in
        representation space, stored along the optimization process

    References
    ----------
    .. [1] Geodesics of learned representations
        O J Hénaff and E P Simoncelli
        Published in Int'l Conf on Learning Representations (ICLR), May 2016.
        http://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Henaff16b
    '''

    def __init__(self, imgA, imgB, model, n_steps=10, init='straight'):
        super().__init__()

        self.n_steps = n_steps
        self.image_shape = imgA.shape
        self.pixelfade = self._initialize('straight', imgA, imgB, n_steps)

        self.xA, x, self.xB = torch.split(self._initialize(init, imgA, imgB,
                                                          n_steps).view(
                                          n_steps+1, torch.numel(imgA[0])),
                                          [1, n_steps-1, 1])
        self.x = nn.Parameter(x.requires_grad_())

        # making sure we are not computing unnecessary gradients
        for p in model.parameters():
            if p.requires_grad:
                p.detach_()
                # TODO: print info
        self.model = model.eval()

        self.loss = []
        self.dist_from_line = []
        self.step_energy = []
        self.step_jerkiness = []

    def _initialize(self, init, imgA, imgB, n_steps):
        """initialize the geodesic

        Parameters
        ----------
        init : {'bridge', 'straight', or torch.Tensor}
            if a tensor is passed it must match the shape of the
            desired geodesic.
        """
        if init == 'bridge':
            x = sample_brownian_bridge(imgA, imgB, n_steps)
        elif init == 'straight':
            x = make_straight_line(imgA, imgB, n_steps)
        else:
            assert init.shape == (n_steps, *self.image_shape[1:])
            x = init
        return x

    def _analyze(self, x):
        """compute the model representation on the current value of
        the optimization variable `x`.

        Note that the optimization variable `x` is a series of vectors,
        it is first reshaped into a tensor of images that the model can
        process, and then the representation is viewed as a vector.
        This is necessary for computation of the regularization of
        path jerkinessthe, which is a vector Jacobian product.
        """
        return self.model(x.view(len(x), *self.image_shape[1:])
                          ).view(len(x), -1)

    def _finite_difference(self, z):
        """compute a discrete approximation to the derivative operator.
        """
        return z[1:] - z[:-1]

    def _step_energy(self, z):
        """compute the energy (ie. squared l2 norm) of each step in `z`.
        """
        velocity = self._finite_difference(z)
        step_energy = torch.norm(velocity, dim=1) ** 2
        return velocity, step_energy

    def _vector_jacobian_product(self, y, x, a):
        """compute vector-jacobian product: $a^T dy/dx = dy/dx^T a$,
        and allow for further gradient computations by retaining,
        and creating the graph.
        """
        accJac = autograd.grad(y, x, a,
                               retain_graph=True,
                               create_graph=True)[0]
        return accJac

    def _step_jerkiness(self, y, velocity):
        """compute the energy of `y`'s acceleration direction when it is pulled
        in the input space by the input-output Jacobian.
        ie. alignment of representation's acceleration to model curvature.
        ie. sensitivity of the Jacobian outer product to the acceleration
        direction.
        """
        acceleration = self._finite_difference(velocity)
        acc_magnitude = torch.norm(acceleration, dim=1, keepdim=True)
        acc_direction = torch.div(acceleration, acc_magnitude)
        accJac = self._vector_jacobian_product(y[1:-1], self.x,
                                               acc_direction)
        step_jerkiness = torch.norm(accJac, dim=1) ** 2
        return step_jerkiness

    def _optimizer_step(self, i, pbar):
        """
        At each step `i` of the optimization, the following is done:
        - compute the representation
        - compute the loss as a sum of:
            - path energy (weighted by mu)
            - path jerkiness (weighted by nu)
            - range constraint (weighted by lambda)
        - compute the gradients
        - make sure that neither the loss or the gradients are NaN
        - [TODO: compute conditional loss and gradients]
        - let the optimizer take a step in the direction of the gradients
        - display some information
        - store some information
        """

        self.optimizer.zero_grad()
        y = self._analyze(torch.cat([self.xA, self.x, self.xB]))

        velocity, step_energy = self._step_energy(y)
        repres_path_energy = step_energy.mean()
        # TODO: rescale torch.div(step_energy, )
        loss = self.mu * repres_path_energy

        # if self.nu > 0:
        step_jerkiness = self._step_jerkiness(y, velocity)
        path_jerkiness = step_jerkiness.mean()
        loss = loss + self.nu * path_jerkiness

        if self.lmbda > 0:
            loss = loss + self.lmbda * penalize_range(self.x, (0, 1))

        if loss.item() != loss.item():
            raise Exception('found a NaN in the loss during optimization')

        # if loss.item() < 1e-6:
        #     raise Exception("""the geodesic matches the representation
        #                     straight line up to floating point
        #                     precision""")

        loss.backward()

        grad_norm = torch.norm(self.x.grad.data)
        if grad_norm.item() != grad_norm.item():
            raise Exception('found a NaN in the gradients during optimization')

        # TODO undercomplete representation case
        # if y.shape[-1`] < dx:
        # repres_grad = x.grad

        # self.optimizer.zero_grad()
        # signal_path_energy = torch.norm(_finite_difference(self.x),
        #                                   dim=1).pow(2).mean()
        # signal_grad = x.grad
        # x.grad = signal_grad - (
        #           signal_grad @ repres_grad
        #                         ) / torch.norm(repres_grad) * repres_grad

        self.optimizer.step()

        # displaying some information
        pbar.set_postfix(OrderedDict([('loss', f'{loss.item():.4e}'),
                         ('gradient norm', f'{grad_norm:.4e}'),
                         ('lr', self.optimizer.param_groups[0]['lr'])]))

        # storing some information
        self.loss.append(loss.item())
        if self.verbose:
            self.step_energy.append(step_energy.detach())
            # if self.nu > 0:
            self.step_jerkiness.append(step_jerkiness.detach())
            self.dist_from_line.append(
                distance_from_line(y.unsqueeze(2).unsqueeze(3)))

    def synthesize(self, max_iter=1000, learning_rate=.001, optimizer='Adam',
                   lmbda=.1, mu=1, nu=0, seed=0, verbose=True):
        """ synthesize a geodesic
        Parameters
        ----------
        max_iter: int, optional
            maximum number of steps taken by the optimization algorithm.

        learning_rate: float, optional
            controls the step sizes of the search algorithm.

        optimizer: {'Adam', 'SGD', torch.optim.Optimizer}, optional
            choice of algorithm that will perform the search
            if an optimizer is passed, its `params` argument should be set to
            `self.x`, where self refers to a previously initialized Geodesic
            class.

        mu: float, optional
            strength of the path length objective (usefull for experimenting
            with only optimizing path jerkiness - under developpement)
            (defaults to one)

        lmbda: float, optional
            strength of the regularizer that enforces the image range in [0, 1]
            (present by default)

        nu: float, optional
            strength of the regularizer that enforces minimal path jerkiness
            (ie. representation path orthogonal to model curvature).
            (absent by default)

        seed: int, optional
            set the random number generator

        verbose: bool, optional
            storing information along the run of the optimization algorithm
        """
        self.lmbda = lmbda
        self.mu = mu
        self.nu = nu
        self.verbose = verbose

        torch.manual_seed(seed)
        if optimizer == 'Adam':
            self.optimizer = optim.Adam([self.x],
                                        lr=learning_rate, amsgrad=True)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD([self.x],
                                       lr=learning_rate, momentum=0.9)
        elif isinstance(optimizer, optim.Optimizer):
            self.optimizer = optimizer

        pbar = tqdm(range(max_iter))
        for i in pbar:
            self._optimizer_step(i, pbar)
        self.populate_geodesic()

    def populate_geodesic(self):
        self.geodesic = torch.cat([self.xA, self.x, self.xB]
                                  ).reshape((self.n_steps+1,
                                             *self.image_shape[1:])
                                            ).detach()

    # def calculate_path_jerkiness(self):
    #     usefull to check path jerkiness to certify a candidate geodesic
    #     for now unnecessary because path jerkiness is computed during main algo
    #     y = self._analyze(torch.cat([self.xA, self.x, self.xB]))
    #     velocity = self._finite_difference(y)
    #     return self._step_jerkiness(y, velocity).detach()

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
        fig: matplotlib.figure.Figure
        """
        if not hasattr(self, 'geodesic'):
            self.populate_geodesic()

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(to_numpy(distance_from_line(self.model(self.pixelfade))),
                'g-o', label='pixelfade')

        if iteration is None:
            distance = distance_from_line(self.model(self.geodesic))
        else:
            distance = self.dist_from_line[iteration]
        ax.plot(to_numpy(distance), 'r-o', label='geodesic')

        if vid is not None:
            ax.plot(to_numpy(distance_from_line(self.model(vid))),
                    'b-o', label='video')
        ax.set(xlabel='projection on representation line',
               ylabel='distance from representation line')
        ax.legend(loc=1)

        return fig

    def animate_distance_from_line(self, vid=None, framerate=25):
        """dynamic visualisation of geodesic linearity along the optimization process

        This animates `plot_distance_from_line` over the steps of the algorithm

        Parameters
        ----------
        vid : torch.Tensor, optional
            natural video that bridges the anchor points
        framerate : int, optional
            set the number of frames per second in the animation

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object. It can be saved (can call anim.save(target_location.mp4)),
            or viewed in a jupyter notebook (needs to be converted to HTML).
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
