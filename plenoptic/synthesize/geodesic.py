from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..tools.optim import penalize_range
from ..tools.straightness import (deviation_from_line, make_straight_line,
                                  sample_brownian_bridge)


class Geodesic(nn.Module):
    r'''Synthesize an approximate geodesic between two images according to a model.

    This method can be used to visualize and refine the invariances of a
    model's representation as described in [1]_.

    Parameters
    ----------
    imgA, imgB: torch.FloatTensor
        Start (resp. stop) anchor points of the geodesic,
        of shape [1, C, H, W] in range [0, 1].

    model: nn.Module
        an analysis model that computes representations on siglals like `imgA`.

    n_steps: int, optional
        the number of steps in the trajectory between the two anchor points.

    init: {'straight', 'bridge'}, optional
        initialize the geodesic with pixel linear interpolation (default),
        or with a brownian bridge between the two anchors.

    Attributes
    ----------
    x: torch.FloatTensor
        the optimization variable of shape: [n_steps-1, D], where D = C x H x W

    geodesic: torch.FloatTensor
        the synthesized sequence of images between the two anchor points that
        minimizes representation path energy

    pixelfade: torch.FloatTensor
        the straight interpolation between the two anchor points,
        used as reference

    step_energy: list of torch.FloatTensor
        step lengths in representation space, stored along the optimization
        process

    dev_from_line: list of torch.FloatTensor
        deviation of the representation to the straight line interpolation,
        measures distance from straight line and distance along straight line,
        stored along the optimization process

    step_jerkiness: list of torch.FloatTensor
        alignment of representation's acceleration with local model curvature,
        stored along the optimization process.
        (TODO this is an experimental feature)

    Note
    ----
    Manifold prior hypothesis: natural images form a manifold 𝑀x embedded
    in signal space (ℝ𝑛), a model warps this manifold to another manifold 𝑀y
    embedded in representation space (ℝ𝑛), and thereby induces a different
    local metric.

    This method computes an approximate geodesics by solving an optimization
    problem: it minimizes the path energy (aka. action functional), which has
    the same minimum as minimizing path length and by Cauchy-Schwarz, reaches
    it with constant-speed minimizing geodesic

    Caveat: depending on the geometry of the manifold, geodesics between two
    anchor points may not be unique and be dependent on the initialization.

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
        start = imgA.clone().view(1, -1)
        stop = imgB.clone().view(1, -1)
        self.pixelfade = self._initialize('straight', start, stop, n_steps
                                          ).view(n_steps+1, *imgA.shape[1:])

        xinit = self._initialize(init, start, stop, n_steps
                                 ).view(n_steps+1, -1)
        self.xA, x, self.xB = torch.split(xinit, [1, n_steps-1, 1])
        self.x = nn.Parameter(x.requires_grad_())

        warn = True
        for p in model.parameters():
            if p.requires_grad:
                p.detach_()
                if warn:
                    print("""we detach model parameters in order to
                             save time on extraneous gradients
                             computations - indeed only pixel values
                             should be modified.""")
                    warn = False
        self.model = model.eval()

        self.loss = []
        self.dev_from_line = []
        self.step_energy = []
        self.step_jerkiness = []

    def _initialize(self, init, start, stop, n_steps):
        """initialize the geodesic

        Parameters
        ----------
        init : {'bridge', 'straight', or torch.Tensor}
            if a tensor is passed it must match the shape of the
            desired geodesic.
        """
        if init == 'bridge':
            x = sample_brownian_bridge(start, stop, n_steps)
        elif init == 'straight':
            x = make_straight_line(start, stop, n_steps)
        else:
            assert init.shape == (n_steps, torch.numel(self.image_shape[1:]))
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
        """compute alignment of representation's acceleration to model curvature.

        More specifically: compute the sensitivity of the model's input-output
        Jacobian outer product to the representation's `y` acceleration
        ie. the squared norm of the acceleration according to the local
        Riemannian metric on the model's tangent space.
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
        - [TODO: compute conditional loss and gradients - with nesting in mind]
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

        if self.nu > 0:
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
        #                                   dim=1).pow(2).mean(
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
            if self.nu > 0:
                self.step_jerkiness.append(step_jerkiness.detach())
            self.dev_from_line.append(
                deviation_from_line(y.detach()))

    def synthesize(self, max_iter=1000, learning_rate=.001, optimizer='Adam',
                   lmbda=.1, mu=1, nu=0, seed=0, verbose=True):
        """Synthesize a geodesic.

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
            with only optimizing path jerkiness - TODO under developpement)
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
        elif isinstance(optimizer, optim.Optimizer):
            self.optimizer = optimizer

        pbar = tqdm(range(max_iter))
        for i in pbar:
            self._optimizer_step(i, pbar)
        self.populate_geodesic()

    def populate_geodesic(self):
        """Help format the current optimization variable `x` into a geodesic
        attribute that can be used later.

        It joins the endpoints, reshapes the variable into a sequence
        of images, detaches the gradients and clips the range to [0, 1].
        """
        self.geodesic = torch.clip(torch.cat([self.xA, self.x, self.xB]
                                             ).reshape(
                                    (self.n_steps+1, *self.image_shape[1:])
                                      ).detach(), 0, 1)

    def calculate_path_jerkiness(self):
        """check path jerkiness, which can certify a candidate geodesic
        """
        y = self._analyze(torch.cat([self.xA, self.x, self.xB]))
        velocity = self._finite_difference(y)
        return self._step_jerkiness(y, velocity).detach()

    def plot_loss(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.loss)
        ax.set(yscale='log',
               xlabel='iter step',
               ylabel='loss value')
        return fig

    def plot_deviation_from_line(self, video=None, iteration=None,
                                 figsize=(7, 5)):
        """visual diagnostic of geodesic linearity in representation space.

        Parameters
        ----------
        video : torch.Tensor, optional
            natural video that bridges the anchor points
        iteration : int, optional
            plot the geodesic at a given step number of the optimization
            TODO remove if animation no longer supported
        figsize : tuple, optional
            set the dimension of the figure

        Returns
        -------
        fig: matplotlib.figure.Figure
        """
        if not hasattr(self, 'geodesic'):
            self.populate_geodesic()

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(*deviation_from_line(self.model(self.pixelfade
                                                ).view(self.n_steps+1, -1)
                                     ), 'g-o', label='pixelfade')

        if iteration is None:
            deviation = deviation_from_line(
                            self.model(self.geodesic
                                       ).view(self.n_steps+1, -1))
        else:
            deviation = self.dev_from_line[iteration]
        ax.plot(*deviation, 'r-o', label='geodesic')

        if video is not None:
            ax.plot(*deviation_from_line(self.model(video
                                                    ).view(self.n_steps+1, -1)
                                         ), 'b-o', label='video')
        ax.set(xlabel='distance along representation line',
               ylabel='distance from representation line',
               title='deviation from the straight line')
        ax.legend(loc=1)

        return fig
