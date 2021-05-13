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

    Note
    ----
    Manifold prior hypothesis: natural images form a manifold 𝑀ˣ embedded
    in signal space (ℝⁿ), a model warps this manifold to another manifold 𝑀ʸ
    embedded in representation space (ℝᵐ), and thereby induces a different
    local metric.

    This method computes an approximate geodesics by solving an optimization
    problem: it minimizes the path energy (aka. action functional), which has
    the same minimum as minimizing path length and by Cauchy-Schwarz, reaches
    it with constant-speed minimizing geodesic

    Caveat: depending on the geometry of the manifold, geodesics between two
    anchor points may not be unique and may depend on the initialization.

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
        return step_energy

    def _optimizer_step(self, pbar):
        """
        At each step of the optimization, the following is done:
        - compute the representation
        - compute the loss as a sum of:
            - representation's path energy
            - range constraint (weighted by lambda)
        - compute the gradients
        - make sure that neither the loss or the gradients are NaN
        - let the optimizer take a step in the direction of the gradients
        - display some information
        - store some information
        - return delta_x, the norm of the step just taken
        """
        xprev = self.x.clone()
        self.optimizer.zero_grad()

        x = torch.cat([self.xA, self.x, self.xB])
        y = self._analyze(x)

        # representation's path energy
        step_energy = self._step_energy(y)
        loss = step_energy.mean()

        self.loss.append(loss.item())
        if self.lmbda > 0:
            loss = loss + self.lmbda * penalize_range(self.x, (0, 1))

        if not torch.isfinite(loss):
            raise Exception('found a NaN in the loss during optimization')
        loss.backward()

        grad_norm = torch.norm(self.x.grad.data)
        if not torch.isfinite(grad_norm):
            raise Exception('found a NaN in the gradients during optimization')
        self.optimizer.step()

        delta_x = torch.norm(self.x - xprev)
        # displaying some information
        pbar.update(1)
        pbar.set_postfix(OrderedDict([('loss', f'{loss.item():.4e}'),
                         ('gradient norm', f'{grad_norm:.4e}'),
                         ('delta_x', f"{delta_x.item():.5e}")]))
        # storing some information
        if self.verbose:
            self.step_energy.append(step_energy.detach())
            self.dev_from_line.append(
                deviation_from_line(y.detach()))

        return delta_x

    def synthesize(self, max_iter=1000, learning_rate=.001, optimizer='Adam',
                   lmbda=.1, tol=None, seed=0, verbose=True):
        """Synthesize a geodesic via optimization.

        Parameters
        ----------
        max_iter: int, optional
            maximum number of steps taken by the optimization algorithm.
        learning_rate: float, optional
            controls the step sizes of the search algorithm.
        optimizer: {'Adam', torch.optim.Optimizer}, optional
            choice of algorithm that will perform the search
            if an optimizer is passed, its `params` argument should be set to
            `self.x`, where self refers to a previously initialized Geodesic
            class.
        lmbda: float, optional
            strength of the regularizer that enforces the image range in [0, 1]
            (strictly positive by default)
        tol: float, optional
            tolerance threshold used to terminate algorithm before `max_iter`
            if the optimization stopped making progress
        seed: int, optional
            set the random number generator
        verbose: bool, optional
            storing information along the run of the optimization algorithm
        """
        self.lmbda = lmbda
        self.verbose = verbose
        if tol is None:
            tol = self.pixelfade.norm() / 1e4 * (1 + 5 ** .5) / 2
        print(f"\n threshold for delta_x, tolerance = {tol:.5e}")

        torch.manual_seed(seed)
        if optimizer == 'Adam':
            self.optimizer = optim.Adam([self.x],
                                        lr=learning_rate, amsgrad=True)
        elif isinstance(optimizer, optim.Optimizer):
            self.optimizer = optimizer

        delta_x = torch.ones(1)
        i = 0
        with tqdm(range(max_iter)) as pbar:
            # project onto set of representational geodesics
            while delta_x > tol and i < max_iter:
                delta_x = self._optimizer_step(pbar)
                i += 1
        self._populate_geodesic()

    def plot_loss(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.loss)
        ax.set(yscale='log',
               xlabel='iter step',
               ylabel='loss value')
        return fig

    def _populate_geodesic(self):
        """Help format the current optimization variable `x` into a geodesic
        attribute that can be used later.

        It joins the endpoints, reshapes the variable into a sequence
        of images, detaches the gradients and clips the range to [0, 1].
        """
        self.geodesic = torch.cat([self.xA, self.x, self.xB]
                                  ).reshape(
                                (self.n_steps+1, *self.image_shape[1:])
                                            ).clamp(0, 1).detach()

    def plot_deviation_from_line(self, video=None, figsize=(7, 5)):
        """visual diagnostic of geodesic linearity in representation space.

        Parameters
        ----------
        video : torch.Tensor, optional
            natural video that bridges the anchor points
        figsize : tuple, optional
            set the dimension of the figure

        Returns
        -------
        fig: matplotlib.figure.Figure

        Notes
        -----
        This plot illustrates the deviation from the straight line connecting
        the representations of a pair of images, for different paths
        in representation space.
        Axes are in the same units, normalized by the distance separating
        the end point representations.
        Knots along each curve indicate samples used to compute the path.

        When the representation is non-linear it may not be feasible for the
        geodesic to be straight (for example if the representation is
        normalized, all paths are constrained to live on a hypershpere).
        Nevertheless, if the representation is able to linearize the
        transformation between the anchor images, then we expect that both
        the ground tuth video sequence and the geodesic will deviate from
        straight line similarly. By contrast the pixel-based interpolation
        will deviate significantly more from a straight line.
        """
        if not hasattr(self, 'geodesic'):
            self._populate_geodesic()

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(*deviation_from_line(self.model(self.pixelfade
                                                ).view(self.n_steps+1, -1)
                                     ), 'g-o', label='pixelfade')

        deviation = deviation_from_line(
                        self.model(self.geodesic
                                   ).view(self.n_steps+1, -1))
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

    def _vector_jacobian_product(self, y, x, a):
        """compute vector-jacobian product: $a^T dy/dx = dy/dx^T a$,
        and allow for further gradient computations by retaining,
        and creating the graph.
        """
        accJac = autograd.grad(y, x, a,
                               retain_graph=True,
                               create_graph=True)[0]
        return accJac

    def _step_jerkiness(self, y):
        velocity = self._finite_difference(y)
        acceleration = self._finite_difference(velocity)
        acc_magnitude = torch.norm(acceleration, dim=1, keepdim=True)
        acc_direction = torch.div(acceleration, acc_magnitude)
        accJac = self._vector_jacobian_product(y[1:-1], self.x,
                                               acc_direction)
        step_jerkiness = torch.norm(accJac, dim=1) ** 2
        return step_jerkiness

    def calculate_jerkiness(self):
        """Compute the alignment of representation's acceleration
        to model local curvature (here called "jerkiness").
        This is the first order optimality condition for a geodesic,
        and can be used to assess the validity of the solution obtained
        by optimization.
        """
        y = self._analyze(torch.cat([self.xA, self.x, self.xB]))
        return self._step_jerkiness(y).detach()
