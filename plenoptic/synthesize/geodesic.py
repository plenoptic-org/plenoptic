from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

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
            - if regularized: signal's path energy (weighted by mu)
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
        # note that we also keep track of the signal path energy
        signal_path_energy = self._step_energy(x).mean()
        self.loss.append((loss.item(), signal_path_energy.item()))

        if self.lmbda > 0:
            loss = loss + self.lmbda * penalize_range(self.x, (0, 1))

        if self.regularized and self.mu > 0:
            loss = loss + self.mu * signal_path_energy

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

    def _outer_optimizer_step(self):
        """
        At each step of the outer optimization, the following is done:
        - compute the representation path energy
        - compute the signal path energy
        - project the representational gradient out of the signal gradient
        - take a step in this direction

        - store some information
        """
        # representation path energy
        self.x.grad *= 0
        x = torch.cat([self.xA, self.x, self.xB])
        y = self._analyze(x)
        repres_loss = self._step_energy(y).mean()
        repres_loss.backward()
        repres_grad = self.x.grad.clone()

        # signal path energy
        self.x.grad *= 0
        signal_loss = self._step_energy(x).mean()
        signal_loss.backward()
        signal_grad = self.x.grad.clone()
        # project out representational gradient
        projected_signal_grad = signal_grad - (
                        (signal_grad.flatten()@repres_grad.flatten()) /
                        torch.norm(repres_grad)**2) * repres_grad
        # step
        self.x.grad = projected_signal_grad
        self.optimizer_outer.step()
        # self.x.data = self.x.data - self.alpha * projected_signal_grad

        # store some information
        self.loss.append((repres_loss.item(), signal_loss.item()))

    def synthesize(self, max_iter=1000, learning_rate=.001, optimizer='Adam',
                   regularized=True, mu=None, lmbda=.1,
                   conditional=False,  tol=None, seed=0,
                   verbose=True):
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
        regularized: bool, optional
            If True the loss will contain an additional regularization term
            which penalizes signal path energy, True by default
        mu: float, optional
            Strength of the regularization, this hyperparameter should be
            carefully chosen, the default value is only an example
        lmbda: float, optional
            strength of the regularization term that penalizes the optimization
            variable when it exceeds the [0, 1] range, it is strictly positive
            by default (TODO implement this via differentiable parametrization)
        conditional: bool, optional
            If True, search for the geodesic that is shortest in pixel space,
            ie. among paths that minimizes representation path energy, search
            for one with minimal signal path energy. If the representation
            is undercomplete there is a set of candidate geodesics, and this
            would resolve the degeneracy in the solution.
            Note: this is a non-linear analogue to picking the least square
            solution out of a subspace of solutions in an underdetermined
            system of equations
            Else if conditional is set to False, search for the geodesic
            that minimizes representation path energy, irrespective of its
            signal path energy
        tol: float, optional
            tolerance threshold used to terminate algorithm before `max_iter`
            if the optimization stopped making progress
        seed: int, optional
            set the random number generator
        verbose: bool, optional
            storing information along the run of the optimization algorithm
        """
        self.verbose = verbose
        torch.manual_seed(seed)
        self.lmbda = lmbda
        assert not (conditional and regularized), "one thing at a time"

        self.regularized = regularized
        if mu is None and self.regularized:
            # tentative default value
            mu = 2 * (self.model(self.pixelfade).pow(2).mean().pow(.5) /
                      self.pixelfade.pow(2).mean().pow(.5))
        if self.regularized:
            print(f"\n tradeoff parameter, mu = {mu:.4e}")
        self.mu = mu

        self.conditional = conditional
        if tol is None:
            # tentative default value
            tol = self.pixelfade.norm() / 1e4 * (1 + 5 ** .5) / 2
        print(f"\n threshold for delta_x, tolerance = {tol:.5e}")

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
            last_r_loss = self.loss[-1][0]
            if self.verbose:
                if self.regularized:
                    print(f"""found a regularized geodesic after {i} iterations,
            achieving a representational path energy of {last_r_loss:.2e}""")
                else:
                    print(f"""found a representational geodesic after {i} iterations,
            achieving a representational path energy of {last_r_loss:.2e}""")

            if self.conditional and i < max_iter:
                if self.verbose:
                    print("""starting search for conditional geodesic""")
                delta_x_outer = torch.ones(1)
                # self.alpha = .1
                self.optimizer_outer = optim.Adam([self.x], lr=learning_rate,
                                                  amsgrad=True)
                self.outer_step_stamp = []
                while delta_x_outer > tol and i < max_iter:
                    self.outer_step_stamp.append(i)
                    xprev_outer = self.x.clone()
                    self._outer_optimizer_step()
                    delta_x = torch.norm(self.x - xprev_outer)

                    # project back onto set of representational geodesics
                    # delta_x > tol and
                    current_r_loss = self._step_energy(
                        self._analyze(torch.cat([self.xA, self.x, self.xB]))
                                                       ).mean().item()
                    while last_r_loss < current_r_loss and i < max_iter:
                        _ = self._optimizer_step(pbar)
                        i += 1
                        current_r_loss = self.loss[-1][0]

                    delta_x_outer = torch.norm(self.x - xprev_outer)
                    # display some information
                    pbar.set_postfix(OrderedDict([
                            ('delta_x_outer', f"{delta_x_outer.item():.5e}")]))
                    pbar.update(1)
                    i += 1

        self._populate_geodesic()

    def plot_loss(self, share_y=False, show_switches=True):
        """display the evolution of representation and signal path energies
        along the optimisation process.

        Parameters
        ----------
        share_y : bool, optional
            whether the representational path energy and signal path energy
            share a common y axis, by default False because these quantities
            have different "units"
        show_switches : bool, optional
            indicate steps of the outer loop by vertical lines (relevant to
            the conditional geodesic implementation), by default True

        Returns
        -------
        fig: matplotlib.figure.Figure
        """
        if share_y:
            fig, ax = plt.subplots(1, 1)
            ax.plot(self.loss)
            ax.set(yscale='log',
                   xlabel='iteration number',
                   ylabel='loss value',
                   title='evolution of path energy')
            ax.legend(labels=[r'$E[f(\gamma)]$', r'$E[\gamma]$'], loc=1)
        else:
            r_loss = [loss[0] for loss in self.loss]
            s_loss = [loss[1] for loss in self.loss]
            t = range(len(r_loss))

            fig, ax1 = plt.subplots()
            ax1.set(title='evolution of path energy')

            color = 'C0'
            ax1.set_xlabel('iteration number')
            ax1.set(yscale='log')
            ax1.set_ylabel(r'$E[f(\gamma)]$', color=color)
            ax1.plot(t, r_loss, color=color)
            ax1.tick_params(axis='y')  #labelcolor=color

            ax2 = ax1.twinx()
            color = 'C1'
            ax2.set(yscale='log')
            ax2.set_ylabel(r'$E[\gamma]$', color=color)
            ax2.plot(t, s_loss, color=color)
            ax2.tick_params(axis='y')  #labelcolor=color
        if show_switches and hasattr(self, 'outer_step_stamp'):
            for t in self.outer_step_stamp:
                fig.axes[0].axvline(t, alpha=.1)
        fig.tight_layout()
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

    def plot_PC_projections(self, video=None):
        """Two dimensional visual comparison of the geodesic and pixelfade sequences
        both in signal space and in representation space

        Parameters
        ----------
        video : torch.Tensor, optional
            natural video that bridges the anchor points

        Returns
        -------
        fig: matplotlib.figure.Figure
        """
        if not hasattr(self, 'geodesic'):
            self._populate_geodesic()

        g = self.geodesic.view(self.n_steps+1, -1)
        p = self.pixelfade.view(self.n_steps+1, -1)
        X = torch.cat([g, p], 0)
        if video is not None:
            v = video.view(self.n_steps+1, -1)
            X = torch.cat([X, v], 0)

        print("signal PR", torch.trace(X @ X.T) ** 2 / torch.norm(X @ X.T, p='fro') ** 2)
        U, s, V = torch.svd(X)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        ax1.plot(V[:, 0] @ g.T, V[:, 1] @ g.T, 'r-o', label='geodesic')
        ax1.plot(V[:, 0] @ p.T, V[:, 1] @ p.T, 'g-o', label='pixelfade')
        if video is not None:
            ax1.plot(V[:, 0] @ v.T, V[:, 1] @ v.T, 'b-o', label='video')

        ax1.set(xlabel='PC1',
                ylabel='PC2',
                title="pixel space")

        g = self.model(self.geodesic).view(self.n_steps+1, -1)
        p = self.model(self.pixelfade).view(self.n_steps+1, -1)
        X = torch.cat([g, p], 0)
        if video is not None:
            v = self.model(video).view(self.n_steps+1, -1)
            X = torch.cat([X, v], 0)

        print("representation PR", torch.trace(X @ X.T) ** 2 / torch.norm(X @ X.T, p='fro') ** 2)
        U, s, V = torch.svd(X)

        ax2.plot(V[:, 0] @ g.T, V[:, 1] @ g.T, 'r-o', label='geodesic')
        ax2.plot(V[:, 0] @ p.T, V[:, 1] @ p.T, 'g-o', label='pixelfade')
        if video is not None:
            ax2.plot(V[:, 0] @ v.T, V[:, 1] @ v.T, 'b-o', label='video')

        ax2.set(xlabel='PC1',
                ylabel='PC2',
                title="representation space")
        ax2.legend(loc='best')

        plt.tight_layout()
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
