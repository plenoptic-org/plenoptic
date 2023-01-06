from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor
from tqdm.auto import tqdm
import warnings
from typing import Union, Tuple
from typing_extensions import Literal

from ..tools.optim import penalize_range
from ..tools.validate import validate_input, validate_model
from ..tools.straightness import (deviation_from_line, make_straight_line,
                                  sample_brownian_bridge)


class Geodesic(nn.Module):
    r'''Synthesize an approximate geodesic between two images according to a model.

    This method can be used to visualize and refine the invariances of a
    model's representation as described in [1]_.

    Parameters
    ----------
    image_a, image_b:
        Start and stop anchor points of the geodesic, of shape [1, C, H, W]
        with values in range [0, 1].
    model:
        an analysis model that computes representations on signals like `image_a`.
    n_steps:
        the number of steps (i.e., transitions) in the trajectory between the
        two anchor points.
    init:
        initialize the geodesic with pixel linear interpolation
        (``'straight'``), or with a brownian bridge between the two anchors
        (``'bridge'``).
    range_penalty_lambda :
        strength of the regularizer that enforces the allowed_range. Must be
        non-negative.
    allowed_range :
        Range (inclusive) of allowed pixel values. Any values outside this
        range will be penalized.

    Attributes
    ----------
    geodesic: Tensor
        the synthesized sequence of images between the two anchor points that
        minimizes representation path energy, of shape ``(n_steps+1, C, H,
        W)``. It starts with image_a and ends with image_b.
    pixelfade: Tensor
        the straight interpolation between the two anchor points,
        used as reference
    step_energy: list of Tensor
        step lengths in representation space, stored along the optimization
        process
    dev_from_line: list of Tensor
        deviation of the representation to the straight line interpolation,
        measures distance from straight line and distance along straight line,
        stored along the optimization process

    Notes
    -----
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

    def __init__(self, image_a: Tensor, image_b: Tensor,
                 model: torch.nn.Module, n_steps: int = 10,
                 init: Literal['straight', 'bridge'] = 'straight',
                 allowed_range: Tuple[float, float] = (0, 1),
                 range_penalty_lambda: float = .1):
        super().__init__()
        validate_input(image_a, no_batch=True, allowed_range=allowed_range)
        validate_input(image_b, no_batch=True, allowed_range=allowed_range)
        validate_model(model)

        self.n_steps = n_steps
        self.image_shape = image_a.shape
        self.model = model.eval()
        start = image_a.clone().view(1, -1)
        stop = image_b.clone().view(1, -1)
        self.pixelfade = self._initialize('straight', start, stop, n_steps
                                          ).view(n_steps+1, *image_a.shape[1:])

        if range_penalty_lambda < 0:
            raise Exception("range_penalty_lambda must be non-negative!")
        self.range_penalty_lambda = range_penalty_lambda
        self.allowed_range = allowed_range
        xinit = self._initialize(init, start, stop, n_steps
                                 ).view(n_steps+1, -1)
        self._xA, x, self._xB = torch.split(xinit, [1, n_steps-1, 1])
        self._x = nn.Parameter(x.requires_grad_())
        # HAVE self.geodesic be the optimized variable, and just zero out the
        # relevant parts of its gradient and reshape in calculate jerkiness.
        # prevents geodesic and self.x getting out of sync and makes it easier
        # for user to set the relevant parameter
        #
        self._x = nn.Parameter(xinit.requires_grad_())

        self.optimizer = None
        self.loss = []
        self.dev_from_line = []
        self.step_energy = []

    def _initialize(self, init, start, stop, n_steps):
        """initialize the geodesic

        Parameters
        ----------
        init:
            initialize the geodesic with pixel linear interpolation
            (``'straight'``), or with a brownian bridge between the two anchors
            (``'bridge'``).
        """
        if init == 'bridge':
            x = sample_brownian_bridge(start, stop, n_steps)
        elif init == 'straight':
            x = make_straight_line(start, stop, n_steps)
        else:
            raise Exception(f"Don't know how to handle init={init}")
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
        xprev = self._x.clone()
        self.optimizer.zero_grad()

        x = torch.cat([self._xA, self._x, self._xB])
        y = self._analyze(x)

        # representation's path energy
        step_energy = self._step_energy(y)
        loss = step_energy.mean()

        self.loss.append(loss.item())
        loss = loss + self.range_penalty_lambda * penalize_range(self._x, self.allowed_range)

        if not torch.isfinite(loss):
            raise Exception('found a NaN in the loss during optimization')
        loss.backward()
        # self._x.grad[0] = 0
        # self._x.grad[-1] = 0

        grad_norm = torch.norm(self._x.grad.data)
        if not torch.isfinite(grad_norm):
            raise Exception('found a NaN in the gradients during optimization')
        self.optimizer.step()

        # NAME THE SAME as metamer
        delta_x = torch.norm(self._x - xprev)
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

    def _init_optimizer(self, optimizer):
        """Initialize optimizer."""
        if optimizer is None:
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam([self._x],
                                                  lr=.001, amsgrad=True)
        else:
            if self.optimizer is not None:
                raise Exception("When resuming synthesis, optimizer arg must be None!")
            params = optimizer.param_groups[0]['params']
            if len(params) != 1 or not torch.equal(params[0], self._x):
                raise Exception("For geodesic synthesis, optimizer must have one "
                                "parameter, the metamer we're synthesizing.")
            self.optimizer = optimizer

    # - CHANGE functionality for renamed arguments
    def synthesize(self, max_iter: int = 1000,
                   optimizer: Union[None, torch.optim.Optimizer] = None,
                   verbose=True, tol=1e-3,
                   store_progress: Union[bool, int] = False,
                   stop_criterion: Union[float, None] = None,
                   stop_iters_to_check: int = 50) -> Tensor:
        """Synthesize a geodesic via optimization.

        Parameters
        ----------
        max_iter: int, optional
            maximum number of steps taken by the optimization algorithm.
        optimizer :
            The optimizer to use. If None and this is the first time calling
            synthesize, we use Adam(lr=.01, amsgrad=True); if synthesize has
            been called before, this must be None and we reuse the previous
            optimizer.
        store_progress :
            Whether we should store the representation of the metamer
            and the metamer image in progress on every iteration. If
            False, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress``
            iterations (note then that 0 is the same as False and 1 the
            same as True).
        stop_criterion :
            If the loss over the past ``stop_iters_to_check`` has changed
            less than ``stop_criterion``, we terminate synthesis. If None,
            we pick a default value based on the norm of ``self.pixelfade``.
        stop_iters_to_check :
            How many iterations back to check in order to see if the
            loss has stopped decreasing (for ``stop_criterion``).

        """
        self.verbose = verbose
        if tol is None:
            # semi arbitrary default choice of tolerance
            tol = self.pixelfade.norm() / 1e4 * (1 + 5 ** .5) / 2
        print(f"\n threshold for delta_x, tolerance = {tol:.5e}")

        self._init_optimizer(optimizer)

        delta_x = torch.ones(1)
        i = 0
        with tqdm(range(max_iter)) as pbar:
            # project onto set of representational geodesics
            while delta_x > tol and i < max_iter:
                delta_x = self._optimizer_step(pbar)
                i += 1
        # self._populate_geodesic()

    def _populate_geodesic(self):
        """Help format the current optimization variable `x` into a geodesic
        attribute that can be used later.

        It joins the endpoints, reshapes the variable into a sequence
        of images, detaches the gradients and clips the range to [0, 1].
        """
        self.geodesic = torch.cat([self._xA, self._x, self._xB]
                                  ).reshape(
                                (self.n_steps+1, *self.image_shape[1:])
                                            ).clamp(0, 1).detach()

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
        accJac = self._vector_jacobian_product(y[1:-1], self._x,
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
        y = self._analyze(torch.cat([self._xA, self._x, self._xB]))
        return self._step_jerkiness(y).detach()

def plot_loss(geodesic: Geodesic,
              ax: Union[mpl.axes.Axes, None] = None,
              **kwargs) -> mpl.axes.Axes:
    """Plot synthesis loss.

    Parameters
    ----------
    geodesic :
        Geodesic object whose synthesis loss we want to plot.
    ax :
        If not None, the axis to plot this representation on. If
        None, we call ``plt.gca()``
    kwargs :
        passed to plt.semilogy

    Returns
    -------
    ax :
        Axes containing the plot.

    """
    if ax is None:
        ax = plt.gca()
    ax.semilogy(geodesic.loss, **kwargs)
    ax.set(xlabel='Synthesis iteration',
           ylabel='Loss')
    return ax

def plot_deviation_from_line(geodesic: Geodesic,
                             video: Union[Tensor, None] = None,
                             ax: Union[mpl.axes.Axes, None] = None
                             ) -> mpl.axes.Axes:
    """Visual diagnostic of geodesic linearity in representation space.

    This plot illustrates the deviation from the straight line connecting
    the representations of a pair of images, for different paths
    in representation space.

    Parameters
    ----------
    geodesic :
        Geodesic object to visualize.
    video :
        Natural video that bridges the anchor points
    ax :
        If not None, the axis to plot this representation on. If
        None, we call ``plt.gca()``

    Returns
    -------
    ax:
        Axes containing the plot

    Notes
    -----
    Axes are in the same units, normalized by the distance separating
    the end point representations.

    Knots along each curve indicate samples used to compute the path.

    When the representation is non-linear it may not be feasible for the
    geodesic to be straight (for example if the representation is
    normalized, all paths are constrained to live on a hypershpere).
    Nevertheless, if the representation is able to linearize the
    transformation between the anchor images, then we expect that both
    the ground truth video sequence and the geodesic will deviate from
    straight line similarly. By contrast the pixel-based interpolation
    will deviate significantly more from a straight line.

    """
    if not hasattr(geodesic, 'geodesic'):
        geodesic._populate_geodesic()

    if ax is None:
        ax = plt.gca()

    ax.plot(*deviation_from_line(geodesic.model(geodesic.pixelfade
                                                ).view(geodesic.n_steps+1, -1)
                                 ), 'g-o', label='pixelfade')

    deviation = deviation_from_line(
        geodesic.model(geodesic.geodesic
                       ).view(geodesic.n_steps+1, -1))
    ax.plot(*deviation, 'r-o', label='geodesic')

    if video is not None:
        ax.plot(*deviation_from_line(geodesic.model(video
                                                    ).view(geodesic.n_steps+1, -1)
                                     ), 'b-o', label='video')
    ax.set(xlabel='distance along representation line',
           ylabel='distance from representation line',
           title='deviation from the straight line')
    ax.legend(loc=1)

    return ax
