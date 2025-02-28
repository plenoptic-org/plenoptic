import warnings
from collections import OrderedDict
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
from deprecated.sphinx import deprecated
from torch import Tensor
from tqdm.auto import tqdm

from ..tools.convergence import pixel_change_convergence
from ..tools.data import to_numpy
from ..tools.optim import penalize_range
from ..tools.straightness import (
    deviation_from_line,
    make_straight_line,
    sample_brownian_bridge,
)
from ..tools.validate import validate_input, validate_model
from .synthesis import OptimizedSynthesis


@deprecated(
    "Geodesic is not robust enough yet, see https://github.com/plenoptic-org/geodesics for ongoing development",  # noqa: E501
    "1.1.0",
)
class Geodesic(OptimizedSynthesis):
    r"""Synthesize an approximate geodesic between two images according to a model.

    This method can be used to visualize and refine the invariances of a
    model's representation as described in [1]_.

    NOTE: This synthesis method is still under construction. It will run, but
    it might not find the most informative geodesic.

    Parameters
    ----------
    image_a, image_b
        Start and stop anchor points of the geodesic, of shape (1, channel,
        height, width).
    model
        an analysis model that computes representations on signals like `image_a`.
    n_steps
        the number of steps (i.e., transitions) in the trajectory between the
        two anchor points.
    initial_sequence
        initialize the geodesic with pixel linear interpolation
        (``'straight'``), or with a brownian bridge between the two anchors
        (``'bridge'``).
    range_penalty_lambda
        strength of the regularizer that enforces the allowed_range. Must be
        non-negative.
    allowed_range
        Range (inclusive) of allowed pixel values. Any values outside this
        range will be penalized.

    Attributes
    ----------
    geodesic: Tensor
        the synthesized sequence of images between the two anchor points that
        minimizes representation path energy, of shape ``(n_steps+1, channel,
        height, width)``. It starts with image_a and ends with image_b.
    pixelfade: Tensor
        the straight interpolation between the two anchor points,
        used as reference
    losses : Tensor
        A list of our loss over iterations.
    gradient_norm : list
        A list of the gradient's L2 norm over iterations.
    pixel_change_norm : list
        A list containing the L2 norm of the pixel change over iterations
        (``pixel_change_norm[i]`` is the pixel change norm in
        ``geodesic`` between iterations ``i`` and ``i-1``).
    step_energy: Tensor
        step lengths in representation space, stored along the optimization
        process.
    dev_from_line: Tensor
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
        https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Henaff16b

    """

    def __init__(
        self,
        image_a: Tensor,
        image_b: Tensor,
        model: torch.nn.Module,
        n_steps: int = 10,
        initial_sequence: Literal["straight", "bridge"] = "straight",
        range_penalty_lambda: float = 0.1,
        allowed_range: tuple[float, float] = (0, 1),
    ):
        super().__init__(range_penalty_lambda, allowed_range)
        validate_input(image_a, no_batch=True, allowed_range=allowed_range)
        validate_input(image_b, no_batch=True, allowed_range=allowed_range)
        validate_model(
            model,
            image_shape=image_a.shape,
            image_dtype=image_a.dtype,
            device=image_a.device,
        )

        self.n_steps = n_steps
        self._model = model
        self._image_a = image_a
        self._image_b = image_b
        self.pixelfade = make_straight_line(image_a, image_b, n_steps)
        self._initialize(initial_sequence, image_a, image_b, n_steps)
        self._dev_from_line = []
        self._step_energy = []

    def _initialize(self, initial_sequence, start, stop, n_steps):
        """initialize the geodesic

        Parameters
        ----------
        initial_sequence
            initialize the geodesic with pixel linear interpolation
            (``'straight'``), or with a brownian bridge between the two anchors
            (``'bridge'``).
        """
        if initial_sequence == "bridge":
            geodesic = sample_brownian_bridge(start, stop, n_steps)
        elif initial_sequence == "straight":
            geodesic = make_straight_line(start, stop, n_steps)
        else:
            raise ValueError(
                f"Don't know how to handle initial_sequence={initial_sequence}"
            )
        _, geodesic, _ = torch.split(geodesic, [1, n_steps - 1, 1])
        self._initial_sequence = initial_sequence
        geodesic.requires_grad_()
        self._geodesic = geodesic

    def synthesize(
        self,
        max_iter: int = 1000,
        optimizer: torch.optim.Optimizer | None = None,
        store_progress: bool | int = False,
        stop_criterion: float | None = None,
        stop_iters_to_check: int = 50,
    ):
        """Synthesize a geodesic via optimization.

        Parameters
        ----------
        max_iter
            The maximum number of iterations to run before we end synthesis
            (unless we hit the stop criterion).
        optimizer
            The optimizer to use. If None and this is the first time calling
            synthesize, we use Adam(lr=.001, amsgrad=True); if synthesize has
            been called before, this must be None and we reuse the previous
            optimizer.
        store_progress
            Whether we should store the step energy and deviation of the
            representation from a straight line. If False, we don't save
            anything. If True, we save every iteration. If an int, we save
            every ``store_progress`` iterations (note then that 0 is the same
            as False and 1 the same as True).
        stop_criterion
            If pixel_change_norm (i.e., the norm of the difference in
            ``self.geodesic`` from one iteration to the next) over the past
            ``stop_iters_to_check`` has been less than ``stop_criterion``, we
            terminate synthesis. If None, we pick a default value based on the
            norm of ``self.pixelfade``.
        stop_iters_to_check
            How many iterations back to check in order to see if
            pixel_change_norm has stopped decreasing (for ``stop_criterion``).

        """
        if stop_criterion is None:
            # semi arbitrary default choice of tolerance
            stop_criterion = (
                torch.linalg.vector_norm(self.pixelfade, ord=2) / 1e4 * (1 + 5**0.5) / 2
            )
        print(f"\n Stop criterion for pixel_change_norm = {stop_criterion:.5e}")

        self._initialize_optimizer(optimizer, "_geodesic", 0.001)

        # get ready to store progress
        self.store_progress = store_progress

        pbar = tqdm(range(max_iter))
        for _ in pbar:
            self._store(len(self.losses))

            loss = self._optimizer_step(pbar)

            if not torch.isfinite(loss):
                raise ValueError("Found a NaN in loss during optimization.")

            if self._check_convergence(stop_criterion, stop_iters_to_check):
                warnings.warn("Pixel change norm has converged, stopping synthesis")
                break

        pbar.close()

    def objective_function(self, geodesic: Tensor | None = None) -> Tensor:
        """Compute geodesic synthesis loss.

        This is the path energy (i.e., squared L2 norm of each step) of the
        geodesic's model representation, with the weighted range penalty.

        Additionally, caches:

        - ``self._geodesic_representation = self.model(geodesic)``

        - ``self._most_recent_step_energy = self._calculate_step_energy(
            self._geodesic_representation)``

        These are cached because we might store them (if ``self.store_progress
        is True``) and don't want to recalcualte them

        Parameters
        ----------
        geodesic
            Geodesic to check. If None, we use ``self.geodesic``.

        Returns
        -------
        loss

        """
        if geodesic is None:
            geodesic = self.geodesic
        self._geodesic_representation = self.model(geodesic)
        self._most_recent_step_energy = self._calculate_step_energy(
            self._geodesic_representation
        )
        loss = self._most_recent_step_energy.mean()
        range_penalty = penalize_range(self.geodesic, self.allowed_range)
        return loss + self.range_penalty_lambda * range_penalty

    def _calculate_step_energy(self, z):
        """calculate the energy (i.e. squared l2 norm) of each step in `z`."""
        velocity = torch.diff(z, dim=0)
        step_energy = torch.linalg.vector_norm(velocity, ord=2, dim=[1, 2, 3]) ** 2
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
        - return pixel_change_norm, the norm of the step just taken
        """
        last_iter_geodesic = self._geodesic.clone()
        loss = self.optimizer.step(self._closure)
        self._losses.append(loss.item())

        grad_norm = torch.linalg.vector_norm(self._geodesic.grad.data, ord=2, dim=None)
        self._gradient_norm.append(grad_norm)

        pixel_change_norm = torch.linalg.vector_norm(
            self._geodesic - last_iter_geodesic, ord=2, dim=None
        )
        self._pixel_change_norm.append(pixel_change_norm)
        # displaying some information
        pbar.set_postfix(
            OrderedDict(
                [
                    ("loss", f"{loss.item():.4e}"),
                    ("gradient norm", f"{grad_norm.item():.4e}"),
                    ("pixel change norm", f"{pixel_change_norm.item():.5e}"),
                ]
            )
        )
        return loss

    def _check_convergence(
        self, stop_criterion: float, stop_iters_to_check: int
    ) -> bool:
        """Check whether the pixel change norm has stabilized and, if so, return True.

         Have we been synthesizing for ``stop_iters_to_check`` iterations?
         | |
        no yes
         | '---->Is ``(self.pixel_change_norm[-stop_iters_to_check:] < stop_criterion).all()``?
         |        |
         |      no |
         |       | yes
         <-------' |
         |         '------> return ``True``
         |
         '---------> return ``False``

        Parameters
        ----------
        stop_criterion
            If the pixel change norm has been less than ``stop_criterion`` for all
            of the past ``stop_iters_to_check``, we terminate synthesis.
        stop_iters_to_check
            How many iterations back to check in order to see if the
            pixel change norm has stopped decreasing (for ``stop_criterion``).

        Returns
        -------
        loss_stabilized :
            Whether the pixel change norm has stabilized or not.

        """  # noqa: E501
        return pixel_change_convergence(self, stop_criterion, stop_iters_to_check)

    def calculate_jerkiness(self, geodesic: Tensor | None = None) -> Tensor:
        """
        Compute the alignment of representation's acceleration to model local curvature.

        This is the first order optimality condition for a geodesic, and can be
        used to assess the validity of the solution obtained by optimization.

        Parameters
        ----------
        geodesic
            Geodesic to check. If None, we use ``self.geodesic``. Must have a
            gradient attached.

        Returns
        -------
        jerkiness

        """
        if geodesic is None:
            geodesic = self.geodesic
        geodesic_representation = self.model(geodesic)
        velocity = torch.diff(geodesic_representation, dim=0)
        acceleration = torch.diff(velocity, dim=0)
        acc_magnitude = torch.linalg.vector_norm(
            acceleration, ord=2, dim=[1, 2, 3], keepdim=True
        )
        acc_direction = torch.div(acceleration, acc_magnitude)
        # we slice the output of the VJP, rather than slicing geodesic, because
        # slicing interferes with the gradient computation:
        # https://stackoverflow.com/a/54767100
        accJac = self._vector_jacobian_product(
            geodesic_representation[1:-1], geodesic, acc_direction
        )[1:-1]
        step_jerkiness = torch.linalg.vector_norm(accJac, dim=[1, 2, 3], ord=2) ** 2
        return step_jerkiness

    def _vector_jacobian_product(self, y, x, a):
        """compute vector-jacobian product: $a^T dy/dx = dy/dx^T a$,
        and allow for further gradient computations by retaining,
        and creating the graph.
        """
        accJac = autograd.grad(y, x, a, retain_graph=True, create_graph=True)[0]
        return accJac

    def _store(self, i: int) -> bool:
        """Store step_energy and dev_from_line, if appropriate.

        if it's the right iteration, we update ``step_energy`` and
        ``dev_from_line``.

        Parameters
        ----------
        i
            the current iteration

        Returns
        -------
        stored
            True if we stored this iteration, False if not.

        """
        if self.store_progress and (i % self.store_progress == 0):
            # want these to always be on cpu, to reduce memory use for GPUs
            try:
                self._step_energy.append(
                    self._most_recent_step_energy.detach().to("cpu")
                )
                self._dev_from_line.append(
                    torch.stack(
                        deviation_from_line(
                            self._geodesic_representation.detach().to("cpu")
                        )
                    ).T
                )
            except AttributeError:
                # the first time _store is called (i.e., before optimizer is
                # stepped for first time) those attributes won't be
                # initialized
                geod_rep = self.model(self.geodesic)
                self._step_energy.append(
                    self._calculate_step_energy(geod_rep).detach().to("cpu")
                )
                self._dev_from_line.append(
                    torch.stack(deviation_from_line(geod_rep.detach().to("cpu"))).T
                )
            stored = True
        else:
            stored = False
        return stored

    def save(self, file_path: str):
        r"""Save all relevant variables in .pt file.

        See ``load`` docstring for an example of use.

        Parameters
        ----------
        file_path : str
            The path to save the Geodesic object to

        """
        save_io_attrs = [
            ("_model", ("_geodesic",)),
        ]
        save_state_dict_attrs = ["_optimizer"]
        save_attrs = [
            k
            for k in vars(self)
            if k not in [k[0] for k in save_io_attrs] + save_state_dict_attrs
        ]
        super().save(file_path, save_attrs, save_io_attrs, save_state_dict_attrs)

    def to(self, *args, **kwargs):
        r"""Moves and/or casts the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)

        .. function:: to(dtype, non_blocking=False)

        .. function:: to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module

        """
        attrs = [
            "_image_a",
            "_image_b",
            "_geodesic",
            "_model",
            "_step_energy",
            "_dev_from_line",
            "pixelfade",
        ]
        super().to(*args, attrs=attrs, **kwargs)

    def load(
        self,
        file_path: str,
        map_location: str | None = None,
        **pickle_load_args,
    ):
        r"""Load all relevant stuff from a .pt file.

        This should be called by an initialized ``Geodesic`` object.

        Note this operates in place and so doesn't return anything.

        Parameters
        ----------
        file_path : str
            The path to load the synthesis object from
        map_location : str, optional
            map_location argument to pass to ``torch.load``. If you save
            stuff that was being run on a GPU and are loading onto a
            CPU, you'll need this to make sure everything lines up
            properly. This should be structured like the str you would
            pass to ``torch.device``
        pickle_load_args :
            any additional kwargs will be added to ``pickle_module.load`` via
            ``torch.load``, see that function's docstring for details.

        Examples
        --------
        >>> geo = po.synth.Geodesic(img_a, img_b, model)
        >>> geo.synthesize(max_iter=10, store_progress=True)
        >>> geo.save('geo.pt')
        >>> geo_copy = po.synth.Geodesic(img_a, img_b, model)
        >>> geo_copy.load('geo.pt')

        Note that you must create a new instance of the Synthesis object and
        *then* load.

        """
        check_attributes = [
            "_image_a",
            "_image_b",
            "n_steps",
            "_initial_sequence",
            "_range_penalty_lambda",
            "_allowed_range",
            "pixelfade",
        ]
        check_io_attrs = [("_model", ("_geodesic",))]
        super().load(
            file_path,
            "losses",
            map_location=map_location,
            check_attributes=check_attributes,
            check_io_attributes=check_io_attrs,
            state_dict_attributes=["_optimizer"],
            **pickle_load_args,
        )
        # make this require a grad again
        self._geodesic.requires_grad_()
        # these are always supposed to be on cpu, but may get copied over to
        # gpu on load (which can cause problems when resuming synthesis), so
        # fix that.
        if len(self._dev_from_line) and self._dev_from_line[0].device.type != "cpu":
            self._dev_from_line = [dev.to("cpu") for dev in self._dev_from_line]
        if len(self._step_energy) and self._step_energy[0].device.type != "cpu":
            self._step_energy = [step.to("cpu") for step in self._step_energy]

    @property
    def model(self):
        return self._model

    @property
    def image_a(self):
        return self._image_a

    @property
    def image_b(self):
        return self._image_b

    # self._geodesic contains the portion we're optimizing, but self.geodesic
    # combines this with the end points
    @property
    def geodesic(self):
        return torch.cat([self.image_a, self._geodesic, self.image_b])

    @property
    def step_energy(self):
        """
        Squared L2 norm of transition between geodesic frames in representation space.

        Has shape ``(np.ceil(synth_iter/store_progress), n_steps)``, where
        ``synth_iter`` is the number of iterations of synthesis that have
        happened.

        """
        return torch.stack(self._step_energy)

    @property
    def dev_from_line(self):
        """Deviation of representation each from of ``self.geodesic`` from a straight
        line.

        Has shape ``(np.ceil(synth_iter/store_progress), n_steps+1, 2)``, where
        ``synth_iter`` is the number of iterations of synthesis that have
        happened. For final dimension, the first element is the Euclidean
        distance along the straight line and the second is the Euclidean
        distance to the line.

        """
        return torch.stack(self._dev_from_line)


@deprecated(
    "Geodesic is not robust enough yet, see https://github.com/plenoptic-org/geodesics for ongoing development",  # noqa: E501
    "1.1.0",
)
def plot_loss(
    geodesic: Geodesic, ax: mpl.axes.Axes | None = None, **kwargs
) -> mpl.axes.Axes:
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

    Notes
    -----

    """
    if ax is None:
        ax = plt.gca()
    ax.semilogy(geodesic.losses, **kwargs)
    ax.set(xlabel="Synthesis iteration", ylabel="Loss")
    return ax


@deprecated(
    "Geodesic is not robust enough yet, see https://github.com/plenoptic-org/geodesics for ongoing development",  # noqa: E501
    "1.1.0",
)
def plot_deviation_from_line(
    geodesic: Geodesic,
    natural_video: Tensor | None = None,
    ax: mpl.axes.Axes | None = None,
) -> mpl.axes.Axes:
    """Visual diagnostic of geodesic linearity in representation space.

    This plot illustrates the deviation from the straight line connecting
    the representations of a pair of images, for different paths
    in representation space.

    Parameters
    ----------
    geodesic :
        Geodesic object to visualize.
    natural_video :
        Natural video that bridges the anchor points, for comparison.
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
    geodesic to be straight (for example if the representation is normalized,
    all paths are constrained to live on a hypershpere). Nevertheless, if the
    representation is able to linearize the transformation between the anchor
    images, then we expect that both the ground truth natural video sequence
    and the geodesic will deviate from straight line similarly. By contrast the
    pixel-based interpolation will deviate significantly more from a straight
    line.

    """
    if ax is None:
        ax = plt.gca()

    pixelfade_dev = deviation_from_line(geodesic.model(geodesic.pixelfade))
    ax.plot(*[to_numpy(d) for d in pixelfade_dev], "g-o", label="pixelfade")

    geodesic_dev = deviation_from_line(geodesic.model(geodesic.geodesic).detach())
    ax.plot(*[to_numpy(d) for d in geodesic_dev], "r-o", label="geodesic")

    if natural_video is not None:
        video_dev = deviation_from_line(geodesic.model(natural_video))
        ax.plot(*[to_numpy(d) for d in video_dev], "b-o", label="natural video")

    ax.set(
        xlabel="Distance along representation line",
        ylabel="Distance from representation line",
        title="Deviation from the straight line",
    )
    ax.legend(loc=1)

    return ax
