---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

:::{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`ds_away.ipynb`**!

Run it in your browser: **{binder}`ds_away.ipynb`**!

:::

# away

In this notebook, we will create a datasaurus metamer where we have no points near the center of the scatter plot.

This notebook is intentionally brief: most of the code is hidden (you can expand the cells if you would like to see more details), and we only explain the penalty. See [](datasaurus-index) for an overview of the datasaurus dozen dataset.

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

import plenoptic as po

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72

plt.rcParams["animation.html"] = "html5"
# use single-threaded ffmpeg for animation writer
plt.rcParams["animation.writer"] = "ffmpeg"
plt.rcParams["animation.ffmpeg_args"] = ["-threads", "1"]
plt.rcParams["savefig.bbox"] = "tight"

po.set_seed(0)
# To guarantee reproducibility for this example on the GPU, we must tell torch to use
# deterministic algorithms. Note this will make things slower! See "Reproducibility and
# Compatibility" in the docs for more details.
torch.use_deterministic_algorithms(True)


# Model definition, as in top-level notebook
class DatasaurusModel(torch.nn.Module):
    def __init__(self, n_pts=None, dtype=None, include_redundant_stats=True):
        """
        Create model to measure datasaurus stats.

        Parameters
        ----------
        n_pts
            Number of data points in the dataset we'll use the model for. Used to cache
            a corresponding vector of ones for computing linear regression.
        dtype
            dtype for the dataset we'll use the model for. Used to cache
            a corresponding vector of ones for computing linear regression.
        include_redundant_stats
            Whether to include the redundant stats (linear regression slope and
            intercept, coefficient of determination).
        """
        super().__init__()
        # cache ones to save time
        if n_pts is not None:
            self._ones = torch.ones(n_pts, dtype=dtype)
        else:
            self._ones = None
        self.include_redundant_stats = include_redundant_stats
        # This model has no trainable parameters, so it's always in eval mode
        self.eval()

    def _prepare_X(self, x):
        """Append vector of ones to matrix for linear regression (for intercept)."""
        ones = self._ones if self._ones is None else torch.ones_like(x)
        return torch.stack([ones, x], -1)

    def _compute_linreg(self, x, y):
        """Compute linear regression (with intercept) between x and y."""
        X = self._prepare_X(x)
        # unsqueezing and squeezing needed because of https://github.com/pytorch/pytorch/issues/158169
        return torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze()

    def _compute_coeff_determination(self, x, y, solution):
        """Compute R^2 for linera regression fit."""
        X = self._prepare_X(x)
        pred_y = torch.einsum("x, n x -> n", solution, X)
        ss_res = (y - pred_y).pow(2).sum()
        ss_tot = (y - y.mean()).pow(2).sum()
        return 1 - (ss_res / ss_tot)

    def _vmap_coeff_determination(self, x, solution):
        """vmap _compute_coeff_determiniation across dim=0."""
        f = torch.func.vmap(lambda x, solt: self._compute_coeff_determination(*x, solt))
        return f(x, solution).unsqueeze(-1)

    def forward(self, data):
        """Compute summary statistics on data."""
        if data.ndim == 2:
            data = data.unsqueeze(0)
        elif data.ndim != 3:
            raise ValueError("data must be 2 or 3d!")
        stats = []
        stats.append(data.mean(-1))
        stats.append(data.std(-1))
        if self.include_redundant_stats:
            solution = torch.func.vmap(lambda x: self._compute_linreg(*x))(data)
            stats.append(solution)
        crosscorr = torch.func.vmap(lambda x: torch.corrcoef(x)[0, 1])(data)
        stats.append(crosscorr.unsqueeze(-1))
        if self.include_redundant_stats:
            stats.append(self._vmap_coeff_determination(data, solution))
        return torch.cat(stats, -1)

    def plot_representation(self, data, ax=None, style="stem", figsize=(6, 3)):
        """
        Plot model representation of data.

        We plot the representation as stem plots (if style=="stem") or dashed
        horizontal lines (if style=="lines"), on two separate sub-axes. The grouping
        is determined by their approximate magnitude in the original dino dataset. The
        first contains ["x mean", "y mean", "x std", "y std", "linreg intercept"], while
        the second contains ["linreg slope", "correlation", and "R^2"].

        Parameters
        ----------
        data: torch.Tensor
            The data to show on the plot. Should look like the output of
            forward, with the exact same structure.
        ax: plt.Axes or None
            Axes where we will plot the data. If a plt.Axes instance, will
            subdivide into 2 new axes. If None, we create a new figure.
        style: {"stem", "lines"}
            If "stem", plot data as stem plot. If "lines", plot as dashed
            horizontal lines.
        figsize: tuple[int]
            The size of the figure to create. Ignored if ax is not None.

        Returns
        -------
        axes
            List of two axes containing the subplots.
        """
        data = po.to_numpy(data).squeeze()
        # Set up grid spec
        if ax is None:
            # we add 2 to order because we're adding one to get the
            # number of orientations and then another one to add an
            # extra column for the mean luminance plot
            fig = plt.figure(figsize=figsize, layout="constrained")
            gs = mpl.gridspec.GridSpec(1, 2, fig, width_ratios=[5, 3], wspace=0.35)
            axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
        elif isinstance(ax, mpl.axes.Axes) or len(ax) == 1:
            # want to make sure the axis we're taking over is basically invisible.
            ax = po.plot.display._clean_up_axes(
                ax, False, ["top", "right", "bottom", "left"], ["x", "y"]
            )
            gs = ax.get_subplotspec().subgridspec(
                1, 2, width_ratios=[5, 3], wspace=0.35
            )
            fig = ax.figure
            axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
        else:
            axes = ax
            fig = axes[0].figure

        labels = [
            r"$\bar{x}$",  # noqa: RUF027
            r"$\bar{y}$",  # noqa: RUF027
            r"$\sigma_x$",
            r"$\sigma_y$",
            r"$\beta_0$",
            r"$\beta_1$",
            "$r$",
            "$R^2$",
        ]
        cutoff = 5
        linewidth = 1
        for i, ax in enumerate(axes):
            if i == 0:
                slicer = slice(0, cutoff)
            elif i == 1:
                slicer = slice(cutoff, len(labels) + 1)
            y = data[slicer]
            labs = labels[slicer]
            x = np.arange(len(labs))

            if style == "stem":
                ax.stem(y)
            elif style == "lines":
                ax.hlines(y, x - linewidth / 2, x + linewidth / 2, "k", "--")
            ax.set_xticks(x, labs)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(1))
            ax.yaxis.set_minor_locator(mpl.ticker.AutoLocator())
        return axes


def single_scatter(xy, ax, title=None, xlim=(0, 100), ylim=(0, 100), **scatter_kwargs):
    ax.scatter(*xy, **scatter_kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect(1)
    ax.set(xlim=xlim, ylim=ylim)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(1))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(1))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoLocator())
    ax.xaxis.set_minor_locator(mpl.ticker.AutoLocator())
    return ax
```

The following plot shows the `away` dataset from the original datasaurus dozen, along with its representation.

```{code-cell} ipython3
:tags: [hide-input]

data = torch.load(po.data.fetch_data("datasaurus.tar.gz") / "datasaurus.pt")
categories = np.load(
    po.data.fetch_data("datasaurus.tar.gz") / "categories.npy", allow_pickle=True
)

model = DatasaurusModel(data.shape[1], data.dtype)

fig, axes = plt.subplots(
    2, 3, figsize=(8, 6), width_ratios=[5, 5, 3], layout="compressed"
)
for i, title in enumerate(["dino (target)", "away"]):
    d = data[categories == title].squeeze()
    single_scatter(d, axes[i, 0], title)
    model.plot_representation(model(d), axes[i, 1:])
    model.plot_representation(model(data)[0], axes[i, 1:], "lines")
```

Our intended shape here, as can be seen above, is a somewhat randomly-distributed bunch of points, with absolutely no points found in the center of the distribution. To encourage metamer synthesis to find such a dataset, we create a function, `away_penalty`, which computes the distance of each point to some (user-specified) center point and returns a penalty based on a Gaussian with mean 0 and a user-specified standard deviation. To use this penalty with synthesis, we define a center and standard deviation (try changing these to different values!) and combine the resulting value with a range penalty which requires all points to lie between 0 and 100.

```{code-cell} ipython3
def away_penalty(data, target_ctr, std):
    # Turn center into a tensor
    target_ctr = torch.as_tensor(target_ctr).unsqueeze(-1)
    # compute distance from that center
    r = (data - target_ctr).pow(2).sum(0).sqrt()
    # penalty is given by a gaussian with that center and specified std dev
    return torch.exp(-r.pow(2) / (2 * std**2)).mean()


# Try changing these to different values!
ctr = (50, 50)
std = 5


def penalty(x):
    range_penalty = po.regularize.penalize_range(x, (0, 100))
    away = away_penalty(x, ctr, std)
    return range_penalty + away


# data[0] is the dinosaur
met = po.Metamer(data[0], model, penalty_function=penalty, penalty_lambda=1)
met.setup(initial_image=100 * torch.rand_like(data[0]), optimizer=torch.optim.LBFGS)
met.synthesize(50, store_progress=True)
```

```{code-cell} ipython3
:tags: [hide-input]

# use one of our helper functions here.
from plenoptic.plot.display import _rescale_ylim, _update_stem


def penalty_draw(ax):
    ax.scatter(*ctr, marker="+", c="red")


def animate_datasaurus_metamer(
    met,
    model=None,
    initial_ylim=None,
    n_frames=50,
    highlight_swap_frame=None,
):
    if model is None:
        model = met.model
    # Initialize figure by plotting the first iteration
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9, 3),
        width_ratios=[5, 5, 3],
        layout="compressed",
        gridspec_kw={"wspace": 0.25},
    )
    plot_data = met.saved_metamer
    ani_data = po.to_numpy(plot_data)
    ani_rep = po.to_numpy(model(plot_data))
    data_ax = single_scatter(ani_data[0], axes[0])
    penalty_draw(data_ax)

    rep_axes = model.plot_representation(model(data)[0], axes[1:], "lines")
    model.plot_representation(ani_rep[0], rep_axes)
    if initial_ylim is not None:
        rep_axes[1].set(ylim=initial_ylim)
    fig.set_layout_engine("none")

    # In order to avoid this potentially taking a long time, make sure we animate at
    # most n_frames
    frame_step = max(len(plot_data) // n_frames, 1)
    frames = range(0, len(plot_data), frame_step)
    rescale_frames = list(frames)[::10][3:-1]
    if highlight_swap_frame is not None:
        color = "#ffff8144"
        default_color = fig.axes[1].get_facecolor()
        fig.axes[1].set_facecolor(color)
        fig.axes[2].set_facecolor(color)
        fig.suptitle("Phase 1: match statistics")

    # Update the data for each saved iteration.
    def animate(i):
        data_ax.collections[0].set_offsets(ani_data[i].T)
        _update_stem(rep_axes[0].containers[0], ani_rep[i, :5])
        _update_stem(rep_axes[1].containers[0], ani_rep[i, 5:])
        if initial_ylim is not None and i in rescale_frames:
            _rescale_ylim(rep_axes[1], ani_rep[i, 5:])
        color_now = fig.axes[0].get_facecolor()
        if (
            highlight_swap_frame is not None
            and i > highlight_swap_frame
            and color_now == default_color
        ):
            fig.axes[0].set_facecolor(color)
            fig.axes[1].set_facecolor(default_color)
            fig.axes[2].set_facecolor(default_color)
            fig.suptitle("Phase 2: match penalty")

    ani = mpl.animation.FuncAnimation(fig, animate, frames, repeat=False)
    plt.close(fig)

    # This will view the video if running in a jupyter notebook. If you are running
    # outside of a notebook (e.g., in ipython), first save it and then open it with
    # something that can view video files (e.g., your browser) by running:
    # ani.save("ds_away.mp4")
    return ani


animate_datasaurus_metamer(met, highlight_swap_frame=14)
```

In the video of synthesis above, we can see the dataset first becomes metameric (the stem heads quickly align themselves with the horizontal dashed lines in the second and third plots) and then the points gradually move away from the penalty's center (marked with a red plus sign). You can see that, as points get moved away from the center, other points farther away shift around (generally, moving closer) in order to keep the model's statistics identical.

Unlike the corresponding dataset in the original datasaurus dozen, we have a clearly-visible circle in this dataset. This happens because, as the synthesis procedures continues, the points gets pushed away from any locations where the penalty is non-zero. If we would like our dataset to look more like the original, we could've ended synthesis a little earlier, or tried coming up with a different penalty.

```{code-cell} ipython3
:tags: [remove-cell]

import os

from plenoptic.tensors import _check_tensor_equality

if os.environ.get("DATASAURUS_CHECK", False):
    # This cell just tests for reproducibility. As a user, you should skip it -- because
    # pytorch doesn't guarantee reproducibility across CPU/GPU and GPU types, it's
    # unlikely that your results will exactly match ours. (Though it should look
    # approximately as good -- if not, open an issue!)
    cached_met = po.data.fetch_data("datasaurus_metamers.tar.gz") / "datasaurus-away.pt"
    # just load in the metamer tensor, instead of the whole object
    cached_met = torch.load(cached_met)["_metamer"]
    _check_tensor_equality(
        met.metamer,
        cached_met,
        "Notebook",
        "OSF",
        1e-5,
        1e-7,
        "metamer has different {error_type}! Update the OSF version.",
    )
```
