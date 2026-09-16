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

Download the executed notebook: **{nb-download}`ds_oval.ipynb`**!

Run it in your browser: **{binder}`ds_oval.ipynb`**!

:::

# oval

In this notebook, we will create a datasaurus metamer not present in the original set, an oval.

This notebook is intentionally brief: most of the code is hidden (you can expand the cells if you would like to see more details), and we only explain the penalty. See [](datasaurus-index) for an overview of the datasaurus dozen dataset.

```{code-cell} ipython3
:tags: [hide-input]

import einops
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

# set seed for reproducibility. for strict reproducibility, we'd also need to set
# torch.use_deterministic_algorithms(True) here, but we don't need to be so strict here.
po.set_seed(0)
# To guarantee reproducibility for this example on the GPU, we must tell torch to use
# deterministic algorithms. Note this will make things slower! See "Reproducibility and
# Compatibility" in the docs for more details.
torch.use_deterministic_algorithms(True)


# Model definition, as in top-level notebook
class DatasaurusModel(torch.nn.Module):
    def __init__(self, n_pts=None, dtype=None):
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
        """
        super().__init__()
        # cache ones to save time
        if n_pts is not None:
            self._ones = torch.ones(n_pts, dtype=dtype)
        else:
            self._ones = None
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
        solution = torch.func.vmap(lambda x: self._compute_linreg(*x))(data)
        stats.append(solution)
        crosscorr = torch.func.vmap(lambda x: torch.corrcoef(x)[0, 1])(data)
        stats.append(crosscorr.unsqueeze(-1))
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
            gs = mpl.gridspec.GridSpec(1, 2, fig, width_ratios=[5, 3])
            axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
        elif isinstance(ax, mpl.axes.Axes) or len(ax) == 1:
            # want to make sure the axis we're taking over is basically invisible.
            ax = po.plot.display._clean_up_axes(
                ax, False, ["top", "right", "bottom", "left"], ["x", "y"]
            )
            gs = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[5, 3])
            fig = ax.figure
            axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
        else:
            axes = ax
            fig = axes[0].figure

        labels = [
            "x mean",
            "y mean",
            "x std",
            "y std",
            "linreg intercept",
            "linreg slope",
            "correlation",
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
            ax.set_xticks(x, labs, rotation=30, ha="right")
        return axes
```

The penalties this example was developed while working on the [](ds_hwidelines.md) and [](ds_vwidelines.md) examples (you can see them also being used there). After developing `polygon_penalty` (as described in [](ds_polygons.md)), while the resulting groups were all approximately the same size and shape, they were not evenly distributed across the space. To encourage this behavior, an additional penalty was written, `centroid_penalty`, which breaks points into small groups and computes the pairwise Euclidean distance between the centroids of all groups, and then computes the MSE between these distances and some specifvied target. Like `polygon_penalty`, because the total number of points may not be divisible by the group size, some small number of points are completely unconstrained and can fly to any part of the plot.

To use this penalty with synthesis, we define the size of these groups and the target distances for both penalties (try changing these to different values!) and combine the resulting value with a range penalty which requires all points to lie between 0 and 100.

```{code-cell} ipython3
def polygon_penalty(data, target_dist, nbr):
    # break data into "neighborhoods" of nbr points each and tries to make each of their
    # distances match target i.e., form regular polygons of target size
    pts = einops.rearrange(
        data[..., : nbr * (data.shape[-1] // nbr)], "d (n1 n2) -> n1 n2 d", n2=nbr
    )
    dist = torch.cdist(pts, pts)
    tril_idx = torch.tril_indices(pts.shape[1], pts.shape[1], -1)
    dist = dist[:, tril_idx[0], tril_idx[1]]
    return (dist - target_dist).pow(2).mean()


def centroid_penalty(data, target_dist, nbr):
    pts = einops.rearrange(
        data[..., : nbr * (data.shape[-1] // nbr)], "d (n1 n2) -> n1 n2 d", n2=nbr
    )
    dist = torch.cdist(pts.mean(1), pts.mean(1))
    tril_idx = torch.tril_indices(pts.shape[0], pts.shape[0], -1)
    dist = dist[tril_idx[0], tril_idx[1]]
    return (dist - target_dist).pow(2).mean()


# Try changing these to other values!
polygon_distance = 5
centroid_distance = 25
neighborhood_size = 3


data = torch.load(po.data.fetch_data("datasaurus.tar.gz") / "datasaurus.pt")
model = DatasaurusModel(data.shape[1], data.dtype)


def penalty(x):
    range_penalty = po.regularize.penalize_range(x, (0, 100))
    # Change these values to whatever you want!
    polygon = polygon_penalty(x, polygon_distance, neighborhood_size)
    centroid = centroid_penalty(x, centroid_distance, neighborhood_size)
    return range_penalty + centroid + polygon


# data[0] is the dinosaur
met = po.Metamer(data[0], model, penalty_function=penalty, penalty_lambda=0.001)
met.setup(initial_image=100 * torch.rand_like(data[0]), optimizer=torch.optim.LBFGS)
met.synthesize(50, store_progress=True)
```

```{code-cell} ipython3
:tags: [hide-input]

# use one of our helper functions here.
from plenoptic.plot.display import _update_stem

# Initialize figure by plotting the first iteration
fig, axes = plt.subplots(
    1, 3, figsize=(8, 3), width_ratios=[5, 5, 3], layout="compressed"
)
plot_data = met.saved_metamer
ani_data = po.to_numpy(plot_data)
ani_rep = po.to_numpy(model(plot_data))
path = axes[0].scatter(*ani_data[0])
axes[0].set(xlim=(0, 100), ylim=(0, 100))
axes[0].set_aspect(1)

rep_axes = model.plot_representation(model(data)[0], axes[1:], "lines")
model.plot_representation(ani_rep[0], rep_axes)
fig.set_layout_engine("none")


# Update the data for each saved iteration.
def animate(i):
    path.set_offsets(ani_data[i].T)
    _update_stem(rep_axes[0].containers[0], ani_rep[i, :5])
    _update_stem(rep_axes[1].containers[0], ani_rep[i, 5:])


ani = mpl.animation.FuncAnimation(fig, animate, range(len(plot_data)), repeat=False)
plt.close(fig)
ani
```

In the video of the synthesis above, we can see the dataset shifting itself to become metameric and then the points starting to push away from each other. Both penalties are computed using neighborhood sizes of 3, but triangles are not readily visible (unlike [](ds_polygons.md), where hexagons were clearly visible). Instead, the points have formed an oval located in the center of the plot.

There is no corresponding dataset in the original datasaurus, but this dataset is both metameric and "clearly different and identifiably distinct" from the others!

```{code-cell} ipython3
:tags: [remove-cell]

from plenoptic.tensors import _check_tensor_equality

# This cell just tests for reproducibility. As a user, you should skip it -- because
# pytorch doesn't guarantee reproducibility across CPU/GPU and GPU types, it's unlikely
# that your results will exactly match ours. (Though it should look approximtaely as
# good -- if not, open an issue!)
cached_met = po.data.fetch_data("datasaurus_metamers.tar.gz") / "datasaurus-oval.pt"
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
