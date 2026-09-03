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

Download the executed notebook: **{nb-download}`ds_hlines.ipynb`**!

Run it in your browser: **{binder}`ds_hlines.ipynb`**!

:::

# Synthesize the datasaurus hlines

In this notebook, we will create a datasaurus metamer shaped like multiple horizontal lines. See [](datasaurus-index) for an overview of the datasaurus dozen dataset.

```{code-cell} ipython3
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
```

Don't discuss model, already explained in intro

```{code-cell} ipython3
:tags: [hide-input]

class DatasaurusModel(torch.nn.Module):
    def __init__(self, n_pts=None, dtype=None):
        super().__init__()
        # cache ones to save time
        if n_pts is not None:
            self._ones = torch.ones(n_pts, dtype=dtype)
        else:
            self._ones = None

    def _prepare_X(self, x):
        ones = self._ones if self._ones is None else torch.ones_like(x)
        return torch.stack([ones, x], -1)

    def _compute_linreg(self, x, y):
        X = self._prepare_X(x)
        # unsqueezing and squeezing needed because of https://github.com/pytorch/pytorch/issues/158169
        return torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze()

    def _compute_coeff_determination(self, x, y, solution):
        X = self._prepare_X(x)
        pred_y = torch.einsum("x, n x -> n", solution, X)
        ss_res = (y - pred_y).pow(2).sum()
        ss_tot = (y - y.mean()).pow(2).sum()
        return 1 - (ss_res / ss_tot)

    def _vmap_coeff_determination(self, x, solution):
        f = torch.func.vmap(lambda x, solt: self._compute_coeff_determination(*x, solt))
        return f(x, solution).unsqueeze(-1)

    def forward(self, data):
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

Explain figure

```{code-cell} ipython3
data = torch.load(po.data.fetch_data("datasaurus.tar.gz") / "datasaurus.pt")
categories = np.load(
    po.data.fetch_data("datasaurus.tar.gz") / "categories.npy", allow_pickle=True
)

model = DatasaurusModel(data.shape[1], data.dtype)
model.eval()

fig, axes = plt.subplots(
    2, 3, figsize=(8, 6), width_ratios=[5, 5, 3], layout="compressed"
)
for i, title in enumerate(["dino (target)", "h_lines"]):
    d = data[categories == title].squeeze()
    axes[i, 0].scatter(*d)
    axes[i, 0].set_title(title)
    axes[i, 0].set(xlim=(0, 100), ylim=(0, 100))
    axes[i, 0].set_aspect(1)
    model.plot_representation(model(d), axes[i, 1:])
    model.plot_representation(model(data)[0], axes[i, 1:], "lines")
    if i == 0:
        axes[i, 1].set(xticklabels=[])
        axes[i, 2].set(xticklabels=[])
```

Explain penalty: two circles with same center. arbitrarily split points in half

```{code-cell} ipython3
def predict_line(data, intercepts, slope):
    return slope * data[0] + intercepts


def lines_penalty(data, intercepts, slope):
    # intercepts must be shape [n, 1], slope a scalar or same number of elements as
    # intercepts
    errors = []
    n = data.shape[-1] // intercepts.shape[0]
    if hasattr(slope, "__len__") and len(slope) != 1:
        assert len(slope) == len(intercepts)
    else:
        slope = len(intercepts) * [slope]
    for i, (inter, sl) in enumerate(zip(intercepts, slope)):
        if i != len(intercepts) - 1:
            split = data[..., i * n : (i + 1) * n]
        else:
            # extra entries on last one
            split = data[..., i * n :]
        pred_y = predict_line(split, inter, sl)
        errors.append((split[1] - pred_y).pow(2))
    return torch.mean(torch.cat(errors))


def hlines_penalty(data, y_vals=[10, 30, 50, 70, 90]):
    intercepts = torch.as_tensor(y_vals).unsqueeze(-1)
    return lines_penalty(data, intercepts, 0)
```

Combine hlines penalty and range penalty, then run synthesis.

```{code-cell} ipython3
def penalty(x):
    range_penalty = po.regularize.penalize_range(x, (0, 100))
    # Change these values to whatever you want!
    hlines = hlines_penalty(x)
    return range_penalty + hlines


# data[0] is the dinosaur
met = po.Metamer(data[0], model, penalty_function=penalty, penalty_lambda=0.0005)
met.setup(initial_image=100 * torch.rand_like(data[0]), optimizer=torch.optim.LBFGS)
met.synthesize(50, store_progress=True)
```

Visualize synthesis process:

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

And we've done it. Go back to [](datasaurus-index) or click in the sidebar to go to the next one.

```{code-cell} ipython3
:tags: [remove-cell]

from plenoptic.tensors import _check_tensor_equality

# This cell just tests for reproducibility. As a user, you should skip it -- because
# pytorch doesn't guarantee reproducibility across CPU/GPU and GPU types, it's unlikely
# that your results will exactly match ours. (Though it should look approximtaely as
# good -- if not, open an issue!)
cached_met = po.data.fetch_data("datasaurus_metamers.tar.gz") / "datasaurus-hlines.pt"
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
