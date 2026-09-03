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

Download the executed notebook: **{nb-download}`ds_index.ipynb`**!

Run it in your browser: **{binder}`ds_index.ipynb`**!

:::

(datasaurus-index)=
# Using Penalized Metamer Synthesis to Recreate the Datasaurus Dozen

:::{admonition} Penalty function usage
:class: warning

These pages assume familiarity with the basics of using penalty function in metamer synthesis, as shown in the [](how-to-penalty) notebook.

:::

- discuss original results
      - [paper](https://dl.acm.org/doi/10.1145/3025453.3025912), [data source](https://www.openintro.org/data/index.php?data=datasaurus), [wikipedia](https://en.wikipedia.org/wiki/Datasaurus_dozen#cite_note-Matejka2017-1)
- overview of model
- download and visualize them
- refer back to paper (in particular, "can't match stats directly" and "some shapes are more difficult than others")
- then show our results overview
      - I think can loop through the pt files in the datasaurus_metamers tarball, use torch.load and just grab _saved_metamer
      - compare to original success metric: matches each of those to first N decimal points
            - means, stds, correlation:
      - note that we're not being very careful about efficiency in these notebooks, because the synthesis is so fast. if input was larger or model was slower, would be more important
      - and we're not trying to exactly match datasaurus dozen, just conceptually
      - importantly: we don't actually need our metric value to be very low here, just need them to look like our penalty target
      - and say something like, if you come up with a penalty to do a better job at star, away, thick lines or find a new penalty that does something else interesting
- 3 of these are more difficult. all of them require "composite penalties", combining several penalties to try and get what we want
      - additionally star: hard to synthesize (shape hard to match), so we do it in two parts
- some bonus additional ones: centroids and polygons

```{code-cell} ipython3
import itertools

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
```

Explain model

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
        # This model has no trainable parameters, so it's always in eval mode
        self.eval()

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

```{code-cell} ipython3
datasaurus_tarball = po.data.fetch_data("datasaurus.tar.gz")
data = torch.load(datasaurus_tarball / "datasaurus.pt")
# expand folded cell above to see definition of this model
model = DatasaurusModel(data.shape[1], data.dtype)
categories = np.load(datasaurus_tarball / "categories.npy", allow_pickle=True)
```

Define plotting functions:

```{code-cell} ipython3
:tags: [hide-input]

def plot_datasaurus(data, categories, ax_size=2, scatter_kwargs=None, fig=None):
    if scatter_kwargs is None:
        scatter_kwargs = {}
    scatter_kwargs.setdefault("s", 5)
    n_rows = min(3, len(data) - 1)
    n_cols = int(max(np.ceil((len(data) - 1) / n_rows + 1), 2))
    if fig is None:
        fig = plt.figure(
            figsize=(ax_size * n_cols, ax_size * n_rows), layout="compressed"
        )
    axes = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    if n_rows == 1:
        dino_ax = axes[0, 0]
    if n_rows > 1:
        dino_ax = axes[1, 0]
        axes[0, 0].set_visible(False)
    if n_rows > 2:
        axes[2, 0].set_visible(False)
    axes = [dino_ax] + [ax for ax in axes[:, 1:].T.flatten()]
    for xy, title, ax in zip(data, categories, axes):
        ax.scatter(*xy, **scatter_kwargs)
        ax.set_title(title)
        ax.set_aspect(1)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    for ax in axes[len(data) :]:
        ax.set_visible(False)
    return fig, axes


def update_datasaurus(data, axes):
    artists = []
    for d, ax in zip(data, axes):
        art = ax.collections[0]
        art.set_offsets(d.T)
        artists.append(art)
    return artists


def plot_datasaurus_rep(data, categories, model, ax_size=2, aspect=1.3, fig=None):
    n_rows = min(3, len(data) - 1)
    n_cols = int(max(np.ceil((len(data) - 1) / n_rows + 1), 2))
    if fig is None:
        fig = plt.figure(
            figsize=(ax_size * n_cols * aspect, ax_size * n_rows), layout="compressed"
        )
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.25)
    return_axes = []
    turn_off_axes = []
    if n_rows > 1:
        turn_off_axes.append(0)
    if n_rows > 2:
        turn_off_axes.append(2)
    data_idx = 0
    ylims = [[0, 0], [0, 0]]
    for j, i in itertools.product(range(n_cols), range(n_rows)):
        if j == 0 and i in turn_off_axes:
            continue
        else:
            try:
                y = data[data_idx]
                title = categories[data_idx]
            except IndexError:
                continue
            sgs = gs[i, j].subgridspec(1, 2, width_ratios=[5, 3], wspace=0.15)
            plot_axes = [fig.add_subplot(sgs[i]) for i in range(2)]
            plot_axes[0].set_title(title, x=1)
            plot_axes = model.plot_representation(y, plot_axes)
            model.plot_representation(data[0], plot_axes, "lines")
            for i, (ax, ylim) in enumerate(zip(plot_axes, ylims)):
                ax_ylim = ax.get_ylim()
                ylims[i] = [min(ax_ylim[0], ylim[0]), max(ax_ylim[1], ylim[1])]
            if j == 0:
                try:
                    fig.set_layout_engine("none")
                except AttributeError:
                    # then this is a subfigure
                    fig.figure.set_layout_engine("none")
            else:
                for ax in plot_axes:
                    ax.set(yticklabels=[], xticklabels=[])
        data_idx += 1
        return_axes.append(plot_axes)
    # small adjustment so that dino plots don't overlap with yticklabels
    ax = return_axes[0][0]
    pos = [p for p in ax.get_position().bounds]
    pos[0] -= return_axes[0][1].get_position().bounds[2] / 1.2
    ax.set_position(pos)
    for axes in return_axes:
        for ax, ylim in zip(axes, ylims):
            ax.set_ylim(ylim)
    return fig, return_axes


def update_datasaurus_rep(data, axes, model):
    artists = []
    for d, axs in zip(data, axes):
        artists.append(po.plot.display._update_stem(axs[0].containers[0], d[:5]))
        artists.append(po.plot.display._update_stem(axs[1].containers[0], d[5:]))
    return artists
```

```{code-cell} ipython3
ax_size = 3
n_cols, n_rows = (5, 3)
data_fig = plt.figure(figsize=(ax_size * n_cols, ax_size * n_rows))
plot_datasaurus(data, categories, fig=data_fig);
```

Point to [wikipedia](https://en.wikipedia.org/wiki/Datasaurus_dozen) for summary stats, then plot:

```{code-cell} ipython3
rep_fig = plt.figure(figsize=(ax_size * n_cols, ax_size * n_rows))
plot_datasaurus_rep(model(data), categories, model, fig=rep_fig);
```

Now let's load in our datasaurus fortnight(?):

```{code-cell} ipython3
:tags: [hide-input]

cached_metamers = []
saved_metamers = []
# match order of initial data, plus our extras
titles = [
    "away",
    "hlines",
    "vlines",
    "xshape",
    "star",
    "hwidelines",
    "dots",
    "circle",
    "bullseye",
    "slantup",
    "slantdown",
    "vwidelines",
    "polygons",
    "oval",
]
metamer_tarball = po.data.fetch_data("datasaurus_metamers.tar.gz")
for t in titles:
    # synthesis for star is more complex and so it's saved slightly differently. see
    # its notebook for more details.
    f = metamer_tarball / f"datasaurus-{t}.pt"
    cached_metamers.append(torch.load(f)["_metamer"])
    if t == "star":
        f = metamer_tarball / f"datasaurus-{t}-saved.pt"
        saved_metamers.append(torch.load(f).detach())
    else:
        saved_metamers.append(torch.stack(torch.load(f)["_saved_metamer"]).detach())
cached_metamers = torch.stack([data[0], *cached_metamers])
titles = ["dino (target)"] + titles
saved_metamers = torch.stack(saved_metamers)
```

```{code-cell} ipython3
ax_size = 3
n_cols, n_rows = (5, 3)
fig = plt.figure(figsize=(ax_size * n_cols, ax_size * n_rows * 2))
subfigs = fig.subfigures(2, 1, hspace=-0.2)
plot_datasaurus(cached_metamers, titles, fig=subfigs[0])
plot_datasaurus_rep(model(cached_metamers), titles, model, fig=subfigs[1]);
```

Very pretty. But let's see it ANIMATED

```{code-cell} ipython3
:tags: [hide-input]

# use one of our helper functions here.
from plenoptic.plot.display import _update_stem

ax_size = 3
n_cols, n_rows = (5, 3)
fig = plt.figure(figsize=(ax_size * n_cols, ax_size * n_rows * 2))
subfigs = fig.subfigures(2, 1, hspace=-0.2)
init_metamers = torch.cat([data[:1], saved_metamers[:, 0]])
_, data_axes = plot_datasaurus(init_metamers, titles, fig=subfigs[0])
_, rep_axes = plot_datasaurus_rep(model(init_metamers), titles, model, fig=subfigs[1])
fig.set_layout_engine("none")

ani_data = po.to_numpy(saved_metamers)
ani_rep = po.to_numpy(torch.func.vmap(model)(saved_metamers))


# Update the data for each saved iteration.
def animate(frame):
    i = 0
    for dax, rax in zip(data_axes, rep_axes):
        if dax.get_title() in ["", "dino (target)"]:
            continue
        dax.collections[0].set_offsets(saved_metamers[i, frame].T)
        _update_stem(rax[0].containers[0], ani_rep[i, frame, :5])
        _update_stem(rax[1].containers[0], ani_rep[i, frame, 5:])
        i += 1


ani = mpl.animation.FuncAnimation(
    fig, animate, range(saved_metamers.shape[1]), repeat=False
)
plt.close(fig)
ani
```

::::{card}
:::{toctree}
:maxdepth: 1

ds_circle.md
ds_bullseye.md
ds_dots.md
ds_hlines.md
ds_vlines.md
ds_slantup.md
ds_slantdown.md
ds_xshape.md
ds_away.md
ds_star.md
ds_polygons.md
ds_oval.md
ds_hwidelines.md
ds_vwidelines.md

:::
::::
