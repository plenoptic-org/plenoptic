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

Download the executed notebook: **{nb-download}`ds_plenoptic_logo.ipynb`**!

Run it in your browser: **{binder}`ds_plenoptic_logo.ipynb`**!

:::

# plenoptic logo

In this notebook, we will create a datasaurus metamer starting from the plenoptic logo. Unlike the other examples, we will not add a custom penalty to encourage a particular shape. Instead, we first show the metameric dataset that results when using the same set-up as the other notebooks in this series, and then demonstrate the effect of changing the optimizer and of the inclusion of the redundant statistics in the model. See [](datasaurus-index) for an overview of the datasaurus dozen dataset and these redundant statistics.

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
        return axes
```

For this example, the interesting thing is the initial shape of the dataset, the plenoptic logo. Setting the initial arrangement of dots is not something the original datasaurus dozen algorithm in {cite:alp}`Matejka2017-same-stats` was able to do! Our penalty is simply a constraint on the range, requiring all points to lie between 0 and 100. In the other notebooks in this series, this penalty will be combined with other functions to encourage different shapes.

(lbfgs-full-model)=

```{code-cell} ipython3
data = torch.load(po.data.fetch_data("datasaurus.tar.gz") / "datasaurus.pt")
model = DatasaurusModel(data.shape[1], data.dtype)


def penalty(x):
    range_penalty = po.regularize.penalize_range(x, (0, 100))
    return range_penalty


logo = torch.load(po.data.fetch_data("datasaurus.tar.gz") / "plenoptic_logo.pt")
# data[0] is the dinosaur
met = po.Metamer(data[0], model, penalty_function=penalty)
met.setup(initial_image=logo, optimizer=torch.optim.LBFGS)
met.synthesize(50, store_progress=True)
```

```{code-cell} ipython3
:tags: [hide-input]

# use one of our helper functions here.
from plenoptic.plot.display import _rescale_ylim, _update_stem


def animate_datasaurus_metamer(met, model=None, initial_ylim=None, n_frames=50):
    if model is None:
        model = met.model
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
    if initial_ylim is not None:
        rep_axes[1].set(ylim=initial_ylim)
    fig.set_layout_engine("none")

    # In order to avoid this potentially taking a long time, make sure we animate at
    # most n_frames
    frame_step = max(len(plot_data) // n_frames, 1)
    frames = range(0, len(plot_data), frame_step)
    rescale_frames = list(frames)[::10][3:-1]

    # Update the data for each saved iteration.
    def animate(i):
        path.set_offsets(ani_data[i].T)
        _update_stem(rep_axes[0].containers[0], ani_rep[i, :5])
        _update_stem(rep_axes[1].containers[0], ani_rep[i, 5:])
        if initial_ylim is not None and i in rescale_frames:
            _rescale_ylim(rep_axes[1], ani_rep[i, 5:])

    ani = mpl.animation.FuncAnimation(fig, animate, frames, repeat=False)
    plt.close(fig)

    # This will view the video if running in a jupyter notebook. If you are running
    # outside of a notebook (e.g., in ipython), first save it and then open it with
    # something that can view video files (e.g., your browser) by running:
    # ani.save("ds_plenoptic_logo.mp4")
    return ani


animate_datasaurus_metamer(met)
```

This dataset starts out as the plenoptic logo, roughly centered and axes-aligned. To become a metamer, the dataset gets squished and sheared so that it ends up looking like a rotated but still roughly centered version of the logo.

You can see that this happens very quickly --- without any additional constraints, metamers for this model are not difficult to find! The rest of this notebook investigates the effect of the optimization algorithm and the model's redundant statistics on the synthesis process and resulting metamers.

```{code-cell} ipython3
:tags: [remove-cell]

import os

from plenoptic.tensors import _check_tensor_equality

if os.environ.get("DATASAURUS_CHECK", False):
    # This cell just tests for reproducibility. As a user, you should skip it -- because
    # pytorch doesn't guarantee reproducibility across CPU/GPU and GPU types, it's
    # unlikely that your results will exactly match ours. (Though it should look
    # approximately as good -- if not, open an issue!)
    cached_met = (
        po.data.fetch_data("datasaurus_metamers.tar.gz")
        / "datasaurus-plenoptic-logo.pt"
    )
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

## Different optimizers

In the above example, as in the other notebooks in this series, we use the {class}`torch.optim.LBFGS` optimizer. This [optimization algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS) approximates the second derivative (the Hessian matrix) of the parameters in order to find the optimum solution, rather than just using the first derivative (the gradient). LBFGS thus requires more memory than gradient-only methods such as {class}`torch.optim.Adam`, and each iteration requires more time, but it often finds a better solution in fewer iterations, leading to a faster overall synthesis procedure.

:::{admonition} What optimizer should I use?
:class: note

In this example and several others in the documentation (e.g., [the PortillaSimoncelli texture model](ps-basic-synthesis)), we use {class}`torch.optim.LBFGS`, whereas in others we use {class}`torch.optim.Adam` (the default for {class}`~plenoptic.Metamer` and {class}`~plenoptic.MADCompetition`). In our experience, {class}`~torch.optim.LBFGS` does not work for all synthesis problems (it is more likely to get stuck), but when it does work, it finds a better solution faster.

In your own problems, we thus recommend trying both optimizers and seeing how they behave. Remember that you may have to tweak the optimizer's hyper-parameters! In the following, we change the learning rate, and you can also see examples of how to set the other hyper-parameters for {class}`torch.optim.LBFGS` in [](ps-basic-synthesis).

:::

However, other optimizers are also able to solve this problem in a reasonable amount of time and they find different solutions. Additionally, because of the ease of plotting this dataset, the movies we create allow us to visualize how the different optimizers behave.

First, let us use {class}`torch.optim.SGD`, [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (though technically, since we are computing the gradient on the entire dataset at once instead of on multiple mini-batches / sub-samples, plenoptic performs regular gradient descent):

```{code-cell} ipython3
met = po.Metamer(data[0], model, penalty_function=penalty)
met.setup(initial_image=logo, optimizer_kwargs={"lr": 1}, optimizer=torch.optim.SGD)
met.synthesize(3000, store_progress=True)
animate_datasaurus_metamer(met)
```

In the above video, we can see that the points in the dataset move almost directly towards their final locations (rather than rotating around, as in the {class}`~torch.optim.LBFGS` synthesis), since {class}`~torch.optim.SGD` doesn't do anything beyond updating the points based directly on the gradient required to minimize the loss. The resulting metamer looks different but is of comparable quality --- this shouldn't be surprising, as there are many different metamers for this problem and, in this high-dimensional non-convex optimization problem, no guarantee that different optimizers will find identical solutions!

Note also that we needed to increase the learning rate by a factor of 100, from 0.01 to 1 and the number of iterations from 50 to 3000: gradient-based methods such as {class}`~torch.optim.SGD` only have information about the magnitude of the gradient at the location they are evaluating, whereas, because {class}`~torch.optim.LBFGS` approximates the second derivative, it has information about how the gradient is changing, allowing it to make larger changes in each iteration where possible.

Now, let's use {class}`torch.optim.Adam`, the [Adaptive Moment Estimation](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) algorithm. Adam is a variant of stochastic gradient descent which keeps a running average of both the gradients and their second moments which decay over time (so that recent estimates matter more). The algorithm uses these running averages to compute a form of "signal-to-noise ratio", which is used to scale the optimizer's step sizes, resulting in smaller steps as the optimizer approaches an optimum.

```{code-cell} ipython3
met = po.Metamer(data[0], model, penalty_function=penalty)
met.setup(initial_image=logo, optimizer=torch.optim.Adam, optimizer_kwargs={"lr": 0.1})
met.synthesize(400, store_progress=True)
animate_datasaurus_metamer(met)
```

In the above we can see the effect of this property, which the algorithm's author refer to as "automatic annealing": even with a smaller learning rate than the {class}`~torch.optim.SGD` example above, {class}`~torch.optim.Adam` finds a solution faster, though not as fast as {class}`~torch.optim.LBFGS`. This metamer, unlike the previous two, also appears "broken": the circle containing $\theta$ is split and the central square is completely flattened.

Examining these two examples highlights another property of the course of synthesis when using {class}`~torch.optim.LBFGS`: the points appear to rotate into place. This property is consistent across the metamers found in the other notebooks in this series. This is because, whereas the gradient-based methods update the points by applying some scalar multiplied by the gradient, LBFGS updates them using a matrix (the second derivative) multiplied by the gradient. Generally speaking, matrices can be thought of as performing [geometric transformations](https://en.wikipedia.org/wiki/Transformation_matrix), and so this update rule appears as a combination of rotation, stretching, and shearing.

## Remove redundant stats

In this next section, we'll see what happens when we remove the redundant statistics from our model ($\beta_0,\beta_1,R^2$; revisit [here](datasaurus-redundant-stats) for more details).

```{code-cell} ipython3
reduced_model = DatasaurusModel(
    data.shape[1], data.dtype, include_redundant_stats=False
)
print(f"Full model output: {model(data[0])}")
print(f"Reduced model output: {reduced_model(data[0])}")
```

Because these removed statistics are redundant, any metamer for `reduced_model` will also be a metamer for `model` <!-- skip-lint -->. We can see that in the video below, where the stem plots do eventually align on the dashed horizontal lines for all eight of our statistics:

```{code-cell} ipython3
met = po.Metamer(data[0], reduced_model, penalty_function=penalty)
met.setup(initial_image=logo, optimizer=torch.optim.LBFGS)
met.synthesize(50, store_progress=True)
animate_datasaurus_metamer(met, model, (-1, 1))
```

The above shows the result of using {class}`torch.optim.LBFGS` with the same hyper-parameters as metamer synthesis for [the full model](lbfgs-full-model). While this example does eventually find a good metamer, synthesis takes a different trajectory than the full model's and it takes longer to do so. Notice that the means and standard deviations are still matched fairly quickly, but the Pearson correlation $r$ and thus the redundant stats take a good deal longer.

This pattern holds, and is in fact exaggerated, when using {class}`torch.optim.SGD` or {class}`torch.optim.Adam` to synthesis metamers for the reduced model. Let's first examine Adam:

```{code-cell} ipython3
met = po.Metamer(data[0], reduced_model, penalty_function=penalty)
met.setup(initial_image=logo, optimizer=torch.optim.Adam, optimizer_kwargs={"lr": 1})
met.synthesize(500, store_progress=True, stop_criterion=1e-7)
animate_datasaurus_metamer(met, model)
```

Here, we needed to increase both the number of synthesis iterations and the optimizer's learning rate, while reducing the `stop_criterion` to ensure that metamer synthesis continues to run to a lower loss value.

The same is true for SGD:

```{code-cell} ipython3
met = po.Metamer(data[0], reduced_model, penalty_function=penalty)
met.setup(initial_image=logo, optimizer_kwargs={"lr": 50}, optimizer=torch.optim.SGD)
met.synthesize(6000, store_progress=True, stop_criterion=1e-8)
animate_datasaurus_metamer(met, model, n_frames=100)
```

Here we needed to increase learning rate from 1 to 50 and to run synthesis for 6000 iterations. In the resulting video, the metamer matches the means and standard deviations almost immediately (within the first hundred or so iterations), but takes a very long time to match the Pearson correlation.

## Using a custom loss

Comparing the videos for the SGD and Adam examples with and without the redundant statistics, it is striking how all metamer syntheses quickly match the means and standard deviations, but the Pearson correlation takes much longer to match when the redundant statistics are removed. Looking at the above plots, we can see that the error between the initial and target $r$ has a much smaller magnitude than for the other four non-redundant statistics. As the loss for these synthesis examples is simply {func}`~plenoptic.loss.mse`, the mean-squared error on the model outputs, the contribution of $r$ to the gradient is thus much smaller than that of the other statistics, leading to it being de-prioritized. If we look back at the [definition of the redundant statistics](datasaurus-redundant-stats), we can see that all three of them include $r$, effectively amplifying the contribution of $r$ to the gradient.

With this understanding, a simple solution comes to mind: we need to increase the contribution of $r$ to the gradient. The most straight-forward way to do so is to increase its effect on the loss, which we can do by multiplying its value by a large scalar. Inspired by the [](ps-loss-function) we recommend for {class}`~plenoptic.models.PortillaSimoncelli`, let's write a custom loss function:

```{code-cell} ipython3
weight = torch.ones_like(reduced_model(data[0])).squeeze()
weight[-1] = 10
print(f"{weight=}")


def custom_loss(x, y):
    return po.loss.mse(weight * x, weight * y)
```

The above custom loss computes the mean-squared error between the re-weighted output of the reduced model: we multiply each statistic by 1 except for $r$, which we multiply by 10. In the following block, we use this loss with {class}`torch.optim.SGD`:

```{code-cell} ipython3
met = po.Metamer(data[0], reduced_model, custom_loss, penalty_function=penalty)
met.setup(initial_image=logo, optimizer_kwargs={"lr": 50}, optimizer=torch.optim.SGD)
met.synthesize(200, store_progress=True, stop_criterion=1e-8)
animate_datasaurus_metamer(met, model)
```

With the custom loss and the same learning rate as before, we need fewer than 200 iterations to find a good solution, as opposed to the 6000 iterations we need with the standard {func}`~plenoptic.loss.mse` as the loss function.

In this notebook, we've demonstrated the creation of a simple datasaurus metamer, and further investigating the effect of the optimizer and the model's redundant statistics. The rest of the notebooks in this series use the "standard setup" shown in the first example: {class}`~torch.optim.LBFGS` and the model including the redundant statistics. They focus on the definition of different penalty functions to intentionally shape the resulting metamer, instead of the unintended effects shown here. You are encouraged to try changing the problem in those notebooks, as done here, though note that you may end up in a situation where it is much harder to successfully find a model metamer!
