---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: plenoptic_venv
  language: python
  name: plenoptic_venv
---

:::{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`5_glm.ipynb`**! See the button at the top right to download as markdown or pdf.

:::

# Generalized Linear Model (with NeMoS)

The [NeMoS](https://nemos.readthedocs.io/en/latest/) package, developed by the same team as plenoptic, provides a framework for fitting statistical models for systems neuroscience, including the Generalized Linear Model (GLM). Let's use nemos to fit a model and then plenoptic to synthesize some metamers!

Because nemos relies on jax, while plenoptic relies on pytorch, we cannot use nemos models with plenoptic directly. Instead, we will:
- Fit a GLM model to data, using nemos.
- Implement a small GLM in plenoptic
- Synthesize some metamers

```{code-cell} ipython3
# needed for the plotting/animating:
import matplotlib.pyplot as plt
import matplotlib as mpl
import plenoptic as po
import torch
import pynapple as nap
import nemos as nmo
import numpy as np
from scipy.io import loadmat
import copy
import jax
jax.config.update("jax_enable_x64", True)

plt.rcParams["animation.html"] = "html5"
# use single-threaded ffmpeg for animation writer
plt.rcParams["animation.writer"] = "ffmpeg"
plt.rcParams["animation.ffmpeg_args"] = ["-threads", "1"]
# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Fit the GLM with nemos

In this section, we are doing the same model fit as [this tutorial](https://balzaniedoardo.github.io/nemos_glm_tutorials/tutorials/Sfn-2016-tutorial-GLMs/05_decoding.html), skipping over much of the explanation. If you are interested in learning more about nemos and/or the GLM, you're encouraged to work through those tutorials!

This dataset consists of retinal ganglion cells receiving a one-dimensional input: binary temporal white noise (data from [Uzzell & Chichilnisky, 2004](https://pubmed.ncbi.nlm.nih.gov/15277596/); see [README.txt](https://github.com/pillowlab/GLMspiketraintutorial_python/blob/main/data_RGCs/README.txt) for details). Here, we will build a GLM that predicts a single neuron's firing rate as the result of a linear filter convolved with this input.

This first hidden cell defines the function that we'll use to download the data.

```{code-cell} ipython3
:tags: [hide-cell]

# modified from https://github.com/BalzaniEdoardo/nemos_glm_tutorials/blob/main/src/nemos_tutorials/fetch.py

import pathlib
from typing import Optional, Union

import pooch

DATASETS: dict[str, dict] = {
    "data_RGCs": {
        "files": {
            "SpTimes.mat": "aa0afcb6755fd61ed5dd26c4a8e5b8da91cc13bb3a3640d3294ac37b0193d640",
            "stimtimes.mat": "bdd5cb62a1b7500ebebb2d79beb1bffd97c6d010a2dcce148155fb26806a75e9",
            "Stim.mat": "e6d01592cd08a89740a018294a56d1c94b0254e34ae1a1cf56c468586e22e15e",
        },
        "base_url": "https://raw.githubusercontent.com/pillowlab/GLMspiketraintutorial_python/main/data_RGCs/",
    },
}

# Flat registry and per-file URL map derived from DATASETS
REGISTRY_DATA: dict[str, Optional[str]] = {
    fname: fhash
    for ds in DATASETS.values()
    for fname, fhash in ds["files"].items()
}

FILE_URLS: dict[str, str] = {
    fname: ds["base_url"] + fname
    for ds in DATASETS.values()
    for fname in ds["files"]
}

def _create_retriever(path: Optional[pathlib.Path] = None) -> pooch.Pooch:
    """Create a pooch retriever for fetching datasets.

    Parameters
    ----------
    path :
        Directory where datasets will be stored. Defaults to
        pooch.os_cache('nemos_tutorials').

    Returns
    -------
    :
        A configured pooch retriever.
    """
    if path is None:
        path = pooch.os_cache("nemos_tutorials")

    return pooch.create(
        path=path,
        # base_url is unused when every file has an explicit entry in `urls`,
        # but pooch requires the argument, so pass an empty string.
        base_url="",
        urls=FILE_URLS,
        registry=REGISTRY_DATA,
        retry_if_failed=2,
    )


def fetch_data(
    dataset_name: str,
    path: Optional[Union[pathlib.Path, str]] = None,
) -> dict[str, str]:
    """Download all files belonging to a named dataset.

    Parameters
    ----------
    dataset_name :
        Key from DATASETS (e.g. "data_RGCs").
    path :
        Directory where files will be stored. Defaults to the pooch cache.

    Returns
    -------
    :
        Mapping of {filename: local_path} for every file in the dataset.

    Raises
    ------
    ValueError
        If dataset_name is not found in DATASETS.
    """
    if dataset_name not in DATASETS:
        available = ", ".join(DATASETS)
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Available: {available}"
        )

    retriever = _create_retriever(pathlib.Path(path) if path else None)
    filenames = DATASETS[dataset_name]["files"]

    return {
        fname: retriever.fetch(fname, progressbar=True)
        for fname in filenames
    }
```

Download and prepare the data, using [pynapple](https://pynapple.org/), another package developed at Flatiron CCN:

```{code-cell} ipython3
data_paths = fetch_data("data_RGCs")

# Load and wrap spike times
spike_times = loadmat(data_paths["SpTimes.mat"], simplify_cells=True)["SpTimes"]
units = nap.TsGroup({i: nap.Ts(val) for i, val in enumerate(spike_times)})

# Load and wrap stimulus
stim_times = loadmat(data_paths["stimtimes.mat"], simplify_cells=True)["stimtimes"]
stim = loadmat(data_paths["Stim.mat"], simplify_cells=True)["Stim"]
stimulus = nap.Tsd(stim_times, stim)

# Align, count, resample
units = units.restrict(stimulus.time_support)
bin_size = stimulus.t[1] - stimulus.t[0]
counts = units.count(bin_size, stimulus.time_support)
stimulus = counts.value_from(stimulus, mode="before")

cell_idx = 2
neuron_counts = counts[:, cell_idx]
```

The nemos tutorial shows how to split the data into test and train sets. We're not going to do that here, instead training on the whole data set.

Below we create the basis object that will construct our filter, fit the model, and save the parameters. We also save our stimulus.

```{code-cell} ipython3
basis_stim = nmo.basis.HistoryConv(20, label="stim", conv_kwargs={"shift": False})
X_stim = basis_stim.compute_features(stimulus)

glm_stim = nmo.glm.GLM(observation_model="Poisson")
glm_stim.fit(X_stim, neuron_counts)

glm_stim.save_params("nemos_glm.npz")
np.savez("nemos_stimulus.npz", allow_pickle=False, stimulus=stimulus)
```

## Synthesizing GLM metamers with plenoptic

### Building a GLM in plenoptic

Plenoptic doesn't currently include a GLM model, though it is in our roadmap (see [issue 243](https://github.com/plenoptic-org/plenoptic/issues/243) if you're interested). So here, we will implement a basic GLM that can load in the parameter file we saved above and then use it to synthesize metamers for this simple 1d stimulus.

First, let's define some helper functions for converting jax arrays to torch tensors and which we'll use for plotting our metamer. We hide this cell because the details are not that important:

```{code-cell} ipython3
:tags: [hide-cell]

def jax_to_torch(x, n_unsqueeze=0):
    x = torch.from_numpy(copy.copy(np.asarray(x)))
    for _ in range(n_unsqueeze):
        x = x.unsqueeze(0)
    return x

def plot_model(model, stim):
    fig = plt.figure(layout="constrained", figsize=(10, 5))
    gs = mpl.gridspec.GridSpec(4, 2, width_ratios=[1, 3], figure=fig)
    ax = fig.add_subplot(gs[1:3, 0])
    ax.plot(po.to_numpy(model.conv.weight.squeeze()))
    ax.set_title("Filter")
    ax = fig.add_subplot(gs[:2, 1])
    ax.set_title("Stimuli")
    ax.plot(stim.squeeze(), label="Real stimulus")
    ax.legend()
    n_timepts = stim.shape[-1]
    ax.set(xlim=(0, n_timepts))
    model_stim = model(stim).squeeze()
    init_x = n_timepts - len(model_stim)
    x = np.arange(init_x, n_timepts)
    ax = fig.add_subplot(gs[2:, 1])
    ax.set_title("Model response")
    ax.plot(x, po.to_numpy(model_stim), label="Real stimulus")
    ax.legend()
    ax.set(xlim=(0, n_timepts))
    return fig


def plot_met(mets, labels):
    if not hasattr(mets, "__len__"):
        mets = [mets]
    if isinstance(labels, str):
        labels = [labels]
    gs = mpl.gridspec.GridSpec(4, 2, width_ratios=[1, 3])
    fig = plot_model(mets[0].model, mets[0].image)
    n_timepts = stim.shape[-1]
    for met, label in zip(mets, labels):
        fig.axes[1].plot(po.to_numpy(met.metamer.squeeze()), "--", label=label)
    fig.axes[1].legend()
    fig.axes[1].set(xlim=(0, n_timepts))
    model_stim = met.model(met.image).squeeze()
    init_x = n_timepts - len(model_stim)
    x = np.arange(init_x, n_timepts)
    for met, label in zip(mets, labels):
        fig.axes[2].plot(x, po.to_numpy(met.model(met.metamer).squeeze()), "--", label=label)
    fig.axes[2].legend()
    fig.axes[2].set(xlim=(0, n_timepts))
    return fig
```

Next, let's define our `GLM` model. Note that this class will not work for all nemos GLMs. Currently, only those that use either no basis or {external+nemos:class}`nemos.basis.HistoryConv` (like the above).

```{code-cell} ipython3
class GLM(torch.nn.Module):
    def __init__(self, weight_shape=None, weight=None, bias=None, link_func="exp"):
        """Initialize GLM.

        Exactly one of weight or weight_shape must be set. If weight_shape
        is set, we randomly initialize the weights in the corresponding shape. Else, we
        use the specified weights.

        Supports weight_shape (and weight shape) of 1 through 3 dimensions
        (inclusive), though this has only been tested with 1d weights.
        """
        super().__init__()
        if weight_shape is not None and weight is not None:
            raise ValueError("Exactly one of weight_shape and weight must be set!")
        if weight_shape is None and weight is None:
            raise ValueError("Exactly one of weight_shape and weight must be set!")
        if weight_shape is None:
            weight_shape = weight.shape
            dtype = weight.dtype
        else:
            dtype = torch.float32
        if len(weight_shape) == 1:
            self.conv = torch.nn.Conv1d(1, 1, weight_shape, dtype=dtype)
        elif len(weight_shape) == 2:
            self.conv = torch.nn.Conv2d(1, 1, weight_shape, dtype=dtype)
        elif len(weight_shape) == 3:
            self.conv = torch.nn.Conv3d(1, 1, weight_shape, dtype=dtype)
        state_dict = {}
        if weight is not None:
            state_dict["conv.weight"] = weight.unsqueeze(0).unsqueeze(0)
        if bias is not None:
            state_dict["conv.bias"] = bias
        if link_func == "jax.numpy.exp":
            self.link_func = torch.exp
        else:
            raise ValueError(f"Don't know how to handle {link_func=}")
        self.load_state_dict(state_dict)

    def forward(self, x, **kwargs):
        """Return predicted firing rate."""
        return self.link_func(self.conv(x, **kwargs))

    @classmethod
    def load_nemos_glm(cls, path):
        """Load the output of nemos GLM's save_params method."""
        coeffs_npz = np.load(path)
        try:
            # this is a simple GLM. we reverse the filter because nemos convention is
            # reverse of torch's with respect to time
            weight = jax_to_torch(coeffs_npz["item::strkey:coef_"][::-1])
        except KeyError:
            # this is a GLM that was fit using a pytree, specifying the stimulus filter
            weight = jax_to_torch(coeffs_npz["dict::strkey:coef_::item::strkey:stim"][::-1])
        bias = jax_to_torch(coeffs_npz["item::strkey:intercept_"])
        link_func = coeffs_npz["item::strkey:inverse_link_function"]
        return cls(weight=weight, bias=bias, link_func=link_func)
```

Now, let's initialize our model using the parameters saved above, switching the model to evaluation mode and removing gradient on its parameters, as is standard in plenoptic.

```{code-cell} ipython3
glm = GLM.load_nemos_glm("nemos_glm.npz")
glm.eval()
po.remove_grad(glm)
```

Now, load in the stimulus saved above and convert to a torch tensor. We'll only use the first 200 time points, for simplicity.

```{code-cell} ipython3
stim = jax_to_torch(np.load("nemos_stimulus.npz")["stimulus"], 2)[..., :200]
```

Now, let's visualize our model and its predictions:

```{code-cell} ipython3
plot_model(glm, stim);
```

- The leftmost plot shows the 1d filter of this model.
- The top plot shows the stimulus, the one-dimensional binary noise.
- The bottom shows the model's predicted firing rate, in spikes per second.

### Synthesizing metamers

Synthesize the metamer. In this setup, our only goal is to find a stimulus that gives rise to the same predicted firing rate, with no constraints.

:::{admonition} Seed
:class: note dropdown

Note that we set the seed at the top: it is possible for this problem to hit some NaNs during optimization. With seed 1, we reliably find a good solution.

:::

```{code-cell} ipython3
po.set_seed(1)
met = po.Metamer(stim, glm, penalty_lambda=0)
met.setup(optimizer=torch.optim.LBFGS)
met.synthesize(1000, stop_criterion=1e-20)
plot_met(met, "No penalty");
```

From the bottom subplot, we can see that our metamer is doing a good job: the predicted responses lie directly on top of each other. From the top, we can see that our metamer looks really different from our actual stimulus. In particular, its values are outside the range of our actual stimulus, varying from -0.6 or so up to almost 1.5.

While this may be interesting (our model expects the cell to give similar firing rates to a wider range of stimuli than those tested), in many cases the range of your stimulus is fixed based on the properties of your setup (e.g., you cannot display pixel values outside of some range). In the next section we show how to add a penalty to constrain the range.

### Using penalties to find constrain metamer synthesis

As discussed above, we would like to find a metamer whose values lie within the range of our original stimulus, -0.5 and 0.5. To do so, plenoptic allows you to specify a {external+plenoptic:ref}`metamer-regularization`, which modifies the objective function.

#### Constrain range

The most common way of using `penalty_function` is to constrain the range. By default, plenoptic constrains the range to lie between 0 and 1. This is reasonable for image pixels (the most common plenoptic use-case), but not our stimulus here. Instead, we'd like the range to vary between -0.5 and 0.5

We can do this using {external+plenoptic:func}`plenoptic.regularize.penalize_range`, specifying the `allowed_range` value to `(-0.5, 0.5)`:

```{code-cell} ipython3
range_penalty = lambda x: po.regularize.penalize_range(x, (-0.5, 0.5))
```

`range_penalty` is now a function that accepts a single tensor and returns a scalar, quadratic penalty on any values it contains outside of -0.5 and 0.5:

```{code-cell} ipython3
# all ones -- high penalty
print(range_penalty(torch.ones(10)))
# all zeros -- no penalty
print(range_penalty(torch.zeros(10)))
# random values between 0 and 1 -- medium penalty
print(range_penalty(torch.rand(10)))
```

Now we pass this function to {external+plenoptic:class}`~plenoptic.Metamer` at initialization and run synthesis as before:

```{code-cell} ipython3
met = po.Metamer(stim, glm, penalty_function=range_penalty)
met.setup(optimizer=torch.optim.LBFGS)
met.synthesize(1000, stop_criterion=1e-20)
plot_met(met, "Range Penalty");
```

We can see that our resulting metamer is still a good one, with the predicted firing rates lying on top of each other again. However, our new metamer's values lie between -0.5 and 0.5, as desired. It's not identical to our original stimulus, but it's pretty similar. Is that necessarily the case though? What if we tried to find a metamer that was *really* different from our original stimulus?

#### Uncorrelated

We can do that by using a different penalty, one which encourages the metamer to be uncorrelated with the original stimulus:

```{code-cell} ipython3
# this remaps so that the function's minimum value is at target, at which point remap
# returns 0. this works as long as we have a finite target (if our target was -inf, it
# wouldn't)
def remap(x, target=0):
    return (x-target).pow(2)

# note that this function accepts a metamer tensor and then correlates it with stim,
# which is defined earlier in this notebook -- it's not an argument to the function
def corr_penalty(metamer, target=0):
    # compute pearson R
    penalty = torch.corrcoef(torch.stack([stim.squeeze(), metamer.squeeze()], 0))[0, 1]
    return remap(penalty, target)

uncorr_penalty = lambda x: corr_penalty(x, 0) + po.regularize.penalize_range(x, (-.5, .5))
```

This function is slightly more complicated to the above. It accepts a single tensor and returns a scalar which is the sum of this "uncorrelated penalty" and the range penalty. So a tensor with a really low penalty value will be both uncorrelated with our original stimulus and have all values between -0.5 and 0.5.

We can pass this function to {external+plenoptic:class}`~plenoptic.Metamer` at initialization and run synthesis as before:

```{code-cell} ipython3
met = po.Metamer(stim, glm, penalty_function=uncorr_penalty)
met.setup(optimizer=torch.optim.LBFGS)
met.synthesize(1000, stop_criterion=1e-20)
plot_met(met, "UnCorr+Range Penalty");
```

We end up with a metamer that's quite different from our original stimulus! We can see that our new metamer aligns with the original stimulus for most sustained values (either negative or positive, e.g., around 77 seconds and 140). However, it has the opposite value near rapid transitions (e.g., around 90 seconds)! This highlights that our model is responding to rapid transitions between values, but doesn't care whether they're negative-to-positive or the reverse.

## Try other penalties!

What other penalties can you try? They need to accept a single tensor and return a single scalar. Try writing your own!

Plenoptic includes the {external+plenoptic:func}`plenoptic.validate.validate_penalty` function, which can validate your penalty function. If you call it on your function and the code runs without any errors, then you can use your function for metamer synthesis:

```{code-cell} ipython3
:tags: [skip-execution]

penalty_func = # WRITE SOMETHING HERE
po.validate.validate_penalty(penalty_func, stim.shape, stim.dtype)
```

## Try changing the weights of the GLM

In the above examples, we load in a weights file to set the GLM weights. But you can also specify it yourself! Try initializing the weights to some other tensor and use it to initialize the `GLM` class, and see what metamers result:

```{code-cell} ipython3
:tags: [skip-execution]

weights = torch.tensor() # WRITE SOMETHING HERE
glm = GLM(weight=weight, bias=) # PICK A BIAS
```

## Try making it multi-dimensional

So far, we've only looked at 1d GLMs. But you could build a 2d or even a 3d one, just make `weight` the appropriate shape! Of course, you'll also need a stimulus of the appropriate dimensionality as well.

Plenoptic includes some {external+plenoptic:ref}`LGN-inspired models <models-api>` you can raid for their spatial filter, and then combine them a temporal filter to build a 3d GLM.

```{code-cell} ipython3
model = po.models.CenterSurround(kernel_size=10, on_center=True)
# grab 2d center-surround filter
filt = model.filt.squeeze()
print(filt.shape)
```
