---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: plenoptic
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [hide-input]

import warnings

warnings.filterwarnings(
    "ignore",
    message="Looks like representation is image-like",
    category=UserWarning,
)
```

:::{admonition} Download
:class: important

Download this notebook: **{nb-download}`Metamer.ipynb`**!

:::

(metamer-nb)=
# Metamers

Metamers are an old concept in the study of perception, dating back to the color-matching experiments in the 18th century that first provided support for the existence of three cone types (though it would be another two hundred years before anatomical evidence was found). These color-matching evidences demonstrated that, by combining three colored lights in different proportions, you could generate a color that humans perceived as identical to any other color, even though their physical spectra were different. Perceptual metamers, then, refer to two images that are physically different but perceived as identical.

For the purposes of `plenoptic`, wherever we say "metamers", we mean "model metamers": images that are physically different but have identical representation for a given model, i.e., that the model "perceives" as identical. Like all synthesis methods, it is model-specific, and one potential experiment is to determine if model metamers can serve as human percpetual metamers, which provides support for the model as an accurate representation of the human visual system.

In the Lab for Computational Vision, this goes back to [Portilla and Simoncelli, 2001](https://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf), where the authors created a parametric model of textures and synthesized novel images as a way of demonstrating the cases where the model succeeded and failed. In that paper, the model did purport to have anything to do with human vision, and they did not refer to their images as "metamers", that term did not appear until [Freeman and Simoncelli, 2011](https://www.cns.nyu.edu/pub/eero/freeman10-reprint.pdf), where the authors pool the Portilla and Simoncelli texture statistics in windows laid out in a log-polar fashion to generate putative human perceptual metamers.

This notebook demonstrates how to use the {class}`Metamer <plenoptic.synthesize.metamer.Metamer>` class to generate model metamers.

```{code-cell} ipython3
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

import plenoptic as po

# this notebook runs just about as fast with GPU and CPU
DEVICE = torch.device("cpu")

# so that relative sizes of axes created by po.imshow and others look right
plt.rcParams["figure.dpi"] = 72
# Animation-related settings
plt.rcParams["animation.html"] = "html5"
# use single-threaded ffmpeg for animation writer
plt.rcParams["animation.writer"] = "ffmpeg"
plt.rcParams["animation.ffmpeg_args"] = ["-threads", "1"]


%load_ext autoreload
%autoreload 2
```

(metamer-basic-usage)=
## Basic usage

As with all our synthesis methods, we start by grabbing a target image and initializing our model.

```{code-cell} ipython3
img = po.data.curie().to(DEVICE)
po.imshow(img);
```

For the model, we'll use a simple On-Off model of visual neurons

```{code-cell} ipython3
model = po.simul.OnOff((7, 7))
model.to(DEVICE)
model.eval()
po.tools.remove_grad(model)
```

When this model is called on the image, it returns a 4d tensor. This representation is what the {class}`Metamer <plenoptic.synthesize.metamer.Metamer>` class will try to match.

```{code-cell} ipython3
print(model(img))
```

In order to visualize this, we can use the helper function {func}`plot_representation <plenoptic.tools.display.plot_representation>` (see [Display notebook](display-nb) for more details here). In this case, the representation looks like two images, and so we plot it as such:

```{code-cell} ipython3
po.tools.display.plot_representation(data=model(img), figsize=(11, 5))
```

At the simplest, to use {class}`Metamer <plenoptic.synthesize.metamer.Metamer>`, simply initialize it with the target image and the model, then call {func}`synthesize <plenoptic.synthesize.metamer.Metamer.synthesize>`. By setting `store_progress=True`, we update a variety of attributes (all of which start with `saved_`) on each iteration so we can later examine, for example, the synthesized image over time.

```{code-cell} ipython3
met = po.synth.Metamer(img, model)
met.synthesize(store_progress=True, max_iter=50)
```

We then call the {func}`plot_synthesis_status <plenoptic.synthesize.metamer.plot_synthesis_status>` function to see how things are doing. The image on the left shows the metamer at this moment, while the center plot shows the loss over time, with the red dot pointing out the current loss, and the rightmost plot shows the representation error. If a model has a `plot_representation` <!-- skip-lint --> method, this plot can be more informative, but this plot can always be created.

```{code-cell} ipython3
# model response error plot has two subplots, so we increase its relative width
po.synth.metamer.plot_synthesis_status(
    met, width_ratios={"plot_representation_error": 2}
)
```

{func}`plot_synthesis_status <plenoptic.synthesize.metamer.plot_synthesis_status>` is a helper function to show all of this at once, but the individual components can be created separately:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(25, 5), gridspec_kw={"width_ratios": [1, 1, 2]})
po.synth.metamer.display_metamer(met, ax=axes[0])
po.synth.metamer.plot_loss(met, ax=axes[1])
po.synth.metamer.plot_representation_error(met, ax=axes[2])
```

The loss is decreasing, but clearly there's much more to go. So let's continue.

You can resume synthesis as long as you pass the same argument to `store_progress` <!-- skip-lint --> on each run.

Everything that stores the progress of the optimization ({attr}`losses <plenoptic.synthesize.metamer.Metamer.losses>`, {attr}`saved_metamer <plenoptic.synthesize.metamer.Metamer.saved_metamer>`) will persist between calls and so potentially get very large.

```{code-cell} ipython3
met.synthesize(store_progress=True, max_iter=100)
```

Let's examine the status again. But instead of looking at the latest iteration, let's look at 10 from the end:

```{code-cell} ipython3
po.synth.metamer.plot_synthesis_status(
    met, iteration=-10, width_ratios={"plot_representation_error": 2}
)
```

Since we have the ability to select which iteration to plot (as long as we've been storing the information), we can create an animation showing the synthesis over time. This {class}`matplotlib FuncAnimation <matplotlib.animation.FuncAnimation>` object can either be viewed in the notebook (note that this requires the matplotlib configuration options in the first cell of this notebook) or saved as some video format (e.g., `anim.save('test.mp4'`).

```{code-cell} ipython3
anim = po.synth.metamer.animate(met, width_ratios={"plot_representation_error": 2})
anim
```

Generally speaking, synthesis will run until you hit `max_iter` iterations. However, synthesis can also stop if it looks like the loss has stopped changing. This behavior is controlled with the `loss_thresh` and `loss_change_iter` arguments: if the loss has changed by less than `loss_thresh` over the past `loss_change_iter` iterations, we stop synthesis.

### Moving between devices

{class}`Metamer <plenoptic.synthesize.metamer.Metamer>` has a {func}`to <plenoptic.synthesize.metamer.Metamer.to>` method for moving the object between devices or dtypes. Call it as you would call any {func}`tensor.to <torch.Tensor.to>` and it will move over the necessary attributes.

### Saving and loading

Finally, you probably want to save the results of your synthesis. As mentioned above, you can save the synthesis animation, and all of the plots return regular `matplotlib` Figures and can be manipulated as expected. The synthesized image itself is a tensor and can be detached, converted to a numpy array, and saved (either as an image or array) as you'd expect. {func}`po.to_numpy <plenoptic.tools.data.to_numpy>` is a convenience function we provide for operations like this, which detaches the tensor, sends it to the CPU, and converts it to a numpy array with appropriate dtype. Note that it doesn't squeeze the tensor, so you may want to do that yourself.

```{code-cell} ipython3
met_image = po.to_numpy(met.metamer).squeeze()
# convert from array to int8 for saving as an image
print(f"Metamer range: ({met_image.min()}, {met_image.max()})")
met_image = po.tools.convert_float_to_int(np.clip(met_image, 0, 1))
imageio.imwrite("test.png", met_image)
```

The metamer lies slightly outside the range `[0, 1]`, so we clip before saving as an image. Metamer's objective function has a quadratic penalty on the synthesized image's range, and the weight on this penalty can be adjusted by changing the value of `range_penalty_lambda` <!-- skip-lint --> at initialization.

You can also save the entire {class}`Metamer <plenoptic.synthesize.metamer.Metamer>` object with its {func}`save <plenoptic.synthesize.metamer.Metamer.save>` method. This can be fairly large (depending on how many iterations you ran it for and how frequently you stored progress), but stores all information:

```{code-cell} ipython3
met.save("test.pt")
```

You can then load it back in using the method {func}`load <plenoptic.synthesize.metamer.Metamer.load>`. Note that you need to first instantiate the {class}`Metamer <plenoptic.synthesize.metamer.Metamer>` object and then call {func}`load <plenoptic.synthesize.metamer.Metamer.load>` --- it must be instantiated with the same image, model, and loss function in order to load it in!

```{code-cell} ipython3
met_copy = po.synth.Metamer(img, model)
# it's modified in place, so this method doesn't return anything
met_copy.load("test.pt")
(met_copy.saved_metamer == met.saved_metamer).all()
```

Because the model itself can be quite large, we do not save it along with the {class}`Metamer <plenoptic.synthesize.metamer.Metamer>` object. This is why you must initialize it before loading from disk.

## Reproducibility

You can set the seed before you call {func}`synthesize <plenoptic.synthesize.metamer.Metamer.synthesize>` for reproducibility by using {func}`set_seed <plenoptic.tools.optim.set_seed>`. This will set both the `pytorch` and `numpy` seeds, but note that we can't guarantee complete reproducibility: see [](reproduce) for some caveats.

Also note that pytorch does not guarantee identical results between CPU and GPU, even with the same seed.

## More Advanced Options

The solution found by the end of the [Basic usage section](metamer-basic-usage) is only one possible metamer. In general, optimization in a high-dimensional space with non-linear models is inherently challenging and so we can't guarantee you'll find a model metamer, but we do provide some tools / extra functionality to help.

### Initialization

By default, we initialize the {attr}`metamer <plenoptic.synthesize.metamer.Metamer.metamer>` attribute with with uniformly-distributed random noise between 0 and 1. If you wish to use some other image for initialization, you can initialize it yourself (it must be the same shape as `target_signal`) and pass to the optional {func}`setup <plenoptic.synthesize.metamer.Metamer.setup>` method before calling {func}`synthesize <plenoptic.synthesize.metamer.Metamer.synthesize>`.

### Optimization basics

You can set all the various optimization parameters you'd expect. {func}`setup <plenoptic.synthesize.metamer.Metamer.setup>` has an `optimizer` argument, which accepts an uninitialized pytorch optimizer, and an `optimizer_kwargs` arg, which accepts a dictionary. You can therefore change the optimizer from the default `Adam` and/or specify any of the arguments you would like:

```{code-cell} ipython3
met = po.synth.Metamer(img, model)
met.setup(optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.001})
met.synthesize()
```

{func}`setup <plenoptic.synthesize.metamer.Metamer.setup>` also accepts a `scheduler` argument, so that you can pass a [pytorch scheduler](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate), which modifies the learning rate during optimization.


### Regularization penalty

It is sometimes useful to control properties of the synthesized metamer beyond
those captured by the input model. Some examples include controlling the range
of pixel values, penalizing high frequencies, or matching the spectrum of the target image.

For this purpose, the {class}`Metamer <plenoptic.synthesize.metamer.Metamer>` class
takes an optional `penalty_function` argument at initialization.
The `penalty_function` is a callable that takes as an input the synthesized metamer
image, and returns a scalar penalty. This scalar penalty is added to the loss
during optimization, and it can be used to control certain properties of the
synthesized metamer.

For example, the default `penalty_function` uses the
{func} `penalize_range <plenoptic.tools.regularization.penalize_range>` function
to penalize pixel values that fall outside the range [0, 1], helping to keep the
synthesized metamer within this range. The user can pass custom penalty functions
that control other properties of the synthesized metamer.
For example, we can constrain the image pixels to fall inside a different range,
by using the argument `allowed_range` in the 
{func} `penalize_range <plenoptic.tools.regularization.penalize_range>` function
to define a new range penalization.

```{code-cell} ipython3
from plenoptic.tools import regularization
# Create custom_penalty function, that penalizes pixels outside of [0.2, 0.8] range
def custom_penalty(image):
    penalty = regularization.penalize_range(image, allowed_range=(0.2, 0.8))
)

# Pass the custom_penalty function to the Metamer class, and synthesize the metamer
met = po.synth.Metamer(
    img,
    model,
    penalty_function=custom_penalty,
)
met.synthesize(store_progress=True, max_iter=50)

print("Range of synthesized metamer: "
  f" {met.metamer.min().item():.3f} to {met.metamer.max().item():.3f}")
```

The {class}`Metamer <plenoptic.synthesize.metamer.Metamer>` class also has a
`penalty_lambda` argument, that weighs the contribution of the penalty function
to optimization.


(metamer-coarse-to-fine)=
### Coarse-to-fine optimization

Some models, such as the Portilla-Simoncelli texture statistics, have a multiscale representation of the image, which can complicate the optimization. It's generally recommended that you normalize the representation (or use a specific loss function) so that the different scales all contribute equally to the representation (see [](tips-model-tweak) for more information).

We provide the option to use coarse-to-fine optimization, such that you optimize the different scales separately (starting with the coarsest and then moving progressively finer) and then, at the end, optimizing all of them simultaneously. This was first used in Portilla and Simoncelli, 2000, and can help avoid local optima in image space. Unlike everything else described in this notebook, it will not work for all models. There are two specifications the model must meet (see [Model requirements page](models-coarse-to-fine) for more details):

1. It must have a `scales` <!-- skip-lint --> attribute that gives the scales in the order they should be optimized.
2. Its `forward` <!-- skip-lint --> method must accept a `scales` keyword argument, which accepts a list and causes the model to return only the scale(s) included. See {func}`PortillaSimoncelli.forward <plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli.forward>` for an example.

We can see that the included {class}`PortillaSimoncelli <plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli>` model satisfies these constraints, and that the model returns a subset of its output when the `scales` argument is passed to {func}`PortillaSimoncelli.forward <plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli.forward>`:

```{code-cell} ipython3
# we change images to a texture, which the PS model can do a good job capturing
img = po.data.reptile_skin()
ps = po.simul.PortillaSimoncelli(img.shape[-2:])
print(ps.scales)
print(ps.forward(img).shape)
print(ps.forward(img, scales=[0]).shape)
```

There are two choices for how to handle coarse-to-fine optimization: `'together'` or `'separate'`. In `'together'` (recommended), we start with the coarsest scale and then gradually add each finer scale (this is like blurring the objective function and then gradually adding details). In `'separate'`, we compute the gradient with respect to each scale separately (ignoring the others), then with respect to all of them at the end.

If our model meets the above requirements, then we can use the {class}`MetamerCTF <plenoptic.synthesize.metamer.MetamerCTF>` class, which uses this coarse-to-fine procedure. We specify which of the two above options are used during initialization, and it will work through the scales as described above (and will resume correctly if you resume synthesis). Note that the progress bar now specifies which scale we're on.

```{code-cell} ipython3
met = po.synth.MetamerCTF(
    img, ps, loss_function=po.tools.optim.l2_norm, coarse_to_fine="together"
)
met.synthesize(store_progress=True, max_iter=100)
# we don't show our synthesized image here, because it hasn't gone through all the
# scales, and so hasn't finished synthesizing
```

In order to control when synthesis considers a scale to be "done" and move on to the next one, you can set two arguments: `change_scale_criterion` and `ctf_iters_to_check`. If the scale-specific loss (`current_scale_loss` in the progress bar above) has changed by less than `change_scale_criterion` over the past `ctf_iters_to_check` iterations, we consider that scale to have reached a local optimum and move on to the next. You can also set `change_scale_criterion=None`, in which case we always shift scales after `ctf_iters_to_check` iterations

```{code-cell} ipython3
# initialize with some noise that is approximately mean-matched and with low variance
im_init = torch.rand_like(img) * 0.1 + img.mean()
met = po.synth.MetamerCTF(
    img,
    ps,
    loss_function=po.tools.optim.l2_norm,
    coarse_to_fine="together",
)
met.setup(im_init)
met.synthesize(
    store_progress=10,
    max_iter=500,
    change_scale_criterion=None,
    ctf_iters_to_check=7,
)
po.imshow(
    [met.image, met.metamer],
    title=["Target image", "Synthesized metamer"],
    vrange="auto1",
);
```

And we can see these shifts happening in the animation of synthesis:

```{code-cell} ipython3
po.synth.metamer.animate(met)
```

{class}`MetamerCTF <plenoptic.synthesize.metamer.MetamerCTF>` has several attributes which are used in the course of coarse-to-fine synthesis:

- {attr}`scales_loss <plenoptic.synthesize.metamer.MetamerCTF.scales_loss>`: this list contains the scale-specific loss at each iteration (that is, the loss computed on just the scale(s) we're optimizing on that iteration; which we use to determine when to switch scales).
- {attr}`scales <plenoptic.synthesize.metamer.MetamerCTF.scales>`: this is a list of the scales in optimization order (i.e., from coarse to fine). The last entry will be `'all'` (since after we've optimized each individual scale, we move on to optimizing all at once). This attribute will be modified by the {func}`synthesize <plenoptic.synthesize.metamer.MetamerCTF.synthesize>` method and is used to track which scale we're currently optimizing (the first one). When we've gone through all the scales present, this will just contain a single value: `'all'`.
- {attr}`scales_timing <plenoptic.synthesize.metamer.MetamerCTF.scales_timing>`: this is a dictionary whose keys are the values of scales. The values are lists, with 0 through 2 entries: the first entry is the iteration where we started optimizing this scale, the second is when we stopped (thus if it's an empty list, we haven't started optimizing it yet).
- {attr}`scales_finished <plenoptic.synthesize.metamer.MetamerCTF.scales_finished>`: this is a list of the scales that we've finished optimizing (in the order we've finished). The union of this and {attr}`scales <plenoptic.synthesize.metamer.MetamerCTF.scales>` will be the same as `metamer.model.scales` (e.g., {attr}`PortillaSimoncelli.forward <plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli.scales>`).

A small wrinkle: if `coarse_to_fine=='together'`, then none of these will ever contain the final, finest scale, since that is equivalent to `'all'`.
