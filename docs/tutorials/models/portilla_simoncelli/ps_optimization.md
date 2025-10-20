(ps-optimization)=
# Portilla-Simoncelli optimization details

As you read the other Portilla-Simoncelli notebooks, you may have noticed that the example metamer synthesis code is a bit more complicated than the [basic Metamer usage](metamer-nb). There are two choices that are important for finding good solutions: the LBFGS optimization algorithm and a custom loss function. This notebook will discuss those two choices in a bit more detail, as well as what "good" means.

## What is a "good" metamer?

In theory, model metamers are sets of images that have different pixel values and identical model outputs. In practice, metamer synthesis is a non-convex optimization problem in a high-dimensional space. In such problems, one typically optimizes until optimization has converged (i.e., the loss has stopped decreasing) or your result is "good enough" (i.e., the loss is low enough). Each study must determine what "good enough" means for them, and you are encouraged to read papers that have used model metamers (e.g., {cite:alp}`Freeman2011-metam-ventr-stream`, {cite:alp}`Feather2019-metam`, {cite:alp}`Broderick2025-foveat-metam`) to see the approaches they have taken.

For the purposes of plenoptic, which provides implementations for researchers to use in their own experiments, "best" means "the lowest loss in the shortest time possible". Therefore, for the purposes of plenoptic's {class}`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli` implementation, we evaluated multiple possible configurations of optimizers, optimizer configurations, and optimization loss functions against each other and against the [earlier MATLAB implementation](https://github.com/LabForComputationalVision/textureSynth). This investigation can be found in [Issue #365](https://github.com/plenoptic-org/plenoptic/issues/365); the results are summarized in this notebook.

:::{admonition} What about perceptual quality?
:class: note

For the texture model, good metamer synthesis also means that, when your target image is a natural texture, successful Portilla-Simoncelli metamer synthesis should result in an image which:
1. Looks natural.
2. Belongs to the same perceptual texture class as the target.
3. Is not identical to the original image.

In an ideal world, these criteria would be identical to "has a low loss value". However, the Portilla-Simoncelli texture statistics are not all of equal perceptual importance: the **variance of the highpass residuals**, for example, is related to the presence of high frequencies in the image, where are present in the white noise samples used to initialize metamer synthesis, but which are largely absent from natural images. This statistic only makes a small contribution to the overall loss, but is very important to metamer perceptual quality, which is important for the results laid out in this notebook.

:::

## Optimizer configuration

By default, {class}`~plenoptic.synthesize.metamer.Metamer` uses the {class}`~torch.optim.Adam` optimizer, a variant of stochastic gradient descent. However, we found that {class}`~torch.optim.LBFGS` (the [Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS)) performs much better. LBFGS tracks the history of the optimization parameters (i.e., the metamer image pixels) over time, and uses this to approximate the second derivatives of the gradient. This results in:
- Higher memory usage.
- Longer duration per optimization step.
- Much faster decrease in loss per optimization step.

In our experiments, using LBFGS results in lower loss for a given duration of synthesis time than using Adam or the original MATLAB code, across a variety of target images and initializations. Additionally, the loss continues to decrease beyond the point when the alternatives have started to converge.

LBFGS has many keyword arguments that users are able to tweak. From our experiments, we found that the following dictionary performs best:

```{code-block} python
opt_kwargs = {
    "max_iter": 10,
    "max_eval": 10,
    "history_size": 100,
    "line_search_fn": "strong_wolfe",
    "lr": 1,
}
```

Of these arguments, the only one that we believe users may want to experiment with is `history_size`: a larger history size results in a larger memory footprint (which grows with the square of the value) and slightly longer durations. Loss does not decrease monotonically with `history_size`: we found that loss decreased from `history_size=3` to `history_size=100` and then increased with `history_size=300`. We have decided `history_size=100` is a good tradeoff for the purposes of our documentation and tests, but users may find increased performance with other values.

:::{admonition} How much memory?
:class: note dropdow

As mentioned, increasing `history_size` leads to more memory being used. In our tests, running {class}`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli` metamer synthesis for 100 iterations with the default initialization arguments, `history_size=100`, and a 256 by 256 image had a peak memory usage of about 600MB.

:::

Finally, as with all iterative optimization procedures, we must decide how long to run synthesis for. In our experiments, for 10 to 200 iterations, the loss decreases as approximately one over the number of iterations, and the synthesis duration (in [wall time](https://en.wikipedia.org/wiki/Elapsed_real_time)) increases approximately linearly with the number of iterations (the exact duration will depend on your hardware). For our documentation and tests, we have decided to use `max_iter=100`, though users may want to run synthesis for longer.

If you are interested, more details about the experiments that lead to the above recommendations can be found in [Issue #365](https://github.com/plenoptic-org/plenoptic/issues/365).

(ps-loss-function)=
## Custom loss function

When using the LBFGS optimizer configured as described above, plenoptic's Portilla-Simoncelli metamer synthesis has a lower loss in every component than the MATLAB implementation, with the notable exception of the variance of highpass residuals. The two synthesis implementations (as discussed in [](ps-mat-diffs)) are very different from each other: the MATLAB code does something akin to coordinate descent, where it optimizes each set of statistics separately, whereas the plenoptic code optimizes all of them together by trying to minimize the overall loss. This results in plenoptic weighting each statistic approximately equally and thus, implicitly, being considered equally important. They thus each have an approximately equal error at any given moment during synthesis, whereas the MATLAB code drops the error in the variance of the highpass residuals *incredibly* rapidly, matching it with much higher precision than the other statistics.

In order to make plenoptic's synthesis perform similarly, we need to make this statistic more important than the others. We do this by using a custom loss function which reweights the representation of each image before computing the L2-norm of the difference these representations. We do this using the {func}`~plenoptic.tools.optim.portilla_simoncelli_loss_factory` function, which takes in an instance of the {class}`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli` model and an image, and returns a loss function. (This is necessary because the shape of the model output depends on the parameters used to initialize it.)

We use a similar trick to remove the image's minimum and maximum from the loss, since their gradients are strangely behaved: when computing `torch.max(x, a)`, the gradient is 1 if `x>a` and 0 if `x<a` (and analogously for min). Thus, having many pixels outside the target range will overwhelm the gradient and changing the pixels by a small amount won't affect the gradient at all, making it a difficult problem in optimize. In practice, LBFGS gets stuck when there are many pixels outside the target range, and the range penalty whose behavior is determined by the initialization of  {class}`~plenoptic.synthesize.metamer.Metamer` accomplishes the same task.

To see what this looks like, here is how you could create the custom loss function directly:

```{code-block} python
import plenoptic as po
import torch

image = po.data.einstein()
model = po.simul.PortillaSimoncelli(image.shape[-2:])
weights = model.convert_to_dict(torch.ones_like(model(image)))
# reweight the pixel min/max and the variance of the highpass residuals
weights["pixel_statistics"][..., -2:] = 0
weights["var_highpass_residual"] = 100 * torch.ones_like(weights["var_highpass_residual"])
weights = model.convert_to_tensor(weights)

def loss(x, y):
    return l2_norm(weights * x, weights * y)

met = po.synth.Metamer(image, model, loss_function=loss)
```

This function makes use of the {func}`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli.convert_to_dict` and {func}`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli.convert_to_tensor` methods, which allow us to convert the Portilla-Simoncelli model representation from the standard vector form into a more structured dictionary form and vice versa.

{func}`~plenoptic.tools.optim.portilla_simoncelli_loss_factory` allows the user to specify the weights on the image's min/max and variance of the highpass residuals, though we do not expect this to be necessary.

:::{admonition} How would I do this for my own model?
:class: hint

For your own model, you could do something similar to the above to reweight your model representation as needed. However, you could also simply change the representation of your model, performing the reweighting in your model's `forward` <!-- skip-lint --> method. We did not do that because we want our representation to match that of the original MATLAB implementation. Using a reweighting function may also make testing out a variety of weights more straightforward.

:::
