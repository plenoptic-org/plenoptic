(tips-doc)=

# Tips and Tricks

:::{contents}
:local:
:::

## Why does synthesis take so long?

Synthesis can take a while to run, especially if you are trying to synthesize a large image or using a complicated model. The following might help:

- Reducing the amount of time your model's forward pass takes is the easiest way to reduce the overall duration of synthesis, as the forward pass is called many, many times over synthesis. Try using python's [built-in profiling tools](https://docs.python.org/3/library/profile.html) to check which part of your model's forward pass is taking the longest, and try to make those parts more efficient. Jupyter also has nice [profiling tools](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html). For example, if you have for loops in your code, try and replace them with matrix operations and [einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html).
- If you have access to a GPU, use it! If your inputs are on the GPU before initializing the synthesis methods, the synthesis methods will also make use of the GPU. You can also move the `plenoptic`'s synthesis methods and models over to the GPU after initialization using the `to` <!-- skip-lint --> method.
- Parallelize! When using plenoptic's synthesis method for an experiment, you will probably need to synthesize a large number of images. Doing so is [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) --- synthesizing a given image has no dependence on synthesizing any other image. Therefore, you can run many synthesis processes simultaneously. If you have access to a high performance computing cluster, you can often run 100s of jobs simultaneously, which will greatly reduce the amount of time required to synthesize a full set of images.
- PyTorch appears to greedily grab more [threads](https://en.wikipedia.org/wiki/Thread_(computing)) than are helpful. This can slow you down, especially if you are running many CPU-only jobs in parallel, and if you're using LBFGS. It can thus be helpful to restrict the number of threads each process has access to, using {func}`torch.set_num_threads` and the `OMP_NUM_THREADS` environment variable. See [](threads) for more information, including how to do so.
- PyTorch's [performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) has lots of helpful tips, so it's worth a read.

## Optimization is hard

You should double-check whether synthesis has successfully completed before interpreting the outputs or using them in any experiments. This is not necessary for eigendistortions (see its [notebook](eigendistortion-nb) for more details on why), but is necessary for all the iterative optimization methods.

- For metamers, this means double-checking that the difference between the model representation of the metamer and the target image is small enough. If your model's representation is multi-scale, trying coarse-to-fine optimization may help (see [notebook](metamer-coarse-to-fine) for details).
- For MAD competition, this means double-checking that the reference metric is constant and that the optimized metric has converged at a lower or higher value (depending on the value of `synthesis_target`); use {func}`plot_synthesis_status <plenoptic.synthesize.mad_competition.plot_synthesis_status>` to visualize these values. You will likely need to spend time trying out different values for the `metric_tradeoff_lambda` argument set during initialization to achieve this.

For all of the above, if synthesis has not found a good solution, you may need to run synthesis longer, use a learning-rate scheduler, change the learning rate, or try different optimizers. Each optimized synthesis method's `objective_function` <!-- skip-lint --> method captures the value that we are trying to minimize, but may be influenced by other values (such as the penalty on allowed range values).

See [Issue #365](https://github.com/plenoptic-org/plenoptic/issues/365) for an example of how we worked through finding a good optimization configuration for the {class}`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli` model.

In that investigation, we found that the {class}`~torch.optim.LBFGS` optimizer performs very well, finding good solutions in a relatively short amount of time. We have not had the opportunity to investigate this optimizer in other problems, but we recommend giving it a try.

Additionally, it may be helpful to visualize the progression of synthesis, using each synthesis method's `animate` <!-- skip-lint --> or `plot_synthesis_status` <!-- skip-lint --> helper functions (e.g., {func}`metamer.animate <plenoptic.synthesize.metamer.animate>`, {func}`metamer.plot_synthesis_status <plenoptic.synthesize.metamer.plot_synthesis_status>`).

(tips-model-tweak)=
### Tweaking the model

You can also improve your chances of finding a good synthesis by tweaking the model. For example, the loss function used for metamer synthesis by default is mean-squared error. This implicitly weights all aspects of the model's representation equally. Thus, if there are portions of the representation whose magnitudes are significantly smaller than the others, they might not be matched at the same rate as the others. You can address this using coarse-to-fine synthesis or picking a more suitable loss function, but it's generally a good idea for all of a model's representation to have roughly the same magnitude. You can do this in a variety of ways:

- Principled: compose your representation of statistics that you know lie within the same range. For example, use correlations instead of covariances (see the Portilla-Simoncelli model, and in particular [how plenoptic's implementation differs from matlab](ps-mat-diffs) for an example of this).
- Empirical: measure your model's representation on a dataset of relevant natural images and then use this output to z-score your model's representation on each pass (see {cite:alp}`Ziemba2021-oppos-effec` for an example; this is what the Van Hateren database is used for).
- In the middle: normalize statistics based on their value in the original image (note: not the image the model is taking as input! this will likely make optimization very difficult).
- Hacky: try reweighting different components of your model representation so that optimization matches some statistics more closely than others (see the [Portilla-Simoncelli custom loss function](ps-loss-function) for an example). This can be useful if some part of your representation is more perceptually important than others.

If you are computing a multi-channel representation, you may have a similar problem where one channel is larger or smaller than the others. Here, tweaking the loss function might be more useful. Using something like `logsumexp` (the log of the sum of exponentials, a smooth approximation of the maximum function) to combine across channels after using something like L2-norm to compute the loss within each channel might help.

### My synthesis procedure is getting stuck

It is possible that, even after tweaking your model and trying a variety of different optimization configurations, your optimization is still converging with a weirdly high loss. One possible reason for this is that different parts of your representation are trading off with each other, so that it's hard to reduce error in one part of your representation without increasing it in another. In order to determine whether this is the case, it can be helpful to use the {class}`~plenoptic.synthesize.metamer.MetamerCTF` class, optimizing different components of the representation independently and then together (see [](metamer-coarse-to-fine) for more details about {class}`~plenoptic.synthesize.metamer.MetamerCTF`; this is equivalent to but perhaps more convenient than creating individual sub-models which return the different parts of the representation). If the loss is able to decrease with each component separately, but increases again when optimizing all of them together, your statistics are likely trading off against each other. If this is the case, then you can try removing the offending parts of your representation or reweighting them (see [](ps-loss-function)).

See the discussion of the variance of the highpass residuals and the min/max in [Issue #365](https://github.com/plenoptic-org/plenoptic/issues/365) for an example of how we worked through similar problems for the {class}`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli` model.

## None of the existing synthesis methods meet my needs

`plenoptic` provides three synthesis methods, but you may find you wish to do something slightly outside the capabilities of the existing methods. There are generally two ways to do this: by tweaking your model or by extending one of the methods.

- See the Portilla-Simoncelli notebook on [](ps-extensions) for examples on how to get different metamer results by tweaking your model.
- The coarse-to-fine optimization, discussed in the [metamer notebook](metamer-coarse-to-fine), is an example of changing optimization by extending the {class}`~plenoptic.synthesize.metamer.Metamer` class.
- The [Synthesis extensions notebook](synthesis-extensions) contains a discussion focused on this as well.

If you extend a method successfully or would like help making it work, please let us know by posting a [discussion!](https://github.com/plenoptic-org/plenoptic/discussions)
