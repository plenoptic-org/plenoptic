(tips-doc)=

# Tips and Tricks

## Why does synthesis take so long?

Synthesis can take a while to run, especially if you are trying to synthesize a large image or using a complicated model. The following might help:

- Reducing the amount of time your model's forward pass takes is the easiest way to reduce the overall duration of synthesis, as the forward pass is called many, many times over synthesis. Try using python's [built-in profiling tools](https://docs.python.org/3/library/profile.html) to check which part of your model's forward pass is taking the longest, and try to make those parts more efficient. Jupyter also has nice [profiling tools](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html). For example, if you have for loops in your code, try and replace them with matrix operations and [einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html).
- If you have access to a GPU, use it! If your inputs are on the GPU before initializing the synthesis methods, the synthesis methods will also make use of the GPU. You can also move the `plenoptic`'s synthesis methods and models over to the GPU after initialization using the `.to()` method.

## Optimization is hard

You should double-check whether synthesis has successfully completed before interpreting the outputs or using them in any experiments. This is not necessary for eigendistortions (see its [notebook](eigendistortion-nb) for more details on why), but is necessary for all the iterative optimization methods.

- For metamers, this means double-checking that the difference between the model representation of the metamer and the target image is small enough. If your model's representation is multi-scale, trying coarse-to-fine optimization may help (see [notebook](metamer-coarse-to-fine) for details).
- For MAD competition, this means double-checking that the reference metric is constant and that the optimized metric has converged at a lower or higher value (depending on the value of `synthesis_target`); use [`plot_synthesis_status`](plenoptic.synthesize.mad_competition.plot_synthesis_status) to visualize these values. You will likely need to spend time trying out different values for the `metric_tradeoff_lambda` argument set during initialization to achieve this.

For all of the above, if synthesis has not found a good solution, you may need to run synthesis longer, use a learning-rate scheduler, change the learning rate, or try different optimizers. Each method's `objective_function` method captures the value that we are trying to minimize, but may contain other values (such as the penalty on allowed range values).

Additionally, it may be helpful to visualize the progression of synthesis, using each synthesis method's `animate` or `plot_synthesis_status` helper functions (e.g., [`metamer.plot_synthesis_status`](plenoptic.synthesize.metamer.plot_synthesis_status)).

(tips-model-tweak)=
### Tweaking the model

You can also improve your changes of finding a good synthesis by tweaking the model. For example, the loss function used for metamer synthesis by default is mean-squared error. This implicitly weights all aspects of the model's representation equally. Thus, if there are portions of the representation whose magnitudes are significantly smaller than the others, they might not be matched at the same rate as the others. You can address this using coarse-to-fine synthesis or picking a more suitable loss function, but it's generally a good idea for all of a model's representation to have roughly the same magnitude. You can do this in a principled or empirical manner:

- Principled: compose your representation of statistics that you know lie within the same range. For example, use correlations instead of covariances (see the Portilla-Simoncelli model, and in particular [how plenoptic's implementation differs from matlab](ps-mat-diffs) for an example of this).
- Empirical: measure your model's representation on a dataset of relevant natural images and then use this output to z-score your model's representation on each pass (see {cite:alp}`Ziemba2021-oppos-effec` for an example; this is what the Van Hateren database is used for).
- In the middle: normalize statistics based on their value in the original image (note: not the image the model is taking as input! this will likely make optimization very difficult).

If you are computing a multi-channel representation, you may have a similar problem where one channel is larger or smaller than the others. Here, tweaking the loss function might be more useful. Using something like `logsumexp` (the log of the sum of exponentials, a smooth approximation of the maximum function) to combine across channels after using something like L2-norm to compute the loss within each channel might help.

## None of the existing synthesis methods meet my needs

`plenoptic` provides three synthesis methods, but you may find you wish to do something slightly outside the capabilities of the existing methods. There are generally two ways to do this: by tweaking your model or by extending one of the methods.

- See the [Portilla-Simoncelli texture model notebook](ps-nb) for examples on how to get different metamer results by tweaking your model or extending the `Metamer` class.
- The coarse-to-fine optimization, discussed in the [metamer notebook](metamer-coarse-to-fine), is an example of changing optimization by extending the `Metamer` class.
- The [Synthesis extensions notebook](synthesis-extensions) contains a discussion focused on this as well.

If you extend a method successfully or would like help making it work, please let us know by posting a [discussion!](https://github.com/plenoptic-org/plenoptic/discussions)
