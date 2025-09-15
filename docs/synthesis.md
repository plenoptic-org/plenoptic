(synthesis-objects)=

# Synthesis object design

The following describes how synthesis objects are structured. This is probably most useful if you are creating a new synthesis method that you would like to include in or be compliant with `plenoptic`, rather than using existing ones.

The synthesis methods included in `plenoptic` generate one or more novel images based on the output of a model. These images can be used to better understand the model or as stimuli for an experiment comparing the model against another system. Beyond this rather vague description, however, there is a good deal of variability. We use inheritance in order to try and keep synthesis methods as similar as possible, to facilitate user interaction with them (and testing), but we want to avoid forcing too much similarity.

In the following description:

- *must* connotes a requirement; any synthesis object not meeting this property will not be merged and is not considered "plenoptic-compliant".
- *should* connotes a suggestion; a compelling reason is required if the property is not met.
- *may* connotes an option; these properties may make things easier (for developers or users), but are completely optional.

## All synthesis methods

To that end, all synthesis methods must inherit the {class}`Synthesis <plenoptic.synthesize.synthesis.Synthesis>` class. This requires the synthesis method to have a `synthesize` <!-- skip-lint --> method, and provides helper functions for `save` <!-- skip-lint -->, `load` <!-- skip-lint -->, and `to` <!-- skip-lint -->, which must be used when implementing them. Furthermore:

- the initialization method (`__init__`) must accept any images as its first input(s). If only a single image is accepted, it must be named `image` <!-- skip-lint -->. If more than one, their names must be of the form `image_X`, replacing `X` with a more descriptive string. These must all have type [](torch.Tensor) and they must be validated with {func}`validate_input <plenoptic.tools.validate.validate_input>`. This should be stored in an attribute with the same name as the argument.
- the initialization method's next argument(s) must be any models or metrics that the synthesis will be based on. Similarly, if a single model / metric is accepted, they must be named `model` <!-- skip-lint --> / `metric` <!-- skip-lint -->. If more than one, their names should be of the form `X_model` / `X_metric`, replacing `X` with a more descriptive string. These must be validated with {func}`validate_model <plenoptic.tools.validate.validate_model>` / {func}`validate_metric <plenoptic.tools.validate.validate_metric>`. This should be stored in an attribute with the same name as the argument.
- any other arguments to the initialization method may follow.
- the object must be able to work on GPU and CPU. Users must be able to use the GPU either by initializing the synthesis object with tensors or models already on the GPU or by calling `to` <!-- skip-lint -->. The easiest way to do this is to use `torch.rand_like` and analogous methods when initializing new tensors where possible, and explicitly calling `to` <!-- skip-lint --> on any other newly-created tensors.
- ideally, the same for different float and complex data types (e.g., support both `torch.float32` and `torch.float64`), though this is not a strict requirement if there's a good reason.
- if `synthesize` <!-- skip-lint --> operates in an iterative fashion, it must accept a `max_iter: int` argument to specify how long to run synthesis for and a `stop_criterion: float` argument to allow for early termination if some convergence is reached. *What* exactly is being checked for convergence (e.g., change in loss, change in pixel values) may vary, but it must be clarified in the docstring. A `stop_iters_to_check: int` argument may also be included, which specifies how many iterations ago to check. If it is not included, the number of iterations must be clarified in docstring.
- additionally, if synthesis is iterative, [tqdm.auto.tqdm](https://tqdm.github.io/docs/shortcuts/#tqdmauto) must be used as a progress bar, initialized with `pbar = tqdm(range(max_iter))`, which should present information using `pbar.set_postfix` (such as the loss or whatever else is checked for convergence, as discussed above).
- `synthesize` <!-- skip-lint --> must not return anything. The outputs of synthesis must be stored as attributes of the object. The number of large attributes should be minimized in order to reduce overall size in memory.
- the synthesis output must be stored as an attribute with the same name as the class (e.g., {attr}`Metamer.metamer <plenoptic.synthesize.metamer.Metamer.metamer>`).
- any attribute or method that the user does not need should be hidden (i.e., start with `_`).
- consider using the `@property` decorator to make important attributes write-only or differentiate between the public and private views. For example, {class}`MADCompetition <plenoptic.synthesize.mad_competition.MADCompetition>` tracks the loss of the reference metric in a list, `_reference_metric_loss`, but the {attr}`reference_metric_loss <plenoptic.synthesize.mad_competition.MADCompetition.reference_metric_loss>` attribute converts this list to a tensor before returning it, as that's how the user will most often want to interact with it.
- All attributes should be initialized at object initialization, though they can be "False-y" (e.g., an empty list, `None`). At least one attribute should be `None` or an empty list at initialization, which we use when loading to check if the object has just been initialized. All attributes will be saved using the `save` <!-- skip-lint --> method, inherited from the {class}`Synthesis <plenoptic.synthesize.synthesis.Synthesis>` superclass.

The above are the only requirements that all synthesis methods must meet.

## Helper / display functions

It may also be useful to include some functions for investigating the status or output(s) of synthesis. As a general rule, if a function will be called during synthesis (e.g., to compute a loss value), it should be a method of the object. If it is only called afterwards (e.g., to display the synthesis outputs in a useful way), it should be included as a function in the same file (see {func}`metamer.display_metamer <plenoptic.synthesize.metamer.display_metamer>` for an example).


Functions that show images or videos should be called `display_X`, whereas those that show numbers as a scatter plot, line plot, etc. should be called `plot_X`. These must be axes-level matplotlib functions: they must accept an axis as an optional argument named `ax`, which will contain the plot. If no `ax` is supplied, `matplotlib.pyplot.gca` must be used to create / grab the axis. If a multi-axis figure is called for (e.g., to display the synthesis output and plot the loss), a function named `plot_synthesis_status` <!-- skip-lint --> should be created. This must have an optional `fig` argument, creating a figure if none is supplied. See {func}`metamer.plot_synthesis_status <plenoptic.synthesize.metamer.plot_synthesis_status>` for an example. If possible, this plot should be able to be animated to show progress over time. See {func}`metamer.plot_synthesis_status <plenoptic.synthesize.metamer.plot_synthesis_status>` for an example.

See our {doc}`/tutorials/advanced/Display` notebook for description and examples of the included plotting and display code.

## Optimized synthesis

Many synthesis methods will use an optimizer to generate their outputs. If the method makes use of a [](torch.optim.Optimizer) object, it must inherit the {class}`OptimizedSynthesis <plenoptic.synthesize.synthesis.OptimizedSynthesis>` class (this is a subclass of {func}`Synthesis <plenoptic.synthesize.synthesis.Synthesis>`, so the above all still applies).

Currently, the following are required (if not all of these are applicable to new methods, we may modify {class}`OptimizedSynthesis <plenoptic.synthesize.synthesis.OptimizedSynthesis>`):

- the points about iterative synthesis described above all hold: `synthesize` <!-- skip-lint --> must accept `max_iter`, `stop_criterion`, may accept `stop_iters_to_check`, and must use [tqdm.auto.tqdm](https://tqdm.github.io/docs/shortcuts/#tqdmauto).
- the object must have an `objective_function` <!-- skip-lint --> method, which returns a measure of "how bad" the current synthesis output is. Optimization is minimizing this value.
- the object must have a `_check_convergence` method, which is used (along with `stop_criterion` and, optionally, `stop_iters_to_check`) to determine if synthesis has converged.
- the object must have an `setup` <!-- skip-lint --> method, which initializes the synthesis output (e.g., with an appropriately-shaped sample of noise), optimizer, and scheduler. All of the inputs are optional and should have default behavior. The user can call this method once between initialization and `synthesize` <!-- skip-lint --> and it should be called in `synthesize` <!-- skip-lint --> if it hasn't been called yet.
- the setup method may accept some argument to affect this initialization, which should be named `initial_X` (replacing `X` as appropriate). For example, this could be another image to use for initialization (`initial_image` <!-- skip-lint -->) or some property of noise used to generate an initial image (`initial_noise`).
- the initialization method must accept `range_penalty_lambda: float` and `allowed_range: Tuple[float, float]` arguments, which should be used with {func}`penalize_range <plenoptic.tools.optim.penalize_range>` to constrain the range of synthesis output.
- during synthesis, the object should update the `_losses`, `_gradient_norm`, and `_pixel_change_norm` attributes on each iteration.
- the object may have a `_closure` method, which performs the gradient calculation. This (when passed to `optimizer.step` during the synthesis loop in `synthesize` <!-- skip-lint -->) enables optimization algorithms that perform several evaluations of the gradient before taking a step (e.g., second-order methods). See `OptimizedSynthesis._closure` for the simplest version of this.
- the `synthesize` <!-- skip-lint --> method should accept a `store_progress` argument, which optionally stores additional information over iteration, such as the synthesis output-in-progress. {class}`OptimizedSynthesis <plenoptic.synthesize.synthesis.OptimizedSynthesis>` has a setter method for this attribute, which will ensure things are correct. This argument can be an integer (in which case, the attributes are updated every `store_progress` <!-- skip-lint --> iterations), `True` (same behavior as `1`), or `False` (no updating of attributes). This should probably be done in a method named `_store`.
- the `synthesize` <!-- skip-lint --> method should be callable multiple times with the same object, in which case progress is resumed. On all subsequent calls, `store_progress` <!-- skip-lint -->, `stop_criterion`, and `stop_iters_to_check` must have the same values.
- must implement `get_progress` <!-- skip-lint --> method, which calls the superclass version of that method, specifying the names of the attributes that are updated every iteration and every `store_progress` <!-- skip-lint --> iterations.

## How to order methods

Python doesn't care how you order any of the methods or properties of a class, but doing so in a consistent manner will make reading the code easier, so try to follow these guidelines:

- The caller should (almost always) be above the callee and related concepts should be close together.
- `__init__` should be first, followed by any methods called within it.
- After all those initialization-related methods, `setup` <!-- skip-lint --> should come next.
- After `setup` <!-- skip-lint -->, `synthesize` <!-- skip-lint --> should come next. Again, this should be followed by most of the the methods called within it, ordered roughly by importance. Thus, the first methods should probably be `objective_function` <!-- skip-lint --> and `_optimizer_step`, followed by `_check_convergence`. What shouldn't be included in this section are helper methods that aren't scientifically interesting (e.g., `_initialize_optimizer`, `_store`).
- Next, any other content-related methods, such as helper methods that perform useful computations that are not called by `__init__` or `synthesize` <!-- skip-lint -->.
- Next, the helper functions we ignored from earlier, such as `_initialize_optimizer` and `_store`.
- Next, `save` <!-- skip-lint -->, `load` <!-- skip-lint -->, `to` <!-- skip-lint -->.
- Finally, all the properties.
