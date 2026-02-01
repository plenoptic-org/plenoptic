(models-doc)=

# Model requirements

`plenoptic` provides a model-based synthesis framework, and therefore we require several things of the models used with the package (the {func}`validate_model <plenoptic.tools.validate.validate_model>` function provides a convenient way to check whether your model meets the following requirements, and see [](plenoptic.simulate.models) for some examples). Your model:

- should inherit [](torch.nn.Module) (this is not strictly necessary, but will make meeting the other requirements easier).

- must be callable, be able to accept a [](torch.Tensor) as input, and return a [](torch.Tensor) as output.

  - If you inherit [](torch.nn.Module), implementing the `forward` <!-- skip-lint --> method will make your model callable.
  - Otherwise, implement the `__call__` method.

- the above transformation must be differentiable by [torch](inv:torch:std:doc#index). In practice, this generally means you perform all computations using [torch functions](inv:torch:std:doc#torch) (unless you want to write a custom `backward` method).

- must not have any learnable parameters. This is largely to save time by avoiding calculation of unnecessary gradients, but synthesis is performed with a **fixed** model --- we are optimizing the input, not the model parameters. You can use the helper function {func}`remove_grad <plenoptic.tools.validate.remove_grad>` to detach all parameters. Similarly, your model should probably be in evaluation mode (i.e., call `model.eval`), though this is not strictly required. See the [pytorch documentation](https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc) for the difference between evaluation mode and disabling gradient computation.

Additionally, your model inputs and outputs should be real- or complex-valued and should be *interpretable* for all possible values (within some range). The intention of stimulus synthesis is to facilitate model understanding --- if the synthesized stimulus are meaningless, this defeats the purpose. (Note that domain restrictions, such as requiring integer-valued inputs, can probably be accomplished by adding a penalty to an objective function, but will make your life harder.)

{class}`MADCompetition <plenoptic.synthesize.mad_competition.MADCompetition>` uses metrics, rather than models, which have the following requirements (use the {func}`validate_metric <plenoptic.tools.validate.validate_metric>` function to check whether your metric meets the following requirements and see [](plenoptic.metric) for some examples):

- a metric must be callable, accept two [](torch.Tensor) objects as inputs, and return a scalar as output. It can be a [](torch.nn.Module) object or other callable object, like models, as well as a function.
- when called on two identical inputs, the metric must return a value less than `5e-7` (effectively, zero).
- it must always return a non-negative number.

(models-coarse-to-fine)=
Finally, {class}`MetamerCTF <plenoptic.synthesize.metamer.MetamerCTF>` implements
coarse-to-fine synthesis, as described in {cite:alp}`Portilla2000-param-textur`. To make use of coarse-to-fine synthesis, your model must meet the following additional requirements (use the {func}`validate_coarse_to_fine <plenoptic.tools.validate.validate_coarse_to_fine>` function to check and see {class}`PortillaSimoncelli <plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli>` for an example):

- the model must have a `scales` <!-- skip-lint --> attribute.
- in addition to a [](torch.Tensor), the `forward` <!-- skip-lint --> method must also be able to accept an optional `scales` argument (equivalently, when calling the model, if the model does not inherit [](torch.nn.Module)).
- that argument should be a list, containing one or more values from `model.scales`, and the shape of the output should change when `scales` <!-- skip-lint --> is a strict subset of all possible values.
