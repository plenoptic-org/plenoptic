.. _models:

Model requirements
******************

``plenoptic`` provides a model-based synthesis framework, and therefore we
require several things of the models used with the package (the
:func:`plenoptic.tools.validate.validate_model` function provides a convenient
way to check whether your model meets the following requirements). Your model:

* should inherit ``torch.nn.Module`` (this is not strictly necessary, but will
  make meeting the other requirements easier).
* must be callable, be able to accept a 4d ``torch.Tensor`` as input, and return a
  3d or 4d ``torch.Tensor`` as output.
* must be differentiable by ``torch``. In practice, this generally means you
  perform all computations using ``torch`` functions (unless you want to write a
  custom ``.backward()`` method).
* must not have any learnable parameters. This is largely to save time by
  avoiding calculation of unnecessary gradients, but synthesis is performed with
  a **fixed** model --- we are optimizing the input, not the model parameters.
  You can use the helper function :func:`plenoptic.tools.validate.remove_grad` to detach
  all parameters. Similarly, your model should probably be in evaluation mode
  (i.e., call ``model.eval()``).

Additionally, your model inputs and outputs should be real- or complex-valued
and should be *interpretable* for all possible values (within some range). The
intention of stimulus synthesis is to facilitate model understanding --- if the
synthesized stimulus are meaningless, this defeats the purpose. (Note that
domain restrictions, such as requiring integer-valued inputs, can probably be
accomplished by adding a penalty to an objective function, but will make your
life harder.)

:class:`plenoptic.synthesize.mad_competition.MADCompetition` uses metrics, rather than models,
which have the following requirements (use the
:func:`plenoptic.tools.validate.validate_metric` function to check whether your
metric meets the following requirements):

* a metric must be callable, accept two 4d ``torch.Tensor`` objects as inputs,
  and return a scalar as output.
* when called on two identical inputs, the metric must return a value of 0.

Finally, :class:`plenoptic.synthesize.metamer.Metamer` supports coarse-to-fine synthesis,
as described in [PS]_. To make use of coarse-to-fine synthesis, your model must
meet the following additional requirements (use the
:func:`plenoptic.tools.validate.validate_coarse_to_fine` function to check):

* the model must have a ``scales`` attribute.
* in addition to a ``torch.Tensor``, one must be able to pass a ``scales``
  keyword argument when calling the model.
* that argument should be a list, containing one or more values from
  ``model.scales``, the shape of the output should change when ``scales`` is
  a strict subset of all possible values.

.. [PS] J Portilla and E P Simoncelli. A Parametric Texture Model based on Joint
        Statistics of Complex Wavelet Coefficients. Int'l Journal of Computer
        Vision. 40(1):49-71, October, 2000. `abstract
        <http://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html>`_,
        `paper <http://www.cns.nyu.edu/~lcv/texture/>`_.
