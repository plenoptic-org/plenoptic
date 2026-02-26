.. _models-api:

Models
------

Models give a response to a single stimulus and are compatible with :class:`~plenoptic.synthesize.metamer.Metamer` and :class:`~plenoptic.synthesize.eigendistortion.Eigendistortion`, and can be turned into :ref:`metrics-api` by using the :func:`~plenoptic.metric.model_metric_factory` function.

See :ref:`models-doc` for more details.

.. currentmodule:: plenoptic.models
.. autosummary::
   :signatures: none
   :toctree: generated
   :template: torch_module.rst.jinja

   ~PortillaSimoncelli

.. rubric:: LGN-inspired Models
   :heading-level: 3

These "front end" models are inspired by the lateral geniculate nucleus (LGN; the first non-retinal stage of the primate visual system), come from :cite:alp:`Berardino2017-eigen`, and are nested, increasing in complexity as you move down the list.

.. autosummary::
   :signatures: none
   :toctree: generated
   :template: torch_module.rst.jinja

   ~LinearNonlinear
   ~LuminanceGainControl
   ~LuminanceContrastGainControl
   ~OnOff

The following models are used to construct the models above. They are probably most useful in the construction of other, more complex models, but they are compatible with our synthesis methods.

.. autosummary::
   :signatures: none
   :toctree: generated
   :template: torch_module.rst.jinja

   ~Identity
   ~Linear
   ~Gaussian
   ~CenterSurround
