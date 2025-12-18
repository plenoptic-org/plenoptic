.. _api:

API Reference
=============

Synthesis
---------

.. rubric:: Synthesis objects
   :heading-level: 3
.. currentmodule:: plenoptic.synthesize
.. autosummary::
   :toctree: generated
   :signatures: long
   :template: synthesis_object.rst.jinja

   ~metamer.Metamer
   ~metamer.MetamerCTF
   ~eigendistortion.Eigendistortion
   ~mad_competition.MADCompetition

.. rubric:: Synthesis helper functions
   :heading-level: 3

These helper functions all are intended to help visualize the status and outputs of the synthesis objects above.

.. rubric:: Metamer / MetamerCTF
   :heading-level: 4

.. currentmodule:: plenoptic.synthesize
.. autosummary::
   :signatures: long

   ~metamer.plot_loss
   ~metamer.display_metamer
   ~metamer.plot_pixel_values
   ~metamer.plot_representation_error
   ~metamer.plot_synthesis_status
   ~metamer.animate

.. rubric:: MAD Competition
   :heading-level: 4
.. currentmodule:: plenoptic.synthesize
.. autosummary::
   :signatures: long

   ~mad_competition.display_mad_image
   ~mad_competition.display_mad_image_all
   ~mad_competition.plot_loss
   ~mad_competition.plot_loss_all
   ~mad_competition.plot_pixel_values
   ~mad_competition.plot_synthesis_status
   ~mad_competition.animate

.. rubric:: Eigendistortion
   :heading-level: 4
.. currentmodule:: plenoptic.synthesize
.. autosummary::
   :signatures: long

   ~eigendistortion.display_eigendistortion
   ~eigendistortion.display_eigendistortion_all

.. _models-api:

Models
------

Models give a response to a single stimulus and are compatible with :class:`~plenoptic.synthesize.metamer.Metamer` and :class:`~plenoptic.synthesize.eigendistortion.Eigendistortion`, and can be turned into :ref:`metrics-api` by using the :func:`~plenoptic.metric.model_metric.model_metric_factory` function.

See :ref:`models-doc` for more details.

.. currentmodule:: plenoptic.simulate
.. autosummary::
   :signatures: none

   ~models.portilla_simoncelli.PortillaSimoncelli

.. rubric:: Front End
   :heading-level: 3

These "front end" models are inspired by the retina, come from :cite:alp:`Berardino2017-eigen`, and are nested, increasing in complexity as you move down the list.

.. autosummary::
   :signatures: none

   ~frontend.LinearNonlinear
   ~frontend.LuminanceGainControl
   ~frontend.LuminanceContrastGainControl
   ~frontend.OnOff

.. rubric:: Naive
   :heading-level: 3

These models are used to construct the front end models above. They are probably most useful in the construction of other, more complex models, but they are compatible with our synthesis methods.

.. autosummary::
   :signatures: none

   ~naive.Identity
   ~naive.Linear
   ~naive.Gaussian
   ~naive.CenterSurround

.. _metrics-api:

Metrics
-------

Metrics compare two stimuli and are compatible with :class:`~plenoptic.synthesize.mad_competition.MADCompetition`.

See :ref:`metric-def` for more details.

.. currentmodule:: plenoptic.metric
.. autosummary::
   :signatures: none

   ~naive.mse
   ~model_metric.model_metric_factory
   ~perceptual_distance.ssim
   ~perceptual_distance.ms_ssim
   ~perceptual_distance.nlpd

.. rubric:: Metric components
   :heading-level: 3

These functions are used by the metrics above. While they are not compatible with any of our synthesis methods, they may be useful to better understand the behavior of their respective metrics.

.. autosummary::
   :signatures: none

   ~perceptual_distance.ssim_map
   ~perceptual_distance.normalized_laplacian_pyramid

Canonical Computations
----------------------

These classes and functions may be helpful for constructing your own models. As is, they are not compatible with any of the synthesis methods.

.. rubric:: Image pyramids
   :heading-level: 3

Image pyramids decompose images into bands corresponding to different spatial frequencies and orientations. As we often think of neurons in the early visual system as similarly only responding to a range of spatial frequencies and/or orientations, these pyramids can be used to construct a model of those early areas.

See :external+pyrtools:std:doc:`index` for more information, including links to resources for learning more.

.. currentmodule:: plenoptic.simulate
.. autosummary::
   :signatures: none

   ~canonical_computations.laplacian_pyramid.LaplacianPyramid
   ~canonical_computations.steerable_pyramid_freq.SteerablePyramidFreq

.. rubric:: Filter construction functions
   :heading-level: 3

These convenience functions make it easier to construct some commonly-used filters.

.. currentmodule:: plenoptic.simulate
.. autosummary::
   :signatures: none

   ~canonical_computations.filters.circular_gaussian2d
   ~canonical_computations.filters.gaussian1d

.. rubric:: Non-linearities
   :heading-level: 3

These functions perform some useful image-processing non-linearities on tensors or dictionaries of tensors.

.. currentmodule:: plenoptic
.. autosummary::
   :signatures: none

   ~tools.signal.rectangular_to_polar
   ~tools.signal.polar_to_rectangular
   ~simulate.canonical_computations.non_linearities.local_gain_control
   ~simulate.canonical_computations.non_linearities.local_gain_release
   ~simulate.canonical_computations.non_linearities.rectangular_to_polar_dict
   ~simulate.canonical_computations.non_linearities.polar_to_rectangular_dict
   ~simulate.canonical_computations.non_linearities.local_gain_control_dict
   ~simulate.canonical_computations.non_linearities.local_gain_release_dict
