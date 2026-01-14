.. _api:

API Reference
=============

.. _synthesis-api:

Synthesis objects
-----------------

Synthesis objects generate novel stimuli which allow researchers to better understand how their computational models make sense of their inputs.

.. currentmodule:: plenoptic.synthesize
.. autosummary::
   :toctree: generated
   :signatures: long
   :template: synthesis_object.rst.jinja

   ~metamer.Metamer
   ~metamer.MetamerCTF
   ~eigendistortion.Eigendistortion
   ~mad_competition.MADCompetition

.. _models-api:

Models
------

Models give a response to a single stimulus and are compatible with :class:`~plenoptic.synthesize.metamer.Metamer` and :class:`~plenoptic.synthesize.eigendistortion.Eigendistortion`, and can be turned into :ref:`metrics-api` by using the :func:`~plenoptic.metric.model_metric.model_metric_factory` function.

See :ref:`models-doc` for more details.

.. currentmodule:: plenoptic.simulate
.. autosummary::
   :signatures: none

   ~models.portilla_simoncelli.PortillaSimoncelli

.. rubric:: Front End Models
   :heading-level: 3

These "front end" models are inspired by the retina, come from :cite:alp:`Berardino2017-eigen`, and are nested, increasing in complexity as you move down the list.

.. autosummary::
   :signatures: none

   ~frontend.LinearNonlinear
   ~frontend.LuminanceGainControl
   ~frontend.LuminanceContrastGainControl
   ~frontend.OnOff

The following models are used to construct the front end models above. They are probably most useful in the construction of other, more complex models, but they are compatible with our synthesis methods.

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

Synthesis helper functions
---------------------------

These helper functions all are intended to help visualize the status and outputs of `synthesis objects <synthesis-api>`_.

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

Loading, creating, and dealing with images
------------------------------------------

.. currentmodule:: plenoptic
.. autosummary::
   :signatures: none

   ~tools.data.load_images
   ~tools.data.to_numpy
   ~tools.data.convert_float_to_int

.. rubric:: Example images
   :heading-level: 3

Plenoptic includes a small number of example images, which are used throughout the documentation.

.. currentmodule:: plenoptic
.. autosummary::
   :signatures: none

   ~data.einstein
   ~data.curie
   ~data.parrot
   ~data.reptile_skin
   ~data.color_wheel

Plenoptic includes a helper function for downloading data. These files are used for our internal tests and documentation.

.. autosummary::
   :signatures: none

   ~data.fetch.fetch_data
   ~data.fetch.DOWNLOADABLE_FILES

.. rubric:: Synthetic images
   :heading-level: 3

The following functions create synthetic image tensors. See :external+pyrtools:mod:`pyrtools.tools.synthetic_images` for more synthetic images.

.. currentmodule:: plenoptic.tools
.. autosummary::
   :signatures: none

   ~signal.make_disk
   ~data.polar_radius
   ~data.polar_angle

Validation
----------

The following functions are used to validate that the user-supplied inputs are compatible with our synthesis objects.

.. autosummary::
   :signatures: none

   ~validate.remove_grad
   ~validate.validate_model
   ~validate.validate_input
   ~validate.validate_metric
   ~validate.validate_coarse_to_fine
   ~validate.validate_convert_tensor_dict

Display
-------

The following functions can be used to visualize images, videos, and model representations.

.. autosummary::
   :signatures: none

   ~display.imshow
   ~display.animshow
   ~display.pyrshow
   ~display.plot_representation

The following functions are used internally in the above display functions. They may be helpful,

.. autosummary::
   :signatures: none

   ~display.clean_up_axes
   ~display.clean_stem_plot
   ~display.rescale_ylim
   ~display.update_plot
   ~display.update_stem

Optimization
------------

.. autosummary::
   :signatures: none

   ~optim.set_seed

.. rubric:: Loss functions
   :heading-level: 3

The following functions can be used as the ``loss_function`` argument for :class:`~plenoptic.synthesize.metamer.Metamer` and :class:`~plenoptic.synthesize.metamer.MetamerCTF`.

.. autosummary::
   :signatures: none

   ~optim.mse
   ~optim.l2_norm
   ~optim.relative_sse

.. rubric:: Loss function factories
   :heading-level: 3

The following functions return a function that can be used as a ``loss_function``, as above.

.. autosummary::
   :signatures: none

   ~optim.portilla_simoncelli_loss_factory
   ~optim.groupwise_relative_l2_norm_factory

.. rubric:: Penalty functions
   :heading-level: 3

The following functions operate as penalties.

.. autosummary::
   :signatures: none

   ~optim.penalize_range

.. rubric:: Convergence checkers
   :heading-level: 3

The following functions are used internally to check whether optimization has converged.

.. autosummary::
   :signatures: none

   ~convergence.loss_convergence
   ~convergence.pixel_change_convergence
   ~convergence.coarse_to_fine_enough

Computations
------------

The following functions perform computations or transformations on image-like tensors and may be useful in building new models.

.. autosummary::
   :signatures: none

   ~conv.correlate_downsample
   ~conv.blur_downsample
   ~conv.upsample_convolve
   ~conv.upsample_blur
   ~conv.same_padding
   ~signal.shrink
   ~signal.expand
   ~signal.autocorrelation
   ~signal.rescale
   ~signal.add_noise
   ~signal.center_crop
   ~signal.modulate_phase
   ~stats.variance
   ~stats.skew
   ~stats.kurtosis


Debugging
---------

Functions to help understand when things have gone wrong.

.. autosummary::
   :signatures: none

   ~io.examine_saved_synthesis

External
--------

Functions for interacting with other code and packages.

.. autosummary::
   :signatures: none

   ~external.plot_MAD_results
