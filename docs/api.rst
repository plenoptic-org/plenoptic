.. _api:

API
===

.. _synthesis-api:

Synthesis objects
-----------------

Synthesis objects generate novel stimuli which allow researchers to better understand how their computational models make sense of their inputs.

.. currentmodule:: plenoptic.synthesize
.. autosummary::
   :toctree: generated
   :signatures: none
   :template: synthesis_object.rst.jinja

   ~metamer.Metamer
   ~metamer.MetamerCTF
   ~eigendistortion.Eigendistortion
   ~mad_competition.MADCompetition

.. toctree::
   :hidden:

   generated/plenoptic.simulate.models.portilla_simoncelli
   generated/plenoptic.simulate.frontend
   generated/plenoptic.simulate.naive
   generated/plenoptic.metric.naive
   generated/plenoptic.metric.model_metric
   generated/plenoptic.metric.perceptual_distance
   generated/plenoptic.synthesize.metamer
   generated/plenoptic.synthesize.eigendistortion
   generated/plenoptic.synthesize.mad_competition
   generated/plenoptic.simulate.canonical_computations.laplacian_pyramid
   generated/plenoptic.simulate.canonical_computations.steerable_pyramid_freq
   generated/plenoptic.simulate.canonical_computations.filters
   generated/plenoptic.simulate.canonical_computations.non_linearities
   generated/plenoptic.tools.conv
   generated/plenoptic.tools.signal
   generated/plenoptic.tools.stats
   generated/plenoptic.data
   generated/plenoptic.data.fetch
   generated/plenoptic.tools.data
   generated/plenoptic.tools.validate
   generated/plenoptic.tools.display
   generated/plenoptic.tools.optim
   generated/plenoptic.tools.io
   generated/plenoptic.tools.external

.. _models-api:

Models
------

Models give a response to a single stimulus and are compatible with :class:`~plenoptic.synthesize.metamer.Metamer` and :class:`~plenoptic.synthesize.eigendistortion.Eigendistortion`, and can be turned into :ref:`metrics-api` by using the :func:`~plenoptic.metric.model_metric.model_metric_factory` function.

See :ref:`models-doc` for more details.

.. currentmodule:: plenoptic.simulate
.. autosummary::
   :signatures: none

   ~models.portilla_simoncelli.PortillaSimoncelli

.. rubric:: LGN-inspired Models
   :heading-level: 3

These "front end" models are inspired by the lateral geniculate nucleus (LGN; the first non-retinal stage of the primate visual system), come from :cite:alp:`Berardino2017-eigen`, and are nested, increasing in complexity as you move down the list.

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
   :signatures: none

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
   :signatures: none

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
   :signatures: none

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

.. rubric:: Coordinate transformations
   :heading-level: 3

This related set of functions convert between the rectangular and polar representations of a signal (or computing the norm and direction, which are the analogues for real-valued signals).

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

The following table summarizes the relationship among these functions (where ➡ denotes from rectangular to polar and ⬅ is the inverse):

.. csv-table::
   :header: "function", "dtype", "input object", "direction"
   :widths: auto

   rectangular_to_polar, ``complex*``, ``tensor``, ➡
   polar_to_rectangular, ``complex*``, ``tensor``, ⬅
   local_gain_control, ``float*``, ``tensor``, ➡
   local_gain_release, ``float*``, ``tensor``, ⬅
   rectangular_to_polar_dict, ``complex*``, ``dict``, ➡
   polar_to_rectangular_dict, ``complex*``, ``dict``, ⬅
   local_gain_control_dict, ``float*``, ``dict``, ➡
   local_gain_release_dict, ``float*``, ``dict``, ⬅

.. rubric:: Image resizing
   :heading-level: 3

The following functions return a version of their input whose size has increased or decreased without further modifying its contents.

.. currentmodule:: plenoptic.tools
.. autosummary::
   :signatures: none

   ~conv.correlate_downsample
   ~conv.blur_downsample
   ~conv.upsample_convolve
   ~conv.upsample_blur
   ~conv.same_padding
   ~signal.shrink
   ~signal.expand

.. rubric:: Image modification
   :heading-level: 3

The following functions return a modified version of their input.

.. autosummary::
   :signatures: none

   ~signal.rescale
   ~signal.add_noise
   ~signal.center_crop
   ~signal.modulate_phase

.. rubric:: Image statistics
   :heading-level: 3

The following functions compute summary statistics of their inputs.

.. autosummary::
   :signatures: none

   ~signal.autocorrelation
   ~stats.variance
   ~stats.skew
   ~stats.kurtosis

Loading, creating, and accessing images
---------------------------------------

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

Plenoptic includes a helper function for downloading additional data, which includes example synthesis objects and additional images.

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
