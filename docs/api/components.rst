.. _components-api:

Model and metric components
---------------------------

These classes and functions may be helpful for constructing your own models or metrics. As is, they are not compatible with any of the synthesis methods.

.. rubric:: Metric components
   :heading-level: 3

These functions are used by the :ref:`metrics-api` above. While they are not compatible with any of our synthesis methods, they may be useful to better understand the behavior of their respective metrics.

.. currentmodule:: plenoptic.metric
.. autosummary::
   :signatures: none
   :toctree: generated

   ~ssim_map
   ~normalized_laplacian_pyramid

.. rubric:: Image pyramids
   :heading-level: 3

Image pyramids decompose images into bands corresponding to different spatial frequencies and orientations. As we often think of neurons in the early visual system as similarly only responding to a range of spatial frequencies and/or orientations, these pyramids can be used to construct a model of those early areas.

See :external+pyrtools:std:doc:`index` for more information, including links to resources for learning more.

.. currentmodule:: plenoptic.model_components
.. autosummary::
   :signatures: none
   :toctree: generated
   :template: torch_module.rst.jinja

   ~LaplacianPyramid
   ~SteerablePyramidFreq

.. rubric:: Filter construction functions
   :heading-level: 3

These convenience functions make it easier to construct some commonly-used filters.

.. autosummary::
   :signatures: none
   :toctree: generated

   ~circular_gaussian2d
   ~gaussian1d

.. rubric:: Coordinate transformations
   :heading-level: 3

This related set of functions convert between the rectangular and polar representations of a signal (or computing the norm and direction, which are the analogues for real-valued signals).

.. autosummary::
   :signatures: none
   :toctree: generated

   ~rectangular_to_polar
   ~polar_to_rectangular
   ~local_gain_control
   ~local_gain_release
   ~rectangular_to_polar_dict
   ~polar_to_rectangular_dict
   ~local_gain_control_dict
   ~local_gain_release_dict

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

.. autosummary::
   :signatures: none
   :toctree: generated

   ~correlate_downsample
   ~blur_downsample
   ~upsample_convolve
   ~upsample_blur
   ~same_padding
   ~shrink
   ~expand

.. rubric:: Image modification
   :heading-level: 3

The following functions return a modified version of their input.

.. autosummary::
   :signatures: none
   :toctree: generated

   ~rescale
   ~add_noise
   ~center_crop
   ~modulate_phase

.. rubric:: Image statistics
   :heading-level: 3

The following functions compute summary statistics of their inputs.

.. autosummary::
   :signatures: none
   :toctree: generated

   ~autocorrelation
   ~variance
   ~skew
   ~kurtosis
