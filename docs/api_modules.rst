:orphan:

.. we use this to generate autosummary files for the modules here, but we don't
   link to these directly. we instead link to specific functions from api.rst.
   this allows us to have a table with entries for each of the functions, but
   only have a file per module

.. currentmodule:: plenoptic
.. autosummary::
   :toctree: generated
   :template: synthesis_module.rst.jinja

   synthesize.metamer
   synthesize.eigendistortion
   synthesize.mad_competition

.. autosummary::
   :toctree: generated
   :template: module_source_order.rst.jinja

   simulate.frontend
   simulate.naive
   metric.perceptual_distance

.. autosummary::
   :toctree: generated

   simulate.models.portilla_simoncelli
   metric.naive
   metric.model_metric
   simulate.canonical_computations.laplacian_pyramid
   simulate.canonical_computations.steerable_pyramid_freq
   simulate.canonical_computations.filters
   simulate.canonical_computations.non_linearities
   data
   data.fetch
   tools.signal
   tools.conv
   tools.data
   tools.display
   tools.external
   tools.io
   tools.optim
   tools.stats
   tools.validate
