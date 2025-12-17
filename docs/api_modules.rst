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

   simulate.models.portilla_simoncelli
   simulate.frontend
   simulate.naive
   metric.naive
   metric.perceptual_distance
   metric.model_metric
