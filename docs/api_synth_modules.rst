.. we use this to generate autosummary files for the modules here, but we don't
   link to these directly. we instead link to specific functions from api.rst.
   this allows us to have a table with entries for each of the functions, but
   only have a file per module

.. currentmodule:: plenoptic.synthesize
.. autosummary::
   :toctree: generated
   :template: synthesis_module.rst.jinja

   metamer

.. autosummary::
   :toctree: generated
   :template: synthesis_module.rst.jinja

   metamer
   eigendistortion
   mad_competition
