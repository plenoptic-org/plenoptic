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
   :signatures: none
   :template: synthesis_object.rst.jinja

   Metamer
   MetamerCTF
   Eigendistortion
   MADCompetition

.. rubric:: Synthesis helper functions
   :heading-level: 3

These helper functions all are intended to help visualize the status and outputs of the synthesis objects above.

.. rubric:: Metamer
   :heading-level: 4

These functions work with both :class:`~plenoptic.synthesize.metamer.Metamer` and :class:`~plenoptic.synthesize.metamer.MetamerCTF` objects.

.. currentmodule:: plenoptic.synthesize.metamer
.. autosummary::
   :signatures: long

   plot_loss
   display_metamer
   plot_pixel_values
   plot_representation_error
   plot_synthesis_status
   animate

.. rubric:: MAD Competition
   :heading-level: 4
.. currentmodule:: plenoptic.synthesize.mad_competition
.. autosummary::
   :signatures: long

   display_mad_image
   display_mad_image_all
   plot_loss
   plot_loss_all
   plot_pixel_values
   plot_synthesis_status
   animate

.. rubric:: Eigendistortion
   :heading-level: 4
.. currentmodule:: plenoptic.synthesize.eigendistortion
.. autosummary::
   :signatures: long

   display_eigendistortion
   display_eigendistortion_all
