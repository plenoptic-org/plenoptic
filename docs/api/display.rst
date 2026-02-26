.. _display-api:

Display
-------

The following functions can be used to visualize images, videos, and model representations.

.. currentmodule:: plenoptic.tools
.. autosummary::
   :signatures: none
   :toctree: generated

   ~display.imshow
   ~display.animshow
   ~display.pyrshow
   ~display.plot_representation

The following functions are used internally in the above display functions. They may be helpful,

.. autosummary::
   :signatures: none
   :toctree: generated

   ~display.clean_up_axes
   ~display.clean_stem_plot
   ~display.rescale_ylim
   ~display.update_plot
   ~display.update_stem
