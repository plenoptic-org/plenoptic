.. _images-api:

Loading, creating, and accessing images
---------------------------------------

.. currentmodule:: plenoptic
.. autosummary::
   :signatures: none
   :toctree: generated

   ~load_images
   ~to_numpy
   ~convert_float_to_int

.. rubric:: Example images
   :heading-level: 3

Plenoptic includes a small number of example images, which are used throughout the documentation.

.. currentmodule:: plenoptic.data
.. autosummary::
   :signatures: none
   :toctree: generated

   ~einstein
   ~curie
   ~parrot
   ~reptile_skin
   ~color_wheel

Plenoptic includes a helper function for downloading additional data, which includes example synthesis objects and additional images.

.. autosummary::
   :signatures: none
   :toctree: generated

   ~fetch_data
   ~DOWNLOADABLE_FILES

.. rubric:: Synthetic images
   :heading-level: 3

The following functions create synthetic image tensors. See :external+pyrtools:mod:`pyrtools.tools.synthetic_images` for more synthetic images.

.. autosummary::
   :signatures: none
   :toctree: generated

   ~disk
   ~polar_radius
   ~polar_angle
