.. _optimization-api:

Optimization
------------

.. currentmodule:: plenoptic.tools.optim
.. autosummary::
   :signatures: none
   :toctree: generated

   ~set_seed

.. rubric:: Loss functions
   :heading-level: 3

The following functions can be used as the ``loss_function`` argument for :class:`~plenoptic.synthesize.metamer.Metamer` and :class:`~plenoptic.synthesize.metamer.MetamerCTF`.

.. autosummary::
   :signatures: none
   :toctree: generated

   ~mse
   ~l2_norm
   ~relative_sse

.. rubric:: Loss function factories
   :heading-level: 3

The following functions return a function that can be used as a ``loss_function``, as above.

.. autosummary::
   :signatures: none
   :toctree: generated

   ~portilla_simoncelli_loss_factory
   ~groupwise_relative_l2_norm_factory

.. rubric:: Penalty functions
   :heading-level: 3

The following functions operate as penalties.

.. autosummary::
   :signatures: none
   :toctree: generated

   ~penalize_range
