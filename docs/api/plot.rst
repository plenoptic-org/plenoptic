.. _plot-api:

Plotting functions
------------------

.. rubric:: Synthesis objects
   :heading-level: 3

These functions all are intended to help visualize the status and outputs of `synthesis objects <synthesis-api>`_. The following accept all synthesis classes:

.. currentmodule:: plenoptic.plot
.. autosummary::
   :signatures: none
   :toctree: generated

   ~synthesis_imshow
   ~synthesis_histogram
   ~synthesis_status

The following accept :class:`~plenoptic.Metamer` / :class:`~plenoptic.MetamerCTF` and :class:`~plenoptic.MADCompetition` objects:

.. autosummary::
   :signatures: none
   :toctree: generated

   ~synthesis_loss
   ~synthesis_animshow

The following accept only objects of a single class.

.. rubric:: :class:`~plenoptic.Metamer` / :class:`~plenoptic.MetamerCTF`
   :heading-level: 4

.. autosummary::
   :signatures: none
   :toctree: generated

   ~metamer_representation_error

.. rubric:: :class:`~plenoptic.MADCompetition`
   :heading-level: 4
.. autosummary::
   :signatures: none
   :toctree: generated

   ~mad_imshow_all
   ~mad_loss_all

.. rubric:: :class:`~plenoptic.Eigendistortion`
   :heading-level: 4
.. autosummary::
   :signatures: none
   :toctree: generated

   ~eigendistortion_imshow_all

.. rubric:: Tensors
   :heading-level: 3

The following functions can be used to visualize :class:`torch.Tensor` objects directly: images, videos, and model representations.

.. currentmodule:: plenoptic.plot
.. autosummary::
   :signatures: none
   :toctree: generated

   ~imshow
   ~animshow
   ~pyrshow
   ~plot_representation
   ~stem_plot
   ~histogram
   ~update_plot
