.. _display-api:

Display
-------

.. rubric:: Synthesis objects
   :heading-level: 3

These functions all are intended to help visualize the status and outputs of `synthesis objects <synthesis-api>`_.

.. currentmodule:: plenoptic.plot
.. autosummary::
   :signatures: none
   :toctree: generated

   ~synthesis_loss

.. rubric:: :class:`~plenoptic.Metamer` / :class:`~plenoptic.MetamerCTF`
   :heading-level: 4

.. currentmodule:: plenoptic.plot
.. autosummary::
   :signatures: none
   :toctree: generated

   ~metamer_loss
   ~metamer_imshow
   ~metamer_pixel_values
   ~metamer_representation_error
   ~metamer_synthesis_status
   ~metamer_animshow

.. rubric:: :class:`~plenoptic.MADCompetition`
   :heading-level: 4
.. autosummary::
   :signatures: none
   :toctree: generated

   ~mad_imshow
   ~mad_imshow_all
   ~mad_loss
   ~mad_loss_all
   ~mad_pixel_values
   ~mad_synthesis_status
   ~mad_animshow

.. rubric:: :class:`~plenoptic.Eigendistortion`
   :heading-level: 4
.. autosummary::
   :signatures: none
   :toctree: generated

   ~eigendistortion_imshow
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
   ~update_plot
   ~histogram
