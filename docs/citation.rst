.. _citation:

Citation Guide
**************

If you use ``plenoptic`` in a published piece of academic work, please cite
[VSS2023]_ (and check this guide again before publishing, as we are planning on
having a paper to cite instead of the current presentation).

Additionally, please additionally cite the following paper(s) depending on which
component you use:

-  :class:`plenoptic.synthesize.metamer.Metamer`: or :class:`plenoptic.synthesize.metamer.MetamerCTF`: [Portilla2000]_.
- :class:`plenoptic.synthesize.mad_competition.MADCompetition`: [Wang2008]_.
- :class:`plenoptic.synthesize.eigendistortion.Eigendistortion`: [Berardino2017]_.
- :class:`plenoptic.synthesize.geodesic.Geodesic`: [Henaff2016]_.
- :class:`plenoptic.simulate.canonical_computations.steerable_pyramid_freq.SteerablePyramidFreq`: [Simoncelli1992]_.
- :class:`plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli`: [Portilla2000]_.
- :class:`plenoptic.simulate.models.frontend` (any model): [Berardino2017]_.
- :class:`plenoptic.metric.perceptual_distance.ssim` or :class:`plenoptic.metric.perceptual_distance.ssim_map`: [Wang2004]_ if ``weighted=False``, [Wang2008]_ if ``weighted=True``.
- :class:`plenoptic.metric.perceptual_distance.ms_ssim`: [Wang2003]_.
- :class:`plenoptic.metric.perceptual_distance.nlpd`: [Laparra2017]_.

.. [VSS2023] Lyndon Duong, Kathryn Bonnen, William Broderick, Pierre-Étienne
             Fiquet, Nikhil Parthasarathy, Thomas Yerxa, Xinyuan Zhao, Eero
             Simoncelli; Plenoptic: A platform for synthesizing model-optimized
             visual stimuli. Journal of Vision 2023;23(9):5822.
             https://doi.org/10.1167/jov.23.9.5822.
