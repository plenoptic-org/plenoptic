.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10151130.svg
            :target: https://zenodo.org/doi/10.5281/zenodo.10151130

.. _citation:

Citation Guide
**************

If you use ``plenoptic`` in a published academic article or presentation, please
cite both the code by the DOI as well [VSS2023]_. You can use the following:

- Code: |zenodo|
- Paper:

  .. code-block:: bibtex

     @article{duong2023plenoptic,
       title={Plenoptic: A platform for synthesizing model-optimized visual stimuli},
       author={Duong, Lyndon and Bonnen, Kathryn and Broderick, William and Fiquet, Pierre-{\'E}tienne and Parthasarathy, Nikhil and Yerxa, Thomas and Zhao, Xinyuan and Simoncelli, Eero},
       journal={Journal of Vision},
       volume={23},
       number={9},
       pages={5822--5822},
       year={2023},
       publisher={The Association for Research in Vision and Ophthalmology}
     }

Additionally, please cite the following paper(s) depending on which
component you use:

-  :class:`plenoptic.synthesize.metamer.Metamer`: or :class:`plenoptic.synthesize.metamer.MetamerCTF`: [Portilla2000]_.
- :class:`plenoptic.synthesize.mad_competition.MADCompetition`: [Wang2008]_.
- :class:`plenoptic.synthesize.eigendistortion.Eigendistortion`: [Berardino2017]_.
- :class:`plenoptic.synthesize.geodesic.Geodesic`: [Henaff2016]_.
- :class:`plenoptic.simulate.canonical_computations.steerable_pyramid_freq.SteerablePyramidFreq`: [Simoncelli1995]_ ([Simoncelli1992]_ contains a longer discussion about the motivation and the logic, while [Simoncelli1995]_ describes the implementation that is used here).
- :class:`plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli`: [Portilla2000]_.
- :class:`plenoptic.simulate.models.frontend` (any model): [Berardino2017]_.
- :class:`plenoptic.metric.perceptual_distance.ssim` or :class:`plenoptic.metric.perceptual_distance.ssim_map`: [Wang2004]_ if ``weighted=False``, [Wang2008]_ if ``weighted=True``.
- :class:`plenoptic.metric.perceptual_distance.ms_ssim`: [Wang2003]_.
- :class:`plenoptic.metric.perceptual_distance.nlpd`: [Laparra2017]_.

Note that, the citations given above define the application of the relevant idea
("metamers") to computational models of the visual system that are instantiated
in the algorithms found in ``plenoptic``, but that, for the most part, these
general concepts were not developed by the developers of ``plenoptic`` or the
Simoncelli lab and are, in general, much older -- the idea of metamers goes all
the way back to [Helmholtz1852]_! The papers above generally provide some
discussion of this history and can point you to further reading, if you are
interested.

.. [VSS2023] Lyndon Duong, Kathryn Bonnen, William Broderick, Pierre-Étienne
             Fiquet, Nikhil Parthasarathy, Thomas Yerxa, Xinyuan Zhao, Eero
             Simoncelli; Plenoptic: A platform for synthesizing model-optimized
             visual stimuli. Journal of Vision 2023;23(9):5822.
             https://doi.org/10.1167/jov.23.9.5822.
