(citation-doc)=

# Citation Guide and Bibliography

If you use `plenoptic` in a published academic article or presentation, please cite both the code by the DOI (for the specific version you used) as well as the current publication, {cite:alp}`Duong2023-plenop`. You can use the following:

- Most recent version of code: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10151130.svg)](https://doi.org/10.5281/zenodo.10151130)

  - See [zenodo](https://zenodo.org/search?q=parent.id%3A10151130&f=allversions%3Atrue&l=list&p=1&s=10&sort=version) for DOIs for older versions of the code.

- Paper:

  ```bibtex
  @article{duong2023plenoptic,

  title={Plenoptic: A platform for synthesizing model-optimized visual stimuli},
  author={Duong, Lyndon and Bonnen, Kathryn and Broderick, William and Fiquet, Pierre-{'E}tienne and Parthasarathy, Nikhil and Yerxa, Thomas and Zhao, Xinyuan and Simoncelli, Eero},
  journal={Journal of Vision},
  volume={23},
  number={9},
  pages={5822--5822},
  year={2023},
  publisher={The Association for Research in Vision and Ophthalmology}
  }
  ```

Additionally, please cite the following paper(s) depending on which component you use:

- {class}`~plenoptic.synthesize.metamer.Metamer` or {class}`~plenoptic.synthesize.metamer.MetamerCTF`: {cite:alp}`Portilla2000-param-textur`.
- {class}`~plenoptic.synthesize.mad_competition.MADCompetition`: {cite:alp}`Wang2008-maxim-differ`.
- {class}`~plenoptic.synthesize.eigendistortion.Eigendistortion`: {cite:alp}`Berardino2017-eigen`.
- {class}`~plenoptic.simulate.canonical_computations.steerable_pyramid_freq.SteerablePyramidFreq`: {cite:alp}`Simoncelli1995-steer-pyram` ({cite:alp}`Simoncelli1992-shift-multi` contains a longer discussion about the motivation and the logic, while {cite:alp}`Simoncelli1995-steer-pyram` describes the implementation that is used here).
- {class}`~plenoptic.simulate.models.portilla_simoncelli.PortillaSimoncelli`: {cite:alp}`Portilla2000-param-textur`.
- {mod}`~plenoptic.simulate.frontend` (any model): {cite:alp}`Berardino2017-eigen`.
- {func}`~plenoptic.metric.perceptual_distance.ssim` or {func}`~plenoptic.metric.perceptual_distance.ssim_map`: {cite:alp}`Wang2004-image-qualit-asses` if `weighted=False`, {cite:alp}`Wang2008-maxim-differ` if `weighted=True`.
- {func}`~plenoptic.metric.perceptual_distance.ms_ssim`: {cite:alp}`Wang2003-multis`.
- {func}`~plenoptic.metric.perceptual_distance.nlpd`: {cite:alp}`Laparra2017-percep-optim`.

Note that, the citations given above define the application of the relevant idea ("metamers") to computational models of the visual system that are instantiated in the algorithms found in `plenoptic`, but that, for the most part, these general concepts were not developed by the developers of `plenoptic` or the Simoncelli lab and are, in general, much older -- the idea of metamers goes all the way back to {cite:alp}`Helmholtz1852-lxxxi`! The papers above generally provide some discussion of this history and can point you to further reading, if you are interested.

## Bibliography

```{bibliography} ../references.bib
:style: plain
```
