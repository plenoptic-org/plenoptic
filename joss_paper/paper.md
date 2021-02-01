---
title: 'Plenoptic: synthesis methods for analyzing model representations'
tags:
  - Python
  - PyTorch
  - neural networks
  - computational neuroscience
  - image synthesis
authors:
  - name: Kathryn Bonnen
    orcid: 0000-0002-9210-8275
    affiliation: 1, 2
  - name: William Broderick
    orcid: 0000-0002-8999-9003
    affiliation: 1
  - name: Lyndon R. Duong
    orcid: 0000-0003-0575-1033
    affiliation: 1
  - name: Pierre-Etienne Fiquet
    orcid: 0000-0002-8301-2220
    affiliation: 1
  - name: Nikhil Parthasarathy
    orcid: 0000-0003-2572-6492
    affiliation: 1
  - name: Eero P. Simoncelli
    orcid: 000-0002-1206-527X
    affiliation: 1, 2
affiliations:
 - name: Center for Neural Science, New York University, New York, NY, USA
   index: 1
 - name: Center for Computational Neuroscience, Flatiron Institute, New York, NY, USA
   index: 2
date: April 2021
bibliography: paper.bib
---

# Summary


``Plenoptic`` builds primarily off of ``PyTorch`` [@paszke_pytorch_2019], a Python machine learning library popular in the research community due to its rapid prototyping capability. With ``Plenoptic``, users can build and train models in ``PyTorch``, then use ``Plenoptic`` synthesis methods to assess their internal representations.
Our library is easily extensible, and allows for great flexibility to those who wish to develop or test their own synthesis methods.
Within the library, we also provide an extensive suite of ``PyTorch``-implemented models and activation functions canonical to computational neuroscience.

Many of the methods in ``Plenoptic`` have been developed and used across several studies; however, analyses in these studies used disparate languages and frameworks, and some have yet to be made publicly available.
Here, we have reimplemented the methods central to each of these studies, and unified them under a single, fully-documented API.
Our library includes several Jupyter notebook tutorials designed to be accessible to researchers in the fields of machine learning, and computational neuroscience, and perceptual science.
``Plenoptic`` provides an exciting avenue for researchers to probe their models to gain a deeper understanding of their internal representations.

# Acknowledgements

KB, WB, LRD, PEF, and NP each contributed equally to this work.
EPS was funded by the Howard Hughes Medical Institute. EPS and KB were funded by Simons Institute.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# References

@berardino_eigen-distortions_2017
@henaff_geodesics_2015
@simoncelli_steerable_1995
@freeman_metamers_2011
@wang_maximum_2008
@paszke_pytorch_2019
@portilla_parametric_2000

