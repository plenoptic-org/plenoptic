---
title: 'Plenoptic.py: Synthesizing model-optimized visual stimuli'
tags:
  - Python
  - PyTorch
  - neural networks
  - computational neuroscience
  - image synthesis
authors:
  - name: William F. Broderick
    orcid: 0000-0002-8999-9003
    affiliation: 1
    corresponding: true
  - family-names: Edoardo Balzani
    orcid: 0000-0002-3702-5856
    affiliation: 1
  - name: Kathryn Bonnen
    equal-contrib: true
    orcid: 0000-0002-9210-8275
    affiliation: 2
  - name: Hanna Dettki
    orcid: 0009-0006-7237-6121
  - given-names: Wolf
    surname: De Wulf
    orcid: 0000-0002-0219-3120
    affiliation: 3
  - name: Lyndon R. Duong
    equal-contrib: true
    orcid: 0000-0003-0575-1033
    affiliation: 4
  - name: Pierre-Etienne Fiquet
    equal-contrib: true
    orcid: 0000-0002-8301-2220
    affiliation: 1
  - name: Daniel Herrera-Esposito
    orcid: 0000-0001-8181-9787
    affiliation: 5
  - name: Emily E Meyer
    orcid: 0009-0004-1302-1510
    affiliation: 6
  - name: Nikhil Parthasarathy
    equal-contrib: true
    orcid: 0000-0003-2572-6492
    affiliation: 7
  - name: Erica Shook
    orcid: 0009-0004-3657-7562
    affiliation: 8
  - given-names: Sarah Jo
    surname: Venditto
    orcid: 0000-0003-4681-0538
    affiliation: 1
  - name: Guillaume Viejo
    orcid: 0000-0002-2450-7397
    affiliation: 1
  - name: Thomas E. Yerxa
    equal-contrib: true
    orcid: 0000-0003-2687-0816
    affiliation: 9
  - name: Xinyuan Zhao
    equal-contrib: true
    affiliation: 10
  - given-names: Raffles Xingqi
    surname: Zhu
    orcid: 0009-0006-1930-8912
  - name: Eero P. Simoncelli
    orcid: 000-0002-1206-527X
    affiliation: "1, 10"
affiliations:
 - name: Center for Computational Neuroscience, Flatiron Institute, New York, NY, USA
   index: 1
 - name:  School of Optometry, Indiana University, Bloomington, IN, USA
   index: 2
 - name: School of Informatics, University of Edinburgh, Edinburgh, UK
   index: 3
 - name: Apple Inc., Cupertino, CA, USA
   index: 4
 - name: School of Public and International Affairs, Princeton University, Princeton, NJ, USA
   index: 5
 - name: Department of Psychology, University of Pennsylvania, Philadelphia, PA, USA
   index: 6
 - name: Google DeepMind, London, UK
   index: 7
 - name: Zuckerman Institute, Columbia University, New York, NY, USA
   index: 8
 - name: Meta Inc., Menlo Park, CA, USA
   index: 9
 - name: Center for Neural Science, New York University, New York, NY, USA
   index: 10
date: October 2026
bibliography: references.bib
---

# Summary

In sensory perception and neuroscience, new computational models are most often tested and compared in terms of their ability to fit existing data sets.
However, experimental data are inherently limited in size, quality, and type, and complex models often saturate their explainable variance.
Moreover, it is often difficult to use models to guide the development of future experiments.
Here, building on ideas for optimal experimental stimulus selection  (e.g., QUEST, Watson and Pelli, 1983), we present "Plenoptic", a python software library for generating visual stimuli optimized for testing or comparing models.
Plenoptic provides a unified framework containing four previously-published synthesis methods -- model metamers (Freeman and Simoncelli, 2011), Maximum Differentiation (MAD) competition (Wang and Simoncelli, 2008), eigen-distortions (Berardino et al. 2017), and representational geodesics (Hénaff and Simoncelli, 2015) -- each of which offers visualization of model representations, and generation of images that can be used to experimentally test alignment with the human visual system.
Plenoptic leverages modern machine-learning methods to enable application of these synthesis methods to any computational model that satisfies a small set of common requirements.
The most important of these is that the model must be image-computable, implemented in PyTorch (Paszke et al. 2019), and end-to-end differentiable.
The package includes examples of several low- and mid-level visual models, as well as a set of perceptual quality metrics.
Plenoptic is open source, tested, documented, and extensible, allowing the broader research community to contribute new examples and methods.
In summary, Plenoptic leverages machine learning tools to tighten the scientific hypothesis-testing loop, facilitating investigation of human visual representations.

# Statement of need
<!-- A section that clearly illustrates the research purpose of the software and places it in the context of related work. This should clearly state what problems the software is designed to solve, who the target audience is, and its relation to other work. -->

# State of the field
<!-- A description of how this software compares to other commonly-used packages in the research area. If related tools exist, provide a clear “build vs. contribute” justification explaining your unique scholarly contribution and why existing alternatives are insufficient. -->

# Software design
<!-- An explanation of the trade-offs you weighed, the design/architecture you chose, and why it matters for your research application. This should demonstrate meaningful design thinking beyond a superficial code structure description. -->

# Research impact statement
<!-- Evidence of realized impact (publications, external use, integrations) or credible near-term significance (benchmarks, reproducible materials, community-readiness signals). The evidence should be compelling and specific, not aspirational. -->
<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->

# AI usage disclosure
<!-- Transparent disclosure of any use of generative AI in the software creation, documentation, or paper authoring. If no AI tools were used, state this explicitly. If AI tools were used, describe how they were used and how the quality and correctness of AI-generated content was verified. -->

# Acknowledgements

EPS and KB were funded by Simons Institute.

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
