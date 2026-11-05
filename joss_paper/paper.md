---
title: 'Plenoptic: a python library for model-based synthesis of perceptual stimuli'
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
  - name: Edoardo Balzani
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
    orcid: 0000-0003-1770-631X
  - given-names: Raffles Xingqi
    surname: Zhu
    orcid: 0009-0006-1930-8912
  - name: Eero P. Simoncelli
    orcid: 0000-0002-1206-527X
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
bibliography: paper.bib
---

# Summary

<!-- Summary: A description of the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->

In sensory perception and neuroscience, new computational models are most often tested and compared in terms of their ability to fit existing data sets. However, experimental data are inherently limited in size, quality, and type, and complex models often saturate their explainable variance. Moreover, it is often difficult to use models to guide the development of future experiments. Here, building on ideas for optimal experimental stimulus selection (e.g., QUEST, @Watson1983-quest), we present "Plenoptic", a python software library for model-based synthesis of perceptual stimuli. Plenoptic provides a unified framework containing three previously-published synthesis methods -- model metamers [@freeman_metamers_2011], Maximum Differentiation (MAD) competition [@wang_maximum_2008], eigen-distortions [@berardino_eigen-distortions_2017] -- each of which synthesizes novel stimuli to visualize different aspects of model representations, which can also be used as experimental stimuli to test alignment with human or other biological perceptual systems. Plenoptic leverages modern machine-learning methods to enable application of these synthesis methods to any computational model that satisfies a small set of common requirements: the model must be stimulus-computable, implemented in PyTorch [@paszke_pytorch_2019], and end-to-end differentiable. The package additionally includes PyTorch implementations of several previously-published low- and mid-level visual models, as well as a set of perceptual quality metrics.

# Statement of need
<!-- A section that clearly illustrates the research purpose of the software and places it in the context of related work. This should clearly state what problems the software is designed to solve, who the target audience is, and its relation to other work. -->

All synthesis methods in plenoptic have been published and described elsewhere, as cited in the Summary section. The goal of plenoptic is to make these methods more accessible by providing model-agnostic PyTorch (and thus GPU-accelerated) implementations of them so that researchers can use them with their own PyTorch models. While several papers shared code that contained an implementation of one of these methods, the code was focused on reproducing the specific results presented in said paper, and were generally not installable, documented, tested, or maintained.

The methods in plenoptic allow users to generate stimuli which enable interpretation of model properties through examination of features that are enhanced, suppressed, or discarded. The intended users are researchers in neuroscience, psychology, and machine learning, who are building and applying models that process perceptual stimuli. Plenoptic's synthesis methods enable users to perform "informal psychophysics", using their own visual system to examine the synthesized images, and more importantly, these methods are intended to tighten the model-experiment loop: researchers should develop a set of stimuli that they can use in further perceptual or neural experiments to validate or falsify model predictions.

# State of the field
<!-- A description of how this software compares to other commonly-used packages in the research area. If related tools exist, provide a clear “build vs. contribute” justification explaining your unique scholarly contribution and why existing alternatives are insufficient. -->

Research code has been shared with implementations of the model metamer synthesis method for specific research problems: e.g., [from](https://github.com/LabForComputationalVision/textureSynth) @portilla_parametric_2000, [from](https://github.com/freeman-lab/metamers/) @freeman_metamers_2011, [from](https://github.com/jenellefeather/model_metamers_pytorch) @Feather2023-model-metam. However, the two former implementations are written in MATLAB and only operate on the model described in their respective papers, while the latter focuses on reproducing the results presented in the paper. Additionally, no public implementation exists of the other two synthesis methods (eigendistortions and MAD competition). MATLAB and numpy implementations exist for some of the models included in plenoptic (maintained by some of the same authors), but no PyTorch implementation.

Finally, the [lucent](https://github.com/greentfrapp/lucent) package is a PyTorch implementation of the TensorFlow [lucid](https://github.com/tensorflow/lucid) library, which provides a series of tools for interpretability in deep neural networks. This is a related cluster of ideas and, indeed, several of the visualizations from lucent / lucid (such as adversarial examples) can also be synthesized with plenoptic. However, those libraries are specifically focused on "neural network interpretability", whereas plenoptic is model-agnostic and focuses on methods which investigate the relationship between the representations in artificial and biological perceptual systems. Additionally, those libraries are no longer maintained.

# Software design

<!-- An explanation of the trade-offs you weighed, the design/architecture you chose, and why it matters for your research application. This should demonstrate meaningful design thinking beyond a superficial code structure description. -->

The primary goal for plenoptic is to make the included synthesis methods as broadly applicable as possible. All of these methods require computing the gradient of the model with respect to its input and using this information in some way to update an input. When the first of these methods (model metamers) was developed, that required manually computing the derivatives for each model. However, the development and proliferation of automatic differentiation in software packages such as PyTorch simplifies this process: it allowed us to set up the problem and have the package compute the derivatives as needed. Thus, plenoptic must be built on top of one of these libraries, and we wanted to use one in python, which is the most commonly-used open source language within the neuroscience community. When plenoptic's development started in 2019, the primary python choices were PyTorch or TensorFlow. We chose PyTorch as we found its syntax more "pythonic" and easier to read and write.

In order to make our synthesis objects as broadly applicable as possible, we sought to design the synthesis methods as objects which:

- Are model-agnostic: any model that is callable, accepts and returns a tensor, and can be differentiated by PyTorch, will work with plenoptic's synthesis methods.
- Are flexible: two of the synthesis methods (model metamers and MAD Competition) are iterative optimization procedures. For these, the user is able to specify the loss function, an optional penalty (which is included in the objective function as a way of e.g., encouraging all pixel values to lie between 0 and 1 for images), the optimizer, and the scheduler. Our documentation explains what properties are required for these, and they are compatible with existing PyTorch objects (e.g., any optimizer in `torch.optim`).
- Reuse PyTorch conventions where possible: while the models included in plenoptic inherit PyTorch's `torch.nn.Module` class (commonly used for models and their components), PyTorch does not have any built-in objects that represent optimization problems, and so we have written our own classes. They do match the `torch.nn.Module` conventions where it makes sense: they have `to`, `load`, and `save` methods with similar call signatures.
- Are consistent with each other: the synthesis methods all operate in slightly different ways, but their methods and method signatures are as consistent as possible. For example, both model metamers and MAD Competition are iterative optimization procedures, while eigendistortion is not. Thus, while all three corresponding objects have `synthesize` methods, `Metamer` and `MADCompetition` additionally have `setup` methods, which allow users to set the optimizer, scheduler, and optimization initialization tensor; `Eigendistortion` has no such method.

Plenoptic's primary focus is on the synthesis methods, but we include a variety of models, image quality metrics, and image processing methods. These are included both because they are useful for demonstrating how to use the included synthesis methods, and also because they are useful in and of themselves for vision science research. To that end, these objects and functions are stand-alone (can be used independently of plenoptic's synthesis methods) and are designed to meet the above desiderata as well.

Plenoptic also includes a variety of functions for the creation of `matplotlib` plots, found within the appropriately-named `plot` module. These are intended to make the creation of good-enough diagnostic plots straightforward. There are three classes of plotting functions:

- those that operate on tensors. These functions accept tensors and create simple plots: a histogram of the distribution of values, displaying the image without interpolation, arraying many images into a single figure, or animating a video without interpolation. These functions can operate on single axes objects, allowing them to be combined into more complex plots, or can create multi-axes figures, depending on user input.
- those that operate on synthesis objects. These axis-level functions accept a single synthesis object and visualize the corresponding synthesized image. The user can specify the axis for the plot and the axis is returned by the function, allowing these plots to be combined in larger, more complex plots, both by plenoptic itself (in the functions described in the following point) or by users.
- those that combine synthesis object plots. These figure-level functions (`synthesis_status` and `synthesis_animate`) combine selections of the above plots into a single figure (`synthesis_status`) or animation (`synthesis_animate`) in order to allow users to quickly create diagnostic plots and visualize the synthesis procedure.

The above plotting methods are all intended for (and tested on) images, represented as 4-dimensional tensors. While the synthesis methods work with other stimulus modalities (in particular, audio), we have not yet developed similar diagnostic methods.

# Research impact statement
<!-- Evidence of realized impact (publications, external use, integrations) or credible near-term significance (benchmarks, reproducible materials, community-readiness signals). The evidence should be compelling and specific, not aspirational. -->
<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->

<!-- - papers / posters -->
<!-- - workshops -->
<!-- - bringing in interns to work on package -->
<!-- - reproducible materials: much of documentation shows how to use plenoptic to do something interesting, is reproducible -->

Plenoptic has been used in several published studies, pre-prints, and PhD theses: @Broderick2026-foveat-metam, @olianezhad2025visual, @feather2025discriminating, @parthasarathy2024layerwise, @mahncke2025visibility, @meyer2026transformation, @maruya2025phantom, @Yerxa-phd, @Woo2023-dynam-patter. <!-- codespell:ignore theses -->

The developers of plenoptic take part in a variety of outreach methods. In addition to presenting at relevant conferences [@Duong2023-plenop,@Broderick2025-plenop,@Broderick2026-plenop,@Broderick2026-biasing-optim] the developers reach users by teaching at summer schools ([Cold Spring Harbor Labs, Computational Neuroscience: Vision](https://meetings.cshl.edu/courses.aspx?course=c-visi) course, 2024 and 2026) and running workshops ([at VSS 2025](https://web.archive.org/web/20260317095307/https://www.visionsciences.org/2025-plenoptic-satellite/)). The goal is to present users with an opportunity to interact more directly with the developers and to try out the package directly with live-coding and exercises.

As the plenoptic developers are employees of the Simons Foundation's Flatiron Institute, we take part in the Simons Foundation [summer internship program](https://www.simonsfoundation.org/summer-at-simons/) and have done so for the past three summers. We have had three interns work on plenoptic or related packages for their internship, and one additional intern contribute to the package after their internship concluded. All four former interns (Hanna Dettki, Emily Meyer, Raffles Zhu, and Wolf De Wulf) are included as authors on this paper.

Plenoptic's documentation contains many reproducible examples, as well as reproductions of published results. Our documentation is a sphinx site written using [myst-parser](https://myst-parser.readthedocs.io/en/latest/index.html) and [myst-nb](https://myst-nb.readthedocs.io/en/latest/index.html) and thus the vast majority of the code in the documentation is executed at build time, ensuring reproducibility. The included notebooks also have links to run them on [mybinder](https://mybinder.org/). Several pages in our documentation demonstrate how to reproduce results from the literature, and we are always adding more. Currently we show how to use plenoptic to reproduce some of the results from  @portilla_parametric_2000, @Szegedy2013, @Matejka2017-same-stats (the ["datasaurus dozen"](https://en.wikipedia.org/wiki/Datasaurus_dozen)), @berardino_eigen-distortions_2017, and @Feather2023-model-metam. Note that several of these examples require more time than is desirable during the documentation build, and thus these examples are run as part of the tests instead, checking for exact reproducibility against cached versions of earlier runs, and are only loaded in the documentation.

# AI usage disclosure
<!-- Transparent disclosure of any use of generative AI in the software creation, documentation, or paper authoring. If no AI tools were used, state this explicitly. If AI tools were used, describe how they were used and how the quality and correctness of AI-generated content was verified. -->

The majority of plenoptic has been written without the use of generative AI tools. The exceptions are in the implementation of penalty functions in [PR 411](https://github.com/plenoptic-org/plenoptic/pull/411) and the addition of support for color images to the Portilla-Simoncelli model in (the currently unmerged) [PR 482](https://github.com/plenoptic-org/plenoptic/pull/482), both by Daniel Herrera-Esposito, whose disclosures follow.

For PR 411:

1. Tool verification: GPT-5.2, used for the code.
2. Scope of assistance: Code autocomplete.
3. I affirm reviewed, modified, and validated all AI-generated content while making primary architectural and design decisions.

For PR 482:

1. Tool verification: Codex 0.156.0 with GPT-5.6-Sol, used for the code.
2. Scope of assistance: Code generation
3. I affirm reviewed, modified, and validated all AI-generated content while making primary architectural and design decisions.

In addition to the PR author, the above contributions were reviewed by William F. Broderick.

# Acknowledgments
<!-- Acknowledgement of any financial support. -->

Plenoptic's development and maintenance is supported by the [Center for Computational Neuroscience](https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/), in the Flatiron Institute of the Simons Foundation.

# References
