.. |license-shield| image:: https://img.shields.io/badge/license-MIT-yellow.svg
			    :target: https://github.com/LabForComputationalVision/plenoptic/blob/main/LICENSE

.. |python-version-shield| image:: https://img.shields.io/badge/python-3.7%7C3.8%7C3.9%7C3.10-blue.svg

.. |build| image:: https://github.com/LabForComputationalVision/plenoptic/workflows/build/badge.svg
		     :target: https://github.com/LabForComputationalVision/plenoptic/actions?query=workflow%3Abuild

.. |tutorials| image:: https://github.com/LabForComputationalVision/plenoptic/workflows/tutorials/badge.svg
		         :target: https://github.com/LabForComputationalVision/plenoptic/actions?query=workflow%3Atutorials

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3995057.svg
            :target: https://doi.org/10.5281/zenodo.3995057

.. plenoptic documentation master file, created by
   sphinx-quickstart on Thu Jun 20 15:56:27 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

plenoptic
=====================================

|license-shield| |python-version-shield| |build| |tutorials| |zenodo|

``plenoptic`` is a python library for model-based stimulus synthesis. It
provides tools to help researchers understand their model by synthesizing novel
informative stimuli, which help build intuition for what features the model
ignores and what it is sensitive to. They can be used in future experiments for
further investigation.

Getting started
---------------

- If you are unfamiliar with stimulus synthesis, see the :ref:`conceptual-intro`
  for a more in-depth introduction.
- If you understand the basics of synthesis and want to get started using
  ``plenoptic`` quickly, see the `Quickstart <tutorials/00_quickstart.html>`_
  tutorial.

Installation
^^^^^^^^^^^^

.. highlight:: bash

The best way to install ``plenoptic`` is via ``pip``. For now, you must do this
from github directly::

$ pip install git+https://github.com/LabForComputationalVision/plenoptic.git

See the `README
<https://github.com/LabForComputationalVision/plenoptic/#setup>`_ for more
details, including how to set up an isolated virtual environment.

Why perform stimulus synthesis?
-------------------------------

- scientific models are normally evaluated on their ability to fit data or
  perform a task, such as how well a model performs on ImageNet or how closely a
  model tracks firing rate in some collected data. However, many models can
  perform a task equally or comparably well [#]_. Stimulus synthesis provides
  another method for discriminating between competing models.
- models perform unexpectedly on out-of-distribution data, as adversarial
  examples show [Szegedy2013]_. It's impossible to evaluate models on *all*
  possible images, but stimulus synthesis allows for an exploration of image
  space in a targeted manner.
- both of the above are specific examples of a general point: stimulus synthesis
  allows for the generation of novel, informative stimuli, which can be used in
  further experiments. These stimuli can emphasize where models succeed or fail
  to capture the phenomena under study.
- investigating synthesized stimuli can help scientists reason about the types
  of information that their models disregard and the information that it
  consider essential, facilitating model comprehension.

See :ref:`conceptual-intro` and tutorials for longer discussions of the
scientific reasoning enabled by stimulus synthesis.

.. _package-contents:
Contents
--------

.. figure:: images/example_synth.svg
   :figwidth: 100%
   :alt: The four synthesis methods included in plenoptic

``plenoptic`` includes the following synthesis methods:

- `Metamers <tutorials/06_Metamer.html>`_: given a model and a reference image,
  stochastically generate a new image whose model representation is identical to
  that of the reference image. This method investigates what image features the
  model disregards entirely.

  - Example papers: [Portilla2000]_, [Freeman2011]_
- `Eigendistortions <tutorials/02_Eigendistortions.html>`_: given a model and a
  reference image, compute the image perturbation that produces the smallest and
  largest changes in the model response space. This method investigates the
  image features the model considers the least and most important.

  - Example papers: [Berardino2017]_
- `Maximal differentiation (MAD) competition
  <tutorials/07_MAD_Competition.html>`_: given two metrics that measure distance
  between images and a reference image, generate pairs of images that optimally
  differentiate the models. Specifically, synthesize a pair of images that the
  first model says are equi-distant from the reference while the second model
  says they are maximally/minimally distant from the reference. Then synthesize
  a second pair with the roles of the two models reversed. This method allows
  for efficient comparison of two metrics, highlighting the aspects in which
  their sensitivities differ.

  - Example papers: [Wang2008]_
- `Geodesics <tutorials/05_Geodesics.html>`_: given a model and two images,
  synthesize a sequence of images that lie on the shortest ("geodesic") path in
  the model's representation space. This method investigates how a model
  represents motion and what changes to an image it consider reasonable.

  - Example papers: [Henaff2016]_, [Henaff2020]_

``plenoptic`` also contains the following models, metrics, and model components
that might be useful:

- Portilla-Simoncelli texture model, [Portilla2000]_, which measures the
  statistical properties of visual textures, here defined as "repeating visual
  patterns."
- Steerable pyramid, [Simoncelli1995]_, a multi-scale oriented image
  decomposition. The basis are oriented (steerable) filters, localized in space
  and frequency. Among other uses, the steerable pyramid serves as a good
  representation from which to build a primary visual cortex model. See the
  `pyrtools documentation
  <https://pyrtools.readthedocs.io/en/latest/index.html>`_ for more details on
  image pyramids in general and the steerable pyramid in particular.
- Structural Similarity Index (SSIM), [Wang2004]_, is a perceptual similarity
  metric, returning a number between -1 (totally different) and 1 (identical)
  reflecting how similar two images are. This is based on the images' luminance,
  contrast, and structure, which are computed convolutionally across the images.
- Multiscale Structrual Similarity Index (MS-SSIM), [Wang2003]_, is a perceptual
  similarity metric similar to SSIM, except it operates at multiple scales
  (i.e., spatial frequencies).
- Normalized Laplacian distance, [Laparra2016]_ and [Laparra2017]_, is an
  perceptual distance metric based on transformations associated with the early
  visual system: local luminance subtraction and local contrast gain control, at
  six scales.

If anything is unclear, please `open an issue
<https://github.com/LabForComputationalVision/plenoptic/issues>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   conceptual_intro
   reproducibility
   tips
   api/modules

.. toctree::
   :titlesonly:
   :glob:
   :caption: Tutorials and examples

   tutorials/*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [#] for example, as of February 2022, more than 100 models have above 95% top
  5 accuracy on ImageNet, with 9 models within a percent of the top performer at
  99.02%. Furthermore, the state of the art top 5 accuracy has been at or above
  95% since 2016, with an improvement of only 4% in the past six years.

.. [Portilla2000] Portilla, J., & Simoncelli, E. P. (2000). A parametric texture
   model based on joint statistics of complex wavelet coefficients.
   International journal of computer vision, 40(1), 49–70.
   https://www.cns.nyu.edu/~lcv/texture/.
   https://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf
.. [Freeman2011] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
   ventral stream. Nature Neuroscience, 14(9), 1195–1201.
   http://www.cns.nyu.edu/pub/eero/freeman10-reprint.pdf
.. [Berardino2017] Berardino, A., Laparra, V., J Ball\'e, & Simoncelli, E. P.
   (2017). Eigen-distortions of hierarchical representations. In I. Guyon, U.
   Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett,
   Adv. Neural Information Processing Systems (NIPS*17) (pp. 1–10). : Curran
   Associates, Inc. https://www.cns.nyu.edu/~lcv/eigendistortions/
   http://www.cns.nyu.edu/pub/lcv/berardino17c-final.pdf
.. [Wang2008] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation
   (MAD) competition: A methodology for comparing computational models of
   perceptual discriminability. Journal of Vision, 8(12), 1–13.
   https://ece.uwaterloo.ca/~z70wang/research/mad/
   http://www.cns.nyu.edu/pub/lcv/wang08-preprint.pdf
.. [Henaff2016] H\'enaff, O.~J., & Simoncelli, E.~P. (2016). Geodesics of
   learned representations. ICLR.
   http://www.cns.nyu.edu/pub/lcv/henaff16b-reprint.pdf
.. [Henaff2020] O Hénaff, Y Bai, J Charlton, I Nauhaus, E P Simoncelli and R L T
   Goris. Primary visual cortex straightens natural video trajectories Nature
   Communications, vol.12(5982), Oct 2021.
   https://www.cns.nyu.edu/pub/lcv/henaff20-reprint.pdf
.. [Simoncelli1995] Simoncelli, E. P., & Freeman, W. T. (1995). The steerable
   pyramid: A flexible architecture for multi-scale derivative computation. In ,
   Proc 2nd IEEE Int'l Conf on Image Proc (ICIP) (pp. 444–447). Washington, DC:
   IEEE Sig Proc Society. http://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf
.. [Wang2004] Wang, Z., Bovik, A., Sheikh, H., & Simoncelli, E. (2004). Image
   quality assessment: from error visibility to structural similarity. IEEE
   Transactions on Image Processing, 13(4), 600–612.
   https://www.cns.nyu.edu/~lcv/ssim/.
   http://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
.. [Wang2003] Z Wang, E P Simoncelli and A C Bovik. Multiscale structural
   similarity for image quality assessment Proc 37th Asilomar Conf on Signals,
   Systems and Computers, vol.2 pp. 1398--1402, Nov 2003.
   http://www.cns.nyu.edu/pub/eero/wang03b.pdf
.. [Laparra2017] Laparra, V., Berardino, A., Johannes Ball\'e, &
   Simoncelli, E. P. (2017). Perceptually Optimized Image Rendering. Journal of
   the Optical Society of America A, 34(9), 1511.
   http://www.cns.nyu.edu/pub/lcv/laparra17a.pdf
.. [Laparra2016] Laparra, V., Ballé, J., Berardino, A. and Simoncelli,
   E.P., 2016. Perceptual image quality assessment using a normalized Laplacian
   pyramid. Electronic Imaging, 2016(16), pp.1-6.
   http://www.cns.nyu.edu/pub/lcv/laparra16a-reprint.pdf
.. [Szegedy2013] Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D.,
   Goodfellow, I., & Fergus, R. (2013). Intriguing properties of neural
   networks. https://arxiv.org/abs/1312.6199
