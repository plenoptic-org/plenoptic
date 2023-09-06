.. |pypi-shield| image:: https://img.shields.io/pypi/v/plenoptic.svg
			 :target: https://pypi.org/project/plenoptic/

.. |license-shield| image:: https://img.shields.io/badge/license-MIT-yellow.svg
                    :target: https://github.com/LabForComputationalVision/plenoptic/blob/main/LICENSE

.. |python-version-shield| image:: https://img.shields.io/badge/python-3.7%7C3.8%7C3.9%7C3.10-blue.svg

.. |build| image:: https://github.com/LabForComputationalVision/plenoptic/workflows/build/badge.svg
		     :target: https://github.com/LabForComputationalVision/plenoptic/actions?query=workflow%3Abuild

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3995057.svg
            :target: https://doi.org/10.5281/zenodo.3995057

.. |codecov| image:: https://codecov.io/gh/LabForComputationalVision/plenoptic/branch/main/graph/badge.svg?token=EDtl5kqXKA
             :target: https://codecov.io/gh/LabForComputationalVision/plenoptic

.. |binder| image:: https://mybinder.org/badge_logo.svg
		    :target: https://mybinder.org/v2/gh/LabForComputationalVision/plenoptic/1.0.1?filepath=examples

.. plenoptic documentation master file, created by
   sphinx-quickstart on Thu Jun 20 15:56:27 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


plenoptic
*********

|pypi-shield| |license-shield| |python-version-shield| |build| |zenodo| |codecov| |binder|


.. image:: images/plenoptic_logo_wide.svg
   :align: center
   :alt: plenoptic logo

``plenoptic`` is a python library for model-based synthesis of perceptual stimuli. 
Most examples are visual, but the tools can also be used for auditory models. 
The generated stimuli enable interpretation of model properties through examination of features that are 
enhanced, suppressed, or descarded.
More importantly, they can facilitate the scientific proceess, through use in perceptual or neural experiments 
aimed at validating/falsifying model predictions.

Getting started
---------------

- If you are unfamiliar with stimulus synthesis, see the :ref:`conceptual-intro`
  for an in-depth introduction.
- Otherwise, see the `Quickstart <tutorials/00_quickstart.nblink>`_
  tutorial.

Installation
^^^^^^^^^^^^

The best way to install ``plenoptic`` is via ``pip``::

$ pip install plenoptic

See the :ref:`install` page for more details, including how to set up an isolated
virtual environment (recommended).

ffmpeg and videos
^^^^^^^^^^^^^^^^^

Some methods in this package generate videos. There are several backends
available for saving the animations to file (see `matplotlib documentation
<https://matplotlib.org/stable/api/animation_api.html#writer-classes>`_
).
To convert them to HTML5 for viewing (for example, in a
jupyter notebook), you'll need `ffmpeg <https://ffmpeg.org/download.html>`_
installed. 
To change the backend, run ``matplotlib.rcParams['animation.writer'] = writer``
before calling any of the animate functions. If you try to set that ``rcParam``
with a random string, ``matplotlib`` will list the available choices.


.. _package-contents:
Contents
--------

.. figure:: images/example_synth.svg
   :figwidth: 100%
   :alt: The four synthesis methods included in plenoptic

Synthesis methods
^^^^^^^^^^^^^^^^^

- `Metamers <tutorials/06_Metamer.nblink>`_: given a model and a reference image,
  stochastically generate a new image whose model representation is identical to
  that of the reference image (a "metamer", as originally defined in the literature on Trichromacy). 
  This method makes explicit those features that the model retains/discards.

  - Example papers: [Portilla2000]_, [Freeman2011]_, [Deza2019]_,
    [Feather2019]_, [Wallis2019]_, [Ziemba2021]_
- `Eigendistortions <tutorials/02_Eigendistortions.nblink>`_: given a model and a
  reference image, compute the image perturbations that produce the smallest/largest 
  change in the model response space. These are the 
  image changes to which the model is least/most sensitive, respectively.

  - Example papers: [Berardino2017]_
- `Maximal differentiation (MAD) competition
  <tutorials/07_MAD_Competition.nblink>`_: given a reference image and two models that measure distance
  between images, generate pairs of images that optimally
  differentiate the models. Specifically, synthesize a pair of images that are equi-distant from
  the reference image according to model-1, but maximally/minimally distant according to model-2.  Synthesize
  a second pair with the roles of the two models reversed. This method allows
  for efficient comparison of two metrics, highlighting the aspects in which
  their sensitivities most differ.

  - Example papers: [Wang2008]_
- `Geodesics <tutorials/05_Geodesics.nblink>`_: given a model and two images,
  synthesize a sequence of images that lie on the shortest ("geodesic") path in
  the model's representation space. This method allows examination of the larger-scale geometric
  properties of model representation (as opposed to the local properties captured by 
  the eigendistortions).

  - Example papers: [Henaff2016]_, [Henaff2020]_

Models, Metrics, and Model Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Steerable pyramid, [Simoncelli1992]_, a multi-scale oriented image
  decomposition. Images are decomposed with a family of oriented filters, localized in space
  and frequency, similar to the "Gabor functions" commonly used to model receptive fields in primary visual cortex.  
  The critical difference is that the pyramid organizes these filters so as to effeciently cover the 4D space of 
  (x,y) positions, orientations, and scales, enabling efficient interpolation and interpretation 
  (`further info <https://www.cns.nyu.edu/~eero/STEERPYR/>`_ ). See the `pyrtools documentation
  <https://pyrtools.readthedocs.io/en/latest/index.html>`_ for more details on
  python tools for image pyramids in general and the steerable pyramid in particular.
- Portilla-Simoncelli texture model, [Portilla2000]_, which computes a set of image statistics
  that capture the appearance of visual textures (`further info <https://www.cns.nyu.edu/~lcv/texture/>`_).
- Structural Similarity Index (SSIM), [Wang2004]_, is a perceptual similarity
  metric, that takes two images and returns a value between -1 (totally different) and 1 (identical)
  reflecting their similarity (`further info <https://www.cns.nyu.edu/~lcv/ssim>`_).
- Multiscale Structural Similarity Index (MS-SSIM), [Wang2003]_, is an extension of SSIM
  that operates jointly over multiple scales.
- Normalized Laplacian distance, [Laparra2016]_ and [Laparra2017]_, is a
  perceptual distance metric based on transformations associated with the early
  visual system: local luminance subtraction and local contrast gain control, at
  six scales (`further info <https://www.cns.nyu.edu/~lcv/NLPyr/>`_).

Getting help
------------

We communicate via several channels on Github:

- To report a bug, open an `issue
  <https://github.com/LabForComputationalVision/plenoptic/issues>`_.
- To send suggestions for extensions or enhancements, please post in the `ideas
  section
  <https://github.com/LabForComputationalVision/plenoptic/discussions/categories/ideas>`_
  of discussions first. We'll discuss it there and, if we decide to pursue it,
  open an issue to track progress.
- To ask usage questions, discuss broad issues, or
  show off what you've made with plenoptic, go to `Discussions
  <https://github.com/LabForComputationalVision/plenoptic/discussions>`_.
- To contribute to the project, see the `contributing guide
  <https://github.com/LabForComputationalVision/plenoptic/blob/main/CONTRIBUTING.md>`_.

In all cases, we request that you respect our `code of conduct
<https://github.com/LabForComputationalVision/plenoptic/blob/main/CODE_OF_CONDUCT.md>`_.

.. toctree::
   :titlesonly:
   :caption: Basic concepts
   :glob:

   install
   conceptual_intro
   models
   tutorials/*

.. toctree::
   :titlesonly:
   :glob:
   :caption: Synthesis method introductions

   tutorials/intro/*

.. toctree::
   :titlesonly:
   :glob:
   :caption: Models and metrics

   tutorials/models/*

.. toctree::
   :titlesonly:
   :glob:
   :caption: Synthesis method examples

   tutorials/applications/*

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Advanced usage

   synthesis
   tips
   reproducibility
   Modules <api/modules>
   tutorials/advanced/*

.. [Portilla2000] Portilla, J., & Simoncelli, E. P. (2000). A parametric texture
   model based on joint statistics of complex wavelet coefficients.
   International journal of computer vision, 40(1), 49–70.
   https://www.cns.nyu.edu/~lcv/texture/.
   https://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf
.. [Freeman2011] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
   ventral stream. Nature Neuroscience, 14(9), 1195–1201.
   http://www.cns.nyu.edu/pub/eero/freeman10-reprint.pdf
.. [Deza2019] Deza, A., Jonnalagadda, A., & Eckstein, M. P. (2019). Towards
   metamerism via foveated style transfer. In , International Conference on
   Learning Representations.
.. [Feather2019] Feather, J., Durango, A., Gonzalez, R., & McDermott, J. (2019).
   Metamers of neural networks reveal divergence from human perceptual systems.
   In NeurIPS (pp. 10078–10089).
.. [Wallis2019] Wallis, T. S., Funke, C. M., Ecker, A. S., Gatys, L. A.,
   Wichmann, F. A., & Bethge, M. (2019). Image content is more important than
   bouma's law for scene metamers. eLife. http://dx.doi.org/10.7554/elife.42512
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
.. [Ziemba2021] Ziemba, C.M., and Simoncelli, E.P. (2021). Opposing effects of selectivity and invariance in peripheral vision.
   Nature Communications, vol.12(4597).
   https://dx.doi.org/10.1038/s41467-021-24880-5

This package is supported by the 'Center for Computational Neuroscience 
<https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/>'_, 
in the Flatiron Institute of the Simons Foundation.

.. image:: images/CCN-logo-wText.png
   :align: center
   :alt: Flatiron Institute Center for Computational Neuroscience logo
