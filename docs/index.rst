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

If you are unfamiliar with stimulus synthesis, see the :ref:`conceptual-intro`
for a more in-depth introduction.

If you understand the basics of synthesis and want to get started using
``plenoptic`` quickly, see the :ref:`Quickstart <tutorials/00_quickstart.html>`
tutorial.

- quickstart
- contents
- quick "why do this?"
- quickstart: merge with thesis example, bring in Eero's notes

If anything is unclear, please `open an issue <https://github.com/LabForComputationalVision/plenoptic/issues>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   conceptual_intro
   reproducibility
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
