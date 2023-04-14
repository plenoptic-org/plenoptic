.. _tips:

Tips and Tricks
***************

Why does synthesis take so long?
================================

Synthesis can take a while to run, especially if you are trying to synthesize a
large image or using a complicated model. The following might help:

- Reducing the amount of time your model's forward pass takes is the easiest way
  to reduce the overall duration of synthesis, as the forward pass is called
  many, many times over synthesis. Try using python's `built-in profiling tools
  <https://docs.python.org/3/library/profile.html>`_ to check which part of your
  model's forward pass is taking the longest, and try to make those parts more
  efficient. Jupyter also has nice `profiling tools
  <https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html>`_.
  For example, if you have for loops in your code, try and replace them with
  matrix operations and `einsum
  <https://pytorch.org/docs/stable/generated/torch.einsum.html>`_.
- If you have access to a GPU, use it! If your inputs are on the GPU before
  initializing the synthesis methods, the synthesis methods will also make use
  of the GPU. You can also move the ``plenoptic``'s synthesis methods and models
  over to the GPU after initialization using the ``.to()`` method.

Optimization is hard
====================

You should double-check whether synthesis has successfully completed before
interpreting the outputs or using them in any experiments. This is not necessary
for eigendistortions (see its `notebook <tutorials/02_Eigendistortions.html>`_
for more details on why), but is necessary for all the iterative optimization
methods.

- For metamers, this means double-checking that the difference between the model
  representation of the metamer and the target image is small enough. If your
  model's representation is multi-scale, trying coarse-to-fine optimization may
  help (see `notebook <tutorials/06_Metamer.html#Coarse-to-fine-optimization>`_
  for details).
- For MAD competition, this means double-checking that the reference metric is
  constant and that the optimized metric has converged at a lower or higher
  value (depending on the value of ``synthesis_target``); use
  :func:`plenoptic.synthesize.mad_competition.plot_synthesis_status` to
  visualize these values. You will likely need to spend time trying out
  different values for the ``metric_tradeoff_lambda`` argument set during
  initialization to achieve this.
- For geodesics, check that your geodesic's path energy is small enough and that
  the deviation from a straight line in representational space is minimal (use
  :func:`plenoptic.synthesize.geodesic.plot_deviation_from_line`)

For all of the above, if synthesis has not found a good solution, you may need
to run synthesis longer, use a learning-rate scheduler, change the learning
rate, or try different optimizers. Each method's ``objective_function`` method
captures the value that we are trying to minimize, but may contain other values
(such as the penalty on allowed range values).

Additionally, it may be helpful to visualize the progression of synthesis, using
each synthesis method's ``animate`` or ``plot_synthesis_status`` helper
functions (e.g., :func:`plenoptic.synthesize.metamer.plot_synthesis_status`).

None of the existing synthesis methods meet my needs
====================================================

``plenoptic`` provides four synthesis methods, but you may find you wish to do
something slightly outside the capabilities of the existing methods. There are
generally two ways to do this: by tweaking your model or by extending one of the
methods.

- See the `Portilla-Simoncelli texture model notebook
  <tutorials/Metamer-Portilla-Simoncelli.html>`_ for examples on how to get
  different metamer results by tweaking your model or extending the ``Metamer``
  class.
- The coarse-to-fine optimization, discussed in the `metamer notebook
  <tutorials/06_Metamer.html#Coarse-to-fine-optimization>`_, is an example of
  changing optimization by extending the ``Metamer`` class.
- The `Synthesis extensions notebook <tutorials/Synthesis_extensions.html>`_
  contains a discussion focused on this as well.

If you extend a method successfully or would like help making it work, please
let us know by posting a `discussion!
<https://github.com/Flatiron-CCN/plenoptic/discussions>`_
