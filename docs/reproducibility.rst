.. _reproduce:

Reproducibility
***************

plenoptic includes several results reproduced from the literature and aims to
facilitate reproducible research. However, we are limited by our dependencies
and PyTorch, in particular, comes with the `caveat
<https://pytorch.org/docs/stable/notes/randomness.html>`_ that "Completely
reproducible results are not guaranteed across PyTorch releases, individual
commits, or different platforms. Furthermore, results may not be reproducible
between CPU and GPU executions, even when using identical seeds" (quote from the
v1.12 documentation).

This means that you should note the ``plenoptic`` version and the ``pytorch``
version your synthesis used in order to guarantee reproduciblity (some versions
of ``pytorch`` will give consistent results with each other, but it's not
guaranteed and hard to predict). We do not believe reproducibility depends on
the python version or any other packages. In general, the CPU and GPU will
always give different results.

We reproduce several results from the literature and validate these as part of
our tests. We are therefore aware of the following changes that broke
reproducibility:

- PyTorch 1.8 and 1.9 give the same results, but 1.10 changes results in
  changes, probably due to the difference in how the sub-gradient for
  ``torch.min`` and ``torch.max`` are computed (`see this PR
  <https://github.com/LabForComputationalVision/plenoptic/pull/96#issuecomment-973318291>`_).

- PyTorch 1.12 breaks reproducibility with 1.10 and 1.11, unclear why (`see this
  issue <https://github.com/LabForComputationalVision/plenoptic/issues/165>`_).
