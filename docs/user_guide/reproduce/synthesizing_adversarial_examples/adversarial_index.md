# Synthesizing Adversarial Examples

The following notebooks demonstrate how to synthesize adversarial examples using Plenoptic. Adversarial examples are images with subtle, imperceptible perturbations designed to deceive a Deep Neural Networks into making an incorrect classification ({cite:alp}`Szegedy2013`, {cite:alp}`goodfellow_explaining_2015`). In these notebooks, we show two ways to synthesize adversarial examples using plenoptic's machinery:
1. Using {class}`~plenoptic.MADCompetition`, we can synthesize an adversarial example by maximizing the distance in the model's representation while keeping the distance in pixel space small.
2. Using {class}`~plenoptic.Metamer`, we can synthesize an adversarial example by starting with one image and turning it into metamers of a different class.

:::::{card}
::::{toctree}
:maxdepth: 2

adversarial_examples_mad
adversarial_examples_metamer

::::
:::::
