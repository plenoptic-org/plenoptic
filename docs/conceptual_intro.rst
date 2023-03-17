.. _conceptual-intro:

Conceptual Introduction
***********************

``plenoptic`` is a python library for "model-based stimulus synthesis". If
you've never heard this phrase before, it may seem mysterious: what is stimulus
synthesis and what types of scientific investigation does it facilitate?

Synthesis is a framework for exploring models that takes advantage of the fact
that we can create stimuli, not just rely on existing ones. Computational models
take a stimulus as input, perform some computations based on parameters, and
return an output. In visual models, the focus of ``plenoptic``, the inputs are
typically images and the outputs are some abstractions of representation, which
are used to predict neural activity or behavior of some kind. Most commonly,
researchers use these models alongside experiments, simulating model responses
(with fixed parameters) to a variety of stimuli that are compared against other
models, neural responses, or animal behavior. Researchers also often fit the
parameters of their model, using optimization to find the parameter values that
best align model responses with the output of interest for the tested set of
inputs. However, stimuli are not special, and researchers can similarly hold
parameters and responses fixed, while using optimization to generate new stimuli
(see :numref:`synthesis-schematic` for a schematic comparing these procedures).
We refer to this process as **synthesis** and it facilitates the exploration of
input space to improve our understanding of a model's representation space.

.. _synthesis-schematic:
.. figure:: images/model_sim-fit-infer.svg
   :figwidth: 100%
   :alt: Schematic describing relationship between simulate, fit, and synthesis.

   Schematic describing relationship between simulate, fit, and synthesis.

This is related to a long and fruitful thread of research in vision science that
focuses on what humans cannot see, that is, the information they are insensitive
to. Perceptual metamers --- images that are physically distinct but perceptually
indistinguishable --- provide direct evidence of such information loss in visual
representations. Color metamers were instrumental in the development of the
Young-Helmholtz theory of trichromacy [trichrom]_. In this context, metamers
demonstrate that the human visual system projects the infinite dimensionality of
the physical signal to three dimensions.

To make this more concrete, let's walk through an example. Humans can see
visible light, which is electromagnetic radiation with wavelengths between 400
and 700 nanometers (nm). We often want to be able to recreate the colors in a
natural scene, such as when we take a picture. In order to do so, we can ask:
what information do we need to record in order to do so? Let's start with a
solid patch of uniform color. If we wanted to recreate the complete energy
spectra of the color, we would need to record a lot of numbers: even if we
subsampled the wavelengths so that we only recorded the energy every 5 nm, we
would need 61 numbers per color! But we know that most modern electronic screens
only use three numbers, often called RGB (red, green, and blue) --- why can we
get away with throwing away so much information? Trichromacy and color metamers
can help explain.

Trichromacy can be demonstrated with the standard color-matching experiment, as
visualized in :numref:`trichromacy`: an observer matches a monochromatic test
color (i.e., a light with energy at only a single wavelength) with the physical
mixture of three different monochromatic stimuli, called **primaries**. Perhaps
surprisingly, not only is this possible for any test color, it is also possible
for just about any selection of primaries (within the visible light spectrum).

.. _trichromacy:
.. figure:: images/trichromacy.svg
   :figwidth: 100%
   :alt: Color matching experiment.

   Color matching experiment

Color matching experiments demonstrate human trichromacy, but...

* following MathTools trichromacy HW question, step through an example of the
  above: random light, any three primaries, can make the perception match
* point out this is because the outputs of the cones match, show cone
  fundamentals
* point out this is simple in this case (though I'm not showing the math)
  because it's a linear system and so we can do it with linear algebra / matrix
  math
* but the point holds generally: visual system is constantly throwing out
  information (just like all models) and once info is thrown out, it cannot be
  recovered
* and so we can match a representation at some other point. switch to foveated
  metamers
* here, things are nonlinear and so we need a different way to generate them.
* make point that this helps tighten the model-experiment loop, makes theorists
  participants in that cycle, to help direct further experiments.

.. plot:: scripts/conceptual_intro.py plot_cone_fundamentals

   The cone fundamentals.


.. [trichrom] Helmholtz, H. (1852). LXXXI. on the theory of compound colours.
  The London, Edinburgh, and Dublin Philosophical Magazine and Journal of
  Science, 4(28), 519–534. http://dx.doi.org/10.1080/14786445208647175
