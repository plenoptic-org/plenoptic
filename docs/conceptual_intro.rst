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
Young-Helmholtz theory of trichromacy [1]_. In this context, metamers
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

One demonstration of trichromacy is the standard color-matching experiment, as
visualized in :numref:`trichromacy`: an observer matches a monochromatic test
color (i.e., a light with energy at only a single wavelength) with the physical
mixture of three different monochromatic stimuli, called **primaries**. Thus,
the goal is to create two perceptually-indistinguishable stimuli, known as
**metamers**. Perhaps surprisingly, not only is this possible for any test
color, it is also possible for just about any selection of primaries (within the
visible light spectrum). Generally, this requires three primaries: for most
people, there are many colors that cannot be matched with only two primaries.
However, there are some people, for whom two primaries are sufficient.

.. _trichromacy:
.. figure:: images/trichromacy.svg
   :figwidth: 100%
   :alt: Color matching experiment.

   Color matching experiment

Requiring three primaries for most people, but two for some might hint at the
reason why this is possible: most people have three cone classes (generally
referred to as S, M, and L, for "short", "medium", and "long"), while some only
have two (which two depends on the type of colorblindness). From the results of
the color matching, we can infer that color metamers are created when cone
responses have been matched. Human cones transform colors from a
high-dimensional space (i.e., a vector describing the energy at each wavelength)
to a three-dimensional one (i.e., a vector describing how active each cone class
is). This means a large amount of wavelength information is discarded.

A worked example may help demonstrate this point more clearly. Let's match the
random light shown on the left below using the primaries shown on the right.

.. _primaries:
.. plot:: scripts/conceptual_intro.py primaries

   Left: Random light whose appearance we will match. Right: primaries.

The only way we can change the matching light is multiply those primaries by
different numbers, moving them up and down. You might look at them and wonder
how we can match the light shown on the left, with all its random wiggles. The
important point is that we **will not** match those wiggles. We will instead
match the cone activation levels, which we get by matrix multiplying our light
by the cone fundamentals, shown below.

.. plot:: scripts/conceptual_intro.py cones

   Left: the cone sensitivity curves. Right: the response of each cone class to
   the random light in :numref:`primaries`

With some linear algebra, we can compute another light that has very different
amounts of energy at each wavelength but identical cone responses, shown below.

.. plot:: scripts/conceptual_intro.py matched_light

If we look at the plot on the left, we can see that the two lights are very
different physically, but we can see on the right that they generate the same
cone responses and thus would be perceived identically.

In this example, the model was a simple linear system of cone responses, and
thus we can generate a metamer, a physically different input with identical
output, via some simple linear algebra. However, while this same idea can be
applied to any model, generating such metamers gets complicated.

NOTES:
======

Color matching experiments demonstrate human trichromacy, but...

* following MathTools trichromacy HW question, step through an example of the
  above: random light, any three primaries, can make the perception match
  * before we answer this, what does it mean for percpetion to match? here, we
    mean that the two colors are indistinguishable.
  * in this situation, that happens when the outputs of the cones match. the
    human visual system throws out a lot of information about wavelengths
    because we only have three cone classes sensitive to visible light. they are
    sensitive to a particular range (which is why some birds, insects can
    respond to ultraviolet light, while we cannot) but they also limit our
    ability to distinguish between physical stimuli *within* our sensitivity
    range
  * analogy is with color blindness. most folks who are colorblind only have two
    cone classes (there are other types of colorblindness), and so throw away
    more information than folks with three classes; they are thus are unable to
    distinguish some colors that folks with three classes are able to, such as
    red and green
  * similarly, people with three classes are unable to distinguish between two
    colors that **are** physically different, e.g., a blue shirt and a picture
    of that shirt.
  * demonstrate with tihs worked example: random light, matrix multiply through
    cone fundamentals to get these three numbers. any light that matches those
    three numbers is indistinguishable. for example, say we had a screen with
    these three primaries. the only thing we can do is multiply these by some
    numbers
  * and we do some linear algebra and get the following light! can see that it
    leads to the same output from the cone fundamentals and thus the same
    perception.
  * nothing special about these primaries, let's change the primaries and try
    again: these are actual CRT monitor phosphors. show same result
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
  * here, throwing out spatial information, depends on where you focus your eyes
    (so you can see the effect by moving your eyes), and we end up with images
    that look blurry
  * just luminance one, point out that model is just a bunch of gaussians that
    grow in size with distance from center of image. this means that it's
    basically low-pass filters, with the cutoff shifting to lower and lower
    frequencies as you get more peripheral
  * thus, you can have any information in there: show "blurred" version, plus
    init-white and init-nat?
* here, things are nonlinear and so we need a different way to generate them.
  * actually , luminance is linear -- but don't really want to explain the
    energy one...
* why do this?
  * it can allow for applications: compression of information. in color example,
    we can represent all visible colors with only three pieces of information
    (rather than storing the complete power spectra), which is exploited by
    cameras, digital screens, etc.
  * but deeper... copy/adapt part of my thesis that discussed this (p136)
* make point that this helps tighten the model-experiment loop, makes theorists
  participants in that cycle, to help direct further experiments.

.. [1] Helmholtz, H. (1852). LXXXI. on the theory of compound colours.
  The London, Edinburgh, and Dublin Philosophical Magazine and Journal of
  Science, 4(28), 519–534. http://dx.doi.org/10.1080/14786445208647175
