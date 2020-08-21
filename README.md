# plenoptic

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/LabForComputationalVision/plenoptic/blob/master/LICENSE)
![Python version](https://img.shields.io/badge/python-3.6%7C3.7-blue.svg)
[![Build Status](https://travis-ci.com/LabForComputationalVision/plenoptic.svg?branch=master)](https://travis-ci.com/LabForComputationalVision/plenoptic)
[![Documentation Status](https://readthedocs.org/projects/plenoptic/badge/?version=latest)](https://plenoptic.readthedocs.io/en/latest/?badge=latest)
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3995057.svg)](https://doi.org/10.5281/zenodo.3995057)

In recent years, [adversarial
examples](https://openai.com/blog/adversarial-example-research/) have
demonstrated how difficult it is to understand how complex models process
images. The space of all possible images is impossibly vast and difficult
to explore: even when training on millions of images, a dataset only
represents a tiny fraction of all images.

`plenoptic` is a python library that provides tools to help researchers
better understand their models by using optimization to synthesize novel 
informative images. These images allow us to build our intuition for what
features the model ignores and what it is sensitive to. These synthetic 
images can then be used in psychophysics experiments for further investigation.

More specifically, all models have three components,
inputs `x`, outputs `y`, and parameters `θ`. When working with models,
we typically either simulate, by holding `x` and `θ` constant, and
generating predictions for `y`, or fit, by holding `x` and `y`
constant and using optimization to find the best-fitting `θ`. However,
for optimization purposes, there's nothing special about `x`, so we
can instead hold `y` and `θ` constant and use optimization to
synthesize new inputs `x`. Synthesis methods do exactely that: they
take a model with set parameters, set outputs and generate new inputs.
They allow for better understanding of the model by assessing what
it is sensitive to and, crucially, what it is not sensitive to,
as well as generating novel stimuli for testing the model.

Here's a table summarizing this:

**{x, y, θ}** : inputs, outputs, parameters

|            	|   fixed  	| variable  |
|:----------:	|:------:	|:------:	|
|  simulate  	| {x, θ} 	|   {y}  	|
|   learn    	| {x, y} 	|   {θ}  	|
| synthesize 	| {y, θ} 	|   {x}  	|

`plenoptic` contains the following four synthesis methods (with links
to the papers describing them):

- [metamers](https://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf):
  given a model and an image, synthesize a new image which is admits
  a model representation identical to that of the original image.
- [Maximal differentiation (MAD)
  competition](https://www.cns.nyu.edu/pub/lcv/wang08-preprint.pdf):
  given two models and an image, synthesize two pairs of images: two
  that the first model represents identically, while the second model
  represents as differently as possible; and two that the first model
  represents as differently as possible while the second model represents
  identically.
- [Geodesics](https://www.cns.nyu.edu/pub/lcv/henaff16b-reprint.pdf):
  given a model and two images, synthesize a sequence of images that form
  a short path in the model's representation space. That is, a sequence of 
  images that are represented to as close as possible to a straight interpolation 
  line between the two anchor images.
- [Eigendistortions](https://www.cns.nyu.edu/pub/lcv/berardino17c-final.pdf):
  given a model and an image, synthesize the most and least noticeable
  distortion on the image (with a constant mean-squared error in
  pixels). That is, if you can change all pixel values by a total of
  100, how does the model think you should do it to make it as obvious
  as possible, and how does the model think you should do it to make
  it unnoticeable.
  
(where for all of these, "identical (resp. different) representation",
stands for small (resp. large) l2-distance in a model's representation space)

# Status

This project is currently in alpha, under heavy development. Not all features
have been implemented, and there will be breaking changes.

# Roadmap

See the [github
project](https://github.com/LabForComputationalVision/plenoptic/projects/1)
for a more detailed roadmap, but at the high level:

- Short term:
  1. Finalize Portilla-Simoncelli texture statistics
  2. Recreate existing `MADCompetition` examples.
- Medium term:
  1. Finalize geodesics
  2. Get eigendistortion and geodesics to use `Synthesis` superclass
  3. Write more documentation and tutorials
  4. Finalize model API, create superclass
  5. Add more models
- Long term:
  1. Present poster at conference to advertise to users
  2. Submit paper to Journal of Open Source Software to get something
     for people to cite

# NOTE

We only support python 3.6 and 3.7

# Setup

These are instructions for how to install and run the development
version of the `plenoptic` package (we are currently pre-release and
not on `pip`, so this is the only way to use `plenoptic`).

The following instructions will work on Linux or Mac. If you're on
Windows, I recommend looking into the [Windows Subsystem for
Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

## System

If you have not setup python on your system before: install
[miniconda](https://conda.io/miniconda.html) (this just contains
python and `conda`, a very nifty package manager; choose python
3.7). Conda is separate from python: it's a package manager, which
makes it easy to install a variety of python libraries. If you've ever
used `apt` on Ubuntu or [`brew` on Macs](https://brew.sh/), then
you've used a package manager before. Python has its own package
manager, `pip`, which generally works very well, but in my experience
I've found conda tends to work with fewer issues. [See
here](https://stackoverflow.com/questions/20994716/what-is-the-difference-between-pip-and-conda)
for some more details, but the gist seems to be: conda can handle
external (non-python) dependencies, whereas pip cannot, and conda can
create virtual environments (see item 3 in this list), whereas pip
cannot (the standard python way is to use `virtualenv`, which also
works well). See
[here](https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/)
for a blog post from Jake VanderPlas with more details on conda.

You will probably need to restart your terminal for this to take
effect. You should see `base` somewhere on your command line prompt.

Once you've done that, open the command-line and navigate to wherever
you want to download this repository (for example, `~/Documents`), and
check that you have `git` installed on your system:

```
cd ~/Documents
which git
```

assuming the second command returns something (e.g., `/usr/bin/git`),
`git` is installed and you're good to go. If nothing gets printed out,
then you need to [install
git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). See
[this cheatsheet](https://neuroplausible.com/github) for some more
explanation of git, Github, and the associated terminology.

## plenoptic

Once git is installed, you can clone the repository:

```
git clone https://github.com/LabForComputationalVision/plenoptic.git
```

Enter your username and password and, after a bit of a wait, you
should have a brand new plenoptic folder, `plenoptic`. Let's navigate
to that folder, create a new virtual environment, and install the
package:

```
cd plenoptic
conda create -n plenoptic python==3.7
conda activate plenoptic
pip install -e .
```

We have now created a new virtual environment called `plenoptic`,
which originally contains only the bare requirements (`python`, `pip`,
etc.). We then activate it (which means that everything we do to
interact with python will use this virtual environment, so only the
python version and packages included there; you should see `plenoptic`
somewhere on your command line prompt) and install `plenoptic`. This
will install all the requirements necessary for plenoptic to run.

## Jupyter

The one additional thing you will want is to install
[JupyterLab](https://jupyterlab.readthedocs.io/en/stable/),
which we use for tutorial and example notebooks:

```
conda install -c conda-forge jupyterlab
```

Note we want to do this within our `plenoptic` environment. If you're
running this section straight through, you won't need to do anything
extra, but if you closed your terminal session after the last section
(for example), you'll need to make sure to activate the correct
environment first: `conda activate plenoptic`.

# Getting started

Once you've set everything up appropriately, navigate to the example
directory, start up JupyterLab (which will open in a new browser tab),
and start exploring the notebooks!

```
cd examples/
jupyter lab
```

The notebooks contain examples and tutorials, and have been numbered
in a recommended order. They're all very much under development, and
we would appreciate any feedback!

# Contributing

For info on how to contribute, see the [CONTRIBUTING](CONTRIBUTING.md)
file, including info on how to test the package and build its
documentation.
