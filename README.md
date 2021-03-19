# plenoptic

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/LabForComputationalVision/plenoptic/blob/master/LICENSE)
![Python version](https://img.shields.io/badge/python-3.6|3.7|3.8-blue.svg)
[![Build Status](https://github.com/LabForComputationalVision/plenoptic/workflows/build/badge.svg)](https://github.com/LabForComputationalVision/plenoptic/actions?query=workflow%3Abuild)
[![Documentation Status](https://readthedocs.org/projects/plenoptic/badge/?version=latest)](https://plenoptic.readthedocs.io/en/latest/?badge=latest)
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3995057.svg)](https://doi.org/10.5281/zenodo.3995057)
[![codecov](https://codecov.io/gh/LabForComputationalVision/plenoptic/branch/master/graph/badge.svg?token=EDtl5kqXKA)](https://codecov.io/gh/LabForComputationalVision/plenoptic)
[![Tutorials Status](https://github.com/LabForComputationalVision/plenoptic/workflows/tutorials/badge.svg)](https://github.com/LabForComputationalVision/plenoptic/actions?query=workflow%3Atutorials)
[![Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LabForComputationalVision/plenoptic/master?filepath=examples)

In recent years, [adversarial
examples](https://openai.com/blog/adversarial-example-research/) have
demonstrated how difficult it is to understand how complex models process
images. The space of all possible images is impossibly vast and difficult to
explore: even when training on millions of images, a dataset only represents a
tiny fraction of all images.

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
to examples that make use of them):

- [Metamers](http://www.cns.nyu.edu/~lcv/texture/):
  given a model and a reference image, stochastically generate a new image whose
  model representation is identical to that of the reference image.
- [Eigendistortions](https://www.cns.nyu.edu/~lcv/eigendistortions/):
  given a model and a reference image, compute the image perturbation that produces
  the smallest and largest changes in the model response space.  These correspond to the
  minimal/maximal eigenvectors of the Fisher Information matrix of the representation (for deterministic models, 
  the minimal/maximal singular vectors of the Jacobian).
- [Maximal differentiation (MAD)
  competition](https://ece.uwaterloo.ca/~z70wang/research/mad/):
  given two models that measure distance between images and a reference image, generate pairs of 
  images that optimally differentiate the models.  Specifically, synthesize a pair of images 
  that the first model says are equi-distant from the reference while the second model says they 
  are maximally/minimally distant from the reference. Synthesize a second pair with the roles of the two models reversed.
- [Geodesics](https://www.cns.nyu.edu/pub/lcv/henaff16b-reprint.pdf):
  given a model and two images, synthesize a sequence of images that lie on 
  the shortest ("geodesic") path in the model's representation space. 
  
# Status

This project is currently under heavy development. Not all features
have been implemented, and there will be breaking changes.

# Roadmap

See the [github
project](https://github.com/LabForComputationalVision/plenoptic/projects/1)
for a more detailed roadmap, but at the high level:

- Short term:
  1. Finalize Portilla-Simoncelli texture model
  2. Recreate existing `MADCompetition` examples.
- Medium term:
  1. Finalize geodesics
  2. Get eigendistortion and geodesics to use `Synthesis` superclass
  3. Write more documentation and tutorials
  4. Finalize model API, create superclass
  5. Add more models
- Long term:
  1. Present at conference to advertise to users
  2. Submit paper to Journal of Open Source Software

# NOTE

We currently support python 3.6, 3.7, and 3.8. See [this
issue](https://github.com/LabForComputationalVision/plenoptic/issues/74) for 3.9
status

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

### ffmpeg

Several methods in this package generate videos. There are several backends
possible for saving the animations to file, see (matplotlib
documentation)[https://matplotlib.org/stable/api/animation_api.html#writer-classes]
for more details. In order convert them to HTML5 for viewing (and thus, to view
in a jupyter notebook), you'll need [ffmpeg](https://ffmpeg.org/download.html)
installed and on your path as well.

To change the backend, run `matplotlib.rcParams['animation.writer'] = writer`
before calling any of the animate functions. If you try to set that `rcParam`
with a random string, `matplotlib` will tell you the available choices.

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

If you wish to locally run the notebooks, you will need to install `jupyter` and
`ipywidgets` (you can also run them in the cloud using
[Binder](https://mybinder.org/v2/gh/LabForComputationalVision/plenoptic/master?filepath=examples)).
There are two main ways of getting a local `jupyter` install` working with this
package:

1. Install jupyter in the same environment as `plenoptic`. If you followed the
   [instructions above](#plenoptic) to create a `conda` environment named
   `plenoptic`, do the following:

``` sh
conda activate plenoptic
conda install -c conda-forge jupyterlab ipywidgets
```

   This is easy but, if you have multiple conda environments and want to use
   Jupyter notebooks in each of them, it will take up a lot of space.
   
2. Use
   [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels).
   Again, if you followed the instructions above:

``` sh
# activate your 'base' environment, the default one created by miniconda
conda activate 
# install jupyter lab and nb_conda_kernels in your base environment
conda install -c conda-forge jupyterlab ipywidgets
conda install nb_conda_kernels
# install ipykernel in the calibration environment
conda install -n plenoptic ipykernel
```

   This is a bit more complicated, but means you only have one installation of
   jupyter lab on your machine.
   
In either case, to open the notebooks, navigate to the `examples/` directory
under this one on your terminal and activate the environment you install jupyter
into (`plenoptic` for 1, `base` for 2), then run `jupyter` and open up the
notebooks. If you followed the second method, you should be prompted to select
your kernel the first time you open a notebook: select the one named
"plenoptic".

## Keeping up-to-date

Once you've downloaded and set up plenoptic for the first time, you can use `git
pull` to keep it up-to-date. Navigate to the directory (if you downloaded
plenoptic into your Documents folder above, that's `cd ~/Documents/plenoptic`)
and run `git pull origin master`. git may yell at you if you've made local
changes it can't figure out how to resolve. You'll have a merge conflict on your
hands, see
[here](https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts)
for more information and how to proceed.

If you'd like to contribute any of the changes you've made, view the
[Contributing](#contributing) section.

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
