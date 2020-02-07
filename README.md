# plenoptic


**{x, y, θ}**  : inputs, outputs, parameters

|            	|   fixed  	| variable  |
|:----------:	|:------:	|:------:	|
|  simulate  	| {x, θ} 	|   {y}  	|
|   learn    	| {x, y} 	|   {θ}  	|
| synthesize 	| {y, θ} 	|   {x}  	|

# Project Vision

`plenoptic` is a project by graduate students and postdocs in the Lab
for Computational Vision to build a python library that enables
researchers who build computational models that operate on and extract
information from images (in neuroscience, computer vision, or other
fields) to interrogate their models, better understand how they work,
and improve their ability to run experiments for testing them. This is
an open project because the included methods have been described in
the literature but do not have easy-to-use, widely-available
implementations and we believe that science functions best when it is
transparent, accessible, and inclusive.

# Roadmap

See the [github
project](https://github.com/LabForComputationalVision/plenoptic/projects/1)
for a more detailed roadmap, but at the high level:

- Short term:
  1. Finalize Portilla-Simoncelli texture statistics
  2. Add MAD competition
  3. Create `Synthesis` superclass
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
