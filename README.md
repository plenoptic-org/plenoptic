# plenoptic

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/LabForComputationalVision/plenoptic/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.7|3.8|3.9|3.10-blue.svg)
[![Build Status](https://github.com/LabForComputationalVision/plenoptic/workflows/build/badge.svg)](https://github.com/LabForComputationalVision/plenoptic/actions?query=workflow%3Abuild)
[![Documentation Status](https://readthedocs.org/projects/plenoptic/badge/?version=latest)](https://plenoptic.readthedocs.io/en/latest/?badge=latest)
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/main/STABILITY-BADGES.md#alpha)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3995057.svg)](https://doi.org/10.5281/zenodo.3995057)
[![codecov](https://codecov.io/gh/LabForComputationalVision/plenoptic/branch/main/graph/badge.svg?token=EDtl5kqXKA)](https://codecov.io/gh/LabForComputationalVision/plenoptic)
[![Tutorials Status](https://github.com/LabForComputationalVision/plenoptic/workflows/tutorials/badge.svg)](https://github.com/LabForComputationalVision/plenoptic/actions?query=workflow%3Atutorials)
[![Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LabForComputationalVision/plenoptic/main?filepath=examples)

`plenoptic` is a python library that provides tools to help researchers
better understand their models by using optimization to synthesize novel 
informative images. These images allow us to build our intuition for what
features the model ignores and what it is sensitive to. These synthetic 
images can then be used in psychophysics experiments for further investigation.

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
  
  
## Installation

The following instructions will work on Linux or Mac. 

Assuming git is installed, you can clone the repository:

```
git clone https://github.com/LabForComputationalVision/plenoptic.git
```

Navigate to the folder, create a new virtual environment, and install the
package:

```
cd plenoptic
conda create -n plenoptic python==3.10
conda activate plenoptic
pip install -e .
```

### Jupyter

If you wish to locally run the notebooks, you will need to install `jupyter` and
`ipywidgets` (you can also run them in the cloud using
[Binder](https://mybinder.org/v2/gh/LabForComputationalVision/plenoptic/main?filepath=examples)).

The easiest way to do this is to install jupyter in the same environment as `plenoptic`. 

``` sh
conda activate plenoptic
conda install -c conda-forge jupyterlab ipywidgets
```

## Getting started

Once you've set everything up appropriately, navigate to the example
directory, start up JupyterLab (which will open in a new browser tab),
and start exploring the notebooks! The notebooks contain examples and tutorials, and have been numbered
in a recommended order. 

```
cd examples/
jupyter lab
```

## Contributing

For info on how to contribute, see the [CONTRIBUTING](CONTRIBUTING.md)
file, including info on how to test the package and build its
documentation.
