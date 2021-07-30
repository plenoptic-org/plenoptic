#! /usr/bin/env python

from setuptools import setup
from importlib import util

# specify plenoptic.__version__
plenoptic_version_spec = util.spec_from_file_location(
    "plenoptic_version", "plenoptic/version.py"
)
plenoptic_version_module = util.module_from_spec(plenoptic_version_spec)
plenoptic_version_spec.loader.exec_module(plenoptic_version_module)
VERSION = plenoptic_version_module.version

setup(
    name="plenoptic",
    version=VERSION,
    description="Visual Information Processing",
    license="MIT",
    url="https://github.com/LabForComputationalVision/plenoptic",
    author="LabForComputationalVision",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
    ],
    keywords=[
        "Visual Information Processing",
        "PyTorch",
              ],
    packages=[
        "plenoptic",
        "plenoptic.metric",
        "plenoptic.simulate",
        "plenoptic.simulate.models",
        "plenoptic.simulate.canonical_computations",
        "plenoptic.synthesize",
        "plenoptic.tools",
    ],
    install_requires=[
        "numpy>=1.1",
        "torch>=1.8",
        "pyrtools>=1.0.0",
        "scipy>=1.0",
        "matplotlib>=3.1",
        "torchvision>=0.3",
        "tqdm>=4.29",
        "requests>=2.21",
        "imageio>=2.5",
        "pytest>=5.1.2",
        "scikit-image>=0.15.0",
        "dill",
        "einops>=0.3.0"
    ],
    tests="tests",
)
