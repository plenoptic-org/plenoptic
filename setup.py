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

def readlines(fn):
    with open(fn) as f:
        return f.readlines()

setup(
    name="plenoptic",
    version=VERSION,
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    description="Python library for model-based stimulus synthesis.",
    license="MIT",
    url="https://github.com/LabForComputationalVision/plenoptic",
    author="LabForComputationalVision",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
    package_data={'plenoptic.metric': ['DN_sigmas.npy', 'DN_filts.npy']},
    install_requires=readlines('jenkins/requirements.txt'),
    tests="tests",
)
