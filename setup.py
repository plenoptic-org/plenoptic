#! /usr/bin/env python

from setuptools import setup, Extension
import importlib
import os

# copied from kymatio's setup.py: https://github.com/kymatio/kymatio/blob/master/setup.py
plenoptic_version_spec = importlib.util.spec_from_file_location('plenoptic_version',
                                                                'plenoptic/version.py')
plenoptic_version_module = importlib.util.module_from_spec(plenoptic_version_spec)
plenoptic_version_spec.loader.exec_module(plenoptic_version_module)
VERSION = plenoptic_version_module.version

setup(
    name='plenoptic',
    version='0.1',
    description='Visual Information Processing',
    license='MIT',
    url='https://github.com/pehf/plenoptic',
    author='LabForComputationalVision',
    author_email='pef246@nyu.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.7'],
    keywords='Visual Information Processing',
    packages=['plenoptic', 'plenoptic.simulate', 'plenoptic.synthesize',
              'plenoptic.tools'],
    install_requires=['numpy>=1.1',
                      'torch>=1.1',
                      'pyrtools>=0.9.1',
                      'scipy>=1.0',
                      'matplotlib<3.3',
                      'torchvision>=0.3',
                      'tqdm>=4.29',
                      'requests>=2.21',
                      'imageio>=2.5',
                      'pytest',
                      'scikit-image>=0.15.0'],
    tests='tests',
     )
