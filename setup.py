#! /usr/bin/env python

from setuptools import setup, Extension

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
    packages=['plenoptic'],
    install_requires=['numpy>=1.1',
                      'torch>=1.1',
                      'pyrtools>=0.1',
                      'scipy>=1.0',
                      'matplotlib>=2.2']
     )
