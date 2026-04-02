#!/usr/bin/env python3
"""Used by check_versions.yml workflow to find supported python versions."""

import sys

import tomllib

with open(sys.argv[1], "rb") as f:
    toml = tomllib.load(f)

classifier_str = "Programming Language :: Python ::"
supported_version = [
    line.replace(classifier_str, "").strip()
    for line in toml["project"]["classifiers"]
    if classifier_str in line
]
print(",".join(supported_version))
