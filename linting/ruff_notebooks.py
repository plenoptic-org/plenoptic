#!/usr/bin/env python

import os
import pathlib
import subprocess
import sys

paths = []
for p in sys.argv[1:]:
    p = pathlib.Path(p)
    if p.is_dir():
        p = list(p.glob("**/*.md"))
    elif p.suffix == ".md":
        p = [p]
    else:
        p = []
    paths.extend(p)


exitcode = 0
os.makedirs("nb_tmp", exist_ok=True)
for p in paths:
    with open(p) as f:
        md = f.read()
    if not md.startswith("---\njupytext"):
        # then this isn't a markdown notebook
        continue
    ipynb_path = f"nb_tmp{os.sep}{p.stem}.ipynb"
    subprocess.run(
        ["jupytext", p, "-o", ipynb_path, "--from", "myst"], capture_output=True
    )
    proc = subprocess.run(
        ["ruff", "format", "--config=pyproject.toml", ipynb_path], capture_output=True
    )
    stdout = "\n".join(proc.stdout.decode().split("\n")).strip()
    if stdout != "1 file left unchanged":
        exitcode = 1
        print(stdout)
    proc = subprocess.run(
        ["ruff", "check", "--config=pyproject.toml", "--fix", ipynb_path],
        capture_output=True,
    )
    stdout = "\n".join(proc.stdout.decode().split("\n")).strip()
    if stdout != "All checks passed!":
        exitcode = 1
        print(stdout)
    if exitcode:
        subprocess.run(
            ["jupytext", ipynb_path, "-o", p, "--to", "myst"], capture_output=True
        )

sys.exit(exitcode)
