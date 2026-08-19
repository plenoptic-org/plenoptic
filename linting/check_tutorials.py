import pathlib
import re
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

EXERCISE_HEADER = r"""This notebook is an exercise for practicing using plenoptic. You
should work through it on your own, either by clicking on one of the following buttons
or opening up a new notebook on your own machine and following along.

Regardless of which you choose, you should keep this page open for reference, as the
links to other pages in the documentation \(and some images\) are broken in the
downloaded and binder notebooks."""
# This allows us to use re.findall and allow for unwrapped or wrapped-like-above
# versions of the above text. Note the escaped parentheses in the string above!
EXERCISE_HEADER = EXERCISE_HEADER.replace("\n", "[\n ]")

fails = []
for p in paths:
    with open(p) as f:
        md = f.read()
    if not md.startswith("---\njupytext"):
        # then this isn't a markdown notebook
        continue
    filename = p.stem
    # handle tutorials/exercises slightly differently
    if "tutorials" not in str(p):
        if (
            not re.findall("Run this notebook yourself!", md)
            or not re.findall(f"{{nb-download}}`{filename}.ipynb`", md)
            or not re.findall(f"{{binder}}`{filename}.ipynb`", md)
        ):
            fails.append(p)
    else:
        if (
            not re.findall("Do this exercise yourself!", md)
            or not re.findall(EXERCISE_HEADER, md)
            or not re.findall(f"{{nb-download}}`{filename}.ipynb`", md)
            or not re.findall(f"{{binder}}`{filename}.ipynb`", md)
        ):
            fails.append(p)


if fails:
    print(
        "The following markdown notebooks' admonition with links to download the "
        "notebook or run it in binder are misformatted!"
    )
    for p in fails:
        print(p)
    sys.exit(1)
