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


fails = []
for p in paths:
    with open(p) as f:
        md = f.read()
    if not md.startswith("---\njupytext"):
        # then this isn't a markdown notebook
        continue
    filename = p.stem
    if not re.findall(f"{{nb-download}}`{filename}.ipynb`", md):
        fails.append(p)


if fails:
    print("The following markdown notebooks are missing a download notebook button!")
    for p in fails:
        print(p)
    sys.exit(1)
