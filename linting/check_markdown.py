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

xrefs = []
for p in paths:
    with open(p) as f:
        md = f.read()
    if xr := re.findall(r"\[`.*?`\]\(.*?\)", md):
        xrefs.append((p, xr))

if xrefs:
    print("The following markdown documents' crossreferences are misformatted!")
    for p, xr in xrefs:
        print(f"{p}:")
        for x in xr:
            print(f"\t{x}")
