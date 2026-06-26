import pathlib
import re
import sys

# these notebooks have a "download script" admonition on the top, instead of download
# notebook, because the notebook itself is a tangled mess (due to using tabs and showing
# several different paths)
SCRIPT_NBS = ["feature_extractor"]

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
    if filename in SCRIPT_NBS:
        if not re.findall(
            rf"Download the script: \[`{filename}.py`\]\(../../scripts/{filename}.py\)",
            md,
        ):
            fails.append(p)
    elif (
        not re.findall("Run this notebook yourself!", md)
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
