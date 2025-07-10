import pathlib
import re
import sys

objects_to_check = []
try:
    import sphobjinv as soi

    inv = soi.Inventory("docs/_build/html/objects.inv")
    objects = [d.name for d in inv.objects if d.name.startswith("plenoptic.")]
    objects_to_check = []
    for obj in objects:
        # this will give a list of strings like (for
        # plenoptic.tools.validate.validate_model):
        # plenoptic.tools.validate.validate_model,
        # tools.validate.validate_model, validate.validate_model, validate_model
        obj = [".".join(obj.split(".")[i:]) for i in range(obj.count(".") + 1)]
        objects_to_check.extend(obj)
    objects_to_check = set(objects_to_check)
except ModuleNotFoundError:
    print("sphobjinv package not found, will not check for missing cross-references")
except TypeError:
    print(
        "sphinx objects.inv file not found, will not check for missing cross-references"
    )

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
missing_xrefs = []
double_backticks = []
method_format = []
for p in paths:
    with open(p) as f:
        md = f.read()
    if xr := re.findall(r"\[`.*?`\]\(?.*?\)?", md):
        xrefs.append((p, xr))
    miss_xr = []
    for obj in objects_to_check:
        if xr := re.findall(
            rf"`{obj}\(?\)?`(?! keyword)(?! argument)(?! ?<\!-- *?skip-lint *?-->)", md
        ):
            miss_xr.append(xr)
    if miss_xr:
        missing_xrefs.append((p, miss_xr))
    mfmt = []
    # these two strings are basically the same, but one is looking for "starts
    # with ." and the second is "followed by parentheses"
    for test in [
        r"`\.[A-Za-z0-9_]+\(?\)?(?: <.*?>|`)(?! ?<\!-- *?skip-lint *?-->)",
        r"`?[A-Za-z0-9_]+\(\)(?: <.*?>|`)",
    ]:
        mfmt.extend(re.findall(test, md))
    if mfmt:
        method_format.append((p, mfmt))
    if bt := re.findall(r"(?<!`)``(?!`).*(?<!`)``(?!`)", md):
        double_backticks.append((p, bt))

if xrefs or missing_xrefs or double_backticks or method_format:
    if xrefs:
        print("The following markdown documents' crossreferences are misformatted!")
        for p, xr in xrefs:
            print(f"{p}:")
            for x in xr:
                print(f"\t{x}")
    if missing_xrefs:
        print("The following markdown documents' is missing some crossreferences!")
        for p, xr in missing_xrefs:
            print(f"{p}:")
            for x in xr:
                print(f"\t{x}")
    if double_backticks:
        print("The following markdown documents contain double backticks!")
        for p, bt in double_backticks:
            print(f"{p}: {bt}")
    if method_format:
        print(
            "The following markdown documents contain method/attribute references"
            " that contain a . or (), remove them"
        )
        for p, mf in method_format:
            print(f"{p}: {mf}")
    sys.exit(1)
