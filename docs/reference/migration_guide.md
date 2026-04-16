```{eval-rst}
:html_theme.sidebar_secondary.remove:
```

(migrating-guide)=
# Plenoptic 2.0 Migration Guide

Plenoptic version 2.0 introduces a substantial change to the package API: every class and function has been moved, in an attempt to flatten the hierarchy. The names, signature, and usage of individual classes and functions, with few exceptions, have not changed (see [](plotting-func-changes) for the exceptions). Therefore, it should be possible to update your plenoptic 1.0 code to work with 2.0 with only some replacements.

A [](migration-script) is provided below which should hopefully simplify this migration. We also recommend reading through the [](plotting-func-changes) to see if any of the changes to plotting functions are applicable for your usage. Finally, a [](migration-table) can be found on this page which summarizes all changes.

If you have any problems with this process, please open [a Q&A discussion](https://github.com/plenoptic-org/plenoptic/discussions) with "Migration help" in the title, and we'll help with this process!

(plotting-func-changes)=
## Plotting function changes

All plotting functions in plenoptic are now present within the `plot` module. This includes those that operates on tensors (e.g., {func}`plenoptic.plot.imshow`, previously `plenoptic.imshow`) and synthesis objects (e.g., {func}`plenoptic.plot.metamer_loss`, previously `plenoptic.synthesize.metamer.plot_loss`). Therefore, to avoid name collisions and redundancies (e.g., `plenoptic.plot.plot_loss`) and to clarify which functions operate on which synthesis objects, some of the function names have been changed. See [](migration-table) and search for `plot.` for a list.

Additionally, the acceptable values for the `included_plots` arguments for {func}`~plenoptic.plot.metamer_synthesis_status`, {func}`~plenoptic.plot.metamer_animshow`, {func}`~plenoptic.plot.mad_synthesis_status`, and {func}`~plenoptic.plot.mad_animshow` have all changed to reflect the current names of their component functions. Consult their docstrings for details. (These argument names cannot be remapped automatically by the following [](migration-script) and so need to be updated manually.)

(migration-script)=
## Migration script

The following script was used by the developers to update the plenoptic codebase, documentation, and tests during the migration. Hopefully, it will help with your migration, though we make no guarantees.

:::{admonition} Backup your code!
:class: important

The script in this section will make changes **without** asking for confirmation. Therefore it is highly recommended that you either have backups of any script you are modifying or are using version control, so you can easily see all the resulting changes.

:::

This script only changes fully resolvable plenoptic objects. For example, it will replace `plenoptic.synth.Metamer` with {class}`plenoptic.Metamer`, but will not touch `from plenoptic.synthesize import Metamer`.

If you use the module aliases shown in our previous documentation (i.e., `po` for `plenoptic`, `synth` for `synthesize` <!-- skip-lint -->, and `simul` for `simulate`), this script will also handle them. If you use non-standard aliases (e.g., `import plenoptic as plen`), it will not.

To use, copy the following code block into a python script called `plenoptic_rename_api.py` and run from within a virtual environment with `plenoptic>=2.0.0`, passing it whatever files you would like to change. For example: `python plenoptic_rename_api.py my_plenoptic_code.py` or `python plenoptic_rename_api.py my_project/*.py`.

```
from plenoptic import _api_change
import itertools
import pathlib
import sys

deprecated = {}
UPDATED_API = _api_change.API_CHANGE
UPDATED_API.update(_api_change.SYNTH_PLOT_FUNCS)
UPDATED_API.update(_api_change.PLOT_FUNCS)

# check all possible combinations of the module aliases
module_aliases = []
for i in range(1, len(_api_change.MODULE_ALIASES)+1):
    for mods in itertools.combinations(_api_change.MODULE_ALIASES, i):
        module_aliases.append({k: _api_change.MODULE_ALIASES[k] for k in mods})

for p in sys.argv[1:]:
    p = pathlib.Path(p)
    file_contents = p.read_text()
    for old_func, new_func in UPDATED_API.items():
        file_contents = file_contents.replace(old_func, new_func)
        for aliases in module_aliases:
            old_func_check = old_func
            new_func_check = new_func
            for mod, alias in aliases.items():
                old_func_check = old_func_check.replace(mod, alias)
                new_func_check = new_func_check.replace(mod, alias)
                file_contents = file_contents.replace(old_func_check, new_func_check)
    for dep_func in _api_change.DEPRECATED:
        if dep_func in file_contents:
            if dep_func in deprecated:
                deprecated[dep_func].append(p)
            else:
                deprecated[dep_func] = [p]
        for aliases in module_aliases:
            dep_func_check = dep_func
            for mod, alias in aliases.items():
                dep_func_check = dep_func_check.replace(mod, alias)
                if dep_func_check in file_contents:
                    if dep_func_check in deprecated:
                        deprecated[dep_func_check].append(p)
                    else:
                        deprecated[dep_func_check] = [p]
    p.write_text(file_contents)

if deprecated:
    print("The following deprecated functions were found:")
    for dep_func, dep_files in deprecated.items():
        print(f"\t{dep_func}: {dep_files}")
```


(migration-table)=
## Table of API Changes


The table below shows functions as they were called in plenoptic's 1.x releases in the first column, and as they are called in `plenoptic>=2.0.0` in the second. Note that for many functions and classes, there were multiple ways of calling them in plenoptic 1.x (e.g., `plenoptic.synthesize.Metamer` and `plenoptic.synthesize.metamer.Metamer`). This redundancy has been removed in `plenoptic>=2.0.0`.

:::{csv-table}
:header-rows: 1
:class: search-table
:file: migration_table.csv
:::

The following functions are deprecated and no longer accessible:

:::{csv-table} Deprecated functions
:file: deprecated_table.csv
:::
