```{eval-rst}
:html_theme.sidebar_secondary.remove:
```

(migrating-guide)=
# Plenoptic 2.0 Migration Guide

Plenoptic version 2.0 introduces a substantial change to the package API: every class and function has been moved, in an attempt to flatten the hierarchy. The names, signature, and usage of individual classes and functions, with few exceptions, have not changed (see [](plotting-func-changes) for the exceptions). Therefore, it should be possible to update your plenoptic 1.0 code to work with 2.0 with only some replacements.

A [](migration-script) is provided below which should hopefully simplify this migration. We also recommend reading through the [](plotting-func-changes) to see if any of the changes to plotting functions are applicable for your usage. Finally, a [](migration-table) can be found on this page which summarizes all changes.

If you have any problems with this process, please open [a discussion](https://github.com/plenoptic-org/plenoptic/discussions/new?category=plenoptic-2-0-migration-help), and we'll help with this process!

:::{contents}
:local:
:::

(plotting-func-changes)=
## Plotting function changes

All plotting functions in plenoptic are now present within the `plot` module. This includes those that operates on tensors (e.g., {func}`plenoptic.plot.imshow`, previously `plenoptic.imshow`) and synthesis objects (e.g., {func}`plenoptic.plot.metamer_loss`, previously `plenoptic.synthesize.metamer.plot_loss`). Therefore, to avoid name collisions and redundancies (e.g., `plenoptic.plot.plot_loss`) and to clarify which functions operate on which synthesis objects, some of the function names have been changed. See [](migration-table) and search for `plot.` for a list.

Additionally, the acceptable values for the `included_plots` arguments for {func}`~plenoptic.plot.metamer_synthesis_status`, {func}`~plenoptic.plot.metamer_animshow`, {func}`~plenoptic.plot.mad_synthesis_status`, and {func}`~plenoptic.plot.mad_animshow` have all changed to reflect the current names of their component functions. Consult their docstrings for details. (These argument names cannot be remapped automatically by the following [](migration-script) and so need to be updated manually.)

(migration-script)=
## Migration script

The developers wrote a [little tool](https://github.com/plenoptic-org/plenoptic-migrate) to update the plenoptic codebase, documentation, and tests during the migration to plenoptic 2.0. You are welcome to use it on your own code, though we make no guarantees.

The recommended way to use the tool is via [uvx](https://docs.astral.sh/uv/#tools) or [pipx](https://pipx.pypa.io/stable/), both of which will create a temporary isolated virtual environment for the tool:

:::::{tab-set}
::::{tab-item} pipx
```{code-block} console
$ pipx run plenoptic-migrate --help
```
::::

::::{tab-item} uvx
```{code-block} console
$ uvx plenoptic-migrate --help
```
::::
:::::

View the tool's help or [README](https://github.com/plenoptic-org/plenoptic-migrate) for details on how to use.

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
