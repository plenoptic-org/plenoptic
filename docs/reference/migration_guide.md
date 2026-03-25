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

Additionally, the acceptable values for the `included_plots` arguments for {func}`~plenoptic.plot.metamer_synthesis_status`, {func}`~plenoptic.plot.metamer_animate`, {func}`~plenoptic.plot.mad_synthesis_status`, and {func}`~plenoptic.plot.mad_animate` have all changed to reflect the current names of their component functions. Consult their docstrings for details. (These argument names cannot be remapped automatically by the following [](migration-script) and so need to be updated manually.)

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

:::{admonition} Table generation
:class: dropdown note

The following section is automatically added to this file by downloading the following script as `gen_api_table.py` and running `python gen_api_table.py path/to/migration_guide.md` (with `plenoptic>=2.0.0`):

```
from plenoptic import _api_change
import re
import sys
import pathlib

tables = """
The table below shows functions as they were called in plenoptic's 1.x releases in the
first column, and as they are called in `plenoptic>=2.0.0` in the second.

Note that for many functions and classes, there were multiple ways of calling them in
plenoptic 1.x (e.g., `plenoptic.synthesize.Metamer` and
`plenoptic.synthesize.metamer.Metamer`). This redundancy has been removed in
`plenoptic>=2.0.0`.

```{csv-table}
:header-rows: 1
:class: search-table

"plenoptic 1.x", "plenoptic 2.0"
"""

UPDATED_API = _api_change.API_CHANGE
UPDATED_API.update(_api_change.SYNTH_PLOT_FUNCS)
UPDATED_API.update(_api_change.PLOT_FUNCS)

for k, v in UPDATED_API.items():
    k_to_include = k
    i = 0
    # find the . to split at so that the first half of the string is as close to 45
    # characters long as possible (but no longer), because I cannot come up with a good
    # way to do this in javascript/css
    while len(k_to_include.split(" ")[0]) >= 45:
        i -= 1
        k_to_include = ".".join(k.split(".")[:i]) + ". " + ".".join(k.split(".")[i:])
    tables += f'"`{k_to_include}`", "{{func}}`{v}`"\n'

tables += """```

The following functions are deprecated and no longer accessible:

```{csv-table} Deprecated functions

"""

for f in _api_change.DEPRECATED:
    tables += f'"`{f}`"\n'

tables += "```"

if len(sys.argv) != 2:
    raise Exception("Pass the path to `migration_guide.md` to append table!")
p = pathlib.Path(sys.argv[1])
migration_guide = p.read_text()

migration_guide = re.sub(r"(^## Table of API Changes).*",
                         fr"\1\n{tables}",
                         migration_guide,
                         flags=re.MULTILINE | re.DOTALL)

p.write_text(migration_guide)
```

:::

(migration-table)=
## Table of API Changes

The table below shows functions as they were called in plenoptic's 1.x releases in the
first column, and as they are called in `plenoptic>=2.0.0` in the second.

Note that for many functions and classes, there were multiple ways of calling them in
plenoptic 1.x (e.g., `plenoptic.synthesize.Metamer` and
`plenoptic.synthesize.metamer.Metamer`). This redundancy has been removed in
`plenoptic>=2.0.0`.

```{csv-table}
:header-rows: 1
:class: search-table

"plenoptic 1.x", "plenoptic 2.0"
"`plenoptic.synthesize.metamer.Metamer`", "{func}`plenoptic.Metamer`"
"`plenoptic.synthesize.Metamer`", "{func}`plenoptic.Metamer`"
"`plenoptic.synthesize.metamer.MetamerCTF`", "{func}`plenoptic.MetamerCTF`"
"`plenoptic.synthesize.MetamerCTF`", "{func}`plenoptic.MetamerCTF`"
"`plenoptic.synthesize.eigendistortion. Eigendistortion`", "{func}`plenoptic.Eigendistortion`"
"`plenoptic.synthesize.Eigendistortion`", "{func}`plenoptic.Eigendistortion`"
"`plenoptic.synthesize.mad_competition. MADCompetition`", "{func}`plenoptic.MADCompetition`"
"`plenoptic.synthesize.MADCompetition`", "{func}`plenoptic.MADCompetition`"
"`plenoptic.simulate.models. portilla_simoncelli.PortillaSimoncelli`", "{func}`plenoptic.models.PortillaSimoncelli`"
"`plenoptic.simulate.PortillaSimoncelli`", "{func}`plenoptic.models.PortillaSimoncelli`"
"`plenoptic.simulate.models.frontend. LinearNonlinear`", "{func}`plenoptic.models.LinearNonlinear`"
"`plenoptic.simulate.LinearNonlinear`", "{func}`plenoptic.models.LinearNonlinear`"
"`plenoptic.simulate.models.frontend. LuminanceGainControl`", "{func}`plenoptic.models.LuminanceGainControl`"
"`plenoptic.simulate.LuminanceGainControl`", "{func}`plenoptic.models.LuminanceGainControl`"
"`plenoptic.simulate.models.frontend. LuminanceContrastGainControl`", "{func}`plenoptic.models.LuminanceContrastGainControl`"
"`plenoptic.simulate. LuminanceContrastGainControl`", "{func}`plenoptic.models.LuminanceContrastGainControl`"
"`plenoptic.simulate.models.frontend.OnOff`", "{func}`plenoptic.models.OnOff`"
"`plenoptic.simulate.OnOff`", "{func}`plenoptic.models.OnOff`"
"`plenoptic.simulate.models.naive.Identity`", "{func}`plenoptic.models.Identity`"
"`plenoptic.simulate.Identity`", "{func}`plenoptic.models.Identity`"
"`plenoptic.simulate.models.naive.Linear`", "{func}`plenoptic.models.Linear`"
"`plenoptic.simulate.Linear`", "{func}`plenoptic.models.Linear`"
"`plenoptic.simulate.models.naive.Gaussian`", "{func}`plenoptic.models.Gaussian`"
"`plenoptic.simulate.Gaussian`", "{func}`plenoptic.models.Gaussian`"
"`plenoptic.simulate.models.naive. CenterSurround`", "{func}`plenoptic.models.CenterSurround`"
"`plenoptic.simulate.CenterSurround`", "{func}`plenoptic.models.CenterSurround`"
"`plenoptic.metric.naive.mse`", "{func}`plenoptic.metric.mse`"
"`plenoptic.metric.model_metric. model_metric_factory`", "{func}`plenoptic.metric.model_metric_factory`"
"`plenoptic.metric.perceptual_distance.ssim`", "{func}`plenoptic.metric.ssim`"
"`plenoptic.metric.perceptual_distance.ms_ssim`", "{func}`plenoptic.metric.ms_ssim`"
"`plenoptic.metric.perceptual_distance.nlpd`", "{func}`plenoptic.metric.nlpd`"
"`plenoptic.metric.perceptual_distance. ssim_map`", "{func}`plenoptic.metric.ssim_map`"
"`plenoptic.metric.perceptual_distance. normalized_laplacian_pyramid`", "{func}`plenoptic.metric.normalized_laplacian_pyramid`"
"`plenoptic.simulate.canonical_computations. laplacian_pyramid.LaplacianPyramid`", "{func}`plenoptic.model_components.LaplacianPyramid`"
"`plenoptic.simulate.LaplacianPyramid`", "{func}`plenoptic.model_components.LaplacianPyramid`"
"`plenoptic.simulate.canonical_computations. steerable_pyramid_freq.SteerablePyramidFreq`", "{func}`plenoptic.model_components.SteerablePyramidFreq`"
"`plenoptic.simulate.SteerablePyramidFreq`", "{func}`plenoptic.model_components.SteerablePyramidFreq`"
"`plenoptic.simulate.canonical_computations. filters.circular_gaussian2d`", "{func}`plenoptic.model_components.circular_gaussian2d`"
"`plenoptic.simulate.circular_gaussian2d`", "{func}`plenoptic.model_components.circular_gaussian2d`"
"`plenoptic.tools.signal.rectangular_to_polar`", "{func}`plenoptic.model_components.rectangular_to_polar`"
"`plenoptic.tools.rectangular_to_polar`", "{func}`plenoptic.model_components.rectangular_to_polar`"
"`plenoptic.tools.signal.polar_to_rectangular`", "{func}`plenoptic.model_components.polar_to_rectangular`"
"`plenoptic.tools.polar_to_rectangular`", "{func}`plenoptic.model_components.polar_to_rectangular`"
"`plenoptic.simulate.canonical_computations. non_linearities.local_gain_control`", "{func}`plenoptic.model_components.local_gain_control`"
"`plenoptic.simulate.non_linearities. local_gain_control`", "{func}`plenoptic.model_components.local_gain_control`"
"`plenoptic.simulate.canonical_computations. non_linearities.local_gain_release`", "{func}`plenoptic.model_components.local_gain_release`"
"`plenoptic.simulate.non_linearities. local_gain_release`", "{func}`plenoptic.model_components.local_gain_release`"
"`plenoptic.simulate.canonical_computations. non_linearities.rectangular_to_polar_dict`", "{func}`plenoptic.model_components.rectangular_to_polar_dict`"
"`plenoptic.simulate.non_linearities. rectangular_to_polar_dict`", "{func}`plenoptic.model_components.rectangular_to_polar_dict`"
"`plenoptic.simulate.canonical_computations. non_linearities.polar_to_rectangular_dict`", "{func}`plenoptic.model_components.polar_to_rectangular_dict`"
"`plenoptic.simulate.non_linearities. polar_to_rectangular_dict`", "{func}`plenoptic.model_components.polar_to_rectangular_dict`"
"`plenoptic.simulate.canonical_computations. non_linearities.local_gain_control_dict`", "{func}`plenoptic.model_components.local_gain_control_dict`"
"`plenoptic.simulate.non_linearities. local_gain_control_dict`", "{func}`plenoptic.model_components.local_gain_control_dict`"
"`plenoptic.simulate.canonical_computations. non_linearities.local_gain_release_dict`", "{func}`plenoptic.model_components.local_gain_release_dict`"
"`plenoptic.simulate.non_linearities. local_gain_release_dict`", "{func}`plenoptic.model_components.local_gain_release_dict`"
"`plenoptic.tools.conv.correlate_downsample`", "{func}`plenoptic.model_components.correlate_downsample`"
"`plenoptic.tools.correlate_downsample`", "{func}`plenoptic.model_components.correlate_downsample`"
"`plenoptic.tools.conv.blur_downsample`", "{func}`plenoptic.model_components.blur_downsample`"
"`plenoptic.tools.blur_downsample`", "{func}`plenoptic.model_components.blur_downsample`"
"`plenoptic.tools.conv.upsample_convolve`", "{func}`plenoptic.model_components.upsample_convolve`"
"`plenoptic.tools.upsample_convolve`", "{func}`plenoptic.model_components.upsample_convolve`"
"`plenoptic.tools.conv.upsample_blur`", "{func}`plenoptic.model_components.upsample_blur`"
"`plenoptic.tools.upsample_blur`", "{func}`plenoptic.model_components.upsample_blur`"
"`plenoptic.tools.conv.same_padding`", "{func}`plenoptic.model_components.same_padding`"
"`plenoptic.tools.same_padding`", "{func}`plenoptic.model_components.same_padding`"
"`plenoptic.tools.signal.shrink`", "{func}`plenoptic.model_components.shrink`"
"`plenoptic.tools.shrink`", "{func}`plenoptic.model_components.shrink`"
"`plenoptic.tools.signal.expand`", "{func}`plenoptic.model_components.expand`"
"`plenoptic.tools.expand`", "{func}`plenoptic.model_components.expand`"
"`plenoptic.tools.signal.rescale`", "{func}`plenoptic.model_components.rescale`"
"`plenoptic.tools.rescale`", "{func}`plenoptic.model_components.rescale`"
"`plenoptic.tools.signal.add_noise`", "{func}`plenoptic.model_components.add_noise`"
"`plenoptic.tools.add_noise`", "{func}`plenoptic.model_components.add_noise`"
"`plenoptic.tools.signal.center_crop`", "{func}`plenoptic.model_components.center_crop`"
"`plenoptic.tools.center_crop`", "{func}`plenoptic.model_components.center_crop`"
"`plenoptic.tools.signal.modulate_phase`", "{func}`plenoptic.model_components.modulate_phase`"
"`plenoptic.tools.modulate_phase`", "{func}`plenoptic.model_components.modulate_phase`"
"`plenoptic.tools.signal.autocorrelation`", "{func}`plenoptic.model_components.autocorrelation`"
"`plenoptic.tools.autocorrelation`", "{func}`plenoptic.model_components.autocorrelation`"
"`plenoptic.tools.stats.variance`", "{func}`plenoptic.model_components.variance`"
"`plenoptic.tools.variance`", "{func}`plenoptic.model_components.variance`"
"`plenoptic.tools.stats.skew`", "{func}`plenoptic.model_components.skew`"
"`plenoptic.tools.skew`", "{func}`plenoptic.model_components.skew`"
"`plenoptic.tools.stats.kurtosis`", "{func}`plenoptic.model_components.kurtosis`"
"`plenoptic.tools.kurtosis`", "{func}`plenoptic.model_components.kurtosis`"
"`plenoptic.tools.data.to_numpy`", "{func}`plenoptic.to_numpy`"
"`plenoptic.tools.to_numpy`", "{func}`plenoptic.to_numpy`"
"`plenoptic.tools.data.convert_float_to_int`", "{func}`plenoptic.convert_float_to_int`"
"`plenoptic.tools.convert_float_to_int`", "{func}`plenoptic.convert_float_to_int`"
"`plenoptic.data.fetch.fetch_data`", "{func}`plenoptic.data.fetch_data`"
"`plenoptic.data.fetch.DOWNLOADABLE_FILES`", "{func}`plenoptic.data.DOWNLOADABLE_FILES`"
"`plenoptic.tools.data.make_disk`", "{func}`plenoptic.data.disk`"
"`plenoptic.tools.make_disk`", "{func}`plenoptic.data.disk`"
"`plenoptic.tools.data.polar_radius`", "{func}`plenoptic.data.polar_radius`"
"`plenoptic.tools.polar_radius`", "{func}`plenoptic.data.polar_radius`"
"`plenoptic.tools.data.polar_angle`", "{func}`plenoptic.data.polar_angle`"
"`plenoptic.tools.polar_angle`", "{func}`plenoptic.data.polar_angle`"
"`plenoptic.tools.validate.remove_grad`", "{func}`plenoptic.remove_grad`"
"`plenoptic.tools.remove_grad`", "{func}`plenoptic.remove_grad`"
"`plenoptic.tools.validate.validate_model`", "{func}`plenoptic.validate.validate_model`"
"`plenoptic.tools.validate_model`", "{func}`plenoptic.validate.validate_model`"
"`plenoptic.tools.validate.validate_input`", "{func}`plenoptic.validate.validate_input`"
"`plenoptic.tools.validate_input`", "{func}`plenoptic.validate.validate_input`"
"`plenoptic.tools.validate.validate_metric`", "{func}`plenoptic.validate.validate_metric`"
"`plenoptic.tools.validate_metric`", "{func}`plenoptic.validate.validate_metric`"
"`plenoptic.tools.validate. validate_coarse_to_fine`", "{func}`plenoptic.validate.validate_coarse_to_fine`"
"`plenoptic.tools.validate_coarse_to_fine`", "{func}`plenoptic.validate.validate_coarse_to_fine`"
"`plenoptic.tools.validate. validate_convert_tensor_dict`", "{func}`plenoptic.validate.validate_convert_tensor_dict`"
"`plenoptic.tools.validate_convert_tensor_dict`", "{func}`plenoptic.validate.validate_convert_tensor_dict`"
"`plenoptic.tools.optim.set_seed`", "{func}`plenoptic.set_seed`"
"`plenoptic.tools.set_seed`", "{func}`plenoptic.set_seed`"
"`plenoptic.tools.optim.mse`", "{func}`plenoptic.optim.mse`"
"`plenoptic.tools.mse`", "{func}`plenoptic.optim.mse`"
"`plenoptic.tools.optim.l2_norm`", "{func}`plenoptic.optim.l2_norm`"
"`plenoptic.tools.l2_norm`", "{func}`plenoptic.optim.l2_norm`"
"`plenoptic.tools.optim.relative_sse`", "{func}`plenoptic.optim.relative_sse`"
"`plenoptic.tools.relative_sse`", "{func}`plenoptic.optim.relative_sse`"
"`plenoptic.tools.optim. portilla_simoncelli_loss_factory`", "{func}`plenoptic.optim.portilla_simoncelli_loss_factory`"
"`plenoptic.tools. portilla_simoncelli_loss_factory`", "{func}`plenoptic.optim.portilla_simoncelli_loss_factory`"
"`plenoptic.tools.optim. groupwise_relative_l2_norm_factory`", "{func}`plenoptic.optim.groupwise_relative_l2_norm_factory`"
"`plenoptic.tools. groupwise_relative_l2_norm_factory`", "{func}`plenoptic.optim.groupwise_relative_l2_norm_factory`"
"`plenoptic.tools.optim.penalize_range`", "{func}`plenoptic.regularization.penalize_range`"
"`plenoptic.tools.regularization.penalize_range`", "{func}`plenoptic.regularization.penalize_range`"
"`plenoptic.tools.penalize_range`", "{func}`plenoptic.regularization.penalize_range`"
"`plenoptic.tools.io.examine_saved_synthesis`", "{func}`plenoptic.io.examine_saved_synthesis`"
"`plenoptic.tools.examine_saved_synthesis`", "{func}`plenoptic.io.examine_saved_synthesis`"
"`plenoptic.tools.external.plot_MAD_results`", "{func}`plenoptic.external.plot_MAD_results`"
"`plenoptic.tools.plot_MAD_results`", "{func}`plenoptic.external.plot_MAD_results`"
"`plenoptic.synthesize.metamer.plot_loss`", "{func}`plenoptic.plot.metamer_loss`"
"`plenoptic.synthesize.metamer.display_metamer`", "{func}`plenoptic.plot.metamer_image`"
"`plenoptic.synthesize.metamer. plot_pixel_values`", "{func}`plenoptic.plot.metamer_pixel_values`"
"`plenoptic.synthesize.metamer. plot_representation_error`", "{func}`plenoptic.plot.metamer_representation_error`"
"`plenoptic.synthesize.metamer. plot_synthesis_status`", "{func}`plenoptic.plot.metamer_synthesis_status`"
"`plenoptic.synthesize.metamer.animate`", "{func}`plenoptic.plot.metamer_animate`"
"`plenoptic.synthesize.mad_competition. display_mad_image`", "{func}`plenoptic.plot.mad_image`"
"`plenoptic.synthesize.mad_competition. display_mad_image_all`", "{func}`plenoptic.plot.mad_image_all`"
"`plenoptic.synthesize.mad_competition. plot_loss`", "{func}`plenoptic.plot.mad_loss`"
"`plenoptic.synthesize.mad_competition. plot_loss_all`", "{func}`plenoptic.plot.mad_loss_all`"
"`plenoptic.synthesize.mad_competition. plot_pixel_values`", "{func}`plenoptic.plot.mad_pixel_values`"
"`plenoptic.synthesize.mad_competition. plot_synthesis_status`", "{func}`plenoptic.plot.mad_synthesis_status`"
"`plenoptic.synthesize.mad_competition.animate`", "{func}`plenoptic.plot.mad_animate`"
"`plenoptic.synthesize.eigendistortion. display_eigendistortion`", "{func}`plenoptic.plot.eigendistortion_image`"
"`plenoptic.synthesize.eigendistortion. display_eigendistortion_all`", "{func}`plenoptic.plot.eigendistortion_image_all`"
"`plenoptic.tools.display.imshow`", "{func}`plenoptic.plot.imshow`"
"`plenoptic.tools.imshow`", "{func}`plenoptic.plot.imshow`"
"`plenoptic.imshow`", "{func}`plenoptic.plot.imshow`"
"`plenoptic.tools.display.animshow`", "{func}`plenoptic.plot.animshow`"
"`plenoptic.tools.animshow`", "{func}`plenoptic.plot.animshow`"
"`plenoptic.animshow`", "{func}`plenoptic.plot.animshow`"
"`plenoptic.tools.display.pyrshow`", "{func}`plenoptic.plot.pyrshow`"
"`plenoptic.tools.pyrshow`", "{func}`plenoptic.plot.pyrshow`"
"`plenoptic.pyrshow`", "{func}`plenoptic.plot.pyrshow`"
"`plenoptic.tools.display.plot_representation`", "{func}`plenoptic.plot.plot_representation`"
"`plenoptic.tools.plot_representation`", "{func}`plenoptic.plot.plot_representation`"
"`plenoptic.tools.display.clean_stem_plot`", "{func}`plenoptic.plot.clean_stem_plot`"
"`plenoptic.tools.clean_stem_plot`", "{func}`plenoptic.plot.clean_stem_plot`"
"`plenoptic.tools.display.update_plot`", "{func}`plenoptic.plot.update_plot`"
"`plenoptic.tools.update_plot`", "{func}`plenoptic.plot.update_plot`"
```

The following functions are deprecated and no longer accessible:

```{csv-table} Deprecated functions

"`plenoptic.tools.display.clean_up_axes`"
"`plenoptic.tools.clean_up_axes`"
"`plenoptic.tools.display.rescale_ylim`"
"`plenoptic.tools.rescale_ylim`"
"`plenoptic.tools.display.update_stem`"
"`plenoptic.tools.update_stem`"
```
