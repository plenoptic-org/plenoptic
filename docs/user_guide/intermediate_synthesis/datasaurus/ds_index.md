(datasaurus-index)=
# Using Penalized Metamer Synthesis to Recreate the Datasaurus Dozen

:::{admonition} Penalty function usage
:class: warning

These pages assume familiarity with the basics of using penalty function in metamer synthesis, as shown in the [](how-to-penalty) notebook.

:::

- discuss original results
- overview of model
- download and visualize them
- refer back to paper (in particular, "can't match stats directly" and "some shapes are more difficult than others")
- then show our results overview
      - I think can loop through the pt files in the datasaurus_metamers tarball, use torch.load and just grab _saved_metamer
      - note that we're not being very careful about efficiency in these notebooks, because the synthesis is so fast. if input was larger or model was slower, would be more important
      - and we're not trying to exactly match datasaurus dozen, just conceptually
      - importantly: we don't actually need our metric value to be very low here, just need them to look like our penalty target
      - compare to original success metric: matches each of those to first N decimal points
      - and say something like, if you come up with a penalty to do a better job at star, away, thick lines or find a new penalty that does something else interesting


::::{card}
:::{toctree}
:maxdepth: 1

ds_circle.md
ds_bullseye.md
ds_hlines.md
ds_vlines.md
ds_slantup.md
ds_slantdown.md

:::
::::
