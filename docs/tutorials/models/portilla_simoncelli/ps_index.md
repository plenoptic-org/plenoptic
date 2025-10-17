# Portilla-Simoncelli Texture Model

The following notebooks introduce the Portilla-Simoncelli texture model, as first described in {cite:alp}`Portilla2000-param-textur`, and its implementation in plenoptic.

If you are unfamiliar with the model, I recommended reading the first two notebooks and then reading the notebooks that catch your interest.

If you are familiar with the model and its [matlab implementation](https://github.com/LabForComputationalVision/textureSynth), start with [](ps-basic-synthesis) and check out [](ps-matlab-difference) to see how to use the model in plenoptic and the differences between the two implementations.

If you are planning on synthesizing Portilla-Simoncelli texture metamers for your own research, please read [](ps-optimization) and [](ps-limitations).

:::{toctree}
:glob: true
:titlesonly: true

ps_intro
ps_basic_synthesis
ps_examples
ps_limitations
ps_understand_stats
ps_optimization
ps_extensions
ps_matlab_differences

:::
