(reproduce)=

# Reproducibility and Compatibility

## Reprodubility

`plenoptic` includes several results reproduced from the literature and aims to facilitate reproducible research. However, we are limited by our dependencies and PyTorch, in particular, comes with the [caveat](https://pytorch.org/docs/stable/notes/randomness.html) that "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds" (quote from the [v1.12](https://pytorch.org/docs/1.12/notes/randomness.html) documentation).

This means that you should note the `plenoptic` version and the `pytorch` version your synthesis used in order to guarantee reproducibility (some versions of `pytorch` will give consistent results with each other, but it's not guaranteed and hard to predict). We do not believe reproducibility depends on the python version or any other packages.

However, in general, the CPU and GPU will always give different results, and you may also end up with different outputs on GPU devices with different CUDA or driver versions.

We reproduce several results from the literature and validate these as part of our tests. We are therefore aware of the following changes that broke reproducibility:

- PyTorch 1.8 and 1.9 give the same results, but 1.10 changes results in changes, probably due to the difference in how the sub-gradient for `torch.min` and `torch.max` are computed ([see this PR](https://github.com/plenoptic-org/plenoptic/pull/96#issuecomment-973318291)).
- PyTorch 1.12 breaks reproducibility with 1.10 and 1.11, unclear why ([see this issue](https://github.com/plenoptic-org/plenoptic/issues/165)).

(compat)=
## Compatibility

While we try to maintain compatibility between `plenoptic` versions, `plenoptic` is under active development and so the API of its objects may change. Similar to the comments on reproducibility above, we cannot guarantee that you will be able to load a `plenoptic` object saved with a different version of `plenoptic` or `pytorch`. The following notes known breaking changes and how, if possible, to load your object anyway.

Note that you should always be able to load in the saved object using `pytorch` directly, like so:

If your object was saved in plenoptic 1.1 or earlier:

```python
import torch
plen_object = torch.load("saved_plen_object.pt", weights_only=False)
```

Note that setting `weights_only=False` allows arbitrary code execution and thus is unsafe --- only run the above code when you know the contents of and trust the saved file! (See [related issue](https://github.com/plenoptic-org/plenoptic/issues/313) for more details.)

If your object was saved in plenoptic 1.2 or later, `weights_only=False` is unnecessary:

```python
import torch
plen_object = torch.load("saved_plen_object.pt")
```

The object you have loaded is a dictionary with strings as keys. Once you've loaded your object, you can extract the outcome of your synthesis (e.g., by looking at the `"metamer"` key). [Post an issue](https://github.com/plenoptic-org/plenoptic/issues/new/choose) if you need help!

### Breaking change to load in plenoptic 1.2

Prior to plenoptic 1.2, we were saving python functions and pytorch optimization objects. This was incompatible with a breaking change made to {func}`torch.load` in `pytorch` version 2.6. Unfortunately, there is no way to make an object saved prior to 1.2 compatible with later `plenoptic` releases; see [](compat) for how to extract the outputs of your synthesis.

### FutureWarning in load in plenoptic 1.4

A small change was made to the {class}`Metamer <plenoptic.synthesize.metamer.Metamer>` and {class}`MADCompetition <plenoptic.synthesize.mad_competition.MADCompetition>` APIs in `plenoptic` 1.4. You will be able to load {class}`Metamer <plenoptic.synthesize.metamer.Metamer>` and {class}`MADCompetition <plenoptic.synthesize.mad_competition.MADCompetition>` objects saved with version 1.2 and 1.3 for some time, but doing so will raise a `FutureWarning` and this compatibility will eventually be removed.

In order to make an object compatible with future releases, you can either load it in with `plenoptic` 1.4 and re-save it, or do the following:

```python
import torch
old_save = torch.load("old_save.pt")
old_save["_current_loss"] = None
torch.save(old_save, "old_save.pt")
# and then load as normal
```
