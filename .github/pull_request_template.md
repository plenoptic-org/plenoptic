**Describe the change in this PR at a high-level**

e.g.,
> This PR adds a new model, originally published in [CITATION], which ...

**Link any related issues, discussions, PRs**

Include links to any related issues, discussion or PRs using GitHub's `#` syntax. e.g.,

> This closes issue #350

**Outstanding questions / particular feedback**

List any questions you would like input on or parts of the PR you want reviewers to pay particular attention to, e.g.,

> - I think this can be made more efficient, any ideas?
> - Not sure if my tests make sense, make sure to check them out.

**Describe changes**

In a list, describe the changes you made. This should be more granular than the description at the beginning of the PR. These descriptions can be short, but this would also be the place to describe why something seemingly unrelated was included. e.g.,

> - added tests of new model
> - added docs describing new model
> - updated old tests for new API
> - small change to synthesis to be compatible with new API
> - while implementing this, a new version of torch was released, which required a small update to perceptual_distance.py

**Benchmarking**

If your test involves improvements to the efficiency of any part of the code base, please include some benchmarking results, to show the performance before (i.e., in the `main` branch) and after. See [PR #377](https://github.com/plenoptic-org/plenoptic/pull/377) for an example.

**Checklist**

Affirm that you have done the following:

- [ ] I have described the changes in this PR, following the template above.
- [ ] I have added any necessary tests.
- [ ] I have added any necessary documentation. This includes docstrings, updates to existing files found in `docs/`, or (for large changes) adding new files to the `docs/` folder.
- [ ] If I noticed any typos (my own or others), I have updated `plenoptic-dict.txt` to include those typos and suggested fixes.
- [ ] If a reproducible example was added: I have taken all required steps to ensure reproducibility, including setting the seed, using float64, calling `torch.use_deterministic_algorithms(True)`. See "Reproducibility and Compatibility" page in the docs for details.
- [ ] If a public new class or function was added: I have double-checked that it is present in the API docs, adding it to one of the `rst` files in `docs/api/` or adding a new file as necessary.
