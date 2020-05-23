# CONTRIBUTING

`plenoptic` is a python library of tools to help researchers better
understand their models. While the majority of authors are from NYU's
[Lab for Computational Vision](https://www.cns.nyu.edu/~lcv/), we
welcome other contributors!

First, please check out the [Code of Conduct](CODE_OF_CONDUCT.md) and
read it before going any further. You may also want to check out the
[README](README.md) for a longer overview of the project and how to
get everything installed, the tutorials (Jupyter notebooks found in
the `examples/` folder) for some examples of how to interact with the
library, and the
[Roadmap](https://github.com/LabForComputationalVision/plenoptic/projects/1)
for our current plans.

If you encounter any issues with `plenoptic`, please open an
[issue](https://github.com/LabForComputationalVision/plenoptic/issues)!
We have a template for bug reports, following it (so you provide the
necessary details) will make solving it easier. Right now, we have
enough on our plate that we're not considering any enhancements or new
features -- we're focusing on implementing what we plan to do.

If you'd like to help improve `plenoptic`, we have a bunch of issues
we're working through. For people who are not already familiar with
the project, it would be most helpful to go through the tutorials,
README, and documentation and let us know if anything is unclear, what
needs more detail (or clearer writing), etc. But if you think you can
make progress on one of the existing issues, please give it a try.

In order to submit changes, create a branch or fork of the project,
make your changes, add documentation and tests, and submit a [Pull
Request](https://github.com/LabForComputationalVision/plenoptic/pulls). The
amount and form of documentation to add depends on the size of the
submitted changes. For a significant change (a new model or synthesis
method), please include a new tutorial notebook that walks through how
to use them. For enhancements of existing methods, you can probably
just modify the existing tutorials and add documentation. If unsure,
ask! For docstrings, we follow
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)
style. See later in this file for more details on how to run the tests
and build the documentation (if you create a branch on the main repo,
TravisCI will run tests automatically whenever you push, so you don't
need to worry about running them locally -- but I'm not sure if
everyone can view them?).

COMMUNICATION CHANNELS?

RECOGNITION MODEL?

CONTACT INFO?

## Testing

from the [pytest documentation](http://doc.pytest.org/en/latest/usage.html):

- Run tests by keyword expressions:

```
pytest -k "MyClass and not method"
```

This will run tests which contain names that match the given string expression, which can include Python operators
that use filenames, class names and function names as variables. The example above will run `TestMyClass.test_something`
but not `TestMyClass.test_method_simple`.

- To run a specific test within a module:

```
pytest test_mod.py::test_func
```
Another example specifying a test method in the command line:

```
pytest test_mod.py::TestClass::test_method
```

## Build the documentation

NOTE: We currently don't have a readthedocs page set up, because they
don't support private repos for free. Once we make this repo public,
we'll set one up.

So for now, in order to view the documentation, it must be built
locally. You would do this if you've made changes locally to the
documentation (or the docstrings) that you would like to examine
before pushing. The virtual environment required to do so is defined
in `docs/environment.yml`, so to create that environment and build the
docs, do the following from the project's root directory:

```
# install sphinx and required packages to build documentation
conda env create -f docs/environment.yml
# activate the environment
conda activate plenoptic_docs
# install plenoptic
pip install -e .
# build documentation
cd docs/
sphinx-apidoc -f -o . ../plenoptic
make html
```

The index page of the documentation will then be located at
`docs/_build/html/index.html`, open it in your browser to navigate
around.

The `plenoptic_docs` environment you're creating contains the package
`sphinx` and several extensions for it that are required to build the
documentation. You also need to install `plenoptic` from your local
version so that `sphinx` can import the library and grab all of the
docstrings (you're installing the local version so you can see all the
changes you've made).

And then whenever you want to recreate / update your local
documentation, run:

```
conda activate plenoptic_docs
cd docs/
sphinx-apidoc -f -o . ../plenoptic
make html
```
