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

In order to submit changes, create a branch or fork of the project, make your
changes, add documentation and tests, and submit a [Pull
Request](https://github.com/LabForComputationalVision/plenoptic/pulls). The
amount and form of documentation to add depends on the size of the submitted
changes. For a significant change (a new model or synthesis method), please
include a new tutorial notebook that walks through how to use them. For
enhancements of existing methods, you can probably just modify the existing
tutorials and add documentation. If unsure, ask! For docstrings, we follow
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style. See
later in this file for more details on how to run the tests and build the
documentation (if you create a branch on the main repo, Github Actions will run
tests automatically whenever you push, so you don't need to worry about running
them locally).

COMMUNICATION CHANNELS?

RECOGNITION MODEL?

CONTACT INFO?

## Contributing to the code

We welcome contributions to `plenoptic`! In order to contribute, please createb
your own branch, make sure the tests pass, and open a Pull Request. If you're a
member of the [LCV github](https://github.com/LabForComputationalVision/), you
can create a branch directly in the repository: from within the `plenoptic`
repository:

```
git checkout -b my_cool_branch # create the branch
# make some changes
git commit -a -m "A helpful message explaining my changes" # commit your changes
git push origin my_cool_branch # push to the origin remote
```

Once you're happy with your changes, [add tests](#adding-tests) to check that
they run correctly, then make sure the rest of the [tests](#testing) all run
successfully, that your branch is up-to-date with master, and then open a pull
request (see [here](https://yangsu.github.io/pull-request-tutorial/) for a
tutorial).

If you're not a member of the LCV github, you'll need to first [create a
fork](https://docs.github.com/en/enterprise/2.20/user/github/getting-started-with-github/fork-a-repo)
of the repository to your own github account, and then proceed as above.

## Testing

To run all tests, run `pytest tests/` from the main `plenoptic` directory. This
will take a while, as we have many tests, broken into categories. There are
several choices for how to run a subset of the tests:

- Run tests from one file: `pytest tests/test_mod.py`

- Run tests by keyword expressions: `pytest -k "MyClass and not method"`. This
  will run tests which contain names that match the given string expression,
  which can include Python operators that use filenames, class names and
  function names as variables. The example above will run
  `TestMyClass.test_something` but not `TestMyClass.test_method_simple`.

- To run a specific test within a module: `pytest tests/test_mod.py::test_func`

- Another example specifying a test method in the command line: `pytest
  test_mod.py::TestClass::test_method`

View the [pytest documentation](http://doc.pytest.org/en/latest/usage.html) for
more info.

### Adding tests 

New tests can be added in any of the existing `tests/test_*.py` scripts. Tests
should be functions, contained within classes. The class contains a bunch of
related tests (e.g., metamers, metrics), and each test should ideally be a unit
test, only testing one thing. The classes should be named `TestSomething`, while
test functions should be named `test_something` in snakecase.

If you're adding a substantial bunch of tests that are separate from the
existing ones, you can create a new test script. Its name must begin with
`test_` and it must be contained within the `tests` directory. Additionally, you
should add its name to the `build:strategy:matrix:test_script` section of
`.github/workflows/ci.yml` (this enables us to run tests in parallel). For
example, say you create a new script `tests/test_awesome.py`. You should then
open up `ci.yml` and add a new item to the `test_script` list containing
`awesome`. **Do not** edit the anything else -- if you did the above correctly,
Github Actions will correctly run your new script.

### Testing notebooks

We use [treebeard](https://github.com/treebeardtech/treebeard) to test our
notebooks and make sure everything runs. `treebeard` is still in development and
so their documentation may not be up-to-date. You can run it locally to try and
debug some errors (though errors that result from environment issues obviously
will be harder to figure out locally).

WARNING: If you run `treebeard` locally (with default options, so it doesn't use
`repo2docker` ), then it will restart, re-run, and overwrite your local
notebooks. Make sure this is okay with you.

`treebeard` uses [papermill](https://papermill.readthedocs.io/en/latest/) under
the hood, so if you have problems getting it to run at all, `papermill` may be
where to look. When running papermill locally, I've had issues with papermill
correctly determining which kernel to use (this happens since I use
[nb_conda](https://github.com/Anaconda-Platform/nb_conda_kernels) to specify
conda environments as notebook kernels), which leads to `NoSuchKernel` errors.
If you run into this problem, [this
page](https://papermill.readthedocs.io/en/latest/troubleshooting.html) has
troubleshooting info. I also got an error when trying to run the example
`jupyter kernelspec install` command given, and had to use the solution [on this
page](https://github.com/jupyter/jupyter_client/issues/72) instead:

```
conda activate my-env
python -m ipykernel install --user --name my-env --display-name "my-env"
```

And then `papermill` worked. (You may also need to specify the kernel using `-k
my-env` when calling `papermill`).

Once you've gotten that taken care of, you should be able to run `treebeard`
locally by running `treebeard run` in the `examples/` directory (which contains
all the notebooks). This will re-run all notebooks. You can specify specific
notebook using the `-n` flag.

Similar to adding new [test scripts](#adding-tests), you need to add new
tutorials to the corresponding build matrix so they can be tested. For this, go
to `.github/workflows/treebeard.yml` and add the name of the new notebook
(without the `.ipynb` extension) to the `notebook` field (under
`run:strategy:matrix`). So if your new tutorial was
`examples/100_awesome_tutorial.ipynb`, you would add `100_awesome_tutorial` as
the a new item in the `notebook` list.

## Documentation

### Adding documentation

Documentation in `plenoptic` is built using Sphinx and lives on readthedocs. If
that means nothing to you, don't worry!

Documentation comes in two types: `.rst` files (reST, the markup language used
by Sphinx, see
[here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
for a primer), which contain only text (including math) and images, and `.ipynb`
files (Jupyter notebooks), which also contain code.

Jupyter notebooks are tutorials and show how to use the various functions and
classes contained in the package, and they all should be located in the
`examples/` directory. If you add or change a substantial amount of code, please
add a tutorial showing how to use it. Once you've added a tutorial, see
[here](#add-tutorials) for how to include it on the readthedocs page.

reST files contain everything else, especially discussions about why you should
use some code in the package and the theory behind it, and should all be located
in the `docs/` directory. Add it to the table of contents in `index.rst` by
adding the name of the file (without extension, but with any subfolders) to the
`toctree` block. For example, if you add two new files, `docs/my_cool_docs.rst`
and `docs/some_folder/even_cooler_docs.rst`, you should edit `docs/index.rst`
like so:

```
.. toctree
   :maxdepth: 2
   # other existing docs
   my_cool_docs
   some_folder/even_cooler_docs
```

In order for table of contents to look good, your `.rst` file must be well
structured. Similar to [tutorials](#add-tutorials), it must have a single H1
header (you can have as many sub-headers as you'd like).

You should [build the docs yourself](#build-the-documentation) to ensure it
looks correct before pushing.
 
### Docstrings

All public-facing functions and classes should have complete docstrings, which
start with a one-line short summary of the function, a medium-length description
of the function / class and what it does, and a complete description of all
arguments and return values. Math should be included in a `Notes` section when
necessary to explain what the function is doing, and references to primary
literature should be included in a `References` section when appropriate.
Docstrings should be relatively short, providing the information necessary for a
user to use the code. Longer discussions of why you would use one method over
another, an explanation of the theory behind a method, and extended examples
should instead be part of the tutorials or documentation.

Private functions and classes should have sufficient explanation that other
developers know what the function / class does and how to use it, but do not
need to be as extensive.

We follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/) conventions
for docstring structure.

### Build the documentation

NOTE: If you just want to read the documentation, you do not need to do this;
documentation is built automatically on
[readthedocs](https://plenoptic.readthedocs.io/).

However, it can be built locally as well. You would do this if you've
made changes locally to the documentation (or the docstrings) that you
would like to examine before pushing. The virtual environment required
to do so is defined in `docs/environment.yml`, so to create that
environment and build the docs, do the following from the project's
root directory:

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

### Add tutorials

We build tutorials as Jupyter notebooks so that they can be launched in Binder
and people can play with them on their local machine. In order to include them
in the built docs, add a `nblink` file to the `docs/tutorials/` directory. This
is a json file that should contain the path to the notebook, like so, for
`docs/tutorials/my_awesome_tutorial.nblink`:

```
{
    "path": "../../examples/my_tutorial.ipynb"
}
```

note that you *cannot* have a trailing comma there, because json is very
particular. See the [nbsphinx-link](https://github.com/vidartf/nbsphinx-link)
page for more details.

Once you've done that, you should add it to our `index.rst`. Towards the bottom
of that page, you'll find a `toctree` with the caption "Tutorials and examples".
Add your new tutorial by adding the line `tutorials/my_awesome_tutorial.nblink`
after the existing ones. Then, once you run `make html`, your tutorial should
now be included!

*NOTE*: In order for the `toctree` formatting to work correctly, your notebook
should have exactly one H1 title (i.e., line starting with a single `#`), but
you can have as many lower-level titles as you'd like. If you have multiple H1
titles, they'll each show up as different tutorials. If you don't have an H1
title, it won't show up at all.

When you add a new tutorial, don't forget to add it to the `treebeard.yml` file
so it can be tested (see last paragraph of the [testing
notebooks](#testing-notebooks) section for details).
