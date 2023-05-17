# CONTRIBUTING

`plenoptic` is a python library of tools to help researchers better understand
their models. While the majority of authors are members or alumni of NYU's [Lab
for Computational Vision](https://www.cns.nyu.edu/~lcv/), we welcome other
contributors!

First, please check out the [Code of Conduct](CODE_OF_CONDUCT.md) and read it
before going any further. You may also want to check out the [README](README.md)
for a longer overview of the project and how to get everything installed, the
tutorials (Jupyter notebooks found in the `examples/` folder) for some examples
of how to interact with the library, and the
[documentation](https://plenoptic.readthedocs.io/).

If you encounter any issues with `plenoptic`, please open an
[issue](https://github.com/LabForComputationalVision/plenoptic/issues)!
We have a template for bug reports, following it (so you provide the
necessary details) will make solving it easier. Right now, we have
enough on our plate that we're not considering any enhancements or new
features --- we're focusing on implementing what we plan to do.

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

We try to keep all our communication on Github, and we use several channels:

-   [Discussions](https://github.com/LabForComputationalVision/plenoptic/discussions)
    is the place to ask usage questions, discuss issues too broad for a
    single issue, or show off what you've made with plenoptic.
-   If you've come across a bug, open an
    [issue](https://github.com/LabForComputationalVision/plenoptic/issues).
-   If you have an idea for an extension or enhancement, please post in the
    [ideas
    section](https://github.com/LabForComputationalVision/plenoptic/discussions/categories/ideas)
    of discussions first. We'll discuss it there and, if we decide to pursue it,
    open an issue to track progress.

## Contributing to the code

We welcome contributions to `plenoptic`! In order to contribute, please create
your own branch, make sure the tests pass, and open a Pull Request. If you're a
member of the [LCV github](https://github.com/LabForComputationalVision/), you
can create a branch directly in the repository: from within the `plenoptic`
repository:

```
git checkout -b my_cool_branch # create and switch to the branch
# make some changes
git commit -a -m "A helpful message explaining my changes" # commit your changes
git push origin my_cool_branch # push to the origin remote
```

Once you're happy with your changes, [add tests](#adding-tests) to check that
they run correctly, then make sure the rest of the [tests](#testing) all run
successfully, that your branch is up-to-date with main, and then open a pull
request (see [here](https://yangsu.github.io/pull-request-tutorial/) for a
tutorial).

If you're not a member of the LCV github, you'll need to first [create a
fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) of the
repository to your own github account, and then proceed as above.

### Style guide

- Longer, descriptive names are preferred (e.g., `x` is not an appropriate name
  for a variable), especially for anything user-facing, such as methods,
  attributes, or arguments.
- Any public method or function must have a complete type-annotated docstring
  (see [below](#docstrings) for details). Hidden ones do not *need* to have
  complete docstrings, but they probably should.

### Adding models or synthesis methods

In addition to the above, see the documentation for a description of
[models](https://plenoptic.readthedocs.io/en/latest/models.html) and [synthesis
objects](https://plenoptic.readthedocs.io/en/latest/synthesis.html). Any new
models or synthesis objects will need to meet the requirements outlined in those
pages.

## Testing

Before running tests locally, you'll need
[ffmpeg](https://ffmpeg.org/download.html) installed on your system.

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
`test_`, it must have an `.py` extension, and it must be contained within the
`tests` directory. Assuming you do that, our github actions will automatically
find it and add it to the tests-to-run.

### Testing notebooks

We use [jupyter
execute](https://jupyter.readthedocs.io/en/latest/running.html#using-a-command-line-interface)
to test our notebooks and make sure everything runs. You can run it locally to
try and debug some errors (though errors that result from environment issues
obviously will be harder to figure out locally); `jupyter execute` is part of
the standard `jupyter` install as long as you have `nbclient>=0.5.5`.

Similar to adding new [test scripts](#adding-tests), you don't need to
explicitly add new tutorials to `ci.yml` to be tested: as long as your notebook
is in the `examples/` directory and has an `ipynb` extension, our github actions
will automatically find it and test it.

If your notebook needs additional files to run, you should add a [conditional
job](https://docs.github.com/en/actions/using-jobs/using-conditions-to-control-job-execution)
to download them. If you need to upload them, we recommend uploading a tarball
to the [Open Science Framework](https://osf.io/), and they can then be
downloaded using `wget` and extracted. See `Download TID2013 dataset` in
`ci.yml` for an example. We have a [single OSF project](https://osf.io/ts37w/)
containing all the files we've uploaded and will probably want to include yours
there as well.

If your notebook takes more than ~10 minutes on a github runner, you should find
a way to use reduce it for tests. The goal of the tests is only to check that
each cell runs successfully. For example, the Portilla-Simoncelli texture model
notebook runs several metamer syntheses to completion. This allows the user to
better understand how the model works and confirm that we are able to reproduce
the paper, as well as serving as a convenient way for the developers to ensure
that we maintain this over time. However, the tests are *only intended* to
ensure that everything runs, so we can reduce the number of iterations those
metamer instances run for. We do this using
[papermill](https://papermill.readthedocs.io/), which requires several steps:

- Add a cell to the top of the notebook (under the import cell), add the
  parameter tag (see [papermill
  documentation]https://papermill.readthedocs.io/en/latest/usage-parameterize.html()),
  and create a variable for each synthesis duration (e.g., `vgg16_synth_max_iter
  = 1000`).
- Where synthesis is called later in the notebook, replace the number with the
  variable (e.g., `metamer.synthesize(max_iter=vgg16_max_iter)`).
- Add a conditional job to `ci.yml` for your notebook which installs papermill
  and calls it with the syntax: `papermill ${{ matrix.notebook }} ${{
  matrix.notebook }}_output.ipynb -p PARAM1 VAL1 -p PARAM2 VAL2 -k python3 --cwd
  examples/`, replacing `PARAM1 VAL1` and `PARAM2 VAL2` as appropriate (e.g.,
  `vgg16_synth_max_iter 10`; note that you need a `-p` for each parameter and
  you should change nothing else about that line). See the block with `if: ${{
  matrix.notebook == 'examples/Demo_Eigendistortion.ipynb' }}` for an example.
  
A similar procedure could be used to reduce the size of an image or other steps
that could similarly reduce the total time necessary to run a notebook.

### Test parameterizations and fixtures

#### Parametrize

If you have many variants on a test you wish to run, you should probably make
use of pytests' `parametrize` mark. There are many examples throughout our
existing tests (and see official [pytest
docs](https://docs.pytest.org/en/stable/parametrize.html)), but the basic idea
is that you write a function that takes an argument and then use the
`@pytest.mark.parametrize` decorator to show pytest how to iterate over the
arguments. For example, instead of writing:

```python
def test_basic_1():
    assert int('3') == 3

def test_basic_2():
    assert int('5') == 5
```

You could write:

```python
@pytest.mark.parametrize('a', [3, 5])
def test_basic(a):
    if a == '3':
        test_val = 3
    elif a == '5':
        test_val = 5
    assert int(a) == test_val

```

This starts to become very helpful when you have multiple arguments you wish to
iterate over in this manner.

#### Fixtures

If you are using an object that gets used in multiple tests (such as an image or
model), you should make use of fixtures to avoid having to load or initialize
the object multiple times. Look at `conftest.py` to see those fixtures available
for all tests, or you can write your own (though pay attention to the
[scope](https://docs.pytest.org/en/stable/fixture.html#scope-sharing-fixtures-across-classes-modules-packages-or-session)).
For example, `conftest.py` contains several images that you can use for your
tests, such as `basic_stimuli`, `curie_img`, or `color_img`. To use them, simply
add them as arguments to your function:

```python
def test_img(curie_img):
    img = po.load_images('data/curie.pgm')
    assert torch.allclose(img, curie_img)
```

WARNING: If you're using fixtures, make sure you don't modify them in your test
(or you reset them to their original state at the end of the test). The fixture
is a single object that will get reused across tests, so modifying it will lead
to unexpected behaviors in other tests depending on which tests were run and
their execution order.

#### Combining the two

You can combine fixtures and parameterization, which is helpful for when you
want to test multiple models with a synthesis method, for example. This is
slightly more complicated and relies on pytest's [indirect
parametrization](https://docs.pytest.org/en/stable/example/parametrize.html#indirect-parametrization)
(and requires `pytest>=5.1.2` to work properly). For example, `conftest.py` has
a fixture, `model`, which accepts a string and returns an instantiated model on
the right device. Use it like so:

```python
@pytest.mark.parametrize('model', ['SPyr', 'LNL'], indirect=True)
def test_synth(curie_img, model):
    met = po.synth.Metamer(curie_img, model)
    met.synthesize()
```

This model will be run twice, once with the steerable pyramid model and once
with the Linear-Nonlinear model. See the `get_model` function in `conftest.py`
for the available strings. Note that unlike in the simple
[parametrize](#parametrize) example, we add the `indirect=True` argument here.
If we did not include that argument, `model` would just be the strings `'SPyr'`
and `'LNL'`!

## Documentation

### Adding documentation

Documentation in `plenoptic` is built using Sphinx and lives on readthedocs. If
that means nothing to you, don't worry!

Documentation comes in two types: `.rst` files (reST, the markup language used
by Sphinx, see
[here](https://www.sphinx-doc.org/en/main/usage/restructuredtext/basics.html)
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
 
#### Images and plots

You can include images in `.rst` files in the documentation as well. Simply
place them in the `docs/images/` folder and use the `figure` role, e.g.,:

```rst
.. figure:: images/path_to_my_image.svg
   :figwidth: 100%
   :alt: Alt-text describing my image.
   
   Caption describing my image.
   
```

To refer to it directly, you may want to use the [numref
role](https://docs.readthedocs.io/en/stable/guides/cross-referencing-with-sphinx.html#the-numref-role)
(which has been enabled for our documentation).

If you have plots or other images generated by code that you wish to include,
you can include them in the file directly without either saving the output in
`docs/images/` or turning the page into a notebook. This is useful if you want
to show something generated by code but the code itself isn't the point. We
handle this with [matplotlib plot
directive](https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html)
(which has already been enabled). Add a python script to `docs/scripts/` and
write a function that creates the matplotlib figure you wish to display. Then,
in your documentation, add:

```rst
.. plot:: scripts/path_to_my_script.py my_useful_plotting_function

   Caption describing what is in my plot.
```

Similar to figures, you can use `numref` to refer to plots as well.

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
make html
```

### Add tutorials

We build tutorials as Jupyter notebooks so that they can be launched in Binder
and people can play with them on their local machine. In order to include them
in the built docs, add a `nblink` file to the `docs/tutorials/` directory. We
check for this during the tests, so you won't be able to merge your pull request
into `main` unless you've done this!

This is a json file that should contain the path to the notebook, like so, for
`docs/tutorials/my_awesome_tutorial.nblink`:

```
{
    "path": "../../examples/my_tutorial.ipynb"
}
```

note that you *cannot* have a trailing comma there, because json is very
particular.
If you have extra media (such as images) that are rendered in the notebook, you
need to specify them as well, otherwise they won't render in the documentation:

```
{
    "path": "../../examples/my_tutorial.ipynb",
    "extra-media": ["../../examples/assets/my_folder"]
}

```

note that `extra-media` must be a list, even with a single path.

See the [nbsphinx-link](https://github.com/vidartf/nbsphinx-link)
page for more details.

*NOTE*: In order for the `toctree` formatting to work correctly, your notebook
should have exactly one H1 title (i.e., line starting with a single `#`), but
you can have as many lower-level titles as you'd like. If you have multiple H1
titles, they'll each show up as different tutorials. If you don't have an H1
title, it won't show up at all.
