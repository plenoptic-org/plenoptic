# Contributing

`plenoptic` is a python library of tools to help researchers better understand
their models. We welcome and encourage contributions from everyone!

First, please check out the [Code of Conduct](https://github.com/plenoptic-org/plenoptic/blob/main/CODE_OF_CONDUCT.md) and read it
before going any further. You may also want to check out the [main page of the
documentation](index-doc) for a longer
overview of the project and how to get everything installed, as well as pointers
for further reading, depending on your interests.

If you encounter any issues with `plenoptic`, first search the existing
[issues](https://github.com/plenoptic-org/plenoptic/issues) and
[discussions](https://github.com/plenoptic-org/plenoptic/discussions)
to see if there's already information to help you. If not, please open a new
[issue](https://github.com/plenoptic-org/plenoptic/issues)! We have
a template for bug reports, and following it (so you provide the necessary
details) will make solving your problem much easier.

If you'd like to help improve `plenoptic`, there are many ways you can
contribute, from improving documentation to writing code. For those not already
familiar with the project, it can be very helpful for us if you go through the
tutorials, README, and documentation and let us know if anything is unclear,
what needs more detail (or clearer writing), etc. For those that want to
contribute code, we also have many
[issues](https://github.com/plenoptic-org/plenoptic/issues) that we
are working through. If you would like to work on one of those, please give it a
try!

In order to submit changes, create a branch or fork of the project, make your
changes, add documentation and tests, and submit a [Pull
Request](https://github.com/plenoptic-org/plenoptic/pulls). See
[contributing to the code below](contributing-to-the-code) for more details on
this process.

We try to keep all our communication on Github, and we use several channels:

-   [Discussions](https://github.com/plenoptic-org/plenoptic/discussions)
    is the place to ask usage questions, discuss issues too broad for a
    single issue, or show off what you've made with plenoptic.
-   If you've come across a bug, open an
    [issue](https://github.com/plenoptic-org/plenoptic/issues).
-   If you have an idea for an extension or enhancement, please post in the
    [ideas
    section](https://github.com/plenoptic-org/plenoptic/discussions/categories/ideas)
    of discussions first. We'll discuss it there and, if we decide to pursue it,
    open an issue to track progress.

## Supported versions

`plenoptic` tries to follow
[SPEC-0](https://scientific-python.org/specs/spec-0000/): we support python
versions are supported for 3 years following initial release. This means that we
support three python feature versions (e.g., 3.10, 3.11, and 3.12) at any one
time and that we'll transition between versions during the fourth quarter of
each year. We run our CPU tests on all three versions, and the GPU tests and
documentation build use the middle version.

(contributing-to-the-code)=
## Contributing to the code

### Contribution workflow

We welcome contributions to `plenoptic`! In order to contribute, please create
your own branch, make sure the tests pass, and open a Pull Request. We follow
the [GitHub
Flow](https://www.gitkraken.com/learn/git/best-practices/git-branch-strategy#github-flow-branch-strategy)
workflow: no one is allowed to push to the `main` branch, all development
happens in separate feature branches (each of which, ideally, implements a
single feature, addresses a single issue, or fixes a single problem), and these
get merged into `main` once we have determined they're ready. Then, after enough
changes have accumulated, we put out a new release, adding a new tag which
increments the version number, and uploading the new release to PyPI (see
[releases](releases) for more details).

In addition to the information that follows, [Github](https://docs.github.com/en/get-started/quickstart/github-flow) (unsurprisingly) has good information on this workflow, as does the [Caiman package](https://github.com/flatironinstitute/CaImAn/blob/main/CONTRIBUTING.md) (though note that the Caiman uses **git** flow, which involves a separate develop branch in addition to main).

Before we begin: everyone finds `git` confusing the first few (dozen) times they encounter it! And even people with a hard-won understanding frequently look up information on how it works. If you find the following difficult, we're happy to help walk you through the process. Please [post on our GitHub discussions page](https://github.com/plenoptic-org/plenoptic/discussions) to get help.

#### Creating a development environment

You'll need a local installation of `plenoptic` which keeps up-to-date with any changes you make. To do so, you will need to fork and clone `plenoptic`:

1. Go to the [plenoptic repo](https://github.com/plenoptic-org/plenoptic/) and click on the `Fork` button at the top right of the page. This creates a copy of plenoptic in your Github account.
2. You should then clone *your fork* to your local machine and create an editable installation. To do so, follow the instructions for an editable install found in our [docs](source), replacing `git clone https://github.com/plenoptic-org/plenoptic.git` with `git clone https://github.com/<YourUserName>/plenoptic.git`.
3. Add the `upstream` branch: `git remote add upstream https://github.com/plenoptic-org/plenoptic.git`. At this point, you have two remotes: `origin` (your fork) and `upstream` (the canonical version). You won't have permission to push to upstream (only `origin`), but this makes it easy to keep your `plenoptic` up to date with the canonical version by pulling from upstream: `git pull upstream`.

You should probably also install all the optional dependencies, so that you can run tests, build the documentation, and run the jupyter notebooks locally. To do so, run `pip install -e ".[docs,dev,nb]"` from within the copy of `plenoptic` on your machine (see [docs](jupyter-doc) of our documentation for information on how to set up jupyter if you don't want an extra copy of it in this environment).

#### Creating a new branch

As discussed above, each feature in `plenoptic` is worked on in a separate branch. This allows us to have multiple people developing multiple features simultaneously, without interfering with each other's work. To create your own branch, run the following from within your `plenoptic` directory:

```bash
# switch to main branch of your fork
git checkout main
# update your fork from your github
git pull origin main
# ensure your fork is in sync with the canonical version
git pull upstream main
# update your fork's main branch with any changes from upstream
git push origin main
# create and switch to the branch
git checkout -b my_cool_branch
```

Then, create new changes on this branch and, when you're ready, add and commit them:

```bash
# stage the changes
git add src/plenoptic/the_file_you_changed.py
# commit your changes
git commit -m "A helpful message explaining my changes"
# push to the origin remote
git push origin my_cool_branch
```

If you aren't comfortable with `git add`, `git commit`, `git push`, I recommend the [Software Carpentry git lesson](https://swcarpentry.github.io/git-novice/).

#### Contributing your change back to plenoptic

You can make any number of changes on your branch. Once you're happy with your changes, [add tests](adding-tests) to check that they run correctly and [add documentation](adding-documentation), then make sure the existing [tests](testing) all run successfully, that your branch is up-to-date with main, and then open a pull request by clicking on the big `Compare & pull request` button that appears at the top of your fork after pushing to your branch (see [here](https://intersect-training.org/collaborative-git/pr.html) for a tutorial).

Your pull request should include information on what you changed and why, referencing any relevant issues or discussions, and highlighting any portion of your changes where you have lingering questions (e.g., "was this the right way to implement this?") or want reviewers to pay special attention. You can look at previous closed pull requests to see what this looks like.

At this point, we will be notified of the pull request and will read it over. We will try to give an initial response quickly, and then do a longer in-depth review, at which point you will probably need to respond to our comments, making changes as appropriate. We'll then respond again, and proceed in an iterative fashion until everyone is happy with the proposed changes. This process can take a while! (The more focused your pull request, the less time it will take.)

If your changes are integrated, you will be added as a Github contributor and as one of the authors of the package. Thank you for being part of `plenoptic`!

### Code Style and Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting our Python code to maintain a consistent code style and catch potential errors early. We run ruff as part of our CI (using pre-commit, see below) and non-compliant code will not be merged! You can see the version of `ruff` that we are currently using in the `.pre-commit-config.yaml` file in the project root .

#### Using Ruff

Ruff is a fast and comprehensive Python formatter and linter that checks for common style and code quality issues. It combines multiple tools, like black, Pyflakes, pycodestyle, isort, and other linting rules into one efficient tool, which are specified in `pyproject.toml`. Before submitting your code, make sure to run Ruff to catch any issues. See other sections of this document for how to use `nox` and `pre-commit` to simplify this process.

Ruff has two components, a [formatter](https://docs.astral.sh/ruff/formatter/) and a [linter](https://docs.astral.sh/ruff/linter/). Formatters and linters are both static analysis tools, but formatters "quickly check and reformat your code for stylistic consistency without changing the runtime behavior of the code", while linters "detect not just stylistic inconsistency but also potential logical bugs, and often suggest code fixes" (per [GitHub's readme project](https://github.com/readme/guides/formatters-linters-compilers)). There are many choices of formatters and linters in python; ruff aims to combine the features of many of them while being very fast.

For both the formatter and the linter, you can run ruff without any additional arguments; our configuration option are stored in the `pyproject.toml` file and so don't need to be specified explicitly.

##### Formatting:

`ruff format` is the primary entrypoint to the formatter. It accepts a list of files or directories, and formats all discovered Python files:
```bash
ruff format                   # Format all files in the current directory.
ruff format path/to/code/     # Format all files in `path/to/code` (and any subdirectories).
ruff format path/to/file.py   # Format a single file.
```
For the full list of supported options, run `ruff format --help`.

##### Using Ruff for Linting:

To run Ruff on your code:
```bash
ruff check .
```
It'll then tell you which lines are violating linting rules and may suggest that some errors are automatically fixable.

To automatically fix lintint errors, run:

```bash
ruff --fix .
```

Be careful with **unsafe fixes**, safe fixes are symbolized with the tools emoji and are listed [here](https://docs.astral.sh/ruff/rules/)!

#### Ignoring Ruff Linting
In some cases, it may be acceptable to suppress lint errors, for example when too long lines (code `E501`) are desired because otherwise the url might not be readable anymore. These ignores will be evaluated on a case-by-case basis.
You can do this by adding the following to the end of the line:

```bash
This line is tooooooo long. # noqa: E501
```
If you want to suppress an error across an entire file, do this at the top of the file:
```bash
# ruff: noqa: E501
Below is my python script
...
...
And any line living in this file can be as long as it wants  ...
...
```


In some cases, you want to not only suppress the error message a linter throws but actually _disable_ a linting rule. An example might be if the import order matters and running `isort` would mess with this.
In these cases, you can introduce an [action comment](https://docs.astral.sh/ruff/linter/#action-comments) like this such that ruff does _not_ sort the following packages alphabetically:

```bash
import numpy as np # isort: skip
import my_package as mp # isort: skip
```

For more details, refer to the [documentation](https://docs.astral.sh/ruff/linter/#error-suppression).

#### General Style Guide Recommendations:

- Longer, descriptive names are preferred (e.g., `x` is not an appropriate name
  for a variable), especially for anything user-facing, such as methods,
  attributes, or arguments.
- Any public method or function must have a complete type-annotated docstring
  (see [below](docstrings) for details). Hidden ones do not *need* to have
  complete docstrings, but they probably should.

#### Pre-Commit Hooks:  Identifying simple issues before submission to code review (and how to ignore those)

[Pre-commit](https://pre-commit.com/) hooks are useful for the developer to check if all the linting and formatting rules (see Ruff above) are honored _before_ committing. That is, when you commit, pre-commit hooks are run and auto-fixed where possible (e.g., trailing whitespace). You then need to add _again_ if you want these changes to be included in your commit. If the problem is not automatically fixable, you will need to manually update your code before you are able to commit.

Using pre-commit is optional. We use [pre-commit.ci](https://pre-commit.ci/) to run pre-commit as part of PRs (auto-fixing wherever possible), but it may simplify your life to integrate pre-commit into your workflow.

In order to use pre-commit, you must install the `pre-commit` package into your development environment, and then install the hooks:

```bash
pip install pre-commit
pre-commit install
```

See [pre-commit docs](https://pre-commit.com/) for more details.

After installation, should you want to ignore pre-commit hooks for some reason (e.g., because you have to run to a meeting and so don't have time to fix all the linting errors but still want your changes to be committed), you can add `--no-verify` to your commit message like this:
```bash
git commit -m <my commit message> --no-verify
```


### Adding models or synthesis methods

In addition to the above, see the documentation for a description of
[models](models-doc) and [synthesis
objects](synthesis-objects). Any new
models or synthesis objects will need to meet the requirements outlined in those
pages.

(releases)=
### Releases

We create releases on Github, deploy on / distribute via
[pypi](https://pypi.org/), and try to follow [semantic
versioning](https://semver.org/):

> Given a version number MAJOR.MINOR.PATCH, increment the:
> 1. MAJOR version when you make incompatible API changes
> 2. MINOR version when you add functionality in a backward compatible manner
> 3. PATCH version when you make backward compatible bug fixes

When doing a new release, the following steps must be taken:
1. In a new PR:
  - Update all the [binder](https://mybinder.org) links, which are of the form
    `https://mybinder.org/v2/gh/plenoptic-org/plenoptic/X.Y.Z?filepath=examples`,
    which are found in `README.md` and `index.md`. Note that the version tag
    must match the github tag (specified in the next step) or the link won't
    work.
  - Update `docs/_static/version_switcher.json`. You will need to add a section
    for the new release and move the `preferred=true` line to that section.
2. After merging the above PR into the `main` branch, [create a Github
   release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
   with a new tag matching that used in the binder link above: `X.Y.Z`. Creating
   the release will trigger the deployment to pypi, via our `deploy` action
   (found in `.github/workflows/deploy.yml`). The built version will grab the
   version tag from the Github release, using
   [setuptools_scm](https://github.com/pypa/setuptools_scm).

Note that the binder link I have been unable to find a way to make binder use the latest github release tag directly (or make [binder](https://mybinder.org) use a `latest` tag, so ensure they match!

Shortly after the deploy to pypi goes through (typically within a day), a PR will be automatically opened on the [conda-forge/plenoptic-feedstock](https://github.com/conda-forge/plenoptic-feedstock) repo. After merging that PR, the [plenoptic version on conda-forge](https://anaconda.org/conda-forge/plenoptic) will also be updated

(testing)=
## Testing

Before running tests locally, you'll need
[ffmpeg](https://ffmpeg.org/download.html) installed on your system, as well as
the `dev` optional dependencies (i.e., you should run `pip install -e ".[dev]"`
from within your local copy of `plenoptic`).

To run all tests, run `pytest` from the main `plenoptic` directory. This
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

View the [pytest documentation](https://doc.pytest.org/en/latest/usage.html) for
more info.

### Running pytest with non-standard cache directories

When running tests on some machines (e.g., nodes of the Flatiron cluster), the default cache directories used by some of the libraries will not exist, so running the tests like normal will result in errors that complain about directories not existing or not having permission to create directories in e.g., `/home/wbroderick`.

To avoid this problem, we can use environment variables to control the behavior of these libraries. To do so for pooch (downloading test files), torch (downloading pretrained models), and matplotlib (caching config files), prepend the following to your pytest command: `PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic` (or replace `~/.cache/` with some other directory).

> [!NOTE]
> This may also be helpful when building the documentation.

See relevant torch [1](https://docs.pytorch.org/docs/stable/notes/cuda.html#just-in-time-compilation) and [2](https://docs.pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved), [pooch](https://www.fatiando.org/pooch/latest/user-defined-cache.html), and [matplotlib](https://matplotlib.org/stable/install/environment_variables_faq.html#envvar-MPLCONFIGDIR) docs.

### Using nox to simplify testing and linting
This section is optional but if you want to easily run tests in an isolated environment
using the [nox](https://nox.thea.codes/en/stable/) command-line tool.

Before proceeding, you'll need to install `nox`. You can do this in your plenoptic environment, but it is more common to install it system-wide using `pipx`: `pipx install nox`. This installs `nox` into a globally-available isolated environment, see [pipx docs](https://pipx.pypa.io/stable/) for more details.

You will also need to install `pyyaml` in the same environment as `nox`. If you used `pipx`, then run `pipx inject nox pyyaml`.

To run all tests, formatters, and linters through `nox`, from the root folder of the
plenoptic package, execute the following command,

```bash
nox
```

`nox` will read the configuration from the `noxfile.py` script.

If you only want to run an individual session (e.g., lint or test), you can first check which sessions are available with the following command:

```bash
nox -l
```

Then you can use

```bash
nox -s <your_nox_session>
```
to run the session of your choice.

Here are some examples:

If you want to run just the tests:

```bash
nox -s tests
```

for running only the linters,

```bash
nox -s lint
```

`nox` offers a variety of configuration options, you can learn more about it from their
[documentation](https://nox.thea.codes/en/stable/config.html).

Note that nox works particularly well with pyenv, discussed later in this file, which makes it easy to install the multiple python versions used in testing.

#### Multi-python version testing with pyenv
Sometimes, before opening a pull-request that will trigger the `.github/workflow/ci.yml` continuous
integration workflow, you may want to test your changes over all the supported python versions locally.

Handling multiple installed python versions on the same machine can be challenging and confusing.
[pyenv](https://github.com/pyenv/pyenv) is a great tool that really comes to the rescue. Note that pyenv just handles python versions --- virtual environments have to be handled separately, using [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)!

This tool doesn't come with the package dependencies and has to be installed separately. Installation instructions
are system specific but the package readme is very details, see
[here](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation).

Follow carefully the instructions to configure pyenv after installation.

Once you have the package installed and configured, you can install multiple python version through it.
First get a list of the available versions with the command,

```bash
pyenv install -l
```

Install the python version you need. For this example, let's assume we want `python 3.10.11` and `python 3.11.8`,

```bash
pyenv install 3.10.11
pyenv install 3.11.8
```

You can check which python version is currently set as default, by typing,

```bash
pyenv which python
```

And you can list all available versions of python with,

```bash
pyenv versions
```
If you want to run `nox` on multiple python versions, all you need to do is:

1. Set your desired versions as `global`.
    ```bash
    pyenv global 3.11.8 3.10.11
    ```
    This will make both version available, and the default python will be set to the first one listed
    (`3.11.8` in this case).
2. Run nox specifying the python version as an option.
    ```bash
    nox -p 3.10
    ```

Note that `noxfile.py` lists the available option as keyword arguments in a session specific manner.

As mentioned earlier, if you have multiple python version installed, we recommend you manage your virtual environments through pyenv using the [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) extension.

This tool works with most of the environment managers including (`venv` and `conda`).
Creating an environment with it is as simple as calling,

```bash
pyenv virtualenv my-python my-enviroment
```

Here, `my-python` is the python version, chosen from those listed by `pyenv versions`, and `my-environment` is your new environment name.

If `my-python` has `conda` installed, it will create a conda environment, if not, it will use `venv`.

You can list the virtual environment only with,

```bash
pyenv virtualenvs
```

And you can uninstall an environment with,

```bash
pyenv uninstall my-environment
```

(adding-tests)=
### Adding tests

New tests can be added in any of the existing `tests/test_*.py` scripts. Tests
should be functions, contained within classes. The class contains a bunch of
related tests (e.g., metamers, metrics), and each test should ideally be a unit
test, only testing one thing. The classes should be named `TestSomething`, while
test functions should be named `test_something` in snakecase.

If you're adding a substantial bunch of tests that are separate from the
existing ones, you can create a new test script. Its name must begin with
`test_`, it must have an `py` extension, and it must be contained within the
`tests` directory. Assuming you do that, our github actions will automatically
find it and add it to the tests-to-run.

Note that we also require that tests raise no warnings (see [PR
#335](https://github.com/plenoptic-org/plenoptic/pull/335)). This allows to stay
on top of deprecation warnings from our dependencies. There are several ways to
avoid warnings in tests, in order from most-to-least preferred:

- Write tests such that they avoid warnings. For example, all synthesis methods
  call {func}`~plenoptic.tools.validate.validate_model`, which will raise a warning if the model is in "training
  mode" (e.g., `model.training` exists and is True). The default behavior of
  `torch.nn.Module` objects is to be in training mode after initialization.
  Thus, we call `model.eval` before passing a model to a synthesis method in a
  test.
- Selectively ignore the warning on a given test using
  `@pytest.mark.filterwarnings`. See our tests for an example or the [pytest
  documentation](https://docs.pytest.org/en/stable/how-to/capture-warnings.html#pytest-mark-filterwarnings) for an explanation.
- Configure pytest to ignore the warning for all tests by updating
  `filterwarnings` in `pyproject.toml` (they must come after `"error"`). These
  should only include warnings that are temporary, such as deprecation warnings
  that we raise or warnings that have been fixed upstream but not released yet.

### Testing notebooks

We use [jupyter
execute](https://jupyter.readthedocs.io/en/latest/running.html#using-a-command-line-interface)
to test our notebooks and make sure everything runs. You can run it locally to
try and debug some errors (though errors that result from environment issues
obviously will be harder to figure out locally); `jupyter execute` is part of
the standard `jupyter` install as long as you have `nbclient>=0.5.5`.

### Long-running synthesis and tutorial notebooks

Occasionally, we want to include one or more synthesis calls within a tutorial
notebook that take a long time to run (for example, because we're reproducing a
result from the literature). In order to avoid having the documentation build
take a long time, we instead write a regression test (in
`tests/test_uploaded_files.py`), which runs the synthesis, saves the output, and
compares it against a cached version stored in our OSF project. See
`tests/test_uploaded_files.py` to see how these tests look. Some important notes:

- The new tests should be added to the `TestTutorialNotebooks` class in
  `test_uploaded_files.py`.
- The synthesize call should be shown in the notebook, in a code block (unlike a
  `code-cell`, `code-block` are not run). This `code-block` should be preceded
  by a markdown comment giving the class and name of the corresponding test.
  with a name that corresponds to the name of the test. So, if our test was
  called `test_berardino_onoff` and found within the `TestDemoEigendistortion`
  notebook, the corresponding code block should look like:

  ````
  <!-- TestDemoEigendistortion.test_berardino_onoff -->
  ```{code-block} python
  eigendist_f.synthesize(k=3, method="power", max_iter=2000)
  ```
  ````
- This block will be checked whether it is part of the corresponding test
  (literally, with `in`).
  - If a variable has a different name in the block and in the test, the
    preceding comment should include square brackets containing a
    comma-separated list of the replacements: `<!--
    TestDemoEigendistortion.test_berardino_onoff[eigendist_f:eig] -->`.
  - If the test has lines we want to ignore (because they're test-specific),
    they should contain `lint_ignore` somewhere on the line, **not in a
    comment** (e.g., `this_variable_lint_ignore = 100` but not `this_variable =
    100 # lint_ignore`).
- `src/plenoptic/data/fetch.py` needs the hash and the URL slug of each new
  file, so make sure to update them. The hash can be computed by calling
  `openssl sha256 path/to/file` on the command line.

We have a linter that checks the conditions above.

`pytest` does not run the tests found under `TestTutorialNotebooks` by default,
since they take a long time. In order to run them, you must explicitly set the
environment variable `RUN_REGRESSION_SYNTH=1` when calling pytest.

#### Exact reproducibility

Exact reproducibility with pytorch is hard. See [issue #368](https://github.com/plenoptic-org/plenoptic/issues/368) for some details, but the tl;dr is: you should not expect to get the same outputs (or even, within floating point precision) when running synthesis for long enough (seems to be > 1000 iterations) on devices with different CUDA versions and driver versions. Small differences in the output of e.g., `torch.einsum` / `torch.matmul` will lead to small differences in the gradient, which will accumulate and eventually lead to fairly different optimization outputs.

To deal with this, your regression tests should save their output into the `uploaded_files` folder after synthesis (and before checking). The contents of that folder will be made available as [Jenkins artifacts](https://www.jenkins.io/doc/pipeline/tour/tests-and-artifacts/), which you can then download after the test. This will allow you to download the output of any failing test, manually verify the results look good, and then upload them to the OSF to test against.

### Test parameterizations and fixtures

(parametrize)=
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
tests, such as `basic_stim`, `curie_img`, or `color_img`. To use them, simply
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
a fixture, `model` <!-- skip-lint -->, which accepts a string and returns an instantiated model on
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
[parametrize](parametrize) example, we add the `indirect=True` argument here.
If we did not include that argument, `model` <!-- skip-lint --> would just be the strings `'SPyr'`
and `'LNL'`!

## Documentation

(adding-documentation)=
### Adding documentation

The amount and form of documentation that need to be added alongside a change
depends on the size of the submitted change. For a significant change (a new
model or synthesis method), please include a new tutorial notebook that walks
through how to use them. For enhancements of existing methods, you can probably
just modify the existing tutorials and add documentation. If unsure, ask!

Documentation in `plenoptic` is built using Sphinx on some of Flatiron's Jenkins
runners and hosted on GitHub pages. If that means nothing to you, don't worry!

All of our documentation is written as markdown files, with the extension `md`. We use the [myst parser](https://myst-parser.readthedocs.io/), along with [myst-nb](https://myst-nb.readthedocs.io). Both process markdown files, but `myst-nb` allows us to write [text-based notebooks](https://myst-nb.readthedocs.io/en/latest/authoring/text-notebooks.html), with python code that gets executed when the documentation is built.

The text-based notebooks are tutorials and show how to use the various functions and classes contained in the package. If you add or change a substantial amount of code, please add a tutorial showing how to use it.

In all markdown files, you should try to use sphinx's cross-reference syntax to refer to code objects in API documentation whenever one is mentioned. For example, you should refer to the {class}`~plenoptic.synthesize.metamer.Metamer` class as

```
{class}`~plenoptic.synthesize.metamer.Metamer`
```

You should similarly refer to code objects in other packages (e.g., pytorch and matplotlib), though the syntax is different. See [myst-parser docs](https://myst-parser.readthedocs.io/en/latest/syntax/cross-referencing.html#reference-roles) for more details and the existing documentation for more examples. As part of the pull request review process, we run linters that will check for missing cross-references. The only objects that can be referred to simply as `monospace` font are function arguments and generic attributes / method (e.g., saying that plenoptic models must have a `forward` <!-- skip-lint --> method). The linter will ignore all monospace font that have the word "argument" or "keyword" after them (e.g., "the `scales` keyword" or "the `scales` argument") or an html comment containing "skip-lint" (e.g., "the `scales` <!-- skip-lint --> method"; html comments are not rendered in sphinx).

The regular markdown files contain everything else, especially discussions about why you should use some code in the package and the theory behind it, and should all be located in one of the subfolders within the `docs/` directory. Decide which subfolder to place it in (ask for your help if you're unsure) and add it to that subfolder's `index.md` by adding the name of the file (without extension) to the `toctree` block.

In order for table of contents to look good, your `md` file must be well structured. All markdown files (text-based notebooks and regular) must have a single H1 header (you can have as many sub-headers as you'd like).

You should [build the docs yourself](build-the-documentation) to ensure it looks correct before pushing.

#### Images and plots

You can include images in `md` files in the documentation as well. Simply
place them in the `docs/images/` folder and use the `figure` role, e.g.,:

```md
:::{figure} images/path_to_my_image.svg
:figwidth: 100%
:alt: Alt-text describing my image.

Caption describing my image.
:::
```

To refer to it directly, you may want to use the [numref
role](https://docs.readthedocs.io/en/stable/guides/cross-referencing-with-sphinx.html#the-numref-role)
(which has been enabled for our documentation).

If you have plots or other images generated by code that you wish to include,
you can include them in the file directly without either saving the output in
`docs/_static/images/` or turning the page into a notebook. This is useful if you want
to show something generated by code but the code itself isn't the point. We
handle this with matplotlib's {doc}`plot directive <matplotlib:api/sphinxext_plot_directive_api>`
(which has already been enabled). Add a python script to `docs/scripts/` and
write a function that creates the matplotlib figure you wish to display. Then,
in your documentation, add:

````md
```{eval-rst}
.. plot:: scripts/path_to_my_script.py my_useful_plotting_function

   Caption describing what is in my plot.
```
````

Similar to figures, you can use `numref` to refer to plots as well.

#### API Documentation

All public functions and classes must be included on the API documentation page.
Therefore, if you add a new public function or class, make sure to add it to
`docs/api.rst` in an appropriate location. If this is not done,
`linting/check_apidocs.py` will fail (this check is included in our pre-commit
config and thus is required to pass for a PR to merge).

If you add a new source file (e.g., `src/plenoptic/synthesize/new_method.py`),
you will also need to add it to `docs/api_modules.rst`. If this is not done,
sphinx will raise an error when building the documentation. You also need to add
that file to the hidden toctree at the top of `docs/api.rst`.

(docstrings)=
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

### Doctests

All public-facing functions and classes should include {mod}`doctest`, which are the standard python way of showing short code examples in docstrings. These should be included in their own `Examples` section of the docstring. Every docstring should include at least one example, which shows the most common way of interacting with the function / class. Additional examples should be included where helpful, to show other common ways of interacting with the object (e.g., setting optional arguments), with brief descriptions describing what each example is doing.

A function's Examples section must run independently from those of other functions (i.e., it can't reuse an object defined in a different function), but different blocks withint he section can depend on each other (so that e.g., you don't have to reimport plenoptic in each block).

Our doctests are tested using [pytest](https://docs.pytest.org/en/stable/how-to/doctest.html) and sphinx builds them as part of the documentation. Some notes about this:

- If you would like to include a figure, use matplotlib's {doc}`plot directive <matplotlib:api/sphinxext_plot_directive_api>`. That means, your example should be structured like:

```python
.. plot::
   :context: close-figs

   >>> import plenoptic as po
   >>> # more example code here...
```

- `:context: close-figs` is important to make sure that the figures are independent across examples (this should probably be `:context: reset` for the first plot directive in a given docstring, `close-figs` thereafter). However, unfortunately,  only sphinx knows how to interpret this directive; pytest ignores it. That means the doctests must be written in such a way that they will not fail if they are run with open figures lying around. One could easily start their doctests by closing any open figures, but this generally goes against the principle of making these examples as compact and useful as possible. Unfortunately, I have not found a good general solution here.


(build-the-documentation)=
### Build the documentation

NOTE: If you just want to read the documentation, you do not need to do this;
documentation is built automatically, pushed to the
[plenoptic-documentation](https://github.com/plenoptic-org/plenoptic-documentation)
github repo and published at http://docs.plenoptic.org/.

However, it can be built locally as well. You would do this if you've made changes locally to the documentation (or the docstrings) that you would like to examine before pushing. All additional requirements are included in the `[docs]` optional dependency bundle, which you can install with `pip install plenoptic[docs]`.

Then, to build the documentation, run: `make -C docs html O="-T -j auto"`. (`-j auto` tells sphinx to parallelize the build, using as many cores as possible, a specific number can be set.)

By default, the text-based notebooks (see [earlier](adding-documentation)) are not run because they take a longish time to do so, especially if you do not have a GPU. In order to run all of them, prepend `RUN_NB=1` to the `make` command above. In order to run specific notebooks, set `RUN_NB` to a globbable comma-separated string in the above, e.g., `RUN_NB=Metamer,MAD` to run `docs/user_guide/synthesis/Metamer`, `docs/user_guide/synthesis/MAD_Competition_1`, and `docs/user_guide/synthesis/MAD_Competition_2`.

The index page of the documentation will then be located at
`docs/_build/html/index.html`, open it in your browser to navigate
around.
