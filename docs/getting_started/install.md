(install-doc)=
# Installation

`plenoptic` should work on Windows, Linux, or Mac. If you have a problem with installation, please open a [bug report](https://github.com/plenoptic-org/plenoptic/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=)!

You can install `plenoptic` from [PyPI](https://pypi.org/project/plenoptic/) (the Python Package Index) or [conda-forge](https://anaconda.org/conda-forge/plenoptic), and we provide separate instructions for the two methods.

If you will be contributing to `plenoptic`, and so need an editable install and [](optional-deps), you should use [pip](pip). Otherwise, you can use whichever you are more familiar with, though we have noticed that it tends to be easier to install [pytorch](https://pytorch.org/) with GPU support using `conda`.

:::{tip}
If you are unfamiliar with python environment management, we recommend [](conda).
:::

(conda)=
## Installing with conda

If you wish to follow these instructions and do not have `conda` installed on your machine, we recommend starting with [miniforge](https://github.com/conda-forge/miniforge).

Once you have `conda` correctly installed and on your path, run the following to create a new virtual environment and install plenoptic:

```{code-block} console
$ conda create --name plenoptic pip python=3.11
$ conda activate plenoptic
$ conda install plenoptic -c conda-forge
```

(pip)=
## Installing with pip

While `conda` handles both virtual environment management **and** package installation, `pip` only installs packages; you'll need to use some other system to manage your virtual environment.

In order to avoid this problem, here we'll use python's built-in [venv](https://docs.python.org/3/library/venv.html) to manage the virtual environment with `pip`.

::::{tab-set}
:::{tab-item} Linux / MacOS
:sync: posix

```{code-block} console
$ # create the environment
$ python -m venv path/to/environments/plenoptic-venv
$ # activate the environment
$ source path/to/environment/plenoptic-venv/bin/activate
$ pip install plenoptic
```
:::
:::{tab-item} Windows
:sync: windows

```{code-block} pwsh-session
PS> # create the environment
PS> python -m venv path\to\environments\plenoptic-venv
PS> # activate the environment
PS> path\to\environment\plenoptic-venv\Scripts\activate
PS> pip install plenoptic
```
:::
::::

Note that when using `venv`, you have to decide where you'd like to place the folder containing all files related to the virtual environment. If the virtual environment is related to the development of a package, as in the [](source) section, it is typical to place them within the repository for that package. Otherwise, it is typical to place them all in a single directory somewhere so they're easy to keep track of.

:::{note}
You can use `conda` to manage your virtual environment and `pip` to install packages, but note that when using pip with a conda environment, all pip commands should come after all the conda commands (e.g., you shouldn't run `conda install matplotlib` after `pip install plenoptic`), because conda is unaware of any changes that pip makes. See [this blog post](https://www.anaconda.com/blog/using-pip-in-a-conda-environment) for more details.
:::

(source)=
### Installing from source (for developers)

You can also install plenoptic directly from source in order to have a local editable copy. This is most useful for developing (for more info, see [our contributing guide](https://github.com/plenoptic-org/plenoptic/blob/main/CONTRIBUTING.md)) or if you want to use the most cutting-edge version.

::::{tab-set}
:::{tab-item} Linux / MacOS
:sync: posix

```{code-block} console
$ git clone https://github.com/plenoptic-org/plenoptic.git
$ cd plenoptic
$ # create the environment (this is typically placed in the package's root folder)
$ python -m venv .venv
$ # activate the environment
$ source .venv/bin/activate
$ # install in editable mode with `-e` or, equivalently, `--editable`
$ pip install -e ".[dev]"
```

:::

:::{tab-item} Windows
:sync: windows

```{code-block} pwsh-session
PS> git clone https://github.com/plenoptic-org/plenoptic.git
PS> cd plenoptic
PS> # create the environment (this is typically placed in the package's root folder)
PS> python -m venv .venv
PS> # activate the environment
PS> .venv\Scripts\activate
PS> # install in editable mode with `-e` or, equivalently, `--editable`
PS> pip install -e ".[dev]"
```
:::
::::

:::{note}
   With an editable copy, which we specified with the `-e` / `--editable` flag, any changes made locally will be automatically reflected in your installation.
:::

In this setup, we're installing the `dev` optional dependencies as well as the core dependencies. This will allow you to run our tests. They are, as the name implies, optional (you can just run `pip install -e .` without the `[dev]`), but if you are developing, you will probably want to be able to run the tests. See the [](optional-deps) section for more details and the other sets of optional dependencies.


Note that, with the above setup, all files related to your virtual environment are stored in a hidden directory named `.venv` <!-- skip-lint --> within the `plenoptic/` directory you cloned. Therefore, if you delete the `plenoptic/` directory, you'll need to rerun the setup above to create a new virtual environment.

:::{attention}
To install `plenoptic` in editable mode, you need `pip >= 21.3` (see pip's [changelog](https://pip.pypa.io/en/stable/news/#id286)). If you run into [an error](https://github.com/plenoptic-org/plenoptic/issues/227) after running the `pip install -e .` command, try updating your pip version with `pip install --upgrade pip`.
:::

(optional-deps)=
### Optional dependencies

In addition to installing plenoptic and its core dependencies, you may also wish to install some of our optional dependencies. These dependencies are specified using square brackets during the `pip install` command and can be installed for either a local, editable install or one directly from PyPI:

- If you would like to run the jupyter notebooks locally: `pip install "plenoptic[nb]"` or `pip install -e ".[nb]"`. This includes `pooch` (for downloading some extra data) `torchvision` (which has some models we'd like to use), `jupyter`, and related libraries. See [](jupyter-doc) for a discussion of several ways to handle jupyter and python virtual environments. Note that you can run our notebooks in the cloud using [Binder](https://mybinder.org/v2/gh/plenoptic-org/plenoptic/1.2.0?filepath=examples), no installation required!
- If you would like to run the tests: `pip install -e ".[dev]"`. This includes `pytest` and related libraries. (This probably only makes sense if you have a local installation.)
- If you would like to locally build the documentation: `pip install -e ".[docs]"`. This includes `sphinx` and related libraries. (This probably only makes sense if you have a local installation.)

These optional dependencies can be joined with a comma, e.g., `pip install -e ".[docs,dev]"`

:::{note}
Note that `conda` does not support optional dependencies, though you can view our optional dependencies in the [pyproject.toml](https://github.com/plenoptic-org/plenoptic/blob/main/pyproject.toml#L35) file, if you wish to install them yourself.
:::

## ffmpeg and videos

Several methods in this package generate videos. There are several backends possible for saving the animations to file, see [matplotlib documentation](https://matplotlib.org/stable/api/animation_api.html#writer-classes) for more details. The default writer uses [ffmpeg](https://ffmpeg.org/download.html), which you'll need installed and on your path in order to save the videos or view them in a jupyter notebook. Depending on your system, this might already be installed, but if not, and you're using [conda to manage your environments](conda), the easiest way is probably through [conda](https://anaconda.org/conda-forge/ffmpeg): `conda install -c conda-forge ffmpeg`.

To change the backend, run `matplotlib.rcParams['animation.writer'] = writer` before calling any of the animate functions. If you try to set that `rcParam` with a random string, `matplotlib` will tell you the available choices.
