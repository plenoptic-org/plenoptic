(jupyter-doc)=

# Using Jupyter to Run Example Notebooks

:::{tip}
You can run the notebooks in the cloud using [Binder](https://mybinder.org/v2/gh/plenoptic-org/plenoptic/1.2.0?filepath=examples), no installation required!
:::

## Installing jupyter and setting up the kernel

If you wish to locally run the notebooks, you will need to install `jupyter`, `ipywidgets`, and (for some of the notebooks) `torchvision` and `pooch` . There are two or three possible ways of getting a local jupyter install working with this package, depending on what tool you are using to manage your virtual environments and how you wish to handle them (see [](install-doc) for how to use `conda` or `pip and venv` to setup a virtual environment and install plenoptic).


:::{hint}
If `plenoptic` is the only environment that you want to run notebooks from and/or you are unfamiliar with virtual environments, go with option 1 (for either `conda` or `pip and venv`).
:::

::::::::{tab-set}
:::::::{tab-item} conda

1. Install jupyter in the same environment as `plenoptic`. This is the easiest but, if you have multiple virtual environments and want to use Jupyter notebooks in each of them, it will take up a lot of space.

   ```{code-block} console
   $ conda activate plenoptic
   $ conda install -c conda-forge jupyterlab ipywidgets torchvision pooch
   ```

   With this setup, when you have another virtual environment that you wish to run jupyter notebooks from, you must reinstall jupyter into that separate virtual environment, which is wasteful.

2. Install jupyter in your `base` environment and use [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels) to automatically manage kernels in all your conda environments. This is a bit more complicated, but means you only have one installation of jupyter lab on your machine:

   ```{code-block} console
   $ # activate your 'base' environment, the default one created by conda/miniconda
   $ conda activate base
   $ # install jupyter lab and nb_conda_kernels in your base environment
   $ conda install -c conda-forge jupyterlab ipywidgets
   $ conda install nb_conda_kernels
   $ # install ipykernel, torchvision, and pooch in the plenoptic environment
   $ conda install -n plenoptic ipykernel torchvision pooch
   ```
   With this setup, you have a single jupyter install that can run kernels from any of your conda environments. All you have to do is install `ipykernel` (and restart jupyter) and you should see the new kernel!

   ::::::{attention}
   This method only works with conda environments.
   ::::::

3. Install jupyter in your `base` environment and manually install the kernel in your `plenoptic` virtual environment. This requires only a single jupyter install and is the most general solution (it will work with conda or any other way of managing virtual environments), but requires you to be a bit more comfortable with handling environments.

   ```{code-block} console
   $ # activate your 'base' environment, the default one created by conda/miniconda
   $ conda activate base
   $ # install jupyter lab in your base environment
   $ conda install -c conda-forge jupyterlab ipywidgets
   $ # install ipykernel and torchvision in the plenoptic environment
   $ conda install -n plenoptic ipykernel torchvision pooch
   $ conda activate plenoptic
   $ python -m ipykernel install --prefix=/path/to/jupyter/env --name 'plenoptic'
   ```

   `/path/to/jupyter/env` is the path to your base conda environment, and depends on the options set during your initial installation. It's probably something like `~/conda` or `~/miniconda` on Linux or MacOS and `C:\Users\username\miniconda\` on Windows. See the [ipython docs](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) for more details.

   With this setup, similar to option 2, you have a single jupyter install that can run kernels from any virtual environment. The main difference is that it can run kernels from **any** virtual environment (not just conda!) and have fewer packages installed in your `base` environment, but that you have to run an additional line after installing `ipykernel`  into the environment (`python -m ipykernel install ...`).
:::::::

:::::::{tab-item} pip and venv

1. Install jupyter in the same environment as `plenoptic`. This is the easiest but, if you have multiple virtual environments and want to use Jupyter notebooks in each of them, it will take up a lot of space.

   ::::::{tab-set}
   :sync-group: os

   :::::{tab-item} Linux / MacOS
   :sync: posix

   ::::{tab-set}
   :sync-group: install

   :::{tab-item} plenoptic installed from source
   :sync: source

   ```{code-block} console
   $ cd path/to/plenoptic
   $ source .venv/bin/activate
   $ pip install -e ".[nb]"
   ```
   :::
   :::{tab-item} plenoptic installed from PyPI
   :sync: pypi

   ```{code-block} console
   $ source path/to/environments/plenoptic-venv/bin/activate
   $ pip install "plenoptic[nb]"
   ```
   :::
   ::::
   :::::

   :::::{tab-item} Windows
   :sync: windows

   ::::{tab-set}
   :sync-group: install

   :::{tab-item} plenoptic installed from source
   :sync: source

   ```{code-block} pwsh-session
   PS> cd path\to\plenoptic
   PS> .venv\Scripts\activate
   PS> pip install -e ".[nb]"
   ```
   :::
   :::{tab-item} plenoptic installed from PyPI
   :sync: pypi

   ```{code-block} pwsh-session
   PS> path\to\environments\plenoptic-venv\Scripts\activate
   PS> pip install "plenoptic[nb]"
   ```
   :::
   ::::
   :::::
   ::::::

   With this setup, when you have another virtual environment that you wish to run jupyter notebooks from, you must reinstall jupyter into that separate virtual environment, which is wasteful.

2. Install jupyter in one environment and manually install the kernel in your ``plenoptic`` virtual environment. This requires only a single jupyter install and is the most general solution (it will work with conda or any other way of managing virtual environments), but requires you to be a bit more comfortable with handling environments.

   ::::::{tab-set}
   :sync-group: os

   :::::{tab-item} Linux / MacOS
   :sync: posix

   ::::{tab-set}
   :sync-group: install

   :::{tab-item} plenoptic installed from source
   :sync: source

   ```{code-block} console
   $ source path/to/jupyter-env/bin/activate
   $ pip install jupyterlab ipywidgets
   $ cd path/to/plenoptic
   $ source .venv/bin/activate
   $ pip install ipykernel torchvision pooch
   $ python -m ipykernel install --prefix=path/to/environments/jupyter-env/ --name 'plenoptic'
   ```
   :::
   :::{tab-item} plenoptic installed from PyPI
   :sync: pypi

   ```{code-block} console
   $ source path/to/environments/jupyter-env/bin/activate
   $ pip install jupyterlab ipywidgets
   $ source path/to/environments/plenoptic-venv/bin/activate
   $ pip install ipykernel torchvision pooch
   $ python -m ipykernel install --prefix=path/to/environments/jupyter-env/ --name 'plenoptic'
   ```
   :::
   ::::
   :::::

   :::::{tab-item} Windows
   :sync: windows

   ::::{tab-set}
   :sync-group: install

   :::{tab-item} plenoptic installed from source
   :sync: source

   ```{code-block} pwsh-session
   PS> path\to\environments\jupyter-venv\Scripts\activate
   PS> pip install jupyterlab ipywidgets
   PS> cd path\to\plenoptic
   PS> .venv\Scripts\activate
   PS> pip install ipykernel torchvision pooch
   PS> python -m ipykernel install --prefix=path\to\environments\jupyter-venv\ --name 'plenoptic'
   ```
   :::
   :::{tab-item} plenoptic installed from PyPI
   :sync: pypi

   ```{code-block} pwsh-session
   PS> path\to\environments\jupyter-venv\Scripts\activate
   PS> pip install jupyterlab ipywidgets
   PS> path\to\environments\plenoptic-venv\Scripts\activate
   PS> pip install ipykernel torchvision pooch
   PS> python -m ipykernel install --prefix=\path\to\environments\jupyter-env\ --name 'plenoptic'
   ```
   :::
   ::::
   :::::
   ::::::

   See the [ipython docs](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) for more details on this process.

   With this setup, you have a single jupyter install that can run kernels from any virtual environment. It can run kernels from *any* virtual environment, but that you have to run an additional line after installing `ipykernel`  into the environment (`python -m ipykernel install ...`).

:::::::
::::::::

The following table summarizes the advantages and disadvantages of these three choices:

:::{list-table}
:header-rows: 1

- - Method
  - Advantages
  - Disadvantages
- - 1. Everything in one environment
  - ✅ Simple
  - ❌ Requires lots of hard drive space
- -
  - ✅ Flexible: works with any virtual environment setup
  -
- - 2. `nb_conda_kernels`
  - ✅ Set up once
  - ❌ Initial setup more complicated
- -
  - ✅ Requires only one jupyter installation
  - ❌ Only works with conda
- -
  - ✅ Automatically finds new environments with `ipykernel` installed
  -
- - 3. Manual kernel installation
  - ✅ Flexible: works with any virtual environment setup
  - ❌ More complicated
- -
  - ✅ Requires only one jupyter installation
  - ❌ Extra step for each new environment
:::

## Running the notebooks

Once you have jupyter installed and the kernel set up, you can download the tutorial notebooks and run them. Each notebook has a `Download this notebook` link at the top. After downloading, navigate to the directory containing the notebook, then run `jupyter` and open up the notebooks. If you did not install `jupyter` into the same environment as `plenoptic`, you should be prompted to select your kernel the first time you open a notebook: select the one named "plenoptic".
