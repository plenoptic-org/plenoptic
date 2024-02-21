.. _install:

Installation
************

``plenoptic`` should work on Windows, Linux, or Mac. If you have a problem with installation, please open a `bug report <https://github.com/LabForComputationalVision/plenoptic/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=>`_!

The easiest way to install ``plenoptic`` is from `PyPI <https://pypi.org/project/plenoptic/>`_  (the Python Package Index) using pip within a new virtual environment. The instructions on this page use `conda <https://docs.conda.io/en/latest/>`_, which we recommend if you are unfamiliar with python environment management, but other virtual environment systems should work. If you wish to follow these instructions and do not have ``conda`` installed on your machine, I recommend starting with `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.::

$ conda create --name plenoptic pip python=3.9
$ conda activate plenoptic
$ pip install plenoptic

Our dependencies include `pytorch <https://pytorch.org/>`_ and `pyrtools <https://pyrtools.readthedocs.io/en/latest/>`_. Installation should take care of them (along with our other dependencies) automatically, but if you have an installation problem (especially on a non-Linux operating system), it is likely that the problem lies with one of those packages. `Open an issue <https://github.com/LabForComputationalVision/plenoptic/issues>`_ and we'll try to help you figure out the problem!

You can also install it directly from source to have a local editable copy. This is most useful for developing (for more info, see `our contributing guide <https://github.com/LabForComputationalVision/plenoptic/blob/main/CONTRIBUTING.md>`_) or if you want to use the most cutting-edge version::

$ conda create --name plenoptic pip python=3.9
$ conda activate plenoptic
$ # clone the repository
$ git clone https://github.com/LabForComputationalVision/plenoptic.git
$ cd plenoptic
$ # install in editable mode with `-e` or, equivalently, `--editable`
$ pip install -e .

With an editable copy, any changes locally will be automatically reflected in your installation (under the hood, this command uses symlinks).

.. attention:: To install ``plenoptic`` in editable mode, you need ``pip >= 21.3`` (see pip's `changelog <https://pip.pypa.io/en/stable/news/#id286>`_). If you run into `an error <https://github.com/LabForComputationalVision/plenoptic/issues/227>`_ after running the ``pip install -e .`` command, try updating your pip version with ``pip install --upgrade pip``.

.. _optional-deps:
Optional dependencies
---------------------

The above instructions will install plenoptic and its core dependencies. You may also wish to install some additional optional dependencies. These dependencies are specified using square brackets during the pip install command and can be installed for either a local, editable install or one directly from PyPI:

* If you would like to run the jupyter notebooks locally: ``pip install plenoptic[nb]`` or ``pip install -e .[nb]``. This includes ``torchvision``, ``jupyter``, and related libraries. See the :ref:`jupyter section <jupyter>` for more details on how to handle jupyter and python virtual environments. Note that you can run our notebooks in the cloud using `Binder <https://mybinder.org/v2/gh/LabForComputationalVision/plenoptic/1.0.1?filepath=examples>`_, no installation required!
* If you would like to locally build the documentation: ``pip install -e .[docs]``. This includes ``sphinx`` and related libraries. (This probably only makes sense if you have a local installation.)
* If you would like to run the tests: ``pip install -e .[dev]``. This includes ``pytest`` and related libraries. (This probably only makes sense if you have a local installation.)

These optional dependencies can be joined with a comma: ``pip install -e .[docs,dev]``

ffmpeg and videos
-----------------

Several methods in this package generate videos. There are several backends
possible for saving the animations to file, see `matplotlib documentation
<https://matplotlib.org/stable/api/animation_api.html#writer-classes>`_ for more
details. In order to convert them to HTML5 for viewing (and thus, to view in a
jupyter notebook), you'll need `ffmpeg <https://ffmpeg.org/download.html>`_
installed and on your path as well. Depending on your system, this might already
be installed, but if not, the easiest way is probably through `conda
<https://anaconda.org/conda-forge/ffmpeg>`_: ``conda install -c conda-forge
ffmpeg``.

To change the backend, run ``matplotlib.rcParams['animation.writer'] = writer``
before calling any of the animate functions. If you try to set that ``rcParam``
with a random string, ``matplotlib`` will tell you the available choices.

.. _jupyter:
Running notebooks locally
-------------------------

.. tip:: You can run the notebooks in the cloud using `Binder <https://mybinder.org/v2/gh/LabForComputationalVision/plenoptic/1.0.1?filepath=examples>`_, no installation required!

Installing jupyter and setting up the kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to locally run the notebooks, you will need to install ``jupyter``,
``ipywidgets``, and (for some of the notebooks) ``torchvision`` .
There are three possible ways of getting a local jupyter install working with this
package, depending on how you wish to handle virtual environments.

.. hint:: If ``plenoptic`` is the only environment that you want to run notebooks from and/or you are unfamiliar with virtual environments, go with option 1 below.

1. Install jupyter in the same environment as ``plenoptic``. This is the easiest
   but, if you have multiple virtual environments and want to use Jupyter
   notebooks in each of them, it will take up a lot of space. If you followed
   the instructions above to create a ``conda`` environment named ``plenoptic``,
   do the following::

   $ conda activate plenoptic
   $ conda install -c conda-forge jupyterlab ipywidgets torchvision

   With this setup, when you have another virtual environment that you wish to run jupyter notebooks from, you must reinstall jupyuter into that separate virtual environment, which is wasteful.

2. Install jupyter in your ``base`` environment and use `nb_conda_kernels
   <https://github.com/Anaconda-Platform/nb_conda_kernels>`_ to automatically
   manage kernels in all your conda environments. This is a bit more
   complicated, but means you only have one installation of jupyter lab on your
   machine. Again, if you followed the instructions to create a ``conda``
   environment named ``plenoptic``::

   $ # activate your 'base' environment, the default one created by conda/miniconda
   $ conda activate base
   $ # install jupyter lab and nb_conda_kernels in your base environment
   $ conda install -c conda-forge jupyterlab ipywidgets
   $ conda install nb_conda_kernels
   $ # install ipykernel and torchvision in the plenoptic environment
   $ conda install -n plenoptic ipykernel torchvision

   With this setup, you have a single jupyter install that can run kernels from any of your conda environments. All you have to do is install ``ipykernel`` (and restart jupyter) and you should see the new kernel!

   .. attention:: This method only works with conda environments. If you are using another method to manage your python virtual environments, you'll have to use one of the other methods.

3. Install jupyter in your ``base`` environment and manually install the kernel in your virtual environment. This requires only a single jupyter install and is the most general solution (it will work with conda or any other way of managing virtual environments), but requires you to be a bit more comfortable with handling environments. Again, if you followed the instructions to create a ``conda`` environment named ``plenoptic``::

   $ # activate your 'base' environment, the default one created by conda/miniconda
   $ conda activate base
   $ # install jupyter lab and nb_conda_kernels in your base environment
   $ conda install -c conda-forge jupyterlab ipywidgets
   $ # install ipykernel and torchvision in the plenoptic environment
   $ conda install -n plenoptic ipykernel torchvision
   $ conda activate plenoptic
   $ python -m ipykernel install --prefix=/path/to/jupyter/env --name 'plenoptic'

   ``/path/to/jupyter/env`` is the path to your base conda environment, and depends on the options set during your initial installation. It's probably something like ``~/conda`` or ``~/miniconda``. See the `ipython docs <https://ipython.readthedocs.io/en/stable/install/kernel_install.html>`_ for more details.

   With this setup, similar to option 2, you have a single jupyter install that can run kernels from any virtual environment. The main difference is that it can run kernels from *any* virtual environment (not just conda!) and have fewer packages installed in your ``base`` environment, but that you have to run an additional line after installing ``ipykernel``  into the environment (``python -m ipykernel install ...``).

   .. note:: If you're not using conda to manage your environments, the key idea is to install ``jupyter`` and ``ipywidgets`` in one environment, then install ``ipykernel`` and ``torchvision`` in the same environment as plenoptic, and then run the ``ipykernel install`` command **using the plenoptic environment's python**.

The following table summarizes the advantages and disadvantages of these three choices:

.. list-table::
   :header-rows: 1

   *  - Method
      -  Advantages
      -  Disadvantages
   *  - 1. Everything in one environment
      - |:white_check_mark:| Simple
      - |:x:| Requires lots of hard drive space
   *  - 2. ``nb_conda_kernels``
      - |:white_check_mark:| Set up once
      - |:x:| Initial setup more complicated
   *  -
      - |:white_check_mark:| Requires only one jupyter installation
      -
   *  -
      - |:white_check_mark:| Automatically finds new environments with ``ipykernel`` installed
      -
   *  - 3. Manual kernel installation
      - |:white_check_mark:| Flexible: works with any virtual environment setup
      - |:x:| More complicated
   *  -
      - |:white_check_mark:| Requires only one jupyter installation
      - |:x:| Extra step for each new environment

You can install all of the extra required packages using ``pip install -e .[nb]`` (if you have a local copy of the source code) or ``pip install plenoptic[nb]`` (if you are installing from PyPI). This includes jupyter, and so is equivalent to method 1 above. See the :ref:`optional dependencies section <optional-deps>` for more details.

Running the notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have jupyter installed and the kernel set up, navigate to plenoptic's ``examples/`` directory on your terminal and activate the environment you installed jupyter into (``conda activate plenoptic`` for method 1, ``conda activate base`` for methods methods method 2 or 3), then run ``jupyter`` and open up the notebooks. If you followed the second or third method, you should be prompted to select your kernel the first time you open a notebook: select the one named "plenoptic".

.. attention:: If you installed ``plenoptic`` from PyPI, then you will not have the notebooks on your machine and will need to download them directly from `our GitHub repo <https://github.com/LabForComputationalVision/plenoptic/tree/main/examples>`_. If you have a local install (and thus ran ``git clone``), then the notebooks can be found in the ``examples/`` directory.
