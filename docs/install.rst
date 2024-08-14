.. _install:

Installation
************

``plenoptic`` should work on Windows, Linux, or Mac. If you have a problem with installation, please open a `bug report <https://github.com/LabForComputationalVision/plenoptic/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=>`_!

You can install ``plenoptic`` from `PyPI <https://pypi.org/project/plenoptic/>`_  (the Python Package Index) or `conda-forge <https://anaconda.org/conda-forge/plenoptic>`_, and we provide separate instructions for the two methods. If you will be contributing to ``plenoptic``, and so need an editable install and :ref:`optional-deps`, you should use :ref:`pip <pip>`. Otherwise, you can use whichever you are more familiar with, though we have noticed that it tends to be easier to install `pytorch <https://pytorch.org/>`_ with GPU support using ``conda``.

.. tip::
   If you are unfamiliar with python environment management, we recommend :ref:`conda`.

Our dependencies include `pytorch <https://pytorch.org/>`_ and `pyrtools <https://pyrtools.readthedocs.io/en/latest/>`_. Installation should take care of them (along with our other dependencies) automatically, but if you have an installation problem (especially on a non-Linux operating system), it is likely that the problem lies with one of those packages. `Open an issue <https://github.com/LabForComputationalVision/plenoptic/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=>`_ and we'll try to help you figure out the problem!

.. _conda:

Installing with conda
---------------------

.. warning::

   We do not currently support conda installs on Windows, due to the lack of a Windows pytorch package on conda-forge. Therefore, if you are installing on Windows, you must use :ref:`pip <pip>`. See `this issue <https://github.com/conda-forge/pytorch-cpu-feedstock/issues/32>`__ for the status of the conda-forge Windows pytorch build.

If you wish to follow these instructions and do not have ``conda`` installed on your machine, we recommend starting with `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

Once you have ``conda`` correctly installed and on your path, run the following to create a new virtual environment and install plenoptic::

$ conda create --name plenoptic pip python=3.11
$ conda activate plenoptic
$ conda install plenoptic -c conda-forge

.. _pip:

Installing with pip
-------------------

While ``conda`` handles both virtual environment management **and** package installation, ``pip`` only installs packages; you'll need to use some other system to manage your virtual environment. You can use ``conda`` to manage your virtual environment and ``pip`` to install packages, but note that when using pip with a conda environment, all pip commands should come after all the conda commands (e.g., you shouldn't run ``conda install matplotlib`` after ``pip install plenoptic``), because conda is unaware of any changes that pip makes. See `this blog post <https://www.anaconda.com/blog/using-pip-in-a-conda-environment>`_ for more details.

In order to avoid this problem, here we'll use python's built-in `venv <https://docs.python.org/3/library/venv.html>`_ to manage the virtual environment with ``pip``.

.. tab:: Linux / MacOS

   .. code-block:: shell

      # create the environment
      $ python -m venv path/to/environments/plenoptic-venv
      # activate the environment
      $ source path/to/environment/plenoptic-venv/bin/activate
      $ pip install plenoptic

.. tab:: Windows

   .. code-block:: powershell

      # create the environment
      $ python -m venv path\to\environments\plenoptic-venv
      # activate the environment
      $ path\to\environment\plenoptic-venv\bin\activate
      $ pip install plenoptic

Note that when using ``venv``, you have to decide where you'd like to place the folder containing all files related to the virtual environment. If the virtual environment is related to the development of a package, as in :ref:`source`, it is typical to place them within the repository for that package. Otherwise, it is typical to place them all in a single directory somewhere so they're easy to keep track of.

.. _source:

Installing from source (for developers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also install plenoptic directly from source in order to have a local editable copy. This is most useful for developing (for more info, see `our contributing guide <https://github.com/LabForComputationalVision/plenoptic/blob/main/CONTRIBUTING.md>`_) or if you want to use the most cutting-edge version.

.. tab:: Linux / MacOS

   .. code-block:: shell

      $ git clone https://github.com/LabForComputationalVision/plenoptic.git
      $ cd plenoptic
      # create the environment
      $ python -m venv .venv
      # activate the environment
      $ source .venv/bin/activate
      # install in editable mode with `-e` or, equivalently, `--editable`
      $ pip install -e .

.. tab:: Windows

   .. code-block:: powershell

      $ git clone https://github.com/LabForComputationalVision/plenoptic.git
      $ cd plenoptic
      # create the environment
      $ python -m venv .venv
      # activate the environment
      $ .venv\bin\activate
      # install in editable mode with `-e` or, equivalently, `--editable`
      $ pip install -e .

With an editable copy, any changes locally will be automatically reflected in your installation (under the hood, this command uses symlinks).

Note that, with the above setup, all files related to your virtual environment are stored in a hidden directory named ``.venv`` within the ``plenoptic/`` directory you cloned. Therefore, if you delete the ``plenoptic/`` directory, you'll need to rerun the setup above to create a new virtual environment.

.. attention:: To install ``plenoptic`` in editable mode, you need ``pip >= 21.3`` (see pip's `changelog <https://pip.pypa.io/en/stable/news/#id286>`_). If you run into `an error <https://github.com/LabForComputationalVision/plenoptic/issues/227>`_ after running the ``pip install -e .`` command, try updating your pip version with ``pip install --upgrade pip``.

.. _optional-deps:

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

The above instructions will install plenoptic and its core dependencies. You may also wish to install some additional optional dependencies. These dependencies are specified using square brackets during the ``pip install`` command and can be installed for either a local, editable install or one directly from PyPI:

* If you would like to run the jupyter notebooks locally: ``pip install plenoptic[nb]`` or ``pip install -e .[nb]``. This includes ``pooch`` (for downloading some extra data) ``torchvision`` (which has some models we'd like to use), ``jupyter``, and related libraries. See the :ref:`jupyter section <jupyter>` for a discussion of several ways to handle jupyter and python virtual environments. Note that you can run our notebooks in the cloud using `Binder <https://mybinder.org/v2/gh/LabForComputationalVision/plenoptic/1.0.1?filepath=examples>`_, no installation required!
* If you would like to locally build the documentation: ``pip install -e .[docs]``. This includes ``sphinx`` and related libraries. (This probably only makes sense if you have a local installation.)
* If you would like to run the tests: ``pip install -e .[dev]``. This includes ``pytest`` and related libraries. (This probably only makes sense if you have a local installation.)

These optional dependencies can be joined with a comma, e.g., ``pip install -e .[docs,dev]``

.. note:: Note that ``conda`` does not support optional dependencies, though you can view our optional dependencies in the `pyproject.toml <https://github.com/LabForComputationalVision/plenoptic/blob/main/pyproject.toml#L35>`_ file, if you wish to install them yourself.

.. _jupyter:

Running notebooks locally
-------------------------

.. tip:: You can run the notebooks in the cloud using `Binder <https://mybinder.org/v2/gh/LabForComputationalVision/plenoptic/1.0.1?filepath=examples>`_, no installation required!

Installing jupyter and setting up the kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to locally run the notebooks, you will need to install ``jupyter``,
``ipywidgets``, and (for some of the notebooks) ``torchvision`` and ``pooch`` .
There are two or three possible ways of getting a local jupyter install working with
this package, depending on what tool you are using to manage your virtual
environments and how you wish to handle them.

.. hint:: If ``plenoptic`` is the only environment that you want to run notebooks from and/or you are unfamiliar with virtual environments, go with option 1 below.

.. tab:: conda

   1. Install jupyter in the same environment as ``plenoptic``. This is the easiest
      but, if you have multiple virtual environments and want to use Jupyter
      notebooks in each of them, it will take up a lot of space.

      .. code-block:: shell

         $ conda activate plenoptic
         $ conda install -c conda-forge jupyterlab ipywidgets torchvision pooch

      With this setup, when you have another virtual environment that you wish to run jupyter notebooks from, you must reinstall jupyter into that separate virtual environment, which is wasteful.

   2. Install jupyter in your ``base`` environment and use `nb_conda_kernels
      <https://github.com/Anaconda-Platform/nb_conda_kernels>`_ to automatically
      manage kernels in all your conda environments. This is a bit more
      complicated, but means you only have one installation of jupyter lab on your
      machine:

      .. code-block:: shell

         # activate your 'base' environment, the default one created by conda/miniconda
         $ conda activate base
         # install jupyter lab and nb_conda_kernels in your base environment
         $ conda install -c conda-forge jupyterlab ipywidgets
         $ conda install nb_conda_kernels
         # install ipykernel, torchvision, and pooch in the plenoptic environment
         $ conda install -n plenoptic ipykernel torchvision pooch

      With this setup, you have a single jupyter install that can run kernels from any of your conda environments. All you have to do is install ``ipykernel`` (and restart jupyter) and you should see the new kernel!

      .. attention:: This method only works with conda environments.

   3. Install jupyter in your ``base`` environment and manually install the kernel in your ``plenoptic`` virtual environment. This requires only a single jupyter install and is the most general solution (it will work with conda or any other way of managing virtual environments), but requires you to be a bit more comfortable with handling environments.

      .. code-block:: shell

         # activate your 'base' environment, the default one created by conda/miniconda
         $ conda activate base
         # install jupyter lab in your base environment
         $ conda install -c conda-forge jupyterlab ipywidgets
         # install ipykernel and torchvision in the plenoptic environment
         $ conda install -n plenoptic ipykernel torchvision pooch
         $ conda activate plenoptic
         $ python -m ipykernel install --prefix=/path/to/jupyter/env --name 'plenoptic'

      ``/path/to/jupyter/env`` is the path to your base conda environment, and depends on the options set during your initial installation. It's probably something like ``~/conda`` or ``~/miniconda``. See the `ipython docs <https://ipython.readthedocs.io/en/stable/install/kernel_install.html>`_ for more details.

      With this setup, similar to option 2, you have a single jupyter install that can run kernels from any virtual environment. The main difference is that it can run kernels from *any* virtual environment (not just conda!) and have fewer packages installed in your ``base`` environment, but that you have to run an additional line after installing ``ipykernel``  into the environment (``python -m ipykernel install ...``).

.. tab:: pip and venv

   1. Install jupyter in the same environment as ``plenoptic``. This is the easiest but, if you have multiple virtual environments and want to use Jupyter notebooks in each of them, it will take up a lot of space.

      .. tab:: Linux / MacOS

         .. tab:: plenoptic installed from source

            .. code-block:: shell

               $ cd path/to/plenoptic
               $ source .venv/bin/activate
               $ pip install -e .[nb]

         .. tab:: plenoptic installed from PyPI

            .. code-block:: shell

               $ source path/to/environments/plenoptic-venv/bin/activate
               $ pip install plenoptic[nb]

      .. tab:: Windows

         .. tab:: plenoptic installed from source

            .. code-block:: powershell

               $ cd path\to\plenoptic
               $ .venv\bin\activate
               $ pip install -e .[nb]

         .. tab:: plenoptic installed from PyPI

            .. code-block:: shell

               $ source path\to\environments\plenoptic-venv\bin\activate
               $ pip install plenoptic[nb]

      With this setup, when you have another virtual environment that you wish to run jupyter notebooks from, you must reinstall jupyter into that separate virtual environment, which is wasteful.

   2. Install jupyter in one environment and manually install the kernel in your ``plenoptic`` virtual environment. This requires only a single jupyter install and is the most general solution (it will work with conda or any other way of managing virtual environments), but requires you to be a bit more comfortable with handling environments.

      .. tab:: Linux / MacOS

         .. tab:: plenoptic installed from source

            .. code-block:: shell

               $ source path/to/jupyter-env/bin/activate
               $ pip install jupyterlab ipywidgets
               $ cd path/to/plenoptic
               $ source .venv/bin/activate
               $ pip install ipykernel torchvision pooch
               $ python -m ipykernel install --prefix=path/to/environments/jupyter-env/ --name 'plenoptic'

         .. tab:: plenoptic installed from PyPI

            .. code-block:: shell

               $ source path/to/environments/jupyter-env/bin/activate
               $ pip install jupyterlab ipywidgets
               $ source path/to/environments/plenoptic-venv/bin/activate
               $ pip install ipykernel torchvision pooch
               $ python -m ipykernel install --prefix=path/to/environments/jupyter-env/ --name 'plenoptic'

      .. tab:: Windows

         .. tab:: plenoptic installed from source

            .. code-block:: powershell

               $ path\to\environments\jupyter-venv\bin\activate
               $ pip install jupyterlab ipywidgets
               $ cd path\to\plenoptic
               $ .venv\bin\activate
               $ pip install ipykernel torchvision pooch
               $ python -m ipykernel install --prefix=path\to\environments\jupyter-venv\ --name 'plenoptic'

         .. tab:: plenoptic installed from PyPI

            .. code-block:: shell

               $ path\to\environments\jupyter-venv\bin\activate
               $ pip install jupyterlab ipywidgets
               $ source path\to\environments\plenoptic-venv\bin\activate
               $ pip install ipykernel torchvision pooch
               $ python -m ipykernel install --prefix=\path\to\environments\jupyter-env\ --name 'plenoptic'

      See the `ipython docs <https://ipython.readthedocs.io/en/stable/install/kernel_install.html>`_ for more details on this process.

      With this setup, you have a single jupyter install that can run kernels from any virtual environment. It can run kernels from *any* virtual environment, but that you have to run an additional line after installing ``ipykernel``  into the environment (``python -m ipykernel install ...``).

The following table summarizes the advantages and disadvantages of these three choices:

.. list-table::
   :header-rows: 1

   *  - Method
      -  Advantages
      -  Disadvantages
   *  - 1. Everything in one environment
      - |:white_check_mark:| Simple
      - |:x:| Requires lots of hard drive space
   *  -
      - |:white_check_mark:| Flexible: works with any virtual environment setup
      -
   *  - 2. ``nb_conda_kernels``
      - |:white_check_mark:| Set up once
      - |:x:| Initial setup more complicated
   *  -
      - |:white_check_mark:| Requires only one jupyter installation
      - |:x:| Only works with conda
   *  -
      - |:white_check_mark:| Automatically finds new environments with ``ipykernel`` installed
      -
   *  - 3. Manual kernel installation
      - |:white_check_mark:| Flexible: works with any virtual environment setup
      - |:x:| More complicated
   *  -
      - |:white_check_mark:| Requires only one jupyter installation
      - |:x:| Extra step for each new environment


Running the notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have jupyter installed and the kernel set up, navigate to plenoptic's ``examples/`` directory on your terminal and activate the environment you installed jupyter into, then run ``jupyter`` and open up the notebooks. If you did not install ``jupyter`` into the same environment as ``plenoptic``, you should be prompted to select your kernel the first time you open a notebook: select the one named "plenoptic".

.. attention:: If you did not install ``plenoptic`` from source, then you will not have the notebooks on your machine and will need to download them directly from `our GitHub repo <https://github.com/LabForComputationalVision/plenoptic/tree/main/examples>`_. If installed them from source (and thus ran ``git clone``), then the notebooks can be found in the ``examples/`` directory.

ffmpeg and videos
-----------------

Several methods in this package generate videos. There are several backends possible for saving the animations to file, see `matplotlib documentation <https://matplotlib.org/stable/api/animation_api.html#writer-classes>`_ for more details. The default writer uses `ffmpeg <https://ffmpeg.org/download.html>`_, which you'll need installed and on your path in order to save the videos or view them in a jupyter notebook. Depending on your system, this might already be installed, but if not, and you're using :ref:`conda to manage your environments <conda>`, the easiest way is probably through `conda <https://anaconda.org/conda-forge/ffmpeg>`__: ``conda install -c conda-forge ffmpeg``.

To change the backend, run ``matplotlib.rcParams['animation.writer'] = writer``
before calling any of the animate functions. If you try to set that ``rcParam``
with a random string, ``matplotlib`` will tell you the available choices.
