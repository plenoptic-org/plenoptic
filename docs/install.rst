.. _install:

Installation
************

The following instructions will work on Linux or Mac. If you're on Windows, I
recommend looking into the `Windows Subsystem for Linux
<https://docs.microsoft.com/en-us/windows/wsl/install-win10).>`_

The easiest way to install ``plenoptic`` is with pip within a new `conda
<https://docs.conda.io/en/latest/>`_ environment (if you do not have ``conda``
installed on your machine, I recommend starting with `miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_)::

$ conda create --name plenoptic pip python=3.9
$ conda activate plenoptic
$ pip install git+https://github.com/LabForComputationalVision/plenoptic.git

You can also install it directly from source::

$ conda create --name plenoptic pip python=3.9
$ conda activate plenoptic
$ # clone the repository
$ git clone https://github.com/LabForComputationalVision/plenoptic.git
$ cd plenoptic
$ # install in editable mode with `-e` or, equivalently, `--editable`
$ pip install -e .

ffmpeg and videos
-----------------

Several methods in this package generate videos. There are several backends
possible for saving the animations to file, see `matplotlib documentation
<https://matplotlib.org/stable/api/animation_api.html#writer-classes>`_ for more
details. In order convert them to HTML5 for viewing (and thus, to view in a
jupyter notebook), you'll need `ffmpeg <https://ffmpeg.org/download.html>`_
installed and on your path as well. Depending on your system, this might already
be installed.

To change the backend, run ``matplotlib.rcParams['animation.writer'] = writer``
before calling any of the animate functions. If you try to set that ``rcParam``
with a random string, ``matplotlib`` will tell you the available choices.

Jupyter
-------

If you wish to locally run the notebooks, you will need to install ``jupyter``
and ``ipywidgets`` (you can also run them in the cloud using `Binder
<https://mybinder.org/v2/gh/LabForComputationalVision/plenoptic/v0.2?filepath=examples>`_).
There are two main ways of getting a local `jupyter` install` working with this
package:

1. Install jupyter in the same environment as ``plenoptic``. If you followed the
   instructions above to create a ``conda`` environment named ``plenoptic``, do
   the following::

   $ conda activate plenoptic
   $ conda install -c conda-forge jupyterlab ipywidgets

   This is easy but, if you have multiple conda environments and want to use
   Jupyter notebooks in each of them, it will take up a lot of space.

2. Use `nb_conda_kernels
   <https://github.com/Anaconda-Platform/nb_conda_kernels>`_. Again, if you
   followed the instructions to create a ``conda`` environment named
   ``plenoptic``::

   $ # activate your 'base' environment, the default one created by conda/miniconda
   $ conda activate
   $ # install jupyter lab and nb_conda_kernels in your base environment
   $ conda install -c conda-forge jupyterlab ipywidgets
   $ conda install nb_conda_kernels
   $ # install ipykernel in the plenoptic environment
   $ conda install -n plenoptic ipykernel

   This is a bit more complicated, but means you only have one installation of
   jupyter lab on your machine.

In either case, to open the notebooks, navigate to the ``examples/`` directory
under this one on your terminal and activate the environment you install jupyter
into (``plenoptic`` for 1, ``base`` for 2), then run ``jupyter`` and open up the
notebooks. If you followed the second method, you should be prompted to select
your kernel the first time you open a notebook: select the one named
"plenoptic".
