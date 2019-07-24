# plenoptic


**{x, y, θ}**  : inputs, outputs, parameters

|            	|   fixed  	| variable  |
|:----------:	|:------:	|:------:	|
|  simulate  	| {x, θ} 	|   {y}  	|
|   learn    	| {x, y} 	|   {θ}  	|
| synthesize 	| {y, θ} 	|   {x}  	|

NOTE: We only support python 3.6 and 3.7

# Build the documentation

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
