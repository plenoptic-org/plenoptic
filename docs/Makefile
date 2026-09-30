# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
    # issues with cache here, the || true is so that we don't fail if the
    # directory doesn't exist
	rm -r "$(BUILDDIR)" || true
	rm -r "$(SOURCEDIR)/api/generated/" || true
	rm "$(SOURCEDIR)/api/index.rst" || true
    # ignore the pyrtools-thrown warning about dpi
	@ OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore:Ignoring dpi:UserWarning" $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -T -W
