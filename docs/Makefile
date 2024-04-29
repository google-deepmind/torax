# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
DOC_SOURCEDIR     = .
# copybara:strip_begin(internal variant)
PY_SOURCEDIR  = ../.
# copybara:strip_end_and_replace_begin
# PY_SOURCEDIR  = ../torax
# copybara:replace_end
BUILDDIR      = _build
# copybara:strip_begin(internal variant)
EXCLUDE_PATTERNS = "../simulation_app.py"
# copybara:strip_end_and_replace_begin
# EXCLUDE_PATTERNS = "../torax/simulation_app.py"
# copybara:replace_end

# Put it first so that "make" without argument is like "make help".
help:
	echo "The apidoc target is built by sphinx-apidoc".
	echo ""
	echo "Everything else is built by sphinx-build:"
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

apidoc: Makefile
	sphinx-apidoc -o $(DOC_SOURCEDIR)/api "${PY_SOURCEDIR}" ${EXCLUDE_PATTERNS}
	# The content of this file never really changes, so it is feasible to
	# replace it with a static copy manually.
	# We store the static copy as .txt to avoid SPHINXBUILD warning us about
	# the unused storage copy not appearing in a toctree.
	# We need to do this because the links to the page are listed by the
	# page title and the auto-generated page title is just "torax"; our manual
	# copy here lets us set the page title to "Torax API Reference"
	cp modules.txt ${DOC_SOURCEDIR}/api/modules.rst

clean:
	rm -rf api/
	@$(SPHINXBUILD) -M $@ "$(DOC_SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile apidoc
	@$(SPHINXBUILD) -M $@ "$(DOC_SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
