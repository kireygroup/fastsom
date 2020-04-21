# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

docs:
	@rm -rf build/
	@make html
	@find build/html/examples -type f -exec sed -i 's/<navlinks class="/<navlinks class="nav-links /g' {} + 
	@find build/html/examples -type f -exec sed -i 's/<navlinks>/<navlinks class="nav-links"> /g' {} + 
	@find build/html/examples -type f -exec sed -i 's/<navbar /<navbar class="navbar" /g' {} + 
	@find build/html/examples -type f -exec sed -i 's/<sidebar /<sidebar class="sidebar" /g' {} + 
	@find build/html/examples -type f -exec sed -i 's/<page/<div class="page"/g' {} + 
	@find build/html/examples -type f -exec sed -i 's/<\/page/<\/div/g' {} +
	@mv build/html/_static build/html/static
	@mv build/html/_images build/html/images
	@find build/html -type f -exec sed -i 's/_static/static/g' {} +
	@find build/html -type f -exec sed -i 's/_images/images/g' {} +
