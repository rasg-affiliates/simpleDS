# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
"""Configuration for sphinx documtation."""
import os
import sys
import simpleDS

sys.path.insert(0, os.path.abspath("../simpleDS/"))
readme_file = os.path.join(os.path.abspath("../"), "README.md")
index_file = os.path.join(os.path.abspath("../docs"), "index.rst")
dataparams_file = os.path.join(os.path.abspath("../docs"), "dspec_parameters.rst")

# -- Project information -----------------------------------------------------

project = "simpleDS"
copyright = "2019, rasg-affiliates"
author = "rasg-affiliates"

# The full version, including alpha/beta/rc tags
version = simpleDS.__version__
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Controls when a cell will time out (defaults to 30; use -1 for no timeout):
nbsphinx_timeout = -1

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "default"
html_theme_options = {"rightsidebar": "false", "relbarbgcolor": "black"}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "simpleDSdoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}


def build_custom_docs(app):
    """Generate custom doc files."""
    sys.path.append(os.getcwd())
    import make_index
    import make_parameters

    # import make_cal_parameters
    # import make_beam_parameters
    make_index.write_index_rst(readme_file=readme_file, write_file=index_file)
    make_parameters.write_dataparams_rst(write_file=dataparams_file)
    # make_cal_parameters.write_calparams_rst(write_file=calparams_file)
    # make_beam_parameters.write_beamparams_rst(write_file=beamparams_file)


def setup(app):
    """Connect to doc builder."""
    app.connect("builder-inited", build_custom_docs)
