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
import os
import re
import sys

package_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, package_path)

# -- Project information -----------------------------------------------------

project = 'rawhdr'
copyright = '2019 – 2021, Felix Thaler'
author = 'Felix Thaler'

# The full version, including alpha/beta/rc tags
with open(os.path.join(package_path, 'rawhdr', '__init__.py')) as initfile:
    initfile_content = initfile.read()
match = re.search(r"__version__ = [']([^']*)[']", initfile_content, re.M)
if match:
    release = match.group(1)
else:
    raise RuntimeError('Unable to find version string')
match = re.search(r'[0-9]+\.[0-9]+', release)
if match:
    version = match.group()
else:
    raise RuntimeError('Unable to parse version number')

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {'collapse_navigation': False, 'display_version': True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
