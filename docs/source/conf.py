# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Boax'
copyright = '2023, The Boax authors'
author = 'The Boax authors'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  'myst_nb',
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.doctest',
  'sphinx.ext.duration',
  'sphinx.ext.napoleon',
  'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_theme_options = {
  'repository_url': 'https://github.com/Lando-L/boax',
  'use_repository_button': True,
  'use_issues_button': False,
  'path_to_docs': ('docs'), 
  'show_navbar_depth': 1,
}

# -- Options for Jupyter Notebooks -------------------------------------------

nb_execution_mode = 'off'
