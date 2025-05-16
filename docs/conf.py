# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Ollama'
copyright = '2024, Ollama'
author = 'Ollama'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',  # for Markdown support
    'sphinx_copybutton',  # for copy button in code blocks
    'sphinx.ext.mathjax',  # for math support
    'sphinx.ext.githubpages',  # for GitHub Pages support

    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['README.md', 'source', "requirements.txt"]

# -- Options for MyST Parser -------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'tasklist',

]
# Note that the new syntax allowed in github markdown denoting [!NOTE] or [!WARNING] is not yet supported
# by myst-parser. See https://github.com/executablebooks/MyST-Parser/issues/845

myst_heading_anchors = 7


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- PyData theme options -----------------------------------------------------
html_logo = "_static/logo/llama.png"
html_theme_options = {

    "use_edit_page_button": True,
    "github_url": "https://github.com/ollama/ollama",
    "show_toc_level": 5,

    # Add navigation bar at the top
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    # "navbar_end": ["navbar-icon-links"],

    # Configure the top navigation bar
    "navbar_persistent": ["search-button"],

    # Define the top-level navigation items
    "header_links_before_dropdown": 7,
    "icon_links_label": "Quick Links",

    # Define the navigation structure
    "navigation_depth": 7,
    "collapse_navigation": True,
    "show_nav_level": 7,



    # Add primary sidebar navigation
    "primary_sidebar_end": ["sidebar-ethical-ads"],

    # Configure the top navigation bar with important sections
    # "external_links": [
    #     {"name": "Home", "url": "index.html"},
    #     {"name": "Install", "url": "installation/index.html"},
    #     {"name": "Getting Started", "url": "getting_started/quickstart.html"},
    #     {"name": "API Reference", "url": "devs/api.html"},
    #     {"name": "Resources", "url": "resources/faq.html"},
    # ],

}
html_context = {
    "github_user": "ollama",
    "github_repo": "ollama",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# -- Warning options ---------------------------------------------------
suppress_warnings = ['misc.highlighting_failure']  # some highlighting fails (dockerfile, a shell command)
warning_is_error = True