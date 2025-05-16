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

myst_heading_anchors = 3


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- PyData theme options -----------------------------------------------------
html_logo = "_static/logo/llama.png"
html_theme_options = {

    "use_edit_page_button": False,
    "github_url": "https://github.com/ollama/ollama",
    "twitter_url": "https://twitter.com/ollamaAI",
    "show_toc_level": 2,

    # Add navigation bar at the top
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],

    # Configure the top navigation bar
    "navbar_persistent": ["search-button"],

    # Define the top-level navigation items
    "header_links_before_dropdown": 4,
    "icon_links_label": "Quick Links",

    # Define the navigation structure
    "navigation_depth": 3,
    "collapse_navigation": False,
    "show_nav_level": 2,

    # Add primary sidebar navigation
    "primary_sidebar_end": ["sidebar-ethical-ads"],

    # Configure the top navigation bar with important sections
    "external_links": [
        {"name": "Home", "url": "index.html"},
        {"name": "Install", "url": "installation/index.html"},
        {"name": "Getting Started", "url": "getting_started/quickstart.html"},
        {"name": "API Reference", "url": "devs/api.html"},
        {"name": "Resources", "url": "resources/faq.html"},
    ],
}

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}
