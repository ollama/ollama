# Ollama Documentation

This directory contains the Sphinx documentation for Ollama.

## Building the Documentation

To build the documentation, you'll need to have Python and Sphinx installed. Follow these steps:

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Build the HTML documentation:

```bash
make html
```

The generated HTML files will be in the `build/html` directory. Open `build/html/index.html` in your browser to view the documentation.

## Documentation Structure

- `source/`: Contains the source files for the documentation
  - `index.md`: The main entry point for the documentation
  - `getting_started/`: Getting started guides
  - `reference/`: API and feature reference
  - `resources/`: Additional resources and troubleshooting
- `build/`: Contains the generated documentation (created when you run `make html`)
- `Makefile` and `make.bat`: Scripts for building the documentation
- `requirements.txt`: Python package requirements for building the documentation

## Writing Documentation

The documentation is written in Markdown using the MyST parser for Sphinx. See the [MyST documentation](https://myst-parser.readthedocs.io/) for more information on the syntax.