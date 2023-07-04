# Ollama Python bindings

```
pip install ollama
```

## Developing

Ollama is built using Python 3 and uses [Poetry](https://python-poetry.org/) to manage dependencies and build packages.

```
pip install poetry
```

Install ollama and its dependencies:

```
poetry install --extras server --with dev
```

Run ollama server:

```
poetry run ollama server
```

Update dependencies:

```
poetry update --extras server --with dev
poetry lock
poetry export >requirements.txt
```

Build binary package:

```
poetry build
```
