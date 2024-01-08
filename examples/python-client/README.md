# Ollama Python Client

A client for the Ollama API that uses Python with the `aiohttp` library.

# Installation
Latest version from GitHub:
```bash
pip install "ollama-client git+https://github.com/jmorganca/ollama#subdirectory=examples/python-client"
```

# Development
## Setup
```bash
poetry install
```

## Tests
To run stats, start Ollama:
```bash
ollama serve
```

Then run the tests in another terminal:
```bash
PYTHONPATH="src/ollama_client:${PYTHONPATH}" pytest 
```