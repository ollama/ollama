# Development

ollama is built and run using [Poetry](https://python-poetry.org/).

## Running

**Start backend service:**

Install dependencies:

```
poetry install --extras server
```

Run a server:

```
poetry run ollama serve
```

**Start frontend service:**

Install dependencies:

```
cd desktop
npm install
```

Run the UI:

```
npm start
```

## Building

If using Apple silicon, you need a Python version that supports arm64:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

Get the dependencies:

```bash
poetry install --extras server
```

Then build a binary for your current platform:

```bash
poetry build
```

### Building the app

```
cd desktop
npm run package
```

## Update requirements.txt

In the root directory, run:

```
pipreqs . --force
```
