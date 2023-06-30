# Desktop

The Ollama desktop experience. This is an experimental, easy-to-use app for running models with [`ollama`](https://github.com/jmorganca/ollama).

## Running

In the background run the ollama server `ollama.py` server:

```
poetry -C .. run ollama serve
```

Then run the desktop app with `npm start`:

```
npm install
npm start
```

## Coming soon

- Browse the latest available models on Hugging Face and other sources
- Keep track of previous conversations with models
- Switch quickly between models
- Connect to remote Ollama servers to run models
