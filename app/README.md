# Desktop

_Note: the Ollama desktop app is a work in progress and is not ready yet for general use._

This app builds upon Ollama to provide a desktop experience for running models.

## Developing

In the background run the ollama server `ollama.py`:

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
