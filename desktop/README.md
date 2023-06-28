# Desktop

The Ollama desktop experience. This is an experimental, easy-to-use app for running models with [`ollama`](https://github.com/jmorganca/ollama).

## Download

- [macOS](https://ollama.ai/download/darwin_arm64) (Apple Silicon)
- macOS (Intel – Coming soon)
- Windows (Coming soon)
- Linux (Coming soon)

## Running

In the background run the ollama server `ollama.py` server:

```
python ../ollama.py serve --port 7734
```

Then run the desktop app with `npm start`:

```
npm install
npm start
```

## Coming soon

- Browse the latest available models on Hugging Face and other sources
- Keep track of previous conversations with models
- Switch between models
- Connect to remote Ollama servers to run models
