# FAQ

## How can I expose the Ollama server?

```
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

By default, Ollama allows cross origin requests from `127.0.0.1` and `0.0.0.0`. To support more origins, you can use the `OLLAMA_ORIGINS` environment variable:

```
OLLAMA_ORIGINS=http://192.168.1.1:*,https://example.com ollama serve
```

## Where are models stored?

Raw model data is stored under `~/.ollama/models`. 

This can be customized from the default user home directory setting the `OLLAMA_HOME` environment variable when running the server. 

```
OLLAMA_HOME=~/my/custom/path/ ollama serve
```
