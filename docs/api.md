---
title: "Introduction"
---

Ollama's API allows you to run and interact with models programatically.

## Get started

If you're just getting started, follow the [quickstart](/quickstart) documentation to get up and running with Ollama's API.

## Base URL

After installation, Ollama's API is served by default at:

```
http://localhost:11434/api
```

For running cloud models on **ollama.com**, the same API is available with the following base URL:

```
https://ollama.com/api
```

## Example request

Once Ollama is running, its API is automatically available and can be accessed via `curl`:

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3",
  "prompt": "Why is the sky blue?"
}'
```

## Libraries

Ollama has official libraries for Python and JavaScript:

- [Python](https://github.com/ollama/ollama-python)
- [JavaScript](https://github.com/ollama/ollama-js)

Several community-maintained libraries are available for Ollama. For a full list, see the [Ollama GitHub repository](https://github.com/ollama/ollama?tab=readme-ov-file#libraries-1).

## Versioning

Ollama's API isn't strictly versioned, but the API is expected to be stable and backwards compatible. Deprecations are rare and will be announced in the [release notes](https://github.com/ollama/ollama/releases).
