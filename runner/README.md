# jsonl

`jsonl` frontend to llama.cpp that accepts input and output as machine-readable, newline-delimited JSON

### Why do this?

- Alternative to C bindings, how most tools integrate with Llama.cpp
- Run the latest, fastest and optimized version of llama.cpp
- Make it easy to subshell llama.cpp from other apps
- Keep a model in memory for several embeddings and generation

### Building

```
cd build
cmake .. -DLLAMA_METAL=1
make
```

### Completions

```
./main -m <model> <other options>
```

To generate completions:

```
{"method": "completion", "prompt": "Names for a pet pelican"}
```

Results will be streamed to stdout:

```
{"content": "Here"}
{"content": " are"}
{"content": " some"}
{"content": " names"}
{"content": " for"}
...
{"end": true}
```

Errors will be streamed to stderr:

```
{"error": "out of memory"}
```

### Embeddings

```
{"method": "embeddings", "prompt": "Names for a pet pelican"}
```

### TODO

- Cancel generation with signals
- Initialize a model with a JSON object with standard model parameters
- Stream load %
