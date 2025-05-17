# Distributed Inferencing

Ollama supports distributed inferencing using llama.cpp RPC servers.

# Getting Started

Start the RPC server on each machine:
```
ollama rpc [--host 0.0.0.0] [--port 50052]
```

## Configure with Environment Variable

Specify the RPC servers using the OLLAMA_RPC_SERVERS environment variable. Provide a comma-separated list of `host:port` pairs:

```sh
OLLAMA_RPC_SERVERS="127.0.0.1:50052,192.168.0.69:50053" ollama serve
```

## Override with Request Options

You can override the RPC servers per request using the `rpc_servers `option in the API call:

```sh
curl http://localhost:11434/api/generate --json '{
  "model": "llama3.1",
  "prompt": "hello",
  "stream": false,
  "options": {
    "rpc_servers": "127.0.0.1:50053"
  }
}'
```

Ollama will use the RPC server `127.0.0.1:50053` instead of the servers set by `OLLAMA_RPC_SERVERS` environment variable.
