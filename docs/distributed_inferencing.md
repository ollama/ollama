# Distributed Inferencing

Ollama supports distributed using llama.cpp RPC servers.

## Usage

The RPC servers can be specified using the `OLLAMA_RPC_SERVERS` environment variable.
This environment variable contains a comma seperated lists of the RPC servers, e.g. `OLLAMA_RPC_SERVERS="127.0.0.1:50052,192.168.0.69:50053`.
`ollama serve` will take automatically offload the model to the RPC servers when it is set.
See [llama.cpp RPC example](https://github.com/ggerganov/llama.cpp/tree/master/examples/rpc) for instructions on how to start RPC servers.

**Note:** if ollama is having issues connecting to your RPC servers, make sure the RPC server version (commit) is the same as the llama.cpp version used by ollama.

```sh
OLLAMA_RPC_SERVERS="127.0.0.1:50052,192.168.0.69:50053 ollama serve
```
