# Llama

## Updating llama.cpp

`LLAMA_CPP_VERSION` pins Ollama's llama.cpp source. An update can change more
than compilation: it can affect model loading, GPU discovery, scheduler inputs,
runtime logs, streaming, and on-disk conversion. Validate the upstream diff,
the source Ollama actually builds, and the affected local paths.

### Workflow

Record the old ref from the base branch and choose an explicit new llama.cpp
tag or commit. After updating `LLAMA_CPP_VERSION`, materialize the source
through Ollama's normal build path:

```sh
cmake -S llama/server --preset cpu
```

This configure step fetches the pinned, unpatched source. Confirm the checkout, usually
`build/llama-server-cpu/_deps/llama_cpp-src`, resolves to the intended new ref.
Do not trust an old or dirty `_deps/` checkout as validation.
This is only a source check; it is not runtime validation.

Review the upstream diff using Git refs from the llama.cpp checkout:

```sh
git diff <old-ref> <new-ref> -- <path>
git show <new-ref>:<path>
```

For build prerequisites, platform notes, and backend selection, see the
[developer guide](../docs/development.md).

### What to review

- Build option and dependency drift: changed `GGML_*` or `LLAMA_*` options,
  new `find_package` calls, generated assets, shader tools, or backend
  dependencies. Compare against `llama/server/CMakeLists.txt`,
  `llama/server/CMakePresets.json`, `cmake/local.cmake`, Dockerfiles, CI, and
  build scripts as needed.
- Backend discovery contracts: GGML symbols used by `discover/native_probe*.go`,
  `ggml_backend_dev_props`, backend device type enums, backend registry loading,
  device ordering, visible-device filtering, and CUDA/ROCm/Vulkan/Metal runtime
  library behavior.
- llama-server contracts: launch args and defaults, status and error payloads,
  memory/offload log lines, `system_info:`, flash-attention logging,
  `--main-gpu`, split-mode behavior, and scheduler-sensitive flags consumed by
  `llm/llama_server.go` or `server/sched.go`.
- Streaming: any new SSE frame shape, heartbeat, keepalive ping, completion
  marker, or response cadence on paths Ollama parses directly.
- Model and conversion surfaces: new architectures, tensor names, GGUF
  metadata, tokenizer behavior, speculative/MTP paths, sampler defaults, and
  server capabilities that may require updates under `model/`,
  `create/`, `llm/`, or `compatmigrate/`. A model load alone is not enough;
  affected paths should run a real request and assert the expected result.

### Native Build

After changes to `llama/server/CMakeLists.txt` or `cmake/local.cmake`, build the CPU target:

```sh
cmake --build build/llama-server-cpu --target llama-server --parallel 12
```

Configure-only validation can miss missing sources, template instantiation
problems, and link errors. Legacy models are converted by `compatmigrate`
before loading; validate those conversions against the new upstream version.

### Local checks

Run the Go tests:

```sh
go test ./...
```
Then proceed to build the full Ollama release and verify.

### End-to-end Testing

For runtime validation, build the full applicable native payload for the
platform using the [developer guide](../docs/development.md): Metal on macOS
arm64, and the available CUDA, ROCm, and Vulkan backends on Linux and Windows.

Then run the [integration tests](../integration/README.md) on the platforms
being validated. Use them to exercise real Ollama requests and inspect logs for
device discovery, offload, memory accounting, flash attention, and
request/response behavior. macOS, Windows, and Linux behavior must be validated
on those platforms.
