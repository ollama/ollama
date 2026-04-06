## ps: report actual layer counts instead of percentage

Fixes #7602

### Summary

Changes `ollama ps` to display the number of GPU-offloaded layers out of
total layers (e.g. `33/35 GPU`) instead of a percentage (e.g. `100% GPU`).

This provides more actionable information when tuning the `num_gpu` parameter,
and makes the total layer count of a model visible at a glance.

### Before

```
NAME        ID            SIZE    PROCESSOR         UNTIL
llama3:70b  bcfb190ca3a7  42 GB   100% GPU          4 minutes from now
llama3:8b   ab2c3d4e5f67  5 GB    48%/52% CPU/GPU   3 minutes from now
```

### After

```
NAME        ID            SIZE    PROCESSOR        UNTIL
llama3:70b  bcfb190ca3a7  42 GB   81/81 GPU        4 minutes from now
llama3:8b   ab2c3d4e5f67  5 GB    16/33 CPU/GPU    3 minutes from now
```

### Changes

- **`llm/server.go`**: Add `GPULayerCount()` and `TotalLayerCount()` to `LlamaServer` interface, implemented on `llmServer`
- **`server/sched.go`**: Store layer counts in `runnerRef` at load time
- **`api/types.go`**: Add `gpu_layers` and `total_layers` fields to `ProcessModelResponse`
- **`server/routes.go`**: Populate new fields in `PsHandler`
- **`cmd/cmd.go`**: Update `ListRunningHandler` to display `N/M GPU|CPU|CPU/GPU` format
- **Docs**: Updated `api.md`, `openapi.yaml`, `faq.mdx`
- **Tests**: Updated `mockLlm` in `sched_test.go`

### API Changes

The `/api/ps` response now includes two new fields per model:

```json
{
  "models": [
    {
      "name": "llama3:70b",
      "size": 42000000000,
      "size_vram": 42000000000,
      "gpu_layers": 81,
      "total_layers": 81,
      "context_length": 2048
    }
  ]
}
```

### How to Build

```bash
go build ./...
```

### How to Test

**Unit tests:**

```bash
go test ./server/ -run TestSched -count=1 -timeout 30s
go test ./api/ -count=1 -timeout 30s
```

**Manual testing:**

1. Start the server: `go run . serve`
2. Load a model: `go run . run llama3.2:1b --keepalive 5m`
3. Check output: `go run . ps`
4. Check API: `curl -s http://localhost:11434/api/ps | python3 -m json.tool`

**Partial offload** (set `num_gpu` lower than total layers):
```
PROCESSOR
5/17 CPU/GPU
```

**CPU-only** (`CUDA_VISIBLE_DEVICES=""`):
```
PROCESSOR
17/17 CPU
```

**Full GPU:**
```
PROCESSOR
17/17 GPU
