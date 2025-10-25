# Logging & Instrumentation Guide (Vision Debug)

## Goals
Enhance observability when diagnosing missing vision capability or runtime failures in image processing.

## Built-in Environment Variables
| Variable | Effect | Notes |
| -------- | ------ | ----- |
| `OLLAMA_DEBUG=1` | Basic debug logging | Low verbosity |
| `OLLAMA_DEBUG=2` | High verbosity (recommended) | Includes model loading paths |
| `OLLAMA_NUM_PARALLEL` | Controls concurrency | Set to `1` to simplify traces |

## Key Log Patterns
| Pattern | Interpretation |
| ------- | -------------- |
| `clip_model_loader: has vision encoder` | Vision encoder detected (good) |
| `couldn't open model file` | GGUF open error; capability derivation may fail |
| `one or more GPUs ... unable to accurately report free memory` | GPU scheduling heuristic fallback (not root cause but may affect performance) |
| `standalone vision model` (from create path) | Model categorized as vision-only |

## Temporary Code Instrumentation
Add to `server/images.go` inside `Capabilities()` after opening GGUF:
```go
slog.Debug("capability.scan.start", "model", m.Name)
if f.KeyValue("vision.block_count").Valid() {
  slog.Debug("capability.vision.key", "model", m.Name, "block_count_valid", true)
} else {
  slog.Debug("capability.vision.key", "model", m.Name, "block_count_valid", false)
}
slog.Debug("capability.scan.end", "model", m.Name, "caps", capabilities)
```
Remove after root cause analysis.

## Client Request Logging
Capture full JSON payload sent to `/api/chat` or `/api/generate` to confirm image parts present. Differences in structure (e.g. missing `type":"image"`) can mimic capability absence.

## Trace Artifacts to Keep
| Artifact | Why |
| -------- | --- |
| `show` JSON (working vs failing) | Capability diff |
| Vision inference request JSON | Confirm structure |
| Full server log (debug=2) | Metadata & load sequence |
| GGUF key dump (strings / inspector) | Presence of vision keys |
| Docker image digest & `ollama version` | Map to code commit |

## Performance Counters (Optional)
Add timing around model load and first inference to detect regression side-effects:
```go
start := time.Now()
// call Capabilities()
slog.Debug("timing.capabilities", "elapsed_ms", time.Since(start).Milliseconds())
```

## Reducing Noise
Disable unrelated features in prompts (tools, thinking) to limit branches executed.

## Failure Taxonomy
| Symptom | Likely Zone |
| ------- | ----------- |
| Capability missing, no load errors | Metadata absent |
| Capability missing + file open error | I/O / path issue |
| Capability present, inference error referencing images | Request format / projector runtime |
| Vision present but OOM | Memory sizing / scheduling (check `sched.go`) |

## Escalation Checklist
1. Provide digest, model name, and `ollama version`.
2. Include snippet of debug logs around model load.
3. Attach key presence report for `vision.block_count`.
4. State whether projector used (list `ProjectorPaths`).
