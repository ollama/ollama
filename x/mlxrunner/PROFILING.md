# Profiling the MLX runner

Pointers for getting clean, repeatable performance data out of the MLX runner on
Metal and CUDA. The `cmd/bench` tool drives a runner directly (bypassing
`ollama serve`) so the process under a profiler is the runner itself — no
serve→runner fork to confuse `nsys`/`rocprofv3`/`xctrace`.

## The bench driver (both platforms)

Point `bench` at a runner you started yourself, or let it spawn one:

```bash
# Connect to a runner you started (e.g. under a profiler). Auto-detects whether
# the endpoint is an MLX runner or a llama-server.
go run ./cmd/bench -model <model> -runner 127.0.0.1:8081 -mode decode -ignore-eos

# Let bench spawn the runner. An MLX model launches the MLX runner; a GGUF model
# (a .gguf path, or an ollama name like llama3.2:latest) launches a llama-server.
go run ./cmd/bench -model <model> -spawn -mode decode -ignore-eos
```

Key flags:

- `-mode prefill|decode|both` — isolate a single phase so a capture window
  contains one workload. `prefill` varies the prompt per epoch (cache miss) and
  generates 0 tokens; `decode` holds the prompt fixed so the prefix cache hits
  and the window is pure decode; `both` is a normal mixed run.
- `-ignore-eos` — disable stop tokens so generation runs exactly `-max-tokens`,
  making a capture attributable to a known number of decode passes.
- `-prompt-tokens N` / `-max-tokens N` — shape the prompt/generation.

**Prefer two separate captures (`-mode prefill` and `-mode decode`) over one
mixed run** — prefill (compute-bound GEMMs) and decode (bandwidth-bound batch-1
matvecs) have different bottlenecks and aggregate counters average across them.

## Metal (Apple Silicon)

Throughput numbers: just run `bench` as above; it reports per-phase tok/s.

Per-kernel timeline — `xctrace` Metal System Trace, attached to the runner:

```bash
RUNNER_PID=$(lsof -ti :8081)
xctrace record --template 'Metal System Trace' --attach $RUNNER_PID \
  --time-limit 10s --output /tmp/decode.trace
# ...drive load with `bench -mode decode` during the window...
xctrace export --input /tmp/decode.trace --toc          # list tables
```

- **Named kernels:** by default MLX dispatches show as generic "Compute
  Command". Build with `cmake -B build -DMLX_METAL_DEBUG=on .` (the `MLX_`
  prefix is forwarded to the MLX sub-build) and rebuild; dispatches then carry
  the real op graph (`QuantizedMatmul`, `ScaledDotProductAttention`, `RMSNorm`,
  `RoPE`, …). The debug metallib is larger and can cost performance — turn it
  back off (`-DMLX_METAL_DEBUG=off`) for timing runs.
- **Phase markers (optional):** start the runner with `--profile` to emit
  `os_signpost` intervals (`com.ollama.mlx`, points-of-interest) around prefill
  and decode. The `Metal System Trace` template does **not** capture these — use
  a template that includes the os_signpost instrument (e.g. `Logging` or
  `System Trace`) if you want phase intervals. For most work the `-mode`
  isolation above is simpler.

## CUDA (Linux)

Start the runner under Nsight Systems (or attach), with `--profile` so it emits
NVTX ranges; `nsys` captures these natively in the same trace:

```bash
nsys profile --trace=nvtx,cuda,osrt --output /tmp/decode \
  ollama runner --mlx-engine --model <model> --port 8081 --profile
# ...drive load with `bench -runner 127.0.0.1:8081 -mode decode -ignore-eos`...
nsys stats /tmp/decode.nsys-rep --report cuda_gpu_kern_sum
```

The `prefill` / `decode` NVTX ranges show up on the timeline and bound the right
kernels with no extra setup — this is the primary phase-attribution mechanism on
CUDA. Use `ncu` only when you need per-kernel hardware counters.

System-wide capture is an alternative when attaching is awkward; it requires
`perf_event_paranoid=0` and `--cuda-trace-scope=system-wide`.

## Notes

- The runner serves one model and fixes context length at load; `-num-ctx` is
  ignored in direct mode.
- Markers default off and are a near-no-op unless `--profile` is set, so they do
  not affect normal benchmarking.
- ROCm is not a shipping target yet; the Linux marker path leaves a stub for a
  future `roctx` (rocprofv3) backend.
