# Performance Tuning

Part of the MLX porting guide (`x/models/PORTING_GUIDE.md`). Bring-up
evidence lives beside these docs in `x/models/bringup/<model>/` — git-ignored,
never committed; see `artifacts.md`.

Treat performance as a second pass after the correctness gate. Start with
a baseline rather than guesses:

- use Ollama response timings (`prompt_eval_duration`, `eval_duration`, token
  counts) to record prefill and decode throughput separately
- keep prompts, `num_ctx`, `num_predict`, batch settings, model tag, dtype, and
  hardware fixed across runs
- unload other large models and record whether BF16, FP8, or another quantized
  variant is under test

Rebase onto the current MLX revision before retaining an optimization. Kernel
selection changes quickly, so a model-local workaround that was once necessary
may now be slower than the maintained path.

Use this optimization order:

1. Reshape the model forward pass around existing MLX operations and layouts.
2. Put model-specific compositions of existing operations in `mlx.Compile`
   closures. Closures remain readable and let MLX JIT the graph for the active
   backend and hardware.
3. If the bottleneck is a special case of an existing MLX operation, propose or
   validate an upstream MLX optimization instead of forking that operation in
   Ollama.
4. Add a custom kernel only when profiling and A/B benchmarks show that the
   first three options cannot meet the target. Record its incremental benefit,
   correctness evidence, supported backends and OS versions, and fallback.

Start with established MLX fast paths:

- use `mlx.RMSNormFn` or `nn.RMSNorm` for RMSNorm, not hand-written variance
  math
- use `mlx.RoPEWithBase` or `mlx.RoPEWithFreqs` for rotary embeddings
- use `mlx.ScaledDotProductAttentionCausal` or
  `mlx.ScaledDotProductAttentionMasked` for attention when the model semantics
  fit the fast SDPA path
- use compiled activation helpers such as `mlx.SwiGLU`, `mlx.GeGLU`, and
  `mlx.GELUApprox` instead of unfused element-wise chains
- reuse per-forward artifacts such as sliding-window masks when every layer
  sees the same shape and dtype
- use `GatherMM` or `GatherQMM` for MoE dispatch and sort expert indices for
  sufficiently large prefills when the model layout supports it

When replacing an element-wise chain, first compile the equivalent MLX graph
and compare it against the eager graph. Do not translate the chain directly
into Metal source. When an existing matrix operation is slow, inspect which
shape or dtype selects the slow path and test current upstream MLX before
adding a parallel matrix kernel to Ollama.

Host-side tensor construction must be sub-quadratic in tokens, patches, or
windows. Never materialize an O(n²) mask or table in Go and upload it: build
the small O(n) or O(blocks²) precursors host-side and expand on device
(`Take`, broadcast, `arange`, compiled closures). Audit every port for
`make([]T, n*n)`-shaped allocations and `FromValues` payloads that scale with
the square of sequence or patch count — a quadratic host-side mask passes
every numerical-parity test and tok/s benchmark unnoticed, because neither
instrument sees allocation shape. Record
peak runner memory (the pipeline "peak memory" log line) and host→device
upload volume for a representative large-media or long-context request
alongside throughput.

For GPU-level profiling on Apple Silicon, prefer headless `xctrace` Metal
System Trace captures. Capture prefill and decode separately; mixed captures
hide the bottleneck because prefill is usually GEMM-heavy while decode is often
memory-bandwidth bound. Look for many tiny dispatches, CPU gaps between GPU
kernels, unexpected dtype upcasts, repeated host-side tensor construction, and
manual element-wise sequences that should be compiled or replaced with
`mlx.fast` operations.

Only optimize with evidence. Record each step's before/after command,
throughput, memory footprint when available, and any xctrace findings in the
reviewer report. Run correctness checks before every benchmark. If a custom
kernel does not provide a material incremental win over compiled closures and
current standard MLX operations, remove it. Then re-run the relevant
layer/cache/quantized/renderer tests to prove the selected optimization did not
change model behavior.

Promote every cost quantified during profiling or a performance investigation
that is not the current culprit to an explicit follow-up item in the bring-up
artifacts and the review report — never leave it as narration.
Measured-and-dismissed costs (an upload, a host allocation, a re-encode) are
the cheapest future optimizations and are otherwise lost when the
investigation closes.
