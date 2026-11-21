package mlxrunner

import (
	"log/slog"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/mlx"
)

// Prefill and drafting also allocate buffers outside the cache. Reserve a shared
// 4 GiB: the measured 4k–64k workloads on M2/M5 Macs included about 3 GiB of
// additional temporary prefill storage on the older, smaller Mac.
const parallelWorkspace = 4 << 30

// num_ctx is a sizing hint. Requests can still use the full trained context.
func parallelContextLength(trained, requested int) int {
	length := min(trained, 256*1024)
	if requested > 0 {
		length = min(length, requested)
	}
	return length
}

func defaultParallelism(fullContextBytes, headroom int64) int {
	if fullContextBytes <= 0 || headroom <= parallelWorkspace {
		return 1
	}
	for _, parallel := range []int{8, 4, 2} {
		if fullContextBytes <= (headroom-parallelWorkspace)/int64(parallel) {
			return parallel
		}
	}
	return 1
}

func (r *Runner) configureParallelism() {
	if r.parallel > 0 {
		return
	}
	r.parallel = 1
	// Automatic capacity is qualified on Metal. Other backends retain the
	// existing serial default until their device budgets are qualified too.
	if !mlx.MetalIsAvailable() || r.fullContextBytes <= 0 {
		return
	}
	limit, err := mlx.MaxRecommendedWorkingSetSize()
	if err != nil {
		return
	}
	monitor, err := discover.NewMemoryMonitor()
	if err != nil {
		return
	}
	defer monitor.Close()
	memory, err := monitor.Read()
	if err != nil || memory.Pressure >= discover.PressureElevated || memory.FreeMemoryAboveTarget != nil && *memory.FreeMemoryAboveTarget < 0 {
		return
	}
	headroom := min(int64(limit)-int64(mlx.ActiveMemory()), memory.AvailableRAM)
	r.parallel = defaultParallelism(r.fullContextBytes, headroom)
	slog.Info("MLX automatic parallelism", "parallel", r.parallel, "context_length", r.contextLength,
		"projected_context_length", parallelContextLength(r.contextLength, r.parallelContext),
		"full_context_bytes", r.fullContextBytes, "workspace", parallelWorkspace, "headroom", headroom)
}
