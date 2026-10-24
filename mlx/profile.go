package mlx

import (
	"log/slog"
	"sync/atomic"
)

// profilingEnabled gates emission of GPU profiler phase markers. It is off by
// default; the runner turns it on via the --profile CLI flag. Markers are
// os_signpost intervals on macOS (captured by Instruments / xctrace) and NVTX
// ranges on CUDA/Linux (captured by Nsight Systems). When disabled, the push
// and pop calls are a single atomic load and return, off the hot path.
var profilingEnabled atomic.Bool

// SetProfilingEnabled toggles phase-marker emission. Call it before the MLX
// worker starts; markers are then pushed and popped only from that thread,
// since the native marker state is unsynchronized.
func SetProfilingEnabled(on bool) {
	if on && !profileMarkersAvailable() {
		slog.Warn("GPU profiler markers are unavailable on this system; --profile emits nothing")
	}
	profilingEnabled.Store(on)
}

// ProfileRangePush opens a named profiler range. Ranges nest and must be
// balanced with ProfileRangePop. No-op unless profiling is enabled.
func ProfileRangePush(name string) {
	if !profilingEnabled.Load() {
		return
	}
	profileRangePush(name)
}

// ProfileRangePop closes the most recently opened range. No-op unless enabled.
func ProfileRangePop() {
	if !profilingEnabled.Load() {
		return
	}
	profileRangePop()
}
