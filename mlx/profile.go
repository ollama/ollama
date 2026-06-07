package mlx

import "sync/atomic"

// profilingEnabled gates emission of GPU profiler phase markers. It is off by
// default; the runner turns it on via the --profile CLI flag. Markers are
// os_signpost intervals on macOS (captured by Instruments / xctrace) and NVTX
// ranges on CUDA/Linux (captured by Nsight Systems). When disabled, the push
// and pop calls are a single atomic load and return, off the hot path.
var profilingEnabled atomic.Bool

// SetProfilingEnabled toggles phase-marker emission. Safe to call from any
// goroutine.
func SetProfilingEnabled(on bool) { profilingEnabled.Store(on) }

// ProfilingEnabled reports whether phase markers are being emitted.
func ProfilingEnabled() bool { return profilingEnabled.Load() }

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
