package discover

import "time"

// MemoryHeadroom describes host memory available to the calling process.
// All memory quantities are bytes; device budgets are separate constraints.
type MemoryHeadroom struct {
	SampledAt time.Time // When the OS query completed; its counters may update less often.

	// RAM ceiling available to the process: system RAM,
	// capped by OS-imposed process or service allowances. This is the total
	// allowance before considering current usage.
	RAMLimit int64

	// Estimated additional RAM we can use now, accounting for system use and
	// remaining room under those same allowances. Includes reclaimable RAM,
	// but not swap. The runner still subtracts safety and workspace reserves.
	AvailableRAM int64

	// Additional memory the process can commit before reaching an enforced
	// allocation limit. This may include page-file backing, so it is separate
	// from AvailableRAM. Nil means no applicable commitment limit is exposed.
	AvailableCommit *int64

	// Signed margin above the OS's target for immediately available RAM.
	// Negative means a shortfall even if AvailableRAM includes reclaimable
	// pages. Nil means the OS does not expose an equivalent target.
	FreeMemoryAboveTarget *int64

	// Estimated memory relief suggested by new compression or paging pressure
	// since the previous successful Read. Each nonnegative amount is reported
	// once. Zero means no new byte-sized estimate. The runner decides how much
	// of its own cold history to yield.
	ReclaimBytes int64

	// OS pressure after interpreting notifications, compression and paging.
	// GPU budgets and the runner's workspace requirements remain separate.
	Pressure PressureLevel
}

type PressureLevel uint8

const (
	PressureUnknown  PressureLevel = iota // No reliable pressure observation.
	PressureNormal                        // No current pressure signal.
	PressureElevated                      // The OS is reclaiming memory.
	PressureCritical                      // The OS reports an urgent shortage.
)

// MemoryMonitor owns host observations for one consumer. Create it with
// NewMemoryMonitor. Read calls must be serialized and complete before Close.
// Failed reads preserve the last successful baseline. Platform state is private.
type MemoryMonitor struct {
	memoryPlatform
}

// NewMemoryMonitor initializes OS resources without consuming a sample.
// The first successful Read establishes the pressure baseline.
func NewMemoryMonitor() (*MemoryMonitor, error) {
	return newMemoryMonitor()
}
