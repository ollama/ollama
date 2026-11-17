//go:build linux || windows

package discover

import (
	"math"
	"time"
)

// Read reports capacity and new pressure feedback. Failed reads preserve the
// last successful baseline; unavailable activity establishes a new baseline
// when it recovers, without losing valid capacity.
func (m *memoryPlatform) Read() (MemoryHeadroom, error) {
	sample, err := m.read()
	if err != nil {
		return MemoryHeadroom{}, err
	}
	sample.SampledAt = time.Now()
	if !sample.activityKnown {
		m.history.valid = false
		if sample.Pressure < PressureElevated {
			sample.Pressure = PressureUnknown
		}
		return sample.MemoryHeadroom, nil
	}
	reclaim, active, known := m.history.observe(sample.memoryActivity)
	sample.ReclaimBytes = reclaim
	if sample.Pressure != PressureCritical {
		if active {
			sample.Pressure = PressureElevated
		} else if !known && sample.Pressure != PressureElevated {
			sample.Pressure = PressureUnknown
		}
	}
	return sample.MemoryHeadroom, nil
}

// These measurements stay behind the monitor: callers need a bounded reclaim
// estimate, not OS-specific compression stores and cumulative page counters.
type memoryActivity struct {
	compressed   uint64 // Original size of current compressor contents.
	compressions uint64
	pageOut      uint64 // Bytes sent through the swap/page-out path.
}

type memorySample struct {
	MemoryHeadroom
	memoryActivity
	activityKnown bool
}

type memoryHistory struct {
	previous memoryActivity
	valid    bool
}

func (h *memoryHistory) observe(current memoryActivity) (reclaim int64, active, known bool) {
	previous, valid := h.previous, h.valid
	h.previous, h.valid = current, true
	if !valid || current.compressions < previous.compressions || current.pageOut < previous.pageOut {
		return 0, false, false
	}
	growth := current.compressed - min(current.compressed, previous.compressed)
	paging := current.pageOut - previous.pageOut
	// A page can be compressed and then paged out in the same interval. Use
	// the larger signal rather than charging both as independent byte debts.
	return int64(min(max(growth, paging), math.MaxInt64)), growth > 0 || paging > 0 || current.compressions > previous.compressions, true
}
