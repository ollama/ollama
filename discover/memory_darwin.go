package discover

type memoryPlatform struct {
	compressed int64
	swappedOut int64
	observed   bool
}

func newMemoryMonitor() (*MemoryMonitor, error) { return &MemoryMonitor{}, nil }

// Read reports capacity and new pressure feedback. Failed reads preserve the
// last successful baseline; existing compression and swap-outs are not new pressure.
func (m *memoryPlatform) Read() (MemoryHeadroom, error) {
	observation, compressed, swappedOut, err := readMemoryHeadroom()
	if err != nil {
		return MemoryHeadroom{}, err
	}
	observation.ReclaimBytes = m.observePressure(compressed, swappedOut)
	if observation.ReclaimBytes > 0 {
		observation.Pressure = PressureElevated
	}
	return observation, nil
}

func (m *memoryPlatform) observePressure(compressed, swappedOut int64) int64 {
	var growth int64
	if m.observed {
		// Swapping can reduce the compressor's resident count even while the
		// system remains under pressure. Report new swap-outs independently so
		// the consumer can release cold history in that situation as well.
		growth = max(0, compressed-m.compressed) + max(0, swappedOut-m.swappedOut)
	}
	m.compressed, m.swappedOut, m.observed = compressed, swappedOut, true
	return growth
}

// Close releases observation resources. Periodic observers own no resources.
func (m *memoryPlatform) Close() {}
