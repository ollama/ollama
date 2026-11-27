package discover

type memoryPlatform struct {
	compressed int64
	observed   bool
}

func newMemoryMonitor() (*MemoryMonitor, error) { return &MemoryMonitor{}, nil }

// Read reports capacity and new pressure feedback. Failed reads preserve the
// last successful baseline; pre-existing compression is not new pressure.
func (m *memoryPlatform) Read() (MemoryHeadroom, error) {
	observation, compressed, err := readMemoryHeadroom()
	if err != nil {
		return MemoryHeadroom{}, err
	}
	observation.ReclaimBytes = m.observeCompression(compressed)
	if observation.ReclaimBytes > 0 {
		observation.Pressure = PressureElevated
	}
	return observation, nil
}

func (m *memoryPlatform) observeCompression(current int64) int64 {
	var growth int64
	if m.observed {
		growth = max(0, current-m.compressed)
	}
	m.compressed, m.observed = current, true
	return growth
}

// Close releases observation resources. Periodic observers own no resources.
func (m *memoryPlatform) Close() {}
