//go:build !darwin

package discover

import "time"

type memoryPlatform struct{}

func newMemoryMonitor() (*MemoryMonitor, error) { return &MemoryMonitor{}, nil }

// Read preserves the existing CPU-memory fallback on non-Metal platforms.
func (m *memoryPlatform) Read() (MemoryHeadroom, error) {
	observation, _, err := readMemoryHeadroom()
	if err != nil {
		return MemoryHeadroom{}, err
	}
	return observation, nil
}

// Close releases observation resources. Periodic observers own no resources.
func (m *memoryPlatform) Close() {}

// Retention remains disabled until device-local budgets and host pressure have
// been qualified on these platforms.
func readMemoryHeadroom() (MemoryHeadroom, int64, error) {
	mem, err := GetCPUMem()
	return MemoryHeadroom{SampledAt: time.Now(), RAMLimit: int64(mem.TotalMemory), AvailableRAM: int64(mem.FreeMemory)}, 0, err
}
