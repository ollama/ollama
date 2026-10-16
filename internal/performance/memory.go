package performance

import (
	"github.com/shirou/gopsutil/v3/mem"
)

type MemoryStats struct {
	Total     uint64 // in bytes
	Used      uint64 // in bytes
	Available uint64 // in bytes
}

// GetMemoryStats retrieves virtual memory statistics of the system
func GetMemoryStats() (*MemoryStats, error) {
	vm, err := mem.VirtualMemory()
	if err != nil {
		return nil, err
	}
	return &MemoryStats{
		Total:     vm.Total,
		Used:      vm.Used,
		Available: vm.Available,
	}, nil
}
