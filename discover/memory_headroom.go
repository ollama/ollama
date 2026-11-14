//go:build !darwin

package discover

// GetMemoryHeadroom reports host capacity. Device-local memory and compression
// need separate observation before enabling the retention policy on this host.
func GetMemoryHeadroom() (MemoryHeadroom, error) {
	mem, err := GetCPUMem()
	return MemoryHeadroom{Available: int64(mem.FreeMemory), Immediate: int64(mem.FreeMemory)}, err
}
