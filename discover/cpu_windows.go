package discover

import (
	"fmt"
	"syscall"
	"unsafe"
)

type MEMORYSTATUSEX struct {
	length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

var (
	k32                      = syscall.NewLazyDLL("kernel32.dll")
	globalMemoryStatusExProc = k32.NewProc("GlobalMemoryStatusEx")
	sizeofMemoryStatusEx     = uint32(unsafe.Sizeof(MEMORYSTATUSEX{}))
)

func GetCPUMem() (memInfo, error) {
	memStatus := MEMORYSTATUSEX{length: sizeofMemoryStatusEx}
	r1, _, err := globalMemoryStatusExProc.Call(uintptr(unsafe.Pointer(&memStatus)))
	if r1 == 0 {
		return memInfo{}, fmt.Errorf("GlobalMemoryStatusEx failed: %w", err)
	}
	return memInfo{TotalMemory: memStatus.TotalPhys, FreeMemory: memStatus.AvailPhys, FreeSwap: memStatus.AvailPageFile}, nil
}

func IsNUMA() bool {
	// numa support in ggml is linux only
	return false
}
