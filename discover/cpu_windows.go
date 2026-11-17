package discover

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
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
	k32                      = windows.NewLazySystemDLL("kernel32.dll")
	globalMemoryStatusExProc = k32.NewProc("GlobalMemoryStatusEx")
	sizeofMemoryStatusEx     = uint32(unsafe.Sizeof(MEMORYSTATUSEX{}))
)

func GetCPUMem() (memInfo, error) {
	memStatus, err := getMemoryStatus()
	if err != nil {
		return memInfo{}, err
	}
	return memInfo{TotalMemory: memStatus.TotalPhys, FreeMemory: memStatus.AvailPhys, FreeSwap: memStatus.AvailPageFile}, nil
}

func getMemoryStatus() (MEMORYSTATUSEX, error) {
	memStatus := MEMORYSTATUSEX{length: sizeofMemoryStatusEx}
	r1, _, err := globalMemoryStatusExProc.Call(uintptr(unsafe.Pointer(&memStatus)))
	if r1 == 0 {
		return MEMORYSTATUSEX{}, fmt.Errorf("GlobalMemoryStatusEx failed: %w", err)
	}
	return memStatus, nil
}

func IsNUMA() bool {
	// numa support in ggml is linux only
	return false
}
