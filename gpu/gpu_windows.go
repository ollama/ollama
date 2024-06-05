package gpu

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

var CudartGlobs = []string{
	"c:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin\\cudart64_*.dll",
}

var NvmlGlobs = []string{
	"c:\\Windows\\System32\\nvml.dll",
}

var NvcudaGlobs = []string{
	"c:\\windows\\system*\\nvcuda.dll",
}

var OneapiGlobs = []string{
	"c:\\Windows\\System32\\DriverStore\\FileRepository\\*\\ze_intel_gpu64.dll",
}

var CudartMgmtName = "cudart64_*.dll"
var NvcudaMgmtName = "nvcuda.dll"
var NvmlMgmtName = "nvml.dll"
var OneapiMgmtName = "ze_intel_gpu64.dll"

func GetCPUMem() (memInfo, error) {
	memStatus := MEMORYSTATUSEX{length: sizeofMemoryStatusEx}
	r1, _, err := globalMemoryStatusExProc.Call(uintptr(unsafe.Pointer(&memStatus)))
	if r1 == 0 {
		return memInfo{}, fmt.Errorf("GlobalMemoryStatusEx failed: %w", err)
	}
	return memInfo{TotalMemory: memStatus.TotalPhys, FreeMemory: memStatus.AvailPhys}, nil
}
