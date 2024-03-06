//go:build darwin

package gpu

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Foundation -framework CoreGraphics -framework Metal
#include "gpu_info_darwin.h"
*/
import "C"
import (
	"runtime"
)

// CheckVRAM returns the free VRAM in bytes on Linux machines with NVIDIA GPUs
func CheckVRAM() (int64, error) {
	if runtime.GOARCH == "amd64" {
		// gpu not supported, this may not be metal
		return 0, nil
	}
	recommendedMaxVRAM := int64(C.getRecommendedMaxVRAM())
	return recommendedMaxVRAM, nil
}

func GetGPUInfo() GpuInfo {
	mem, _ := getCPUMem()
	if runtime.GOARCH == "amd64" {
		return GpuInfo{
			Library: "cpu",
			Variant: GetCPUVariant(),
			memInfo: mem,
		}
	}
	return GpuInfo{
		Library: "metal",
		memInfo: mem,
	}
}

func getCPUMem() (memInfo, error) {
	return memInfo{
		TotalMemory: 0,
		FreeMemory:  0,
		DeviceCount: 0,
	}, nil
}
