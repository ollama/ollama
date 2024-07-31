<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 73e3b128c2a287e5cf76ba2f9fcda3086476a289
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

	"github.com/ollama/ollama/format"
)

const (
	metalMinimumMemory = 512 * format.MebiByte
)

func GetGPUInfo() GpuInfoList {
	mem, _ := GetCPUMem()
	if runtime.GOARCH == "amd64" {
		return []GpuInfo{
			{
				Library: "cpu",
<<<<<<< HEAD
				Variant: GetCPUCapability(),
=======
				Variant: GetCPUVariant(),
>>>>>>> 73e3b128c2a287e5cf76ba2f9fcda3086476a289
				memInfo: mem,
			},
		}
	}
	info := GpuInfo{
		Library: "metal",
		ID:      "0",
	}
	info.TotalMemory = uint64(C.getRecommendedMaxVRAM())

	// TODO is there a way to gather actual allocated video memory? (currentAllocatedSize doesn't work)
	info.FreeMemory = info.TotalMemory

	info.MinimumMemory = metalMinimumMemory
	return []GpuInfo{info}
}

<<<<<<< HEAD
func GetCPUInfo() GpuInfoList {
	mem, _ := GetCPUMem()
	return []GpuInfo{
		{
			Library: "cpu",
			Variant: GetCPUCapability(),
			memInfo: mem,
		},
	}
}

=======
>>>>>>> 73e3b128c2a287e5cf76ba2f9fcda3086476a289
func GetCPUMem() (memInfo, error) {
	return memInfo{
		TotalMemory: uint64(C.getPhysicalMemory()),
		FreeMemory:  0,
	}, nil
}

func (l GpuInfoList) GetVisibleDevicesEnv() (string, string) {
	// No-op on darwin
	return "", ""
}
<<<<<<< HEAD
=======
=======
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

	"github.com/ollama/ollama/format"
)

const (
	metalMinimumMemory = 512 * format.MebiByte
)

func GetGPUInfo() GpuInfoList {
	mem, _ := GetCPUMem()
	if runtime.GOARCH == "amd64" {
		return []GpuInfo{
			{
				Library: "cpu",
				Variant: GetCPUCapability(),
				memInfo: mem,
			},
		}
	}
	info := GpuInfo{
		Library: "metal",
		ID:      "0",
	}
	info.TotalMemory = uint64(C.getRecommendedMaxVRAM())

	// TODO is there a way to gather actual allocated video memory? (currentAllocatedSize doesn't work)
	info.FreeMemory = info.TotalMemory

	info.MinimumMemory = metalMinimumMemory
	return []GpuInfo{info}
}

func GetCPUInfo() GpuInfoList {
	mem, _ := GetCPUMem()
	return []GpuInfo{
		{
			Library: "cpu",
			Variant: GetCPUCapability(),
			memInfo: mem,
		},
	}
}

func GetCPUMem() (memInfo, error) {
	return memInfo{
		TotalMemory: uint64(C.getPhysicalMemory()),
		FreeMemory:  0,
	}, nil
}

func (l GpuInfoList) GetVisibleDevicesEnv() (string, string) {
	// No-op on darwin
	return "", ""
}
>>>>>>> 0b01490f7a487eae06890c5eabcd2270e32605a5
>>>>>>> 73e3b128c2a287e5cf76ba2f9fcda3086476a289
