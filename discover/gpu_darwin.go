package discover

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Foundation -framework CoreGraphics -framework Metal
#include "gpu_info_darwin.h"
*/
import "C"

import (
	"errors"
	"os"
	"time"

	"github.com/ollama/ollama/format"
	"golang.org/x/sys/unix"
)

const (
	metalMinimumMemory = 512 * format.MebiByte
)

func GetCPUMem() (memInfo, error) {
	return memInfo{
		TotalMemory: uint64(C.getPhysicalMemory()),
		FreeMemory:  uint64(C.getFreeMemory()),
		// FreeSwap omitted as Darwin uses dynamic paging
	}, nil
}

func IsNUMA() bool {
	// numa support in ggml is linux only
	return false
}

// readMemoryHeadroom keeps the existing scheduler estimate separate.
func readMemoryHeadroom() (MemoryHeadroom, int64, error) {
	var available, immediate, compressed C.uint64_t
	if C.getMemoryHeadroom(&available, &immediate, &compressed) == 0 {
		return MemoryHeadroom{}, 0, errors.New("reading host memory statistics")
	}
	pages, err := unix.SysctlUint32("vm.vm_page_free_target")
	if err != nil {
		return MemoryHeadroom{}, 0, err
	}
	margin := memoryHeadroom(uint64(immediate), uint64(pages)*uint64(os.Getpagesize()))
	pressure := PressureNormal
	if margin < 0 {
		pressure = PressureElevated
	}
	physical := uint64(C.getPhysicalMemory())
	reclaimable := uint64(available)
	// Free and file-backed pages omit inactive anonymous memory, which macOS
	// reclaims without pressure. Also accept the kernel's own estimate; the
	// free-target margin and compression feedback still report real pressure.
	if level, err := unix.SysctlUint32("kern.memorystatus_level"); err == nil {
		reclaimable = kernelAvailableMemory(reclaimable, physical, level)
	}
	return MemoryHeadroom{
		SampledAt: time.Now(), RAMLimit: int64(physical),
		AvailableRAM: int64(reclaimable), FreeMemoryAboveTarget: &margin,
		Pressure: pressure,
	}, int64(compressed), nil
}

func memoryHeadroom(available, freeTarget uint64) int64 {
	return int64(available) - int64(freeTarget)
}

// kernelAvailableMemory returns the larger of the page-count estimate and the
// kernel's available-memory percentage (kern.memorystatus_level) of physical RAM.
func kernelAvailableMemory(pageEstimate, physical uint64, level uint32) uint64 {
	if level == 0 || level > 100 {
		return pageEstimate
	}
	return max(pageEstimate, physical*uint64(level)/100)
}
