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

// GetMemoryHeadroom observes host capacity and compression without changing
// GetCPUMem's existing scheduler estimate.
func GetMemoryHeadroom() (MemoryHeadroom, error) {
	var available, immediate, compressed C.uint64_t
	if C.getMemoryHeadroom(&available, &immediate, &compressed) == 0 {
		return MemoryHeadroom{}, errors.New("reading host memory statistics")
	}
	pages, err := unix.SysctlUint32("vm.vm_page_free_target")
	if err != nil {
		return MemoryHeadroom{}, err
	}
	return MemoryHeadroom{
		Available:  int64(available),
		Immediate:  memoryHeadroom(uint64(immediate), uint64(pages)*uint64(os.Getpagesize())),
		Compressed: int64(compressed),
	}, nil
}

func memoryHeadroom(available, freeTarget uint64) int64 {
	return int64(available) - int64(freeTarget)
}
