package discover

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Foundation -framework CoreGraphics -framework Metal
#include "gpu_info_darwin.h"
*/
import "C"

import (
	"log/slog"
	"syscall"

	"github.com/ollama/ollama/format"
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

func GetCPUDetails() []CPU {
	query := "hw.perflevel0.physicalcpu"
	perfCores, err := syscall.SysctlUint32(query)
	if err != nil {
		slog.Warn("failed to discover physical CPU details", "query", query, "error", err)
	}
	query = "hw.perflevel1.physicalcpu"
	efficiencyCores, _ := syscall.SysctlUint32(query) // On x86 xeon this wont return data

	// Determine thread count
	query = "hw.logicalcpu"
	logicalCores, _ := syscall.SysctlUint32(query)

	return []CPU{
		{
			CoreCount:           int(perfCores + efficiencyCores),
			EfficiencyCoreCount: int(efficiencyCores),
			ThreadCount:         int(logicalCores),
		},
	}
}

func IsNUMA() bool {
	// numa support in ggml is linux only
	return false
}
