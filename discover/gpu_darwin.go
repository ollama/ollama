package discover

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Foundation -framework CoreGraphics -framework Metal
#include "gpu_info_darwin.h"
*/
import "C"

func GetCPUMem() (memInfo, error) {
	return memInfo{
		TotalMemory: uint64(C.getPhysicalMemory()),
		FreeMemory:  uint64(C.getFreeMemory()),
		// FreeSwap omitted as Darwin uses dynamic paging
	}, nil
}
