//go:build aix

package discover

/*
#include <unistd.h>
#include <sys/types.h>
#include <sys/vminfo.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func GetCPUMem() (memInfo, error) {
	var mem memInfo
	var vmi C.struct_vminfo
	if ret := C.vmgetinfo(unsafe.Pointer(&vmi), C.VMINFO, C.sizeof_struct_vminfo); ret != 0 {
		return mem, fmt.Errorf("vmgetinfo failed: %d", ret)
	}
	pageSize := uint64(C.sysconf(C._SC_PAGE_SIZE))
	mem.TotalMemory = uint64(vmi.memsizepgs) * pageSize
	mem.FreeMemory = uint64(vmi.numfrb) * pageSize
	return mem, nil
}

func IsNUMA() bool {
	return false
}
