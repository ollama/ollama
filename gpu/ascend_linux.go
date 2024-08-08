//go:build linux

package gpu

/*
#cgo linux LDFLAGS: -lrt -lpthread -ldl -lstdc++ -lm
#cgo windows LDFLAGS: -lpthread

#include "gpu_info_ascend.h"

*/
import "C"
import (
	"fmt"
	"log/slog"
	"unsafe"
)

var AscendLinuxGlobs = []string{
	"/usr/local/Ascend/latest/aarch64-linux/lib64/libascendcl.so*",
	"/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/libascendcl.so*",
}

var AscendMgmtName = "libascendcl.so"

var (
	ascendLibPath string
)

type ascendHandles struct {
	ascend      *C.ascend_handle_t
	deviceCount int
}

type AscendGPUInfo struct {
	GpuInfo
	index int //nolint:unused,nolintlint
}
type AscendGPUInfoList []AscendGPUInfo

func initAscendHandles() *ascendHandles {
	aHandles := &ascendHandles{}

	// Short Circuit if we already know which library to use
	if ascendLibPath != "" {
		aHandles.deviceCount, aHandles.ascend, _ = LoadAscendMgmt([]string{ascendLibPath})
		return aHandles
	}

	ascendLibPaths := FindGPULibs(AscendMgmtName, AscendLinuxGlobs)
	if len(ascendLibPaths) > 0 {
		deviceCount, ascend, libPath := LoadAscendMgmt(ascendLibPaths)
		if ascend != nil {
			slog.Debug("detected GPUs", "count", deviceCount, "library", libPath)
			aHandles.ascend = ascend
			aHandles.deviceCount = deviceCount
			ascendLibPath = libPath
			return aHandles
		}
	}

	return aHandles
}

func LoadAscendMgmt(ascendLibPath []string) (int, *C.ascend_handle_t, string) {
	var resp C.ascend_init_resp_t
	resp.ah.verbose = getVerboseState()
	for _, libPath := range ascendLibPath {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.ascend_init(lib, &resp)
		if resp.err != nil {
			slog.Debug(fmt.Sprintf("Unable to load ascend management library %s: %s", libPath, C.GoString(resp.err)))
			C.free(unsafe.Pointer(resp.err))
		} else {
			return int(resp.num_devices), &resp.ah, libPath
		}
	}
	return 0, nil, ""
}
