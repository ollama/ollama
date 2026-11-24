package mlx

// #include "generated.h"
import "C"

import (
	"strconv"
	"strings"
	"sync"
)

// kernelArchitecture is the backend architecture MLX tunes its native kernels
// on, such as "applegpu_g17s"; see appleGPU.
var kernelArchitecture = sync.OnceValue(func() string {
	info := C.mlx_device_info_new()
	if mlxError(info) != nil {
		return ""
	}
	defer freeDeviceInfo(info)
	if mlxError(C.mlx_device_info_get(&info, DefaultDevice().ctx)) != nil {
		return ""
	}
	arch, err := deviceInfoString(info, "architecture")
	if err != nil {
		return ""
	}
	return arch
})

// MLX JITs custom shaders as Metal 4 on macOS 26 and later, with either
// bundled dylib. Earlier drivers cannot compile the TensorOps header.
var hasMetalTensorOps = sync.OnceValue(func() bool {
	return macOSMajorVersion() >= 26 && isAppleGPU()
})

func isAppleGPU() bool {
	generation, _ := appleGPU(kernelArchitecture())
	return generation > 0
}

// appleGPU returns the two device facts MLX's Metal backend tunes on, read from
// an architecture such as "applegpu_g17s": the generation (M2 14 through M5 17)
// and the size class ('g' or 'p' base, 's' Pro and Max, 'd' Ultra). Other
// devices return zeros.
func appleGPU(architecture string) (generation int, size byte) {
	rest, ok := strings.CutPrefix(architecture, "applegpu_g")
	if !ok || len(rest) < 2 {
		return 0, 0
	}
	generation, err := strconv.Atoi(rest[:len(rest)-1])
	if err != nil {
		return 0, 0
	}
	return generation, rest[len(rest)-1]
}
