package mlx

// #include "generated.h"
import "C"

import (
	"strings"
	"sync"
)

// MLX exposes the backend device name and architecture that select its
// native CUDA and Metal reductions.
var kernelDevice = sync.OnceValues(func() (string, string) {
	info := C.mlx_device_info_new()
	if mlxError(info) != nil {
		return "", ""
	}
	defer freeDeviceInfo(info)
	if mlxError(C.mlx_device_info_get(&info, DefaultDevice().ctx)) != nil {
		return "", ""
	}
	name, err := deviceInfoString(info, "device_name")
	if err != nil {
		return "", ""
	}
	arch, err := deviceInfoString(info, "architecture")
	if err != nil {
		return "", ""
	}
	return name, arch
})

func isM5Pro(name, architecture string) bool {
	return name == "Apple M5 Pro" && architecture == "applegpu_g17s"
}

func isM5MatrixDevice(name, architecture string) bool {
	return (name == "Apple M5 Pro" || name == "Apple M5 Max") && architecture == "applegpu_g17s" ||
		name == "Apple M5 Ultra" && architecture == "applegpu_g17d"
}

// MLX JITs custom shaders as Metal 4 on macOS 26 and later, with either
// bundled dylib. Earlier drivers cannot compile the TensorOps header.
var hasMetalTensorOps = sync.OnceValue(func() bool {
	return macOSMajorVersion() >= 26 && isAppleGPU()
})

func isAppleGPU() bool {
	_, arch := kernelDevice()
	return strings.HasPrefix(arch, "applegpu_")
}
