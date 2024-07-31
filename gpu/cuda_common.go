//go:build linux || windows

package gpu

import (
	"log/slog"
	"os"
	"runtime"
	"strings"
)

// Jetson devices have JETSON_JETPACK="x.y.z" factory set to the Jetpack version installed.
// Included to drive logic for reducing Ollama-allocated overhead on L4T/Jetson devices.
var CudaTegra string = os.Getenv("JETSON_JETPACK")

func cudaGetVisibleDevicesEnv(gpuInfo []GpuInfo) (string, string) {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "cuda" {
			// TODO shouldn't happen if things are wired correctly...
			slog.Debug("cudaGetVisibleDevicesEnv skipping over non-cuda device", "library", info.Library)
			continue
		}
		ids = append(ids, info.ID)
	}
	return "CUDA_VISIBLE_DEVICES", strings.Join(ids, ",")
}

func cudaGetVariant(gpuInfo CudaGPUInfo) string {
	if runtime.GOARCH == "arm64" && CudaTegra != "" {
		ver := strings.Split(CudaTegra, ".")
		if len(ver) > 0 {
			return "jetpack" + ver[0]
		}
	}

	if gpuInfo.computeMajor < 6 || gpuInfo.DriverMajor < 12 {
		return "v11"
	}
	return "v12"
}
