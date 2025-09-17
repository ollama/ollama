//go:build linux || windows

package discover

import (
	"fmt"
	"log/slog"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

// Jetson devices have JETSON_JETPACK="x.y.z" factory set to the Jetpack version installed.
// Included to drive logic for reducing Ollama-allocated overhead on L4T/Jetson devices.
var CudaTegra string = os.Getenv("JETSON_JETPACK")

func cudaVariant(gpuInfos []CudaGPUInfo) string {
	if runtime.GOARCH == "arm64" && runtime.GOOS == "linux" {
		if CudaTegra != "" {
			ver := strings.Split(CudaTegra, ".")
			if len(ver) > 0 {
				return "jetpack" + ver[0]
			}
		} else if data, err := os.ReadFile("/etc/nv_tegra_release"); err == nil {
			r := regexp.MustCompile(` R(\d+) `)
			m := r.FindSubmatch(data)
			if len(m) != 2 {
				slog.Info("Unexpected format for /etc/nv_tegra_release.  Set JETSON_JETPACK to select version")
			} else {
				if l4t, err := strconv.Atoi(string(m[1])); err == nil {
					// Note: mapping from L4t -> JP is inconsistent (can't just subtract 30)
					// https://developer.nvidia.com/embedded/jetpack-archive
					switch l4t {
					case 35:
						return "jetpack5"
					case 36:
						return "jetpack6"
					default:
						slog.Info("unsupported L4T version", "nv_tegra_release", string(data))
					}
				}
			}
		}
	}

	// Check GPU compute capability FIRST, lowest common denominator if multi-gpu
	for _, gpuInfo := range gpuInfos {
		if gpuInfo.computeMajor < 7 || (gpuInfo.computeMajor == 7 && gpuInfo.computeMinor < 5) {
			// GPU is Pascal or older (CC <= 7.4) - use CUDA v12 (supports CC 6.1)
			return "v12"
		}
	}

	// GPU is Turing or newer (CC >= 7.5) - can use newer CUDA
	if len(gpuInfos) > 0 && gpuInfos[0].DriverMajor < 13 {
		// The detected driver is older than 580 (Aug 2025)
		// Warn if their CC is compatible with v13 and they should upgrade their driver to get better performance
		slog.Warn("old CUDA driver detected - please upgrade to a newer driver for best performance", "version", fmt.Sprintf("%d.%d", gpuInfos[0].DriverMajor, gpuInfos[0].DriverMinor))
		return "v12"
	}
	return "v13"
}
