//go:build linux || windows

package discover

import (
	"log/slog"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/ollama/ollama/runners"
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

var (
	hasv11 = false
	hasv12 = false
	once   = sync.Once{}
)

func cudaVariant(gpuInfo CudaGPUInfo) string {
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

	// Adjust algorithm based on available runners
	once.Do(func() {
		noCuda := true
		for name := range runners.GetAvailableServers() {
			if strings.Contains(name, "cuda_v11") {
				slog.Debug("cuda v11 runner detected")
				hasv11 = true
				noCuda = false
			} else if strings.Contains(name, "cuda_v12") {
				slog.Debug("cuda v12 runner detected")
				hasv12 = true
				noCuda = false
			} else if strings.Contains(name, "cuda") {
				noCuda = false
			}
		}
		if noCuda {
			// Detect build from source or other packaging misconfiguration that results in no cuda runners with cuda GPUs detected.
			slog.Warn("no cuda runners detected, unable to run on cuda GPU")
			// TODO - bubble this failure mode up through info API as well
		}
	})

	if (gpuInfo.computeMajor < 6 || gpuInfo.DriverMajor < 12 || (gpuInfo.DriverMajor == 12 && gpuInfo.DriverMinor == 0)) && hasv11 {
		return "v11"
	}
	if hasv12 {
		return "v12"
	}
	return ""
}
