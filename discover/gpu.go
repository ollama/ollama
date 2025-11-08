package discover

import (
	"log/slog"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// Jetson devices have JETSON_JETPACK="x.y.z" factory set to the Jetpack version installed.
// Included to drive logic for reducing Ollama-allocated overhead on L4T/Jetson devices.
var CudaTegra string = os.Getenv("JETSON_JETPACK")

// GetSystemInfo returns the last cached state of the GPUs on the system
func GetSystemInfo() ml.SystemInfo {
	logutil.Trace("performing CPU discovery")
	startDiscovery := time.Now()
	defer func() {
		logutil.Trace("CPU discovery completed", "duration", time.Since(startDiscovery))
	}()

	memInfo, err := GetCPUMem()
	if err != nil {
		slog.Warn("error looking up system memory", "error", err)
	}
	var threadCount int
	cpus := GetCPUDetails()
	for _, c := range cpus {
		threadCount += c.CoreCount - c.EfficiencyCoreCount
	}

	if threadCount == 0 {
		// Fall back to Go's num CPU
		threadCount = runtime.NumCPU()
	}

	return ml.SystemInfo{
		ThreadCount: threadCount,
		TotalMemory: memInfo.TotalMemory,
		FreeMemory:  memInfo.FreeMemory,
		FreeSwap:    memInfo.FreeSwap,
	}
}

func cudaJetpack() string {
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
						// Newer Jetson systems use the SBSU runtime
						slog.Debug("unrecognized L4T version", "nv_tegra_release", string(data))
					}
				}
			}
		}
	}
	return ""
}
