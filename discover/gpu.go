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

// CudaTegra optionally overrides JetPack detection with the installed JetPack
// version ("x.y.z"). It is read from the JETSON_JETPACK environment variable,
// which is primarily useful inside containers where /etc/nv_tegra_release is
// not present.
var CudaTegra string = os.Getenv("JETSON_JETPACK")

// GetSystemInfo returns host memory information used by scheduling.
func GetSystemInfo() ml.SystemInfo {
	logutil.Trace("performing system memory discovery")
	startDiscovery := time.Now()
	defer func() {
		logutil.Trace("system memory discovery completed", "duration", time.Since(startDiscovery))
	}()

	memInfo, err := GetCPUMem()
	if err != nil {
		slog.Warn("error looking up system memory", "error", err)
	}

	return ml.SystemInfo{
		TotalMemory: memInfo.TotalMemory,
		FreeMemory:  memInfo.FreeMemory,
		FreeSwap:    memInfo.FreeSwap,
	}
}

// cudaJetpack returns the bundled CUDA runner directory suffix for the current
// Jetson host, or "" to use the standard CUDA build.
func cudaJetpack() string {
	if runtime.GOARCH != "arm64" || runtime.GOOS != "linux" {
		return ""
	}

	var tegraRelease []byte
	if CudaTegra == "" {
		// On a device this file records the installed L4T release; in
		// containers it is typically absent, so JETSON_JETPACK is used instead.
		tegraRelease, _ = os.ReadFile("/etc/nv_tegra_release")
	}
	return jetpackRunner(CudaTegra, tegraRelease)
}

// jetpackRunner selects the bundled CUDA runner directory suffix for a Jetson
// host. The JETSON_JETPACK override ("x.y.z") takes precedence; otherwise the
// L4T release major is parsed from the /etc/nv_tegra_release contents. Only
// JetPack 5 and 6 ship dedicated runners ("jetpack5"/"jetpack6"); JetPack 7+
// (L4T r38 on Thor, r39 on Orin, and newer) supports SBSA-based CUDA and uses
// the standard cuda_v13 build, returned here as "".
func jetpackRunner(override string, tegraRelease []byte) string {
	if override != "" {
		// JETSON_JETPACK holds the JetPack version, e.g. "6.1".
		switch major, _, _ := strings.Cut(override, "."); major {
		case "5":
			return "jetpack5"
		case "6":
			return "jetpack6"
		default:
			return "" // JetPack 7+ uses the standard SBSA cuda_v13 build
		}
	}

	if len(tegraRelease) == 0 {
		return ""
	}
	// /etc/nv_tegra_release begins with e.g. "# R36 (release), REVISION: 4.0".
	// The L4T major version maps to JetPack non-arithmetically:
	// https://developer.nvidia.com/embedded/jetpack-archive
	m := regexp.MustCompile(` R(\d+) `).FindSubmatch(tegraRelease)
	if len(m) != 2 {
		slog.Info("unexpected format for /etc/nv_tegra_release; set JETSON_JETPACK to select the version")
		return ""
	}
	l4t, _ := strconv.Atoi(string(m[1]))
	switch {
	case l4t == 35:
		return "jetpack5"
	case l4t == 36:
		return "jetpack6"
	case l4t >= 38:
		// JetPack 7+ (L4T r38/r39 and newer) supports SBSA-based CUDA, so the
		// standard cuda_v13 build is used instead of a Jetson-specific runner.
		return ""
	default:
		slog.Debug("unrecognized L4T version", "nv_tegra_release", string(tegraRelease))
		return ""
	}
}
