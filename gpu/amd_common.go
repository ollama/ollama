//go:build linux || windows

package gpu

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Determine if the given ROCm lib directory is usable by checking for existence of some glob patterns
func rocmLibUsable(libDir string) bool {
	slog.Debug("evaluating potential rocm lib dir " + libDir)
	for _, g := range ROCmLibGlobs {
		res, _ := filepath.Glob(filepath.Join(libDir, g))
		if len(res) == 0 {
			return false
		}
	}
	return true
}

func GetSupportedGFX(libDir string) ([]string, error) {
	var ret []string
	files, err := filepath.Glob(filepath.Join(libDir, "rocblas", "library", "TensileLibrary_lazy_gfx*.dat"))
	if err != nil {
		return nil, err
	}
	for _, file := range files {
		ret = append(ret, strings.TrimSuffix(strings.TrimPrefix(filepath.Base(file), "TensileLibrary_lazy_"), ".dat"))
	}
	return ret, nil
}

func amdSetVisibleDevices(ids []int, skip map[int]interface{}) {
	// Set the visible devices if not already set
	// TODO - does sort order matter?
	devices := []string{}
	for i := range ids {
		if _, skipped := skip[i]; skipped {
			continue
		}
		devices = append(devices, strconv.Itoa(i))
	}

	val := strings.Join(devices, ",")
	err := os.Setenv("HIP_VISIBLE_DEVICES", val)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to set env: %s", err))
	} else {
		slog.Info("Setting HIP_VISIBLE_DEVICES=" + val)
	}
}
