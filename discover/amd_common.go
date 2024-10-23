//go:build linux || windows

package discover

import (
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/envconfig"
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

func rocmGetVisibleDevicesEnv(gpuInfo []GpuInfo) (string, string) {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "rocm" {
			// TODO shouldn't happen if things are wired correctly...
			slog.Debug("rocmGetVisibleDevicesEnv skipping over non-rocm device", "library", info.Library)
			continue
		}
		ids = append(ids, info.ID)
	}
	return "HIP_VISIBLE_DEVICES", strings.Join(ids, ",")
}

func commonAMDValidateLibDir() (string, error) {
	// Favor our bundled version

	// Installer payload location if we're running the installed binary
	exe, err := os.Executable()
	if err == nil {
		rocmTargetDir := filepath.Join(filepath.Dir(exe), envconfig.LibRelativeToExe(), "lib", "ollama")
		if rocmLibUsable(rocmTargetDir) {
			slog.Debug("detected ROCM next to ollama executable " + rocmTargetDir)
			return rocmTargetDir, nil
		}
	}

	// Prefer explicit HIP env var
	hipPath := os.Getenv("HIP_PATH")
	if hipPath != "" {
		hipLibDir := filepath.Join(hipPath, "bin")
		if rocmLibUsable(hipLibDir) {
			slog.Debug("detected ROCM via HIP_PATH=" + hipPath)
			return hipLibDir, nil
		}
	}

	// Scan the LD_LIBRARY_PATH or PATH
	pathEnv := "LD_LIBRARY_PATH"
	if runtime.GOOS == "windows" {
		pathEnv = "PATH"
	}

	paths := os.Getenv(pathEnv)
	for _, path := range filepath.SplitList(paths) {
		d, err := filepath.Abs(path)
		if err != nil {
			continue
		}
		if rocmLibUsable(d) {
			return d, nil
		}
	}

	// Well known location(s)
	for _, path := range RocmStandardLocations {
		if rocmLibUsable(path) {
			return path, nil
		}
	}

	return "", errors.New("no suitable rocm found, falling back to CPU")
}
