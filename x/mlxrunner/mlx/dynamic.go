//go:build mlx

package mlx

// #include "dynamic.h"
// #include "generated.h"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"
)

var initError error

// CheckInit returns any error that occurred during MLX dynamic library initialization.
func CheckInit() error {
	return initError
}

// tryLoadFromDir searches a directory for libmlxc.* and tries to load it.
// Returns true if the library was successfully loaded.
func tryLoadFromDir(dir string) bool {
	matches, err := fs.Glob(os.DirFS(dir), "libmlxc.*")
	if err != nil || len(matches) == 0 {
		return false
	}

	for _, match := range matches {
		path := filepath.Join(dir, match)

		cPath := C.CString(path)
		defer C.free(unsafe.Pointer(cPath))

		var handle C.mlx_dynamic_handle
		if C.mlx_dynamic_load(&handle, cPath) != 0 {
			slog.Error("Failed to load MLX dynamic library", "path", path)
			continue
		}

		if C.mlx_dynamic_load_symbols(handle) != 0 {
			slog.Error("Failed to load MLX dynamic library symbols", "path", path)
			C.mlx_dynamic_unload(&handle)
			continue
		}

		return true
	}
	return false
}

func init() {
	switch runtime.GOOS {
	case "darwin":

	case "windows":
	default:
		return
	}

	// Try OLLAMA_LIBRARY_PATH first
	if paths, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH"); ok {
		for _, dir := range filepath.SplitList(paths) {
			if tryLoadFromDir(dir) {
				return
			}
		}
	}

	// Build search paths: executable directory, then build directories
	var searchDirs []string
	if exe, err := os.Executable(); err == nil {
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		searchDirs = append(searchDirs, filepath.Dir(exe))
	}

	if cwd, err := os.Getwd(); err == nil {
		searchDirs = append(searchDirs, filepath.Join(cwd, "build", "lib", "ollama"))
	}

	for _, dir := range searchDirs {
		if tryLoadFromDir(dir) {
			return
		}
	}

	initError = fmt.Errorf("failed to load MLX dynamic library (searched: %v)", searchDirs)
	slog.Warn("MLX dynamic library not available", "error", initError)
}
