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
var initLoadError string

// CheckInit returns any error that occurred during MLX dynamic library initialization.
// When initialization failed, detailed load errors are logged to help diagnose the issue.
func CheckInit() error {
	if initError != nil && initLoadError != "" {
		slog.Error(initLoadError)
	}
	return initError
}

// tryLoadFromDir searches a directory for the mlxc shared library and tries to load it.
// Returns true if the library was successfully loaded.
func tryLoadFromDir(dir string) bool {
	// On Windows, MSVC produces mlxc.dll (no lib prefix)
	// On Unix, it's libmlxc.so or libmlxc.dylib
	pattern := "libmlxc.*"
	if runtime.GOOS == "windows" {
		pattern = "mlxc.*"
	}
	matches, err := fs.Glob(os.DirFS(dir), pattern)
	if err != nil || len(matches) == 0 {
		return false
	}

	for _, match := range matches {
		path := filepath.Join(dir, match)

		cPath := C.CString(path)
		defer C.free(unsafe.Pointer(cPath))

		var handle C.mlx_dynamic_handle
		if C.mlx_dynamic_load(&handle, cPath) != 0 {
			initLoadError = fmt.Sprintf("failed to load MLX dynamic library: path=%s", path)
			continue
		}

		if C.mlx_dynamic_load_symbols(handle) != 0 {
			initLoadError = fmt.Sprintf("failed to load MLX dynamic library symbols: path=%s", path)
			C.mlx_dynamic_unload(&handle)
			continue
		}

		return true
	}
	return false
}

// tryLoadByName attempts to load the library using just its name,
// allowing the system to use rpath, LD_LIBRARY_PATH, or standard search paths.
// Returns true if the library was successfully loaded.
func tryLoadByName() bool {
	libraryName := "libmlxc.dylib"
	switch runtime.GOOS {
	case "windows":
		libraryName = "mlxc.dll"
	case "linux":
		libraryName = "libmlxc.so"
	}

	cPath := C.CString(libraryName)
	defer C.free(unsafe.Pointer(cPath))

	var handle C.mlx_dynamic_handle
	if C.mlx_dynamic_load(&handle, cPath) != 0 {
		return false
	}
	if C.mlx_dynamic_load_symbols(handle) != 0 {
		C.mlx_dynamic_unload(&handle)
		return false
	}

	return true
}

func init() {
	switch runtime.GOOS {
	case "darwin", "linux", "windows":

	default:
		return
	}

	// Try OLLAMA_LIBRARY_PATH first, including mlx_* subdirectories
	if paths, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH"); ok {
		for _, dir := range filepath.SplitList(paths) {
			if tryLoadFromDir(dir) {
				return
			}
			if mlxDirs, err := filepath.Glob(filepath.Join(dir, "mlx_*")); err == nil {
				for _, mlxDir := range mlxDirs {
					if tryLoadFromDir(mlxDir) {
						return
					}
				}
			}
		}
	}

	// Try loading via rpath/standard library search
	if tryLoadByName() {
		return
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

		// Walk up from cwd to find the repo root (containing go.mod) so that
		// tests running from a package subdirectory can find the build output.
		for dir := cwd; ; {
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				if dir != cwd {
					searchDirs = append(searchDirs, filepath.Join(dir, "build", "lib", "ollama"))
				}
				break
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}

	// Also scan mlx_* subdirectories within each search dir
	var expanded []string
	for _, dir := range searchDirs {
		expanded = append(expanded, dir)
		if mlxDirs, err := filepath.Glob(filepath.Join(dir, "mlx_*")); err == nil {
			expanded = append(expanded, mlxDirs...)
		}
	}

	for _, dir := range expanded {
		if tryLoadFromDir(dir) {
			return
		}
	}

	initError = fmt.Errorf("failed to load MLX dynamic library (searched: %v)", searchDirs)
	slog.Debug("MLX dynamic library not available", "error", initError)
}
