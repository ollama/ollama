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

func init() {
	switch runtime.GOOS {
	case "darwin":

	case "windows":
	default:
		return
	}

	paths, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH")
	if !ok {
		slog.Debug("OLLAMA_LIBRARY_PATH not set, skipping mlx dynamic loading")
		return
	}

	for _, path := range filepath.SplitList(paths) {
		matches, err := fs.Glob(os.DirFS(path), "libmlxc.*")
		if err != nil {
			initError = fmt.Errorf("failed to glob for MLX libraries in %s: %w", path, err)
			slog.Warn("MLX dynamic library not available", "error", initError)
			return
		}

		for _, match := range matches {
			path := filepath.Join(paths, match)
			slog.Info("Loading MLX dynamic library", "path", path)

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

			slog.Info("Loaded MLX dynamic library", "path", path)
			return
		}
	}

	initError = fmt.Errorf("failed to load any MLX dynamic library from OLLAMA_LIBRARY_PATH=%s", paths)
	slog.Warn("MLX dynamic library not available", "error", initError)
}
