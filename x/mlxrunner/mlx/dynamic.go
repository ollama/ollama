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
	"strconv"
	"strings"
	"unsafe"
)

var initError error
var initLoadError string
var initLoadedPath string

// CheckInit returns any error that occurred during MLX dynamic library initialization.
func CheckInit() error {
	if initLoadedPath != "" {
		slog.Debug("MLX dynamic library loaded", "path", initLoadedPath)
	}
	if initError != nil && initLoadError != "" {
		slog.Error(initLoadError)
	}
	return initError
}

// tryLoadFromDir searches a directory for the mlxc shared library and loads it.
func tryLoadFromDir(dir string) bool {
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

		initLoadedPath = path
		// Prepend dir to DYLD_LIBRARY_PATH so the dynamic linker finds the
		// colocated libmlx.dylib before any stale copies on the search path.
		if existing := os.Getenv("DYLD_LIBRARY_PATH"); existing != "" {
			os.Setenv("DYLD_LIBRARY_PATH", filepath.Dir(path)+string(filepath.ListSeparator)+existing)
		} else {
			os.Setenv("DYLD_LIBRARY_PATH", filepath.Dir(path))
		}
		return true
	}
	return false
}

// metalVariantName returns the MLX Metal library variant to load.
func metalVariantName() string {
	if requested, ok := os.LookupEnv("OLLAMA_LLM_LIBRARY"); ok && requested != "" {
		return requested
	}
	if runtime.GOOS == "darwin" && macOSMajorVersion() >= 26 {
		return "mlx_metal_v4"
	}
	return "mlx_metal_v3"
}

func macOSMajorVersion() int {
	data, err := os.ReadFile("/System/Library/CoreServices/SystemVersion.plist")
	if err != nil {
		return 0
	}
	s := string(data)
	idx := strings.Index(s, "<key>ProductVersion</key>")
	if idx < 0 {
		return 0
	}
	rest := s[idx:]
	start := strings.Index(rest, "<string>")
	end := strings.Index(rest, "</string>")
	if start < 0 || end < 0 || start >= end {
		return 0
	}
	parts := strings.Split(rest[start+8:end], ".")
	if len(parts) == 0 {
		return 0
	}
	major, _ := strconv.Atoi(parts[0])
	return major
}

// libOllamaRoots returns candidate directories for MLX dynamic libraries.
// Production: exe_dir/lib/ollama (dist tarball) and exe_dir (app bundle).
// Development: build/lib/ollama and build/*/lib/ollama.
func libOllamaRoots() []string {
	var roots []string

	// Production paths relative to executable
	if exe, err := os.Executable(); err == nil {
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		exeDir := filepath.Dir(exe)
		switch runtime.GOOS {
		case "darwin":
			roots = append(roots, filepath.Join(exeDir, "lib", "ollama"))
			roots = append(roots, exeDir) // app bundle: Contents/Resources/
		case "linux":
			roots = append(roots, filepath.Join(exeDir, "..", "lib", "ollama"))
		case "windows":
			roots = append(roots, filepath.Join(exeDir, "lib", "ollama"))
		}
	}

	// Development paths: build/lib/ollama and build/*/lib/ollama
	for _, base := range repoBuildDirs() {
		roots = append(roots, filepath.Join(base, "lib", "ollama"))
		if matches, err := filepath.Glob(filepath.Join(base, "*", "lib", "ollama")); err == nil {
			roots = append(roots, matches...)
		}
	}

	// OLLAMA_LIBRARY_PATH override
	if paths, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH"); ok {
		roots = append(roots, filepath.SplitList(paths)...)
	}

	return roots
}

// repoBuildDirs returns candidate build/ directories relative to cwd and repo root.
func repoBuildDirs() []string {
	var dirs []string
	if cwd, err := os.Getwd(); err == nil {
		dirs = append(dirs, filepath.Join(cwd, "build"))
		for dir := cwd; ; {
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				if dir != cwd {
					dirs = append(dirs, filepath.Join(dir, "build"))
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
	return dirs
}

func init() {
	switch runtime.GOOS {
	case "darwin", "linux", "windows":
	default:
		return
	}

	variant := metalVariantName()
	roots := libOllamaRoots()

	// Try the exact variant subdir in each root
	for _, root := range roots {
		if tryLoadFromDir(filepath.Join(root, variant)) {
			return
		}
	}

	// Fallback: try root dirs directly (no subdir).
	// Handles dev builds where cmake outputs libmlxc.dylib flat to
	// build/lib/ollama/ without an install step.
	for _, root := range roots {
		if tryLoadFromDir(root) {
			return
		}
	}

	initError = fmt.Errorf("failed to load MLX dynamic library (variant=%s, searched: %v)", variant, roots)
}
