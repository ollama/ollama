package oga

// #include "oga.h"
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

// CheckInit returns any error that occurred during ORT GenAI dynamic library initialization.
func CheckInit() error {
	if initError != nil && initLoadError != "" {
		slog.Error(initLoadError)
	}
	return initError
}

// tryLoadFromDir searches a directory for the onnxruntime-genai shared library and tries to load it.
func tryLoadFromDir(dir string) bool {
	pattern := "onnxruntime-genai.*"
	if runtime.GOOS != "windows" {
		pattern = "libonnxruntime-genai.*"
	}
	matches, err := fs.Glob(os.DirFS(dir), pattern)
	if err != nil || len(matches) == 0 {
		return false
	}

	for _, match := range matches {
		path := filepath.Join(dir, match)

		cPath := C.CString(path)
		defer C.free(unsafe.Pointer(cPath))

		var handle C.oga_dynamic_handle
		if C.oga_dynamic_load(&handle, cPath) != 0 {
			initLoadError = fmt.Sprintf("failed to load ORT GenAI dynamic library: path=%s", path)
			continue
		}

		if C.oga_dynamic_load_symbols(handle) != 0 {
			initLoadError = fmt.Sprintf("failed to load ORT GenAI dynamic library symbols: path=%s", path)
			C.oga_dynamic_unload(&handle)
			continue
		}

		slog.Info("loaded ORT GenAI dynamic library", "path", path)
		return true
	}
	return false
}

// tryLoadByName attempts to load the library using just its name,
// allowing the system to use standard search paths.
func tryLoadByName() bool {
	libraryName := "onnxruntime-genai.dll"
	switch runtime.GOOS {
	case "linux":
		libraryName = "libonnxruntime-genai.so"
	case "darwin":
		libraryName = "libonnxruntime-genai.dylib"
	}

	cPath := C.CString(libraryName)
	defer C.free(unsafe.Pointer(cPath))

	var handle C.oga_dynamic_handle
	if C.oga_dynamic_load(&handle, cPath) != 0 {
		return false
	}
	if C.oga_dynamic_load_symbols(handle) != 0 {
		C.oga_dynamic_unload(&handle)
		return false
	}

	slog.Info("loaded ORT GenAI dynamic library", "name", libraryName)
	return true
}

func init() {
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
		return
	}

	// Try OLLAMA_ORT_PATH first
	if path, ok := os.LookupEnv("OLLAMA_ORT_PATH"); ok {
		for _, dir := range filepath.SplitList(path) {
			if tryLoadFromDir(dir) {
				return
			}
		}
	}

	// Try OLLAMA_LIBRARY_PATH, including ortgenai subdirectories
	if paths, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH"); ok {
		for _, dir := range filepath.SplitList(paths) {
			if tryLoadFromDir(dir) {
				return
			}
			if ortDirs, err := filepath.Glob(filepath.Join(dir, "ortgenai*")); err == nil {
				for _, ortDir := range ortDirs {
					if tryLoadFromDir(ortDir) {
						return
					}
				}
			}
		}
	}

	// Try loading via standard library search
	if tryLoadByName() {
		return
	}

	// Build search paths: executable directory, then build directories
	var searchDirs []string
	if exe, err := os.Executable(); err == nil {
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		exeDir := filepath.Dir(exe)
		searchDirs = append(searchDirs, exeDir)
		searchDirs = append(searchDirs, filepath.Join(exeDir, "ortgenai"))
	}

	if cwd, err := os.Getwd(); err == nil {
		searchDirs = append(searchDirs, filepath.Join(cwd, "build", "lib", "ollama"))
		searchDirs = append(searchDirs, filepath.Join(cwd, "build", "lib", "ollama", "ortgenai"))

		for dir := cwd; ; {
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				if dir != cwd {
					searchDirs = append(searchDirs, filepath.Join(dir, "build", "lib", "ollama"))
					searchDirs = append(searchDirs, filepath.Join(dir, "build", "lib", "ollama", "ortgenai"))
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

	for _, dir := range searchDirs {
		if tryLoadFromDir(dir) {
			return
		}
	}

	initError = fmt.Errorf("failed to load ORT GenAI dynamic library (searched: %v)", searchDirs)
	slog.Debug("ORT GenAI dynamic library not available", "error", initError)
}
