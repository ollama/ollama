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
	"sort"
	"strconv"
	"strings"
	"unsafe"
)

var (
	initError      error
	initLoadError  string
	initLoadedPath string
)

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

// LoadedLibraryPath returns the MLX dynamic library path selected by this package.
func LoadedLibraryPath() (string, error) {
	if initLoadedPath != "" {
		return initLoadedPath, nil
	}
	if initError != nil {
		return "", initError
	}
	return "", fmt.Errorf("MLX dynamic library not loaded")
}

// tryLoadFromDir searches a directory for the mlxc shared library and loads it.
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

		initLoadedPath = path
		return true
	}
	return false
}

// libOllamaRoots returns candidate directories for MLX dynamic libraries.
// Production: exe_dir/lib/ollama (Windows release layout),
// exe_dir/../lib/ollama (standard bin/lib layout), and exe_dir (macOS bundle).
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
			roots = append(roots, filepath.Join(exeDir, "..", "lib", "ollama"))
			roots = append(roots, exeDir) // app bundle: Contents/Resources/
		case "linux":
			roots = append(roots, filepath.Join(exeDir, "..", "lib", "ollama"))
		case "windows":
			roots = append(roots, filepath.Join(exeDir, "lib", "ollama"))
			roots = append(roots, filepath.Join(exeDir, "..", "lib", "ollama"))
		}
	}

	// Development paths: build/lib/ollama and build/*/lib/ollama.
	// Reverse-sort and filter the glob results so higher-versioned Metal
	// builds (e.g., metal-v4) are tried before lower ones (metal-v3),
	// and incompatible variants are skipped. Without this, alphabetical
	// order would always pick v3 over v4 in dev builds.
	for _, base := range repoBuildDirs() {
		platform := runtime.GOOS + "-" + runtime.GOARCH
		platformAlt := runtime.GOOS + "_" + runtime.GOARCH
		roots = append(roots, filepath.Join(base, "lib", "ollama"))
		if matches, err := filepath.Glob(filepath.Join(base, "*", "lib", "ollama")); err == nil {
			sort.Sort(sort.Reverse(sort.StringSlice(matches)))
			for _, m := range matches {
				// Extract the build dir name (e.g., "metal-v4" from "build/metal-v4/lib/ollama")
				rel, _ := filepath.Rel(base, m)
				variant := strings.SplitN(rel, string(filepath.Separator), 2)[0]
				if isCompatibleMLXVariant(variant) {
					roots = append(roots, m)
				}
			}
		}
		if matches, err := filepath.Glob(filepath.Join(base, platform, "*", "lib", "ollama")); err == nil {
			sort.Sort(sort.Reverse(sort.StringSlice(matches)))
			for _, m := range matches {
				variant := filepath.Base(filepath.Dir(filepath.Dir(m)))
				if isCompatibleMLXVariant(variant) {
					roots = append(roots, m)
				}
			}
		}
		repoRoot := filepath.Dir(base)
		roots = append(roots, filepath.Join(repoRoot, "dist", platform, "lib", "ollama"))
		roots = append(roots, filepath.Join(repoRoot, "dist", platformAlt, "lib", "ollama"))
		if runtime.GOOS == "darwin" {
			roots = append(roots, filepath.Join(repoRoot, "dist", "darwin", "lib", "ollama"))
		}
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

func libraryPathEnvVar() string {
	switch runtime.GOOS {
	case "windows":
		return "PATH"
	case "darwin":
		return "DYLD_LIBRARY_PATH"
	case "linux":
		return "LD_LIBRARY_PATH"
	default:
		return ""
	}
}

// prependLibraryPaths prepends dirs to the platform's dynamic library search
// path so MLX and any shared sibling backend dependencies resolve before stale
// system copies. It returns a restore function for failed speculative loads.
func prependLibraryPaths(dirs ...string) func() {
	envVar := libraryPathEnvVar()
	if envVar == "" {
		return func() {}
	}
	oldValue, hadValue := os.LookupEnv(envVar)

	var paths []string
	seen := map[string]bool{}
	add := func(path string) {
		if path == "" {
			return
		}
		path = filepath.Clean(path)
		if !seen[path] {
			seen[path] = true
			paths = append(paths, path)
		}
	}
	for _, dir := range dirs {
		add(dir)
	}
	if existing := os.Getenv(envVar); existing != "" {
		for _, dir := range filepath.SplitList(existing) {
			add(dir)
		}
	}
	if len(paths) == 0 {
		return func() {}
	}
	os.Setenv(envVar, strings.Join(paths, string(filepath.ListSeparator)))
	return func() {
		if hadValue {
			os.Setenv(envVar, oldValue)
		} else {
			os.Unsetenv(envVar)
		}
	}
}

func mlxLibraryPathsForDir(dir string) []string {
	paths := []string{dir}
	if depDir, ok := dependencyDir(dir); ok {
		paths = append(paths, depDir)
	}
	return paths
}

func tryLoadFromMLXDir(dir string) bool {
	restore := prependLibraryPaths(mlxLibraryPathsForDir(dir)...)
	if tryLoadFromDir(dir) {
		return true
	}
	restore()
	return false
}

func prependLoadedLibraryPath() {
	if initLoadedPath == "" {
		return
	}
	dir := filepath.Dir(initLoadedPath)
	if strings.HasPrefix(filepath.Base(dir), "mlx_") {
		prependLibraryPaths(mlxLibraryPathsForDir(dir)...)
		setBundledCUDAToolkitPath(dir)
	} else {
		prependLibraryPaths(dir)
	}
}

func setBundledCUDAToolkitPath(dir string) {
	cudaPath, ok := cudaToolkitDirForMLXDir(dir)
	if !ok {
		return
	}
	os.Setenv("CUDA_PATH", cudaPath)
	os.Setenv("CUDA_HOME", cudaPath)
	slog.Debug("MLX CUDA headers", "CUDA_PATH", cudaPath)
}

func init() {
	switch runtime.GOOS {
	case "darwin", "linux", "windows":
	default:
		return
	}

	// OLLAMA_LLM_LIBRARY overrides variant selection (e.g., "mlx_metal_v3").
	// When set to an mlx_* value, only that specific subdir is tried.
	// The GGML runner ignores mlx_* values (see discover/runner.go).
	forcedVariant, _ := os.LookupEnv("OLLAMA_LLM_LIBRARY")
	if forcedVariant != "" && !strings.HasPrefix(forcedVariant, "mlx_") {
		forcedVariant = "" // not an MLX variant, ignore
	}

	found := findMLXLibrary(forcedVariant)
	if !found {
		initError = fmt.Errorf("failed to load MLX dynamic library (searched: %v)", libOllamaRoots())
		return
	}

	prependLoadedLibraryPath()
}

func findMLXLibrary(forcedVariant string) bool {
	for _, root := range libOllamaRoots() {
		if forcedVariant != "" {
			if tryLoadFromMLXDir(filepath.Join(root, forcedVariant)) {
				return true
			}
		} else {
			if tryLoadFromMLXSubdirs(root) {
				return true
			}
			if tryLoadFromStandaloneDir(root) {
				return true
			}
		}
	}
	return false
}

func tryLoadFromStandaloneDir(dir string) bool {
	return tryLoadFromDir(dir)
}

// tryLoadFromMLXSubdirs globs for mlx_* subdirs within dir, filters out
// incompatible variants, tries the remainder in reverse sorted order (so
// higher-versioned variants are preferred), and returns true on first
// successful load.
func tryLoadFromMLXSubdirs(dir string) bool {
	mlxDirs, err := filepath.Glob(filepath.Join(dir, "mlx_*"))
	if err != nil || len(mlxDirs) == 0 {
		return false
	}
	// Reverse sort: mlx_metal_v4 before mlx_metal_v3, mlx_cuda_v13 before v12
	sort.Sort(sort.Reverse(sort.StringSlice(mlxDirs)))
	for _, mlxDir := range mlxDirs {
		if !isCompatibleMLXVariant(filepath.Base(mlxDir)) {
			slog.Debug("skipping incompatible MLX variant", "dir", mlxDir)
			continue
		}
		if tryLoadFromMLXDir(mlxDir) {
			return true
		}
	}
	return false
}

// isCompatibleMLXVariant checks whether an MLX variant directory is
// compatible with the current OS. On macOS, dlopen does NOT enforce
// the deployment target for dynamically loaded libraries, so we must
// check compatibility ourselves to avoid loading Metal 4.x shaders
// on a Metal 3.x driver.
func isCompatibleMLXVariant(name string) bool {
	if runtime.GOOS != "darwin" {
		return true // non-macOS variants use dlopen failure for filtering
	}
	// Metal variant naming:
	//   Production: mlx_metal_v3, mlx_metal_v4
	//   Dev build:  metal-v3, metal-v4
	var verStr string
	switch {
	case strings.HasPrefix(name, "mlx_metal_v"):
		verStr = strings.TrimPrefix(name, "mlx_metal_v")
	case strings.HasPrefix(name, "metal-v"):
		verStr = strings.TrimPrefix(name, "metal-v")
	}
	if verStr != "" {
		metalVer, err := strconv.Atoi(verStr)
		if err != nil {
			return true // unknown format, try it
		}
		// Metal 4.x requires macOS 26+
		if metalVer >= 4 && macOSMajorVersion() < 26 {
			return false
		}
	}
	return true
}
